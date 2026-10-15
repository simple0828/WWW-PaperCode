from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from data_provider.data_factory import data_provider
from language.apo.baseline import BaselineSample, UnimodalBaselinePredictor
from language.apo.bert_embed import BertEmbeddingPath
from language.apo.fusion_mlp import FusionMLP
from language.apo.reward import RewardBreakdown, RewardMode, compute_reward
from models import PatchTST


class CFADownstreamScorer:
    """Score text via official CFA path: BERT embeddings → MLP → PatchTST(CFA)."""

    def __init__(
        self,
        args,
        bundle_path: str | Path,
        baseline: Optional[UnimodalBaselinePredictor] = None,
        device: Optional[str] = None,
        *,
        reward_mode: RewardMode = "cfa_text_delta",
    ):
        self.args = args
        self.reward_mode: RewardMode = reward_mode
        self.device = torch.device(
            device
            or (f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
        )
        self.bundle_path = Path(bundle_path)
        self.baseline = baseline or UnimodalBaselinePredictor(args, device=str(self.device))
        self.pool_type = getattr(args, "pool_type", "avg")
        self._datasets = {}
        self._cfa_raw_pred_cache: dict[tuple[str, int], np.ndarray] = {}
        self._load_bundle(self.bundle_path)

    def _load_bundle(self, bundle_path: Path) -> None:
        bundle = torch.load(bundle_path, map_location=self.device)
        enc_type = bundle.get("encoder_type", "bert_embedding")
        if enc_type not in ("bert_embedding", "bert"):
            raise ValueError(
                f"Bundle encoder_type={enc_type!r}; expected bert_embedding. "
                "Export with language.apo.runpy_cfa.export_runpy_cfa_bundle."
            )
        self.bert = BertEmbeddingPath(
            model_name=bundle.get("encoder_name", "google-bert/bert-base-uncased"),
            max_length=int(bundle.get("max_length", 1536)),
            device=self.device,
        ).to(self.device)

        llm_dim = int(bundle.get("llm_dim", 768))
        text_emb = int(bundle.get("text_emb", self.args.pred_len))
        self.mlp = FusionMLP(llm_dim, max(1, llm_dim // 8), text_emb, dropout=0.3).to(
            self.device
        )
        self.mlp.load_state_dict(bundle["mlp"])
        self.mlp.eval()

        cfa_args = deepcopy(self.args)
        cfa_args.text_injection_mode = "cfa"
        cfa_args.text_emb = text_emb
        self.model = PatchTST.Model(cfa_args).float().to(self.device)
        self.model.load_state_dict(bundle["patchtst"], strict=True)
        self.model.eval()
        self._llm_dim = llm_dim
        self._text_emb = text_emb

    def reload_bundle(self, bundle_path: str | Path | None = None) -> None:
        path = Path(bundle_path) if bundle_path is not None else self.bundle_path
        self.bundle_path = path
        self._cfa_raw_pred_cache.clear()
        self._load_bundle(path)

    def ensure_cfa_raw_cache(
        self,
        samples: list[BaselineSample],
        *,
        batch_size: int = 16,
    ) -> None:
        """Cache CFA predictions on raw facts (reference for cfa_text_delta reward)."""
        missing = [
            s
            for s in samples
            if (s.split, s.window_index) not in self._cfa_raw_pred_cache
        ]
        if not missing:
            return
        preds = self.predict_fusion_batch(
            missing,
            [s.raw_text for s in missing],
            batch_size=batch_size,
        )
        for sample, pred in zip(missing, preds):
            self._cfa_raw_pred_cache[(sample.split, sample.window_index)] = pred

    def _reference_predictions(
        self,
        samples: list[BaselineSample],
        *,
        batch_size: int = 16,
    ) -> list[np.ndarray]:
        self.ensure_cfa_raw_cache(samples, batch_size=batch_size)
        return [
            np.asarray(
                self._cfa_raw_pred_cache[(s.split, s.window_index)], dtype=float
            )
            for s in samples
        ]

    def _pool(self, prompt_emb: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "max":
            return prompt_emb.max(dim=1)[0]
        if self.pool_type == "min":
            return prompt_emb.min(dim=1)[0]
        return prompt_emb.mean(dim=1)

    def _text_context_batch(self, texts: list[str]) -> torch.Tensor:
        hidden = self.bert.encode_token_embeddings(list(texts))
        return self._pool(self.mlp(hidden))

    def _get_dataset(self, split: str):
        if split not in self._datasets:
            self._datasets[split], _ = data_provider(self.args, split)
        return self._datasets[split]

    def predict_fusion(
        self,
        sample: BaselineSample,
        rewrite_text: str,
        dataset=None,
    ) -> np.ndarray:
        return self.predict_fusion_batch([sample], [rewrite_text])[0]

    def predict_fusion_batch(
        self,
        samples: list[BaselineSample],
        rewrite_texts: list[str],
        *,
        batch_size: int = 16,
    ) -> list[np.ndarray]:
        if len(samples) != len(rewrite_texts):
            raise ValueError("samples and rewrite_texts length mismatch")
        if not samples:
            return []

        out: list[np.ndarray] = []
        for start in range(0, len(samples), max(1, batch_size)):
            chunk_samples = samples[start : start + batch_size]
            chunk_texts = rewrite_texts[start : start + batch_size]
            by_split: dict[str, list[int]] = {}
            for i, sample in enumerate(chunk_samples):
                by_split.setdefault(sample.split, []).append(i)

            chunk_preds: list[np.ndarray | None] = [None] * len(chunk_samples)
            for split, idxs in by_split.items():
                dataset = self._get_dataset(split)
                xs, ys, xms, yms, texts = [], [], [], [], []
                for i in idxs:
                    sample = chunk_samples[i]
                    seq_x, seq_y, seq_x_mark, seq_y_mark, _ = dataset[sample.window_index]
                    xs.append(seq_x)
                    ys.append(seq_y)
                    xms.append(seq_x_mark)
                    yms.append(seq_y_mark)
                    texts.append(chunk_texts[i])

                batch_x = torch.tensor(np.stack(xs), dtype=torch.float32).to(self.device)
                batch_y = torch.tensor(np.stack(ys), dtype=torch.float32).to(self.device)
                batch_x_mark = torch.tensor(np.stack(xms), dtype=torch.float32).to(self.device)
                batch_y_mark = torch.tensor(np.stack(yms), dtype=torch.float32).to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                text_context = self._text_context_batch(texts)
                with torch.no_grad():
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        text_context=text_context,
                    )
                pred = outputs[:, -self.args.pred_len :, :].detach().cpu().numpy()
                if dataset.scale and getattr(self.args, "inverse", False):
                    bsz, tlen, cdim = pred.shape
                    pred = dataset.inverse_transform(pred.reshape(-1, cdim)).reshape(
                        bsz, tlen, cdim
                    )
                for j, i in enumerate(idxs):
                    chunk_preds[i] = pred[j].squeeze()

            missing = [i for i, p in enumerate(chunk_preds) if p is None]
            if missing:
                raise RuntimeError(f"Missing CFA preds for chunk indices {missing}")
            out.extend(chunk_preds)  # type: ignore[arg-type]
        return out

    def score(self, sample: BaselineSample, rewrite_text: str) -> RewardBreakdown:
        y_fusion = self.predict_fusion(sample, rewrite_text)
        y_ref = self._reference_predictions([sample])[0]
        return compute_reward(
            sample.y_true,
            y_ref,
            y_fusion,
            scorer="cfa_bert",
            reward_mode=self.reward_mode,
        )

    def score_batch(
        self,
        samples: list[BaselineSample],
        rewrite_texts: list[str],
        *,
        batch_size: int = 16,
    ) -> list[RewardBreakdown]:
        preds = self.predict_fusion_batch(samples, rewrite_texts, batch_size=batch_size)
        refs = self._reference_predictions(samples, batch_size=batch_size)
        return [
            compute_reward(
                s.y_true,
                ref,
                y,
                scorer="cfa_bert",
                reward_mode=self.reward_mode,
            )
            for s, ref, y in zip(samples, refs, preds)
        ]

    def compute_split_mse(
        self,
        samples: list[BaselineSample],
        rewrite_texts: list[str],
        *,
        batch_size: int = 16,
    ) -> float:
        """Mean squared error on explicit windows (scaled space).

        Use the same window indices as run.py Exp.test (see eval_windows.py)
        so results are comparable to evaluate_cfa_test_mse.
        """
        if len(samples) != len(rewrite_texts):
            raise ValueError("samples and rewrite_texts length mismatch")
        if not samples:
            return float("nan")
        preds = self.predict_fusion_batch(
            samples, rewrite_texts, batch_size=batch_size
        )
        sq = []
        for sample, pred in zip(samples, preds):
            sq.append((pred - sample.y_true) ** 2)
        return float(np.mean(np.concatenate(sq)))

    def check_truncation(self, text: str) -> dict:
        full = self.bert.count_tokens(text, truncation=False)
        used = self.bert.count_tokens(text, truncation=True)
        return {
            "encoder": "bert_embedding",
            "model_name": self.bert.model_name,
            "max_length": self.bert.max_length,
            "token_count": full,
            "truncated": full > self.bert.max_length,
            "used_tokens": used,
        }
