"""Phase B: finetune MLP + CFA-PatchTST on rewritten text (standalone module)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from language.apo.baseline import _coerce_text_cell
from language.apo.bert_embed import BertEmbeddingPath
from language.apo.fusion_mlp import FusionMLP
from models import PatchTST


def _pool(prompt_emb: torch.Tensor, pool_type: str) -> torch.Tensor:
    if pool_type == "max":
        return prompt_emb.max(dim=1)[0]
    if pool_type == "min":
        return prompt_emb.min(dim=1)[0]
    return prompt_emb.mean(dim=1)


def _resolve_batch_texts(
    dataset,
    indices: np.ndarray,
    text_overrides: Optional[Dict[int, str]],
) -> list[str]:
    raw = dataset.get_text(indices)
    out: list[str] = []
    for idx, text in zip(indices, raw):
        key = int(idx)
        if text_overrides is not None and key in text_overrides:
            out.append(text_overrides[key])
        else:
            out.append(_coerce_text_cell(text))
    return out


def _save_bundle(
    path: Path,
    *,
    model: PatchTST.Model,
    mlp: FusionMLP,
    bundle_meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "patchtst": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mlp": {k: v.detach().cpu() for k, v in mlp.state_dict().items()},
        "encoder_type": bundle_meta.get("encoder_type", "bert_embedding"),
        "encoder_name": bundle_meta.get("encoder_name", "google-bert/bert-base-uncased"),
        "max_length": int(bundle_meta.get("max_length", 1536)),
        "llm_dim": int(bundle_meta.get("llm_dim", 768)),
        "text_emb": int(bundle_meta.get("text_emb", 6)),
        "source": "phase_b_finetune",
    }
    torch.save(payload, path)


def _run_epoch(
    *,
    model,
    mlp,
    bert: BertEmbeddingPath,
    loader,
    dataset,
    args,
    device: torch.device,
    pool_type: str,
    text_overrides: Optional[Dict[int, str]],
    train: bool,
    optim_model=None,
    optim_mlp=None,
) -> float:
    if train:
        model.train()
        mlp.train()
    else:
        model.eval()
        mlp.eval()

    losses: list[float] = []
    pred_len = int(args.pred_len)
    criterion = nn.MSELoss()

    for batch_x, batch_y, batch_x_mark, batch_y_mark, index in loader:
        index_np = index.numpy() if hasattr(index, "numpy") else np.asarray(index)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        texts = _resolve_batch_texts(dataset, index_np, text_overrides)
        with torch.set_grad_enabled(train):
            hidden = bert.encode_token_embeddings(texts)
            prompt_emb = mlp(hidden)
            text_context = _pool(prompt_emb, pool_type)

            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(
                device
            )
            outputs = model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, text_context=text_context
            )
            pred = outputs[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            loss = criterion(pred, true)
            if train:
                optim_model.zero_grad()
                optim_mlp.zero_grad()
                loss.backward()
                optim_model.step()
                optim_mlp.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def finetune_cfa_bundle(
    args,
    init_bundle_path: str | Path,
    out_bundle_path: str | Path,
    *,
    train_epochs: int = 10,
    patience: int = 5,
    text_overrides: Optional[Dict[int, str]] = None,
    val_text_overrides: Optional[Dict[int, str]] = None,
    device: Optional[str] = None,
) -> Path:
    """Continue training MLP + CFA-PatchTST from a run.py (or prior phase-B) bundle.

    text_overrides: {train_window_index: rewritten_text}
    val_text_overrides: {val_window_index: rewritten_text}; defaults to text_overrides
        when omitted (early stopping on rewrite val).
    """
    init_bundle_path = Path(init_bundle_path)
    out_bundle_path = Path(out_bundle_path)
    val_overrides = val_text_overrides if val_text_overrides is not None else text_overrides

    device_t = torch.device(
        device
        or (f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    )
    bundle = torch.load(init_bundle_path, map_location=device_t, weights_only=False)
    pool_type = getattr(args, "pool_type", "avg")

    bert = BertEmbeddingPath(
        model_name=bundle.get("encoder_name", "google-bert/bert-base-uncased"),
        max_length=int(bundle.get("max_length", 1536)),
        device=device_t,
    ).to(device_t)

    llm_dim = int(bundle.get("llm_dim", 768))
    text_emb = int(bundle.get("text_emb", args.pred_len))
    mlp = FusionMLP(llm_dim, max(1, llm_dim // 8), text_emb, dropout=0.3).to(device_t)
    mlp.load_state_dict(bundle["mlp"])

    cfa_args = deepcopy(args)
    cfa_args.text_injection_mode = "cfa"
    cfa_args.text_emb = text_emb
    model = PatchTST.Model(cfa_args).float().to(device_t)
    model.load_state_dict(bundle["patchtst"])

    optim_model = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    optim_mlp = optim.Adam(mlp.parameters(), lr=float(args.learning_rate2))

    train_data, train_loader = data_provider(args, "train")
    val_data, val_loader = data_provider(args, "val")

    best_val = float("inf")
    best_model_state = None
    best_mlp_state = None
    wait = 0

    print(
        f"Phase B finetune: epochs={train_epochs} patience={patience} "
        f"train_override={len(text_overrides or {})} val_override={len(val_overrides or {})}",
        flush=True,
    )

    for epoch in range(train_epochs):
        train_loss = _run_epoch(
            model=model,
            mlp=mlp,
            bert=bert,
            loader=train_loader,
            dataset=train_data,
            args=args,
            device=device_t,
            pool_type=pool_type,
            text_overrides=text_overrides,
            train=True,
            optim_model=optim_model,
            optim_mlp=optim_mlp,
        )
        val_loss = _run_epoch(
            model=model,
            mlp=mlp,
            bert=bert,
            loader=val_loader,
            dataset=val_data,
            args=args,
            device=device_t,
            pool_type=pool_type,
            text_overrides=val_overrides,
            train=False,
        )
        print(
            f"  epoch {epoch + 1}/{train_epochs} train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_mlp_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch + 1}", flush=True)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        mlp.load_state_dict(best_mlp_state)

    _save_bundle(out_bundle_path, model=model, mlp=mlp, bundle_meta=bundle)
    print(f"Saved Phase B bundle → {out_bundle_path} (best_val={best_val:.6f})", flush=True)
    return out_bundle_path
