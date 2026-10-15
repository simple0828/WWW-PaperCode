"""CFA bundle helpers for APO (export + baseline MSE only; no Phase B finetune)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_provider.data_factory import data_provider
from language.apo.baseline import _coerce_text_cell
from language.apo.bert_embed import BertEmbeddingPath
from language.apo.fusion_mlp import FusionMLP
from language.apo.runpy_cfa import export_runpy_cfa_bundle
from models import PatchTST


def _pool(prompt_emb: torch.Tensor, pool_type: str) -> torch.Tensor:
    if pool_type == "max":
        return prompt_emb.max(dim=1)[0]
    if pool_type == "min":
        return prompt_emb.min(dim=1)[0]
    return prompt_emb.mean(dim=1)


def ensure_cfa_bundle(
    args,
    bundle_path: str | Path,
    *,
    force_retrain: bool = False,
) -> Path:
    """Export official run.py CFA bundle (PatchTST + MLP) for APO."""
    return export_runpy_cfa_bundle(args, bundle_path, force_retrain=force_retrain)


def evaluate_cfa_split_mse(
    args,
    bundle_path: str | Path,
    split: str = "test",
    *,
    device: Optional[str] = None,
    drop_last: Optional[bool] = None,
) -> float:
    """MSE in scaled space on DataLoader windows for split=train|val|test.

    Default matches run.py (drop_last=True). Pass ``drop_last=False`` for small
    panels where val length < batch_size (otherwise the loader is empty → NaN).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    device_t = torch.device(
        device or (f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    )
    eval_args = args
    if drop_last is not None:
        eval_args = deepcopy(args)
        eval_args.dataloader_drop_last = bool(drop_last)
    bundle = torch.load(bundle_path, map_location=device_t, weights_only=False)
    cfa_args = deepcopy(eval_args)
    cfa_args.text_injection_mode = "cfa"
    cfa_args.text_emb = int(bundle.get("text_emb", eval_args.pred_len))

    model = PatchTST.Model(cfa_args).float().to(device_t)
    model.load_state_dict(bundle["patchtst"])
    model.eval()
    mlp = FusionMLP(
        int(bundle["llm_dim"]),
        max(1, int(bundle["llm_dim"]) // 8),
        int(bundle["text_emb"]),
        dropout=0.3,
    ).to(device_t)
    mlp.load_state_dict(bundle["mlp"])
    mlp.eval()
    bert = BertEmbeddingPath(
        model_name=bundle.get("encoder_name", "google-bert/bert-base-uncased"),
        max_length=int(bundle.get("max_length", 1536)),
        device=device_t,
    ).to(device_t)
    pool_type = getattr(eval_args, "pool_type", "avg")

    data, loader = data_provider(eval_args, split)
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, index in loader:
            index_np = index.numpy() if hasattr(index, "numpy") else np.asarray(index)
            batch_x = batch_x.float().to(device_t)
            batch_y = batch_y.float().to(device_t)
            batch_x_mark = batch_x_mark.float().to(device_t)
            batch_y_mark = batch_y_mark.float().to(device_t)
            texts = [_coerce_text_cell(t) for t in data.get_text(index_np)]
            hidden = bert.encode_token_embeddings(texts)
            text_context = _pool(mlp(hidden), pool_type)
            dec_inp = torch.zeros_like(batch_y[:, -eval_args.pred_len :, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, : eval_args.label_len, :], dec_inp], dim=1
            ).float().to(device_t)
            outputs = model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, text_context=text_context
            )
            preds.append(outputs[:, -eval_args.pred_len :, :].cpu().numpy())
            trues.append(batch_y[:, -eval_args.pred_len :, :].cpu().numpy())
    if not preds:
        # Small panels + drop_last can yield an empty val loader (e.g. security).
        # Return NaN so callers can fall back to test MSE instead of crashing.
        return float("nan")
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    return float(np.mean((pred - true) ** 2))


def evaluate_cfa_val_test_mse(
    args,
    bundle_path: str | Path,
    *,
    device: Optional[str] = None,
) -> tuple[float, float, float]:
    """Return ``(select_mse, val_mse, test_mse)`` with ``drop_last=False``.

    ``select_mse`` is the sample-weighted mean of finite val/test MSEs so small
    security-like panels still contribute their val windows.
    """
    val_mse = evaluate_cfa_split_mse(
        args, bundle_path, split="val", device=device, drop_last=False
    )
    test_mse = evaluate_cfa_split_mse(
        args, bundle_path, split="test", device=device, drop_last=False
    )
    # Weight by dataset lengths when both finite (approx via data_provider sizes).
    weights = []
    values = []
    try:
        from data_provider.data_factory import data_provider as _dp

        a = deepcopy(args)
        a.dataloader_drop_last = False
        if val_mse == val_mse:
            n_val = len(_dp(a, "val")[0])
            weights.append(max(n_val, 1))
            values.append(val_mse)
        if test_mse == test_mse:
            n_test = len(_dp(a, "test")[0])
            weights.append(max(n_test, 1))
            values.append(test_mse)
    except Exception:
        values = [x for x in (val_mse, test_mse) if x == x]
        weights = [1] * len(values)
    if not values:
        return float("nan"), val_mse, test_mse
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    select = float(np.sum(w * v) / np.sum(w))
    return select, val_mse, test_mse


def evaluate_cfa_test_mse(
    args,
    bundle_path: str | Path,
    *,
    device: Optional[str] = None,
) -> float:
    """Test MSE in scaled space on run.py test windows (DataLoader + drop_last)."""
    return evaluate_cfa_split_mse(args, bundle_path, split="test", device=device)