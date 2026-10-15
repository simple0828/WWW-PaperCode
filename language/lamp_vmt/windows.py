"""Align cleaned monthly texts to CFA forecast windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from language.apo.baseline import UnimodalBaselinePredictor, _coerce_text_cell
from language.lamp_vmt.residual import pad_residual
from language.lamp_vmt.schema import PRED_LEN, SEQ_LEN


def load_cleaned_by_date(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            date = str(row.get("date") or "")[:10]
            text = row.get("refined_text") or row.get("text") or ""
            if date:
                out[date] = str(text)
    return out


def _norm_date(cell: Any) -> str:
    s = _coerce_text_cell(cell)
    if not s:
        return ""
    return s[:10]


def build_window_record(
    predictor: UnimodalBaselinePredictor,
    cleaned: dict[str, str],
    *,
    split: str,
    window_index: int,
) -> dict[str, Any]:
    sample = predictor.get_sample(split, window_index)
    ds = predictor.get_dataset(split)
    s_begin = window_index
    s_end = window_index + ds.seq_len

    month_blocks = []
    hist_dates = []
    for i in range(s_begin, s_end):
        d = _norm_date(ds.date[i])
        hist_dates.append(d)
        month_blocks.append({"date": d, "text": cleaned.get(d, "")})

    origin_date = (
        _norm_date(ds.date[s_end]) if s_end < len(ds.date) else sample.start_date[:10]
    )
    residual = pad_residual(sample.residual_true, PRED_LEN).tolist()

    return {
        "split": split,
        "window_index": window_index,
        "window_id": f"{split}:{window_index}",
        "seq_len": int(ds.seq_len),
        "pred_len": PRED_LEN,
        "hist_dates": hist_dates,
        "origin_date": origin_date,
        "start_date": sample.start_date,
        "end_date": sample.end_date,
        "month_blocks": month_blocks,
        "n_nonempty_months": sum(1 for b in month_blocks if (b.get("text") or "").strip()),
        "baseline_forecast": sample.baseline_forecast,
        "y_true": [float(x) for x in np.asarray(sample.y_true).reshape(-1)],
        "y_baseline": [float(x) for x in np.asarray(sample.y_baseline).reshape(-1)],
        "residual_true": [float(x) for x in residual],
        "raw_text_preview": (sample.raw_text or "")[:240],
    }


def iter_windows(
    predictor: UnimodalBaselinePredictor,
    cleaned: dict[str, str],
    *,
    split: str,
    limit: Optional[int] = None,
    stride: int = 1,
):
    ds = predictor.get_dataset(split)
    n = len(ds)
    idxs = list(range(0, n, max(1, stride)))
    if limit is not None and limit >= 0:
        idxs = idxs[:limit]
    for i in idxs:
        yield build_window_record(predictor, cleaned, split=split, window_index=i)


def assert_lengths(seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN) -> None:
    if seq_len != SEQ_LEN or pred_len != PRED_LEN:
        raise ValueError(
            f"Expected CFA lengths seq_len={SEQ_LEN}, pred_len={PRED_LEN}; "
            f"got {seq_len}, {pred_len}"
        )
