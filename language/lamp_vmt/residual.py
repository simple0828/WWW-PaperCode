"""Residual helpers for prototype assignment."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from language.lamp_vmt.schema import PRED_LEN


def pad_residual(residual: Sequence[float], pred_len: int = PRED_LEN) -> np.ndarray:
    arr = np.asarray(list(residual), dtype=float).reshape(-1)
    if arr.size < pred_len:
        pad = np.zeros(pred_len, dtype=float)
        pad[: arr.size] = arr
        return pad
    return arr[:pred_len]


def residual_rmse(a: Sequence[float], b: Sequence[float]) -> float:
    x = pad_residual(a)
    y = pad_residual(b)
    return float(np.sqrt(np.mean((x - y) ** 2)))
