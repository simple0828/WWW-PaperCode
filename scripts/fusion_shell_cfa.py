#!/usr/bin/env python3
"""Shared CFA residual helpers: cache I/O + legacy soft_mech (deprecated).

Numeric handcrafted backends (A4 / soft_pool / graph) were removed — SoftLSTM
(or numeric LatentCoT) replaces them. Explicit candidate 1 is now
``language.stage2.scheme1_hier`` (event→window). ``soft_mech`` remains only
for old dual/lamp caches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mse(a, b) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def sign_mean(v) -> int:
    m = float(np.mean(np.asarray(v, float)))
    if abs(m) < 1e-12:
        return 0
    return 1 if m > 0 else -1


def load_cache(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            key = row.get("window_id") or row.get("key")
            if key is None:
                continue
            out[str(key)] = row["response"]
    return out


def soft_mech(dual_act, lamp_act) -> np.ndarray:
    """DEPRECATED — legacy S5 dual ⊕ LAMP vector blend.

    Prefer ``scheme1_hier.attach_scheme1``. Kept for old cache replay only.
    """
    gd, gL = dual_act["gate"], lamp_act["gate"]
    vd, vL = dual_act["vec"], lamp_act["vec"]
    if gd and gL:
        sd, sL = sign_mean(vd), sign_mean(vL)
        if sd != 0 and sL != 0 and sd != sL:
            return vd.copy()
        return 0.55 * vd + 0.45 * vL
    if gd:
        return vd.copy()
    if gL:
        return vL.copy()
    return np.zeros_like(vd)
