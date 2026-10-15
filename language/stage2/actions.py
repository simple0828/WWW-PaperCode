"""Shared numeric parsers for Stage2 residual branches.

Moved from scripts so domain CLIs share one implementation.
Field name ``vmt_effect`` is legacy (= effect on the *target series*).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from language.lamp_vmt.json_utils import extract_json

PLA_W = {"high": 1.0, "medium": 0.6, "low": 0.3}
BIAS = {
    "strong_up": 0.10,
    "up": 0.05,
    "flat": 0.0,
    "down": -0.05,
    "strong_down": -0.10,
    "increase": 0.06,
    "higher": 0.06,
    "decrease": -0.06,
    "lower": -0.06,
    "mixed": 0.0,
    "volatile": 0.0,
}
STRENGTH = {"weak": 0.5, "medium": 1.0, "strong": 1.5, "high": 1.4}


def soft(ev: dict) -> float:
    mechs = ev.get("mechanisms") or []
    if not mechs:
        return {"up": 1.0, "down": -1.0}.get(str(ev.get("vmt_effect") or ""), 0.0)
    num = den = 0.0
    for m in mechs:
        w = PLA_W.get(str(m.get("plausibility") or "medium"), 0.5)
        num += w * {"up": 1.0, "down": -1.0}.get(str(m.get("vmt_effect") or ""), 0.0)
        den += w
    return num / den if den else 0.0


def parse_json(text: str) -> dict:
    obj: Any = None
    try:
        obj = extract_json(text) if text else {}
    except Exception:
        try:
            obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception:
            obj = {}
    return obj if isinstance(obj, dict) else {}


def parse_action(text: str, H: int = 6) -> dict:
    obj = parse_json(text)
    gate = int(obj.get("gate") or 0)
    if gate not in (0, 1):
        gate = 1 if re.search(r'"gate"\s*:\s*1', text or "") else 0
    try:
        conf = float(obj.get("confidence") or 0.5)
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    hb = obj.get("horizon_bias")
    if isinstance(hb, str):
        tok = hb.lower().replace(" ", "_")
        hb = [tok] * H if tok in BIAS else ["flat"] * H
    elif isinstance(hb, list):
        hb = [str(x).lower().replace(" ", "_") for x in hb]
    else:
        hb = ["flat"] * H
    while len(hb) < H:
        hb.append("flat")
    hb = [x if x in BIAS else "flat" for x in hb[:H]]
    vec = np.array([BIAS[x] for x in hb], float)
    if gate == 0:
        vec[:] = 0.0
    else:
        if conf < 0.6:
            gate = 0
            vec[:] = 0.0
        else:
            vec *= 0.35 + 0.65 * conf
    return {
        "gate": gate,
        "conf": conf,
        "vec": vec,
        "hb": hb,
        "ok": bool(obj),
        "cot": str(obj.get("cot") or obj.get("step2") or "")[:220],
        "novel": str(obj.get("novel_inference") or obj.get("forward") or "")[:220],
        "raw": obj,
    }
