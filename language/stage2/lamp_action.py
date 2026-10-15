"""LAMP JSON → gated residual action (shared across domains)."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from language.lamp_vmt.json_utils import extract_json
from language.stage2.actions import STRENGTH


def parse_lamp_json(text: str) -> dict:
    """Robust parse (truncation repair) used by LAMP caches."""
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    obj: Any = {}
    try:
        obj = extract_json(t) if t else {}
    except Exception:
        try:
            obj = json.loads(t[t.find("{") : t.rfind("}") + 1])
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    if not obj and "{" in t:
        frag = t[t.find("{") :]
        if frag.count('"') % 2 == 1:
            frag += '"'
        frag = re.sub(r",\s*\"[^\"]*$", "", frag)
        frag = re.sub(r",\s*$", "", frag)
        while frag.count("{") > frag.count("}"):
            frag += "}"
        while frag.count("[") > frag.count("]"):
            frag += "]"
        try:
            obj = json.loads(frag)
        except Exception:
            try:
                obj = extract_json(frag)
            except Exception:
                obj = {}
    if not isinstance(obj, dict):
        obj = {}
    if "residual_hypothesis" not in obj:
        m = re.search(r'"residual_hypothesis"\s*:\s*"([^"]+)"', t)
        if m:
            obj["residual_hypothesis"] = m.group(1)
    if "gate" not in obj:
        m = re.search(r'"gate"\s*:\s*([01])', t)
        if m:
            obj["gate"] = int(m.group(1))
    if "confidence" not in obj:
        m = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
        if m:
            try:
                obj["confidence"] = float(m.group(1))
            except Exception:
                pass
    if not (obj.get("keep") or obj.get("drivers")):
        dirs = re.findall(
            r'"(?:dir|residual_dir|tag)"\s*:\s*"(higher|lower|increase|decrease|up|down|volatile)"',
            t,
            flags=re.I,
        )
        strengths = re.findall(
            r'"strength"\s*:\s*"(weak|medium|strong|high)"', t, flags=re.I
        )
        if dirs:
            items = []
            for i, d in enumerate(dirs[:6]):
                items.append(
                    {
                        "dir": d.lower(),
                        "residual_dir": d.lower(),
                        "tag": d.lower(),
                        "strength": (
                            strengths[i].lower() if i < len(strengths) else "medium"
                        ),
                    }
                )
            obj["keep"] = items
            if "residual_hypothesis" not in obj:
                up = sum(1 for d in dirs if d.lower() in ("higher", "increase", "up"))
                dn = sum(1 for d in dirs if d.lower() in ("lower", "decrease", "down"))
                obj["residual_hypothesis"] = (
                    "increase" if up > dn else ("decrease" if dn > up else "flat")
                )
                if "gate" not in obj:
                    obj["gate"] = 1 if abs(up - dn) >= 2 else 0
                if "confidence" not in obj:
                    obj["confidence"] = 0.65 if obj["gate"] else 0.45
    return obj


def hyp_to_bias(hyp: str) -> str:
    h = (hyp or "flat").strip().lower()
    if h in ("increase", "higher", "up"):
        return "up"
    if h in ("decrease", "lower", "down"):
        return "down"
    return "flat"


def action_from_lamp(text: str, H: int = 6) -> dict:
    obj = parse_lamp_json(text)
    items = obj.get("keep") or obj.get("drivers") or []
    if not isinstance(items, list):
        items = []
    hyp = str(obj.get("residual_hypothesis") or "").lower()
    if not hyp and items:
        up = dn = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            v = str(it.get("residual_dir") or it.get("dir") or it.get("tag") or "").lower()
            if v in ("increase", "higher", "up"):
                up += 1
            elif v in ("decrease", "lower", "down"):
                dn += 1
        if up > dn:
            hyp = "increase"
        elif dn > up:
            hyp = "decrease"
        else:
            hyp = "flat"
    if not hyp:
        hyp = "flat"
    try:
        conf = float(obj.get("confidence") or (0.65 if items else 0.45))
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0, 1))
    gate = obj.get("gate", None)
    if gate is None:
        gate = 1 if hyp in ("increase", "decrease") and items and conf >= 0.55 else 0
    try:
        gate = int(gate)
    except Exception:
        gate = 0
    if gate not in (0, 1):
        gate = 0
    strength = 1.0
    if items:
        ss = []
        for it in items:
            if isinstance(it, dict):
                ss.append(STRENGTH.get(str(it.get("strength") or "medium"), 1.0))
        if ss:
            strength = float(np.mean(ss))
    hb = obj.get("horizon_bias")
    if isinstance(hb, list) and len(hb) >= H:
        hb = [str(x).lower().replace(" ", "_") for x in hb[:H]]
        hb = [
            x
            if x in ("flat", "up", "down", "strong_up", "strong_down")
            else hyp_to_bias(hyp)
            for x in hb
        ]
    else:
        tok = hyp_to_bias(hyp)
        hb = [tok] * H
    amap = {
        "flat": 0.0,
        "up": 0.05,
        "down": -0.05,
        "strong_up": 0.09,
        "strong_down": -0.09,
    }
    vec = np.array([amap.get(x, 0.0) for x in hb], float) * strength
    if hyp in ("flat", "mixed", "volatile") or conf < 0.55:
        gate = 0
        vec[:] = 0.0
    if gate == 0:
        vec[:] = 0.0
    else:
        vec *= 0.35 + 0.65 * conf
    feat = np.array(
        [
            gate,
            conf,
            strength,
            1.0 if hyp in ("increase", "higher", "up") else 0.0,
            1.0 if hyp in ("decrease", "lower", "down") else 0.0,
            float(np.mean(vec)),
            float(len(items)),
        ],
        np.float32,
    )
    return {
        "gate": gate,
        "conf": conf,
        "hyp": hyp,
        "vec": vec,
        "hb": hb,
        "feat": feat,
        "ok": bool(obj.get("residual_hypothesis") or items),
        "n_keep": int(len(items)),
        "reason": str(obj.get("reason") or "")[:160],
    }
