"""Unified open-loop soft residual: dual (2) ⊕ LAMP (3) → 5 mechanism cards.

Both paths are open-loop LLM predictions. This module only **unifies the JSON
format** and decodes Δ with the same sparse rules that produced ~−1.23%
(legacy ``soft_mech`` / ``s5_lamp_fuse`` soft_blend).

Important: do **not** invent horizon_bias from direct.dir when hb is flat —
filling all agree-windows from dir destroys the gain (turns −1.2% into +1.5%).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from language.stage2.actions import BIAS, STRENGTH, parse_action

DIR_MAP = {
    "increase": "increase",
    "higher": "increase",
    "up": "up",
    "decrease": "decrease",
    "lower": "decrease",
    "down": "down",
    "flat": "flat",
    "mixed": "flat",
    "volatile": "flat",
}


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _norm_dir(x: Any) -> str:
    return DIR_MAP.get(str(x or "flat").lower().strip(), "flat")


def _norm_str(x: Any, default: str = "medium") -> str:
    s = str(x or default).lower().strip()
    return s if s in STRENGTH else default


def _mech(
    mid: str,
    name: str,
    *,
    residual_dir: str = "flat",
    strength: str = "medium",
    plausibility: float = 0.5,
    evidence: str = "",
    horizon_bias: list | None = None,
    source: str = "",
) -> dict:
    return {
        "id": mid,
        "name": str(name)[:80],
        "residual_dir": _norm_dir(residual_dir),
        "strength": _norm_str(strength),
        "plausibility": float(np.clip(float(plausibility), 0.0, 1.0)),
        "evidence": str(evidence or "")[:160],
        "horizon_bias": list(horizon_bias) if horizon_bias else None,
        "source": source,
    }


def _as_hb(hb: Any, H: int) -> list[str] | None:
    if isinstance(hb, str):
        tok = hb.lower().replace(" ", "_")
        return [tok] * H if tok in BIAS else None
    if isinstance(hb, list) and len(hb) >= H:
        out = [str(x).lower().replace(" ", "_") for x in hb[:H]]
        return [x if x in BIAS else "flat" for x in out]
    return None


def dual_to_mechs(dual_text: str, H: int = 6) -> tuple[list[dict], dict]:
    """M1=direct, M2=indirect. Preserve top-level horizon_bias on both cards."""
    obj = _parse_json(dual_text)
    try:
        conf = float(obj.get("confidence") or 0.5)
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    hb = _as_hb(obj.get("horizon_bias"), H)
    out: list[dict] = []
    for mid, key, label in (
        ("M1", "direct", "direct_path"),
        ("M2", "indirect", "indirect_path"),
    ):
        block = obj.get(key) if isinstance(obj.get(key), dict) else {}
        # prefer path-local hb; else share dual top-level hb (legacy schema)
        path_hb = _as_hb(block.get("horizon_bias"), H) or hb
        out.append(
            _mech(
                mid,
                label,
                residual_dir=block.get("dir") or "flat",
                strength=block.get("strength") or "medium",
                plausibility=conf,
                evidence=block.get("why") or block.get("reason") or obj.get("cot") or "",
                horizon_bias=path_hb,
                source="dual",
            )
        )
    meta = {
        "agree": int(obj.get("agree") or 0),
        "gate": int(obj.get("gate") or 0),
        "confidence": conf,
        "horizon_bias": hb,
    }
    return out, meta


def lamp_to_mechs(lamp_text: str, H: int = 6, k: int = 3) -> tuple[list[dict], dict]:
    """M3..M{2+k} from LAMP keep/drivers (default top-3)."""
    obj = _parse_json(lamp_text)
    items = obj.get("keep") or obj.get("drivers") or []
    if not isinstance(items, list):
        items = []
    try:
        conf = float(obj.get("confidence") or (0.65 if items else 0.45))
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    hyp = _norm_dir(obj.get("residual_hypothesis") or "flat")
    hb = _as_hb(obj.get("horizon_bias"), H)
    out: list[dict] = []
    for i, it in enumerate(items[:k]):
        if not isinstance(it, dict):
            continue
        name = str(
            it.get("driver")
            or it.get("name")
            or it.get("channel")
            or (it.get("text") or f"driver_{i+1}")[:48]
        )
        out.append(
            _mech(
                f"M{i+3}",
                name,
                residual_dir=it.get("residual_dir")
                or it.get("dir")
                or it.get("tag")
                or hyp,
                strength=it.get("strength") or "medium",
                plausibility=conf,
                evidence=it.get("evidence") or it.get("reason") or it.get("text") or "",
                horizon_bias=None,  # lamp uses shared hb / vote, not per-driver hb
                source="lamp",
            )
        )
    while len(out) < k:
        j = len(out)
        out.append(
            _mech(
                f"M{j+3}",
                f"lamp_pad_{j+1}",
                residual_dir="flat",
                strength="weak",
                plausibility=0.0,
                evidence="",
                source="lamp",
            )
        )
    meta = {
        "residual_hypothesis": hyp,
        "confidence": conf,
        "gate": obj.get("gate"),
        "horizon_bias": hb,
        "n_items": int(min(len(items), k)),
    }
    return out[:k], meta


def unify_five(dual_text: str, lamp_text: str, H: int = 6) -> dict:
    """Build unified JSON: mechanisms[M1..M5] + metadata."""
    m_dual, dual_meta = dual_to_mechs(dual_text, H=H)
    m_lamp, lamp_meta = lamp_to_mechs(lamp_text, H=H, k=3)
    mechs = m_dual + m_lamp
    dirs = [m["residual_dir"] for m in mechs if m["residual_dir"] not in ("flat",)]
    if not dirs:
        hyp = "flat"
    else:
        up = sum(1 for d in dirs if d in ("increase", "up", "higher"))
        dn = sum(1 for d in dirs if d in ("decrease", "down", "lower"))
        hyp = "increase" if up > dn else ("decrease" if dn > up else "flat")
    return {
        "mechanisms": mechs,
        "residual_hypothesis": hyp,
        "confidence": float(
            0.5 * (dual_meta["confidence"] + lamp_meta["confidence"])
        ),
        "agree_dual": dual_meta["agree"],
        "dual_gate": dual_meta["gate"],
        "dual_horizon_bias": dual_meta["horizon_bias"],
        "lamp_horizon_bias": lamp_meta["horizon_bias"],
        "lamp_hypothesis": lamp_meta["residual_hypothesis"],
        "lamp_confidence": lamp_meta["confidence"],
        "n_active": int(
            sum(1 for m in mechs if m["residual_dir"] not in ("flat",))
        ),
        "format": "unified_5_dual2_lamp3",
    }


def _hb_vec(hb: list[str] | None, conf: float, H: int) -> np.ndarray:
    if not hb:
        return np.zeros(H, dtype=np.float64)
    vec = np.array([BIAS.get(x, 0.0) for x in hb[:H]], dtype=np.float64)
    if conf < 0.6:
        return np.zeros(H, dtype=np.float64)
    return vec * (0.35 + 0.65 * conf)


def action_from_dual_unified(unified: dict, H: int = 6) -> dict:
    """Sparse dual action — same as parse_action(+agree), using M1/M2 + dual hb."""
    mechs = list(unified.get("mechanisms") or [])
    m1 = mechs[0] if mechs else {}
    agree = int(unified.get("agree_dual") or 0)
    conf = float((m1.get("plausibility") if m1 else None) or unified.get("confidence") or 0.5)
    hb = m1.get("horizon_bias") or unified.get("dual_horizon_bias")
    hb = _as_hb(hb, H)
    gate = 1 if agree == 1 and conf >= 0.6 else 0
    # also honor explicit dual_gate=0
    if int(unified.get("dual_gate") or 1) == 0:
        gate = 0
    vec = _hb_vec(hb, conf, H) if gate else np.zeros(H, dtype=np.float64)
    if gate and not np.any(np.abs(vec) > 1e-12):
        # keep gate=1 for overlap stats, but vec stays 0 (legacy behaviour)
        pass
    dd = _norm_dir(m1.get("residual_dir") if m1 else "flat")
    hyp = (
        "increase"
        if dd in ("up", "increase", "higher")
        else ("decrease" if dd in ("down", "decrease", "lower") else "flat")
    )
    return {"gate": int(gate), "conf": conf, "vec": vec, "hyp": hyp, "hb": hb or ["flat"] * H}


def action_from_lamp_unified(unified: dict, H: int = 6) -> dict:
    """Rebuild LAMP-style action from M3–M5 (+ optional shared lamp hb)."""
    mechs = [m for m in (unified.get("mechanisms") or []) if m.get("source") == "lamp"]
    if len(mechs) < 3:
        mechs = list(unified.get("mechanisms") or [])[2:5]
    hyp = str(unified.get("lamp_hypothesis") or unified.get("residual_hypothesis") or "flat").lower()
    # vote from M3–M5 when hyp flat/mixed
    if hyp in ("", "flat", "mixed", "volatile") and mechs:
        up = dn = 0
        for m in mechs:
            v = _norm_dir(m.get("residual_dir"))
            if v in ("increase", "up", "higher"):
                up += 1
            elif v in ("decrease", "down", "lower"):
                dn += 1
        if up > dn:
            hyp = "increase"
        elif dn > up:
            hyp = "decrease"
        else:
            hyp = "flat"
    conf = float(unified.get("lamp_confidence") or unified.get("confidence") or 0.5)
    conf = float(np.clip(conf, 0.0, 1.0))
    active = [m for m in mechs if _norm_dir(m.get("residual_dir")) not in ("flat",)]
    strength = 1.0
    if active:
        strength = float(np.mean([STRENGTH.get(m.get("strength") or "medium", 1.0) for m in active]))
    hb = _as_hb(unified.get("lamp_horizon_bias"), H)
    if hb is None:
        # map hyp → constant hb (same as lamp_on_cfa.hyp_to_bias path)
        tok = {
            "increase": "up",
            "higher": "up",
            "up": "up",
            "decrease": "down",
            "lower": "down",
            "down": "down",
        }.get(hyp, "flat")
        hb = [tok] * H
    amap = {
        "flat": 0.0,
        "up": 0.05,
        "down": -0.05,
        "strong_up": 0.09,
        "strong_down": -0.09,
        "increase": 0.05,
        "decrease": -0.05,
    }
    vec = np.array([amap.get(x, 0.0) for x in hb], float) * strength
    gate = 1 if hyp in ("increase", "decrease", "higher", "lower", "up", "down") and active and conf >= 0.55 else 0
    if hyp in ("flat", "mixed", "volatile") or conf < 0.55:
        gate = 0
        vec[:] = 0.0
    if gate == 0:
        vec[:] = 0.0
    else:
        vec *= 0.35 + 0.65 * conf
    return {
        "gate": int(gate),
        "conf": conf,
        "vec": vec.astype(np.float64),
        "hyp": hyp,
        "hb": hb,
        "n_keep": len(active),
    }


def soft_mech_blend(dual_act: dict, lamp_act: dict) -> np.ndarray:
    """Same as ``fusion_shell_cfa.soft_mech`` (kept local to avoid scripts import)."""
    gd, gL = int(dual_act.get("gate") or 0), int(lamp_act.get("gate") or 0)
    vd = np.asarray(dual_act.get("vec"), float)
    vL = np.asarray(lamp_act.get("vec"), float)

    def _sign(v) -> int:
        m = float(np.mean(v))
        if abs(m) < 1e-12:
            return 0
        return 1 if m > 0 else -1

    if gd and gL:
        sd, sL = _sign(vd), _sign(vL)
        if sd != 0 and sL != 0 and sd != sL:
            return vd.copy()
        return 0.55 * vd + 0.45 * vL
    if gd:
        return vd.copy()
    if gL:
        return vL.copy()
    return np.zeros_like(vd)


def decode_unified_soft(unified: dict, H: int = 6) -> dict:
    """Format-unified Δ: M1–M2 + M3–M5 → soft_mech blend. No learnable params."""
    d = action_from_dual_unified(unified, H=H)
    L = action_from_lamp_unified(unified, H=H)
    vec = soft_mech_blend(d, L)
    gate = 1 if np.any(np.abs(vec) > 1e-12) else 0
    return {
        "gate": gate,
        "vec": np.asarray(vec, float),
        "dual": d,
        "lamp": L,
        "mode": "unified_soft_mech",
        "ok": True,
    }


def decode_from_caches(dual_text: str, lamp_text: str, H: int = 6) -> tuple[dict, dict]:
    """Cache → unified JSON → soft Δ. Also returns classic soft_mech for A/B."""
    unified = unify_five(dual_text, lamp_text, H=H)
    decoded = decode_unified_soft(unified, H=H)
    # classic replay (parse_action + action_from_lamp + soft_mech)
    dual_act = parse_action(dual_text or "", H=H)
    raw = dual_act.get("raw") or {}
    if int(raw.get("agree") or 0) != 1:
        dual_act = {**dual_act, "gate": 0, "vec": np.zeros(H)}
    # lazy import lamp parser
    from scripts.lamp_on_cfa import action_from_lamp  # noqa: WPS433

    lamp_act = action_from_lamp(lamp_text or "", H=H)
    classic_vec = soft_mech_blend(dual_act, lamp_act)
    classic = {
        "gate": 1 if np.any(np.abs(classic_vec) > 1e-12) else 0,
        "vec": classic_vec,
        "mode": "classic_soft_mech",
    }
    return unified, {**decoded, "classic_vec": classic_vec, "classic": classic}


UNIFIED_SYSTEM = (
    "You write exactly 5 causal mechanisms for the CFA residual (true − CFA). "
    "M1=direct path, M2=indirect path, M3–M5=top event drivers. "
    "Open-loop only. Strict JSON. Prefer flat horizon_bias unless confident."
)


def prompt_unified_five(r: dict, by_ym: dict, H: int = 6) -> str:
    """Optional single-fill prompt (eval below replays existing dual+lamp caches)."""
    from language.stage2.explicit_prompts import _hist_block, _month_events

    months = r["months"]
    blocks = [_month_events(by_ym, ym, k=4) for ym in months]
    return (
        f"Domain={r.get('domain', 'traffic')} window={r['wid']} H={H}\n"
        f"CFA forecast (z): {[round(float(x), 3) for x in r['y_cfa']]}\n"
        f"History:\n{_hist_block(r)}\n"
        "Lookback events (newest last):\n" + "\n".join(blocks) + "\n"
        "Return ONLY JSON:\n"
        "{"
        '"mechanisms":['
        '{"id":"M1","name":"direct_path","residual_dir":"up|down|flat",'
        '"strength":"weak|medium|strong","plausibility":0.0,'
        '"evidence":"≤12 words"},'
        '{"id":"M2","name":"indirect_path",...},'
        '{"id":"M3","name":"driver_a",...},'
        '{"id":"M4","name":"driver_b",...},'
        '{"id":"M5","name":"driver_c",...}'
        "],"
        '"agree_dual":0,'
        '"dual_horizon_bias":["flat"]xH,'
        '"lamp_hypothesis":"increase|decrease|flat|mixed",'
        '"lamp_confidence":0.0,'
        '"lamp_horizon_bias":["flat|up|down"]xH,'
        '"residual_hypothesis":"increase|decrease|flat",'
        '"confidence":0.0,'
        '"reason":"≤20 words"'
        "}\n"
        "Rules: M1/M2 = direct vs indirect (agree_dual=1 only if both dirs match). "
        "M3–M5 = distinct residual drivers (LAMP keep≤3). "
        "Leave dual_horizon_bias all-flat unless you are confident on step-wise bias. "
        "Prefer abstain (flat / low conf) when evidence is weak."
    )
