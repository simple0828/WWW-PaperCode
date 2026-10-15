"""Sampled dual: LLM CoT-select lookback → n×(M1 direct, M2 indirect) → Δ.

Pipeline (open-loop, no learnable params):
  1) For each of n rounds: LLM selects ≤2 FOCUS events/month from the L-month
     lookback by CoT on *future VMT residual vs CFA* (not random).
  2) On that subsample only: LLM writes one direct + one indirect pair.
  3) Aggregate the n pairs (2n hypotheses) into a residual vector Δ.

Division of labour: select = evidence; dual = causal adjudication; soft = Δ.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import numpy as np

from language.stage2.actions import BIAS, STRENGTH, parse_action, parse_json
from language.stage2.profiles import Stage2DomainProfile

_UP = {"up", "increase", "higher"}
_DN = {"down", "decrease", "lower"}
_HB_AMAP = {
    "flat": 0.0,
    "up": 0.05,
    "down": -0.05,
    "strong_up": 0.09,
    "strong_down": -0.09,
}

DEFAULT_N = 3
MAX_PER_MONTH = 2
MAX_CAND_PER_MONTH = 6  # show at most this many candidates/month in select prompt


def _clip(s: Any, n: int) -> str:
    return str(s or "")[:n]


def build_candidates(
    months: Sequence[str],
    by_ym: Mapping[str, list],
    profile: Stage2DomainProfile,
    *,
    max_per_month: int = MAX_CAND_PER_MONTH,
) -> list[dict]:
    """Indexed FOCUS events for one window (stable eid = ym#k)."""
    focus = set(profile.focus)
    report = profile.report_channel
    out: list[dict] = []
    for ym in months:
        evs = [e for e in by_ym.get(ym, []) if e.get("channel") in focus]
        evs = sorted(evs, key=lambda e: 0 if e.get("channel") != report else 1)
        for k, e in enumerate(evs[:max_per_month]):
            eid = f"{ym}#{k}"
            hyps = []
            for j, m in enumerate((e.get("mechanisms") or [])[:2], 1):
                hyps.append(
                    {
                        "j": j,
                        "plaus": m.get("plausibility"),
                        "effect": m.get("vmt_effect"),
                        "chain": _clip(m.get("chain"), 70),
                    }
                )
            out.append(
                {
                    "eid": eid,
                    "ym": ym,
                    "channel": e.get("channel"),
                    "fact": _clip(e.get("fact_text"), 100),
                    "hyps": hyps,
                    "raw": e,
                }
            )
    return out


def format_candidate_pool(cands: Sequence[dict]) -> str:
    lines: list[str] = []
    cur = None
    for c in cands:
        if c["ym"] != cur:
            cur = c["ym"]
            lines.append(f"[{cur}]")
        lines.append(f"  {c['eid']}/{c['channel']}: {c['fact']}")
        for h in c["hyps"]:
            lines.append(
                f"    H{h['j']}[{h['plaus']}→{h['effect']}]: {h['chain']}"
            )
    return "\n".join(lines) if lines else "(no FOCUS events)"


def format_subsample(cands: Sequence[dict], selected_eids: Sequence[str]) -> str:
    by_id = {c["eid"]: c for c in cands}
    lines: list[str] = []
    cur = None
    for eid in selected_eids:
        c = by_id.get(eid)
        if not c:
            continue
        if c["ym"] != cur:
            cur = c["ym"]
            lines.append(f"[{cur}]")
        lines.append(f"  {c['eid']}/{c['channel']}: {c['fact']}")
        for h in c["hyps"]:
            lines.append(
                f"    H{h['j']}[{h['plaus']}→{h['effect']}]: {h['chain']}"
            )
    return "\n".join(lines) if lines else "(empty subsample)"


def prompt_select(
    r: dict,
    cands: Sequence[dict],
    profile: Stage2DomainProfile,
    *,
    round_i: int,
    n_rounds: int,
    prev_eids: Sequence[str],
    max_per_month: int = MAX_PER_MONTH,
) -> str:
    target = profile.target_series_short
    prev = ", ".join(prev_eids) if prev_eids else "(none)"
    return "\n".join(
        [
            f"SELECT residual evidence for {target} (round {round_i + 1}/{n_rounds}).",
            "A CFA model already fused cleaned news. You do NOT forecast absolute level.",
            "Pick events whose mechanisms may explain why TRUE future DEVIATES from CFA.",
            f"Origin {r['origin']} CFA={np.round(r['y_cfa'], 4).tolist()}",
            f"History stats: {_clip(r.get('history_stats'), 160)}",
            "Candidate FOCUS events (indexed by eid):",
            format_candidate_pool(cands),
            f"Already used in prior rounds (prefer alternatives): {prev}",
            f"Rules: ≤{max_per_month} events per calendar month; total prefer 4–12 eids;",
            "use CoT on residual drivers (fuel/holiday/policy/weather/WFH/…); avoid pure VMT restatements;",
            "this round should diversify vs prior rounds when possible.",
            "JSON ONLY:",
            "{"
            '"cot":"≤40 words why these affect future residual vs CFA",'
            '"selected":[{"eid":"YYYY-MM#k","why":"≤12 words"}],'
            '"gate_select":0|1'
            "}",
            "gate_select=0 if nothing credible for residual.",
        ]
    )


def prompt_dual_on_sample(
    r: dict,
    subsample_text: str,
    profile: Stage2DomainProfile,
    *,
    round_i: int,
) -> str:
    target = profile.target_series_short
    return "\n".join(
        [
            f"DUAL causality on a SELECTED subsample (round {round_i + 1}).",
            f"Target: {target}. Reason ONLY from the subsample below (ignore other news).",
            f"Origin {r['origin']} CFA={np.round(r['y_cfa'], 4).tolist()}",
            "Selected events + event-level hypotheses:",
            subsample_text,
            f"Write ONE direct path (fact→{target} residual) and ONE indirect path (fact→mediator→{target}).",
            "Apply correction ONLY if direct and indirect agree; else gate=0.",
            "Prefer horizon_bias all-flat unless step-wise residual direction is clear.",
            "JSON ONLY:",
            '{"direct":{"dir":"up|down|flat","why":"≤20 words"},'
            '"indirect":{"dir":"up|down|flat","mediator":"A→B→residual","why":"≤20 words"},'
            '"agree":0|1,"gate":0|1,"confidence":0-1,'
            '"horizon_bias":["flat","flat","flat","flat","flat","flat"],'
            '"cot":"≤20 words"}',
        ]
    )


def parse_selected_eids(text: str, valid: set[str], max_per_month: int = MAX_PER_MONTH) -> dict:
    obj = parse_json(text) if text else {}
    selected = obj.get("selected") or []
    eids: list[str] = []
    per_m: dict[str, int] = {}
    if isinstance(selected, list):
        for it in selected:
            if isinstance(it, dict):
                eid = str(it.get("eid") or "").strip()
            else:
                eid = str(it).strip()
            if eid not in valid or eid in eids:
                continue
            ym = eid.split("#", 1)[0]
            if per_m.get(ym, 0) >= max_per_month:
                continue
            per_m[ym] = per_m.get(ym, 0) + 1
            eids.append(eid)
    try:
        gate = int(obj.get("gate_select") if obj.get("gate_select") is not None else (1 if eids else 0))
    except Exception:
        gate = 1 if eids else 0
    return {
        "eids": eids,
        "gate_select": gate,
        "cot": _clip(obj.get("cot"), 200),
        "raw": obj,
        "ok": bool(obj),
    }


def pair_to_mechs(dual_text: str, round_i: int, H: int = 6) -> list[dict]:
    """One sample → M1 direct + M2 indirect cards."""
    obj = parse_json(dual_text) if dual_text else {}
    try:
        conf = float(obj.get("confidence") or 0.5)
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    hb = obj.get("horizon_bias")
    if isinstance(hb, list) and len(hb) >= H:
        hb = [str(x).lower().replace(" ", "_") for x in hb[:H]]
        hb = [x if x in BIAS else "flat" for x in hb]
    else:
        hb = ["flat"] * H
    out = []
    for mid, key, name in (
        (f"R{round_i}_M1", "direct", "direct_path"),
        (f"R{round_i}_M2", "indirect", "indirect_path"),
    ):
        block = obj.get(key) if isinstance(obj.get(key), dict) else {}
        out.append(
            {
                "id": mid,
                "name": name,
                "round": round_i,
                "residual_dir": str(block.get("dir") or "flat").lower(),
                "strength": "medium",
                "plausibility": conf,
                "evidence": _clip(block.get("why") or obj.get("cot"), 160),
                "mediator": _clip(block.get("mediator"), 120) if key == "indirect" else "",
                "horizon_bias": hb,
                "agree": int(obj.get("agree") or 0),
                "gate": int(obj.get("gate") or 0),
            }
        )
    return out


def action_from_pair(dual_text: str, H: int = 6) -> dict:
    """Same sparse decode as legacy S5 dual (agree + hb; no dir-fill)."""
    act = parse_action(dual_text or "", H=H)
    raw = act.get("raw") or {}
    if int(raw.get("agree") or 0) != 1:
        return {**act, "gate": 0, "vec": np.zeros(H, dtype=np.float64)}
    return act


def decode_sampled_pairs(
    pair_texts: Sequence[str],
    *,
    H: int = 6,
    mode: str = "mean_gated",
    amp: float = 0.06,
) -> dict:
    """Aggregate n dual JSONs → Δ.

    modes:
      mean_gated — mean of agree-gated hb vecs (legacy S5 spirit; empty → 0)
      soft_all   — soft signed mean over all 2n dirs × amp × ramp (fallback denser)
      mean_or_soft — mean_gated if any nonzero else soft_all
    """
    acts = [action_from_pair(t, H=H) for t in pair_texts]
    gated = [a["vec"] for a in acts if int(a.get("gate") or 0) and np.any(np.abs(a["vec"]) > 1e-12)]
    mechs: list[dict] = []
    for i, t in enumerate(pair_texts):
        mechs.extend(pair_to_mechs(t, i, H=H))

    def _soft_all() -> np.ndarray:
        signed = []
        for m in mechs:
            d = str(m.get("residual_dir") or "flat").lower()
            s = BIAS.get(d, 0.0)
            if abs(s) < 1e-12:
                continue
            # only count mechs from agreeing pairs
            if int(m.get("agree") or 0) != 1:
                continue
            signed.append(s * float(m.get("plausibility") or 0.5))
        if not signed:
            return np.zeros(H, dtype=np.float64)
        s = float(np.mean(signed))
        ramp = np.linspace(0.55, 1.0, H, dtype=np.float64)
        return s * amp * ramp

    if mode == "soft_all":
        vec = _soft_all()
    elif mode == "mean_or_soft":
        vec = np.mean(np.stack(gated, 0), 0) if gated else _soft_all()
    else:  # mean_gated
        vec = np.mean(np.stack(gated, 0), 0) if gated else np.zeros(H, dtype=np.float64)

    return {
        "vec": np.asarray(vec, dtype=np.float64),
        "gate": 1 if np.any(np.abs(vec) > 1e-12) else 0,
        "n_pairs": len(pair_texts),
        "n_gated_nonzero": len(gated),
        "n_mechs": len(mechs),
        "mechs": mechs,
        "mode": mode,
        "pair_gates": [int(a.get("gate") or 0) for a in acts],
    }


def _dir_tag(d: str) -> str:
    d = str(d or "flat").lower()
    if d in _UP:
        return "higher"
    if d in _DN:
        return "lower"
    return "flat"


def mechs_to_lamp_drivers(
    mechs: Sequence[dict],
    *,
    agree_only: bool = True,
) -> list[dict]:
    """Map 2n dual hypotheses → LAMP-style driver items."""
    items = []
    for m in mechs:
        if agree_only and int(m.get("agree") or 0) != 1:
            continue
        tag = _dir_tag(m.get("residual_dir"))
        if tag == "flat":
            continue
        items.append(
            {
                "tag": tag,
                "text": _clip(m.get("evidence") or m.get("name"), 80),
                "strength": str(m.get("strength") or "medium"),
                "id": m.get("id"),
                "round": m.get("round"),
            }
        )
    return items


def pairs_to_lamp_drivers(pair_texts: Sequence[str], H: int = 6) -> list[dict]:
    """One driver per agreeing dual pair (collapse M1/M2)."""
    items = []
    for i, t in enumerate(pair_texts):
        mechs = pair_to_mechs(t, i, H=H)
        if not mechs or int(mechs[0].get("agree") or 0) != 1:
            continue
        tags = [_dir_tag(m.get("residual_dir")) for m in mechs]
        # both should match when agree=1; take first non-flat
        tag = next((x for x in tags if x != "flat"), "flat")
        if tag == "flat":
            continue
        conf = float(mechs[0].get("plausibility") or 0.5)
        strength = "strong" if conf >= 0.8 else ("medium" if conf >= 0.6 else "weak")
        items.append(
            {
                "tag": tag,
                "text": _clip(mechs[0].get("evidence"), 80),
                "strength": strength,
                "id": f"R{i}_pair",
                "round": i,
            }
        )
    return items


def action_from_drivers(
    items: Sequence[dict],
    *,
    H: int = 6,
    conf: float | None = None,
    min_margin: int = 1,
    min_items: int = 1,
    conf_floor: float = 0.55,
) -> dict:
    """LAMP-style sparse driver gate (same numeric rules as ``action_from_lamp``)."""
    items = [it for it in items if isinstance(it, dict)]
    up = dn = 0
    for it in items:
        v = str(it.get("tag") or it.get("dir") or "").lower()
        if v in ("increase", "higher", "up"):
            up += 1
        elif v in ("decrease", "lower", "down"):
            dn += 1
    if up > dn and (up - dn) >= min_margin:
        hyp = "increase"
    elif dn > up and (dn - up) >= min_margin:
        hyp = "decrease"
    else:
        hyp = "flat"
    if conf is None:
        conf = 0.65 if items else 0.45
    conf = float(np.clip(float(conf), 0.0, 1.0))
    gate = 1 if hyp in ("increase", "decrease") and len(items) >= min_items and conf >= conf_floor else 0
    strength = 1.0
    if items:
        ss = [STRENGTH.get(str(it.get("strength") or "medium"), 1.0) for it in items]
        strength = float(np.mean(ss))
    tok = {"increase": "up", "decrease": "down"}.get(hyp, "flat")
    hb = [tok] * H
    vec = np.array([_HB_AMAP.get(x, 0.0) for x in hb], float) * strength
    if hyp in ("flat", "mixed", "volatile") or conf < conf_floor:
        gate = 0
        vec[:] = 0.0
    if gate == 0:
        vec[:] = 0.0
    else:
        vec *= 0.35 + 0.65 * conf
    return {
        "gate": int(gate),
        "vec": vec.astype(np.float64),
        "hyp": hyp,
        "conf": conf,
        "strength": strength,
        "n_items": len(items),
        "vote_up": up,
        "vote_dn": dn,
        "ok": True,
    }


def decode_sparse_driver(
    pair_texts: Sequence[str],
    *,
    H: int = 6,
    mode: str = "sparse_agree",
    min_margin: int = 1,
    min_items: int = 2,
    conf_floor: float = 0.55,
) -> dict:
    """Aggregate 2n hypotheses with sparse driver gate (not soft mean).

    modes:
      sparse_agree — each agree-mech is a driver; vote+margin gate
      sparse_all   — all non-flat mechs as drivers (ignore agree)
      sparse_pair  — one driver per agreeing pair
      sparse_pair_m2 — sparse_pair with min_margin=2 (stricter, LAMP-like sparsity)
    """
    mechs: list[dict] = []
    for i, t in enumerate(pair_texts):
        mechs.extend(pair_to_mechs(t, i, H=H))
    if mode == "sparse_all":
        items = mechs_to_lamp_drivers(mechs, agree_only=False)
        confs = [float(m.get("plausibility") or 0.5) for m in mechs if _dir_tag(m.get("residual_dir")) != "flat"]
    elif mode == "sparse_pair" or mode == "sparse_pair_m2":
        items = pairs_to_lamp_drivers(pair_texts, H=H)
        confs = []
        for i, t in enumerate(pair_texts):
            ms = pair_to_mechs(t, i, H=H)
            if ms and int(ms[0].get("agree") or 0) == 1:
                confs.append(float(ms[0].get("plausibility") or 0.5))
        if mode == "sparse_pair_m2":
            min_margin = max(min_margin, 2)
    else:  # sparse_agree
        items = mechs_to_lamp_drivers(mechs, agree_only=True)
        confs = [
            float(m.get("plausibility") or 0.5)
            for m in mechs
            if int(m.get("agree") or 0) == 1 and _dir_tag(m.get("residual_dir")) != "flat"
        ]
    conf = float(np.mean(confs)) if confs else 0.45
    act = action_from_drivers(
        items,
        H=H,
        conf=conf,
        min_margin=min_margin,
        min_items=min_items,
        conf_floor=conf_floor,
    )
    return {
        **act,
        "mechs": mechs,
        "drivers": items,
        "n_pairs": len(pair_texts),
        "n_mechs": len(mechs),
        "mode": mode,
    }


SYS_SELECT = (
    "You select residual evidence vs CFA. CoT on future deviation drivers. "
    "Strict JSON. Diversify across rounds."
)
SYS_DUAL = (
    "You adjudicate DIRECT vs INDIRECT causality on a fixed subsample. "
    "Gate only if both agree. Strict JSON."
)


def cache_key(wid: str, round_i: int, stage: str) -> str:
    return f"{wid}|r{round_i}|{stage}"
