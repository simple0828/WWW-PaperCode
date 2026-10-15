"""Shared KEEP mechanism bank + soft executor for explicit schemes 3/4.

Extracted from the retired Readable-Mech-RL path so α / Q-agent stay domain-generic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.lamp_vmt.json_utils import extract_json

PLA_W = {"high": 1.0, "medium": 0.6, "low": 0.3}
EFF = {"up": 1.0, "down": -1.0, "mixed": 0.0, "unclear": 0.0}
STRENGTH_W = {"weak": 0.55, "medium": 1.0, "strong": 1.45, "high": 1.35}
DIR_SIGN = {
    "increase": 1.0,
    "higher": 1.0,
    "up": 1.0,
    "decrease": -1.0,
    "lower": -1.0,
    "down": -1.0,
}


def _hop_count(chain: str) -> int:
    s = str(chain or "")
    for sep in ("→", "->", "=>", "⟶"):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return max(1, len(parts))
    if any(k in s.lower() for k in (" via ", " through ", " mediate", "→")):
        return 3
    return 1


def _effect_score(m: Mapping[str, Any]) -> float:
    return float(EFF.get(str(m.get("vmt_effect") or "").lower(), 0.0))


def _plaus(m: Mapping[str, Any]) -> float:
    return float(PLA_W.get(str(m.get("plausibility") or "medium").lower(), 0.5))


@dataclass
class MechCard:
    """Human-readable mechanism instance (explicit by construction)."""

    mid: str
    ym: str
    channel: str
    fact: str
    chain: str
    effect: float
    plaus: float
    hops: int
    lag: int

    @property
    def weight(self) -> float:
        return self.plaus * (0.65 + 0.35 / (1.0 + 0.35 * self.lag))

    def as_prompt_line(self) -> str:
        eff = {1.0: "up", -1.0: "down"}.get(self.effect, "unclear")
        pla = {1.0: "high", 0.6: "medium", 0.3: "low"}.get(self.plaus, "medium")
        chain = (self.chain or "")[:90]
        fact = (self.fact or "")[:70]
        return (
            f"  [{self.mid}] ch={self.channel} eff={eff} pla={pla} lag={self.lag} "
            f"| fact: {fact} | mech: {chain}"
        )


def build_mech_bank(
    by_ym: Mapping[str, Sequence[dict]],
    months: Sequence[str],
    focus: Sequence[str],
    *,
    max_per_month: int = 4,
    max_mechs_per_event: int = 3,
) -> list[MechCard]:
    focus_set = set(focus)
    cards: list[MechCard] = []
    for t, ym in enumerate(months):
        ym = str(ym)[:7]
        lag = max(0, len(months) - 1 - t)
        evs = [e for e in by_ym.get(ym, []) if e.get("channel") in focus_set]
        evs = sorted(
            evs,
            key=lambda e: (
                1 if "report" in str(e.get("channel") or "") else 0,
                -abs(
                    sum(
                        _effect_score(m) * _plaus(m)
                        for m in (e.get("mechanisms") or [])[:3]
                    )
                ),
            ),
        )[:max_per_month]
        for ei, e in enumerate(evs):
            mechs = e.get("mechanisms") or []
            if not mechs:
                mechs = [
                    {
                        "chain": e.get("mechanism_chain") or "",
                        "vmt_effect": e.get("vmt_effect") or "unclear",
                        "plausibility": "medium",
                    }
                ]
            for mi, m in enumerate(mechs[:max_mechs_per_event]):
                cards.append(
                    MechCard(
                        mid=f"{ym}/E{ei}/M{mi+1}",
                        ym=ym,
                        channel=str(e.get("channel") or "other"),
                        fact=str(e.get("fact_text") or "").replace("\n", " "),
                        chain=str(m.get("chain") or m.get("mechanism_chain") or ""),
                        effect=_effect_score(m),
                        plaus=_plaus(m),
                        hops=_hop_count(str(m.get("chain") or "")),
                        lag=lag,
                    )
                )
    return cards


def _ramp(H: int) -> np.ndarray:
    return np.linspace(0.85, 1.15, H, dtype=np.float32)


def soft_signed_from_cards(
    cards: Sequence[MechCard],
    H: int,
    amp: float,
    *,
    strength_mul: Optional[Mapping[str, float]] = None,
    dir_override: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Weighted signed pool over KEEP cards → residual template."""
    if not cards:
        return np.zeros(H, np.float32)
    num = den = 0.0
    for c in cards:
        eff = float(dir_override.get(c.mid, c.effect)) if dir_override else c.effect
        if abs(eff) < 1e-9:
            continue
        w = c.weight
        if strength_mul and c.mid in strength_mul:
            w *= float(strength_mul[c.mid])
        num += w * eff
        den += w
    if den < 1e-8:
        return np.zeros(H, np.float32)
    s = float(np.clip(num / den, -1.5, 1.5))
    return (s * amp * _ramp(H)).astype(np.float32)


def parse_keep_response(text: str) -> dict[str, Any]:
    obj: Any = {}
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = extract_json(t) if t else {}
    except Exception:
        try:
            obj = json.loads(t[t.find("{") : t.rfind("}") + 1])
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    keep = obj.get("keep") or []
    if not isinstance(keep, list):
        keep = []
    norm = []
    for it in keep:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or it.get("mid") or "").strip()
        if not mid:
            continue
        d = str(it.get("residual_dir") or it.get("dir") or "").lower()
        st = str(it.get("strength") or "medium").lower()
        why = str(it.get("why") or it.get("reason") or "")[:80]
        norm.append({"id": mid, "residual_dir": d, "strength": st, "why": why})
    gate = obj.get("gate", None)
    try:
        gate = int(gate) if gate is not None else (1 if norm else 0)
    except Exception:
        gate = 1 if norm else 0
    hyp = str(obj.get("residual_hypothesis") or "").lower()
    try:
        conf = float(obj.get("confidence") or (0.65 if norm else 0.4))
    except Exception:
        conf = 0.5
    reason = str(obj.get("reason") or "")[:160]
    return {
        "gate": 1 if gate == 1 else 0,
        "keep": norm[:4],
        "hyp": hyp,
        "conf": float(np.clip(conf, 0, 1)),
        "reason": reason,
        "raw": obj,
    }


def delta_from_selection(
    bank: Sequence[MechCard],
    selection: Mapping[str, Any],
    H: int,
    amp: float,
) -> tuple[np.ndarray, list[MechCard], dict[str, Any]]:
    by_id = {c.mid: c for c in bank}
    keep = selection.get("keep") or []
    if int(selection.get("gate") or 0) != 1 or not keep:
        return np.zeros(H, np.float32), [], {"n_keep": 0, "matched": []}
    chosen: list[MechCard] = []
    strength_mul: dict[str, float] = {}
    dir_override: dict[str, float] = {}
    matched = []
    for it in keep:
        mid = it["id"]
        if mid not in by_id:
            hit = next((k for k in by_id if k.endswith(mid) or mid.endswith(k)), None)
            if not hit:
                continue
            mid = hit
        c = by_id[mid]
        chosen.append(c)
        strength_mul[mid] = STRENGTH_W.get(it.get("strength") or "medium", 1.0)
        d = DIR_SIGN.get(str(it.get("residual_dir") or ""), None)
        if d is not None:
            dir_override[mid] = d
        matched.append(
            {
                "id": mid,
                "channel": c.channel,
                "chain": c.chain[:100],
                "residual_dir": it.get("residual_dir"),
                "strength": it.get("strength"),
                "why": it.get("why"),
            }
        )
    if not chosen:
        return np.zeros(H, np.float32), [], {"n_keep": 0, "matched": []}
    vec = soft_signed_from_cards(
        chosen, H, amp, strength_mul=strength_mul, dir_override=dir_override
    )
    conf = float(selection.get("conf") or 0.65)
    vec = (vec * (0.40 + 0.60 * conf)).astype(np.float32)
    return vec, chosen, {"n_keep": len(chosen), "matched": matched}


# Default KEEP prompt used by scheme α (single-shot selection).
SEED_PROMPTS = {
    "select_residual_v1": """You are a residual mechanism selector for {target}.

A CFA multimodal forecast already exists. Your job is ONLY to select which
EXISTING mechanism hypotheses explain why the TRUE future may DEVIATE from CFA.

Rules:
- Choose from the MECHANISM BANK only. Cite ids exactly.
- KEEP at most 3 mechanisms. Prefer exogenous / delayed / conflicting pathways.
- If evidence is weak or only restates the target series, set gate=0 and keep=[].
- why must be human-readable (<=12 words).

MECHANISM BANK:
{bank}

Origin={origin}  CFA_forecast={cfa}

JSON ONLY:
{{"residual_hypothesis":"increase|decrease|flat|mixed","gate":0|1,"confidence":0-1,
 "reason":"<=20 words",
 "keep":[{{"id":"YYYY-MM/E#/M#","residual_dir":"increase|decrease","strength":"weak|medium|strong","why":"<=12w"}}]}}
""",
}


def render_actor_prompt(
    template: str,
    *,
    target: str,
    origin: str,
    cfa: Sequence[float],
    bank: Sequence[MechCard],
) -> str:
    bank_txt = "\n".join(c.as_prompt_line() for c in bank) or "  (empty)"
    return template.format(
        target=target,
        origin=origin,
        cfa=np.round(np.asarray(cfa, float), 4).tolist(),
        bank=bank_txt,
    )


def mse(a, b) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def reward_delta(yc, yt, delta) -> float:
    return mse(yc, yt) - mse(yc + delta, yt)


def load_jsonl_cache(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["window_id"]] = row["response"]
    return out


def append_jsonl_cache(path: Path, window_id: str, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"window_id": window_id, "response": response}, ensure_ascii=False)
            + "\n"
        )
