"""Domain-agnostic Explicit-branch prompt builders.

Legacy S5 dual + LAMP driver prompts remain for cache replay / ablation.
Candidate 1 now uses ``scheme1_hier`` (channel→window one-shot).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.stage2.profiles import Stage2DomainProfile


def hist_block(
    months: Sequence[str],
    by_ym: Mapping[str, list],
    profile: Stage2DomainProfile,
    *,
    max_per_month: int = 2,
) -> str:
    focus = set(profile.focus)
    report = profile.report_channel
    lines = []
    for ym in months:
        evs = [e for e in by_ym.get(ym, []) if e.get("channel") in focus]
        evs = sorted(evs, key=lambda e: 0 if e.get("channel") != report else 1)[
            :max_per_month
        ]
        if not evs:
            continue
        lines.append(f"[{ym}]")
        for ei, e in enumerate(evs):
            lines.append(f"  E{ei}/{e.get('channel')}: {(e.get('fact_text') or '')[:90]}")
            for j, m in enumerate((e.get("mechanisms") or [])[:2], 1):
                lines.append(
                    f"    M{j}[{m.get('plausibility')}→{m.get('vmt_effect')}]: "
                    f"{(m.get('chain') or '')[:70]}"
                )
    return "\n".join(lines)


def prompt_dual(r: dict[str, Any], by_ym: Mapping[str, list], profile: Stage2DomainProfile) -> str:
    target = profile.target_series_short
    return "\n".join(
        [
            "CFA residual reasoning with DIRECT vs INDIRECT causality (Dr.ECI-style).",
            f"Target series: {target}.",
            f"Origin {r['origin']} CFA={np.round(r['y_cfa'],4).tolist()}",
            hist_block(r["months"], by_ym, profile, max_per_month=2),
            f"direct: one-hop mechanism (fact→{target} residual)",
            f"indirect: mediator path (fact→X→{target} residual)",
            "Apply correction ONLY if direct and indirect agree on direction; else gate=0.",
            'JSON ONLY: {"direct":{"dir":"up|down|flat","why":"..."},'
            '"indirect":{"dir":"up|down|flat","mediator":"...","why":"..."},'
            '"agree":0|1,"gate":0|1,"confidence":0-1,'
            '"horizon_bias":["flat","flat","flat","flat","flat","flat"],"cot":"≤20 words"}',
        ]
    )


def prompt_lamp_driver(
    r: dict[str, Any], by_ym: Mapping[str, list], profile: Stage2DomainProfile
) -> str:
    target = profile.target_series_short
    return "\n".join(
        [
            f"You are a LAMP residual-evidence extractor for {target}.",
            "A CFA multimodal model already fused cleaned news into a baseline forecast.",
            "Extract only clues why TRUE future may DEVIATE from CFA (residual). Not absolute level.",
            f"Origin: {r['origin']} | H=6",
            f"CFA baseline forecast (scaler): {np.round(r['y_cfa'], 4).tolist()}",
            f"History stats: {(r.get('history_stats') or '')[:180]}",
            "Structured events with competing mechanism hypotheses:",
            hist_block(r["months"], by_ym, profile),
            "Prefer exogenous shocks and delayed/anticipatory effects over restatements of the target series.",
            "Output JSON ONLY:",
            "{",
            '  "drivers": [{"tag":"higher|lower|volatile|persist|reverse","text":"<=28 words","strength":"weak|medium|strong"}],',
            '  "conflict": "one bullet or none",',
            '  "residual_hypothesis": "increase|decrease|flat|mixed",',
            '  "reason": "<=20 words",',
            '  "gate": 0|1,',
            '  "confidence": 0-1,',
            '  "horizon_bias": ["flat","flat","flat","flat","flat","flat"]',
            "}",
            "gate=1 only if residual_hypothesis is increase/decrease with non-weak evidence; else 0.",
        ]
    )


LEAD_RESIDUAL = (
    "Predict the residual (TRUE minus CFA) for the forecast horizon. "
    "Use only dated event facts that can move this series after Origin. "
    "Do not restate the CFA level. If facts are weak or conflicting, output near-zero residual."
)


def build_fact_prompt(
    by_ym: Mapping[str, list],
    months: Sequence[str],
    origin: str,
    profile: Stage2DomainProfile,
    *,
    max_months: Optional[int] = None,
    keep_frac: Optional[float] = None,
    min_per_month: int = 1,
    max_per_month_cap: int = 16,
    max_per_month: Optional[int] = None,
    include_mech: bool = False,
    include_cfa: Optional[Sequence[float]] = None,
    fact_chars: int = 120,
    lead: Optional[str] = None,
) -> str:
    """Build Fact-LatentCoT text prefix.

    Prefix lines (before event facts):
      1) profile.fact_prompt_lead (task instruction)
      2) Origin: <ym>
      3) Facts:

    Event filter (no remote LLM):
      - channel ∈ profile.focus
      - rank: non-report first, then |soft(effect×plaus)| descending
      - keep top-k where k = profile.fact_max_per_month (or override),
        or ceil(keep_frac * n) if keep_frac is set and max_per_month is None
      - months: last ``max_months`` nearest origin (profile.fact_max_months)
    """
    import math

    from language.stage2.actions import soft as soft_score

    if max_months is None:
        max_months = int(getattr(profile, "fact_max_months", 4) or 4)
    if max_per_month is None and keep_frac is None:
        max_per_month = int(getattr(profile, "fact_max_per_month", 2) or 2)

    lead = lead or profile.fact_prompt_lead or (
        f"Infer CFA residual for {profile.target_series_short} from event facts only."
    )
    focus = set(profile.focus)
    report = profile.report_channel
    lines = [lead, f"Target: {profile.target_series_short}.", f"Origin: {origin}."]
    if include_cfa is not None:
        arr = np.asarray(include_cfa, float).reshape(-1)
        lines.append("CFA=" + np.round(arr, 4).tolist().__repr__())
    lines.append("Facts:")
    use_months = list(months)[-int(max_months) :]
    n_kept = 0
    for ym in use_months:
        evs = [e for e in by_ym.get(ym, []) if e.get("channel") in focus]
        evs = sorted(
            evs,
            key=lambda e: (
                1 if e.get("channel") == report else 0,
                -abs(float(soft_score(e))),
            ),
        )
        if not evs:
            continue
        if max_per_month is not None:
            k = min(int(max_per_month), len(evs))
        else:
            frac = 0.4 if keep_frac is None else float(keep_frac)
            k = max(min_per_month, int(math.ceil(frac * len(evs))))
            k = min(k, int(max_per_month_cap), len(evs))
        evs = evs[:k]
        lines.append(f"[{ym}]")
        for e in evs:
            fact = (e.get("fact_text") or "").replace("\n", " ").strip()[: int(fact_chars)]
            if not fact:
                continue
            n_kept += 1
            lines.append(f"  {e.get('channel')}: {fact}")
            if include_mech:
                mechs = e.get("mechanisms") or []
                m0 = mechs[0] if mechs else None
                chain = ""
                effect = e.get("vmt_effect") or ""
                if isinstance(m0, dict):
                    chain = (m0.get("chain") or "")[:80]
                    effect = m0.get("vmt_effect") or effect
                elif e.get("mechanism_chain"):
                    chain = str(e.get("mechanism_chain") or "")[:80]
                if effect or chain:
                    lines.append(f"    → {effect}: {chain}".rstrip())
    lines.append("Think about residual vs CFA in latent space.")
    return "\n".join(lines)
