"""Additional APO meta-prompt paradigms (few-shot, CoT, JSON, etc.)."""

from __future__ import annotations

_PLACEHOLDERS = """
Window:
- start_date={start_date}
- end_date={end_date}

Historical stats:
{history_stats}

Unimodal baseline forecast (no text):
{baseline_forecast}

Raw fact:
{raw_fact}
""".strip()

TRAFFIC_PARADIGM_PROMPTS = {
    "few_shot": f"""You rewrite traffic news into forecast-relevant residual evidence. Follow the EXAMPLES, then process the INPUT.

EXAMPLE A (baseline flat, fact bullish):
Baseline: y_baseline=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
Output:
KEEP: (higher, strong) Memorial Day travel surge expected to lift May VMT above baseline path.
DROP: generic highway history.
Residual hypothesis: increase because holiday travel dominates.

EXAMPLE B (baseline up, fact bearish):
Baseline: y_baseline=[110.0, 111.0, 112.0, 113.0, 114.0, 115.0]
Output:
KEEP: (lower, medium) Fuel price spike and remote-work policy may suppress driving vs rising baseline.
DROP: restated long-run VMT averages.
Residual hypothesis: decrease because fuel and policy headwinds dominate.

Now process INPUT using the same OUTPUT style (<=6 KEEP bullets, each <=28 words).

{_PLACEHOLDERS}
""",
    "chain_of_thought": f"""Rewrite raw traffic text for monthly U.S. VMT forecasting. Think step-by-step internally, but output ONLY the final rewrite.

Steps (do not print step numbers):
1) Read baseline forecast path and history stats.
2) List 3-5 candidate signals in the raw fact that could move VMT off baseline.
3) Drop generic/encyclopedic/non-local statements.
4) Keep only signals with clear direction vs baseline (higher/lower/volatile).
5) Emit concise KEEP/DROP bullets and one residual hypothesis.

Output format:
KEEP: ... (direction, strength=weak|medium|strong)
DROP: ...
Residual hypothesis: <increase|decrease|flat|mixed> because <reason>

{_PLACEHOLDERS}
""",
    "json_residual": f"""Extract structured residual hints from the raw fact vs the unimodal VMT baseline.

Output a single JSON object ONLY (no markdown fences), keys:
- "keep": list of {{"text": str, "direction": "up|down|volatile", "strength": "weak|medium|strong"}}
- "drop_reasons": list of str (why generic lines were removed)
- "residual": {{"label": "increase|decrease|flat|mixed", "pct_adjust_hint": float (-20..20), "confidence": float (0..1), "reason": str}}

Rules: max 6 keep items; no invented events; pct_adjust_hint is a soft hint not the true residual.

{_PLACEHOLDERS}
""",
    "self_consistency": f"""You will produce THREE independent draft rewrites (D1,D2,D3) of the raw fact vs baseline, then a FINAL merged rewrite.

Each draft: 3-5 KEEP bullets with direction/strength + residual hypothesis.
Merge rule for FINAL: keep bullets supported by >=2 drafts; prefer specific dated facts; resolve conflicts by window locality.

Output ONLY:
DRAFT1: ...
DRAFT2: ...
DRAFT3: ...
FINAL: <merged KEEP/DROP + Residual hypothesis>

{_PLACEHOLDERS}
""",
    "counterfactual": f"""Counterfactual residual extraction for U.S. monthly VMT.

1) State the baseline story implied by y_baseline (1 sentence).
2) State the counterfactual story implied by the raw fact if baseline were wrong (1 sentence).
3) KEEP only facts that flip or materially change the baseline story.
4) DROP facts that would leave the baseline story unchanged.

Format:
Baseline story: ...
Counterfactual story: ...
KEEP: ... (direction, strength)
DROP: ...
Residual hypothesis: <increase|decrease|flat|mixed> because ...

{_PLACEHOLDERS}
""",
}

AGRI_PARADIGM_PROMPTS = {
    "few_shot": f"""Rewrite broiler/agriculture news into forecast-relevant residual evidence. Follow EXAMPLES, then INPUT.

EXAMPLE A:
KEEP: (higher, strong) Feed cost surge and lower bird placements may lift prices above baseline.
DROP: generic poultry industry definitions.
Residual hypothesis: increase because supply tightens.

EXAMPLE B:
KEEP: (lower, medium) Disease-related culling reported in major producing region.
DROP: long-run average price restatements.
Residual hypothesis: decrease because supply recovery exceeds demand.

Same OUTPUT style for INPUT (<=6 KEEP bullets).

{_PLACEHOLDERS}
""",
    "chain_of_thought": f"""Step-by-step broiler price residual extraction. Output ONLY final KEEP/DROP + hypothesis.

Steps (internal): baseline read → candidate signals → filter generic → direction vs baseline → concise bullets.

{_PLACEHOLDERS}
""",
    "json_residual": f"""Output JSON only for broiler residual hints:
{{"keep":[{{"text","direction","strength"}}], "drop_reasons":[], "residual":{{"label","pct_adjust_hint","confidence","reason"}}}}

{_PLACEHOLDERS}
""",
    "self_consistency": f"""Three drafts then FINAL merged rewrite (broiler prices). Output DRAFT1/2/3 then FINAL.

{_PLACEHOLDERS}
""",
    "counterfactual": f"""Baseline story vs counterfactual story from raw fact; KEEP only story-flipping facts.

{_PLACEHOLDERS}
""",
}

PARADIGM_NAMES = (
    "few_shot",
    "chain_of_thought",
    "json_residual",
    "self_consistency",
    "counterfactual",
)
