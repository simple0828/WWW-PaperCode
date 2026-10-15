"""Stage1 Self-Evo cleaning prompts (domain-parameterized)."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

FACT_FORMAT_CONSTRAINTS = """\
Optional starting hint (NOT a hard lock — evolve from experience/Judge gaps):
- Prefer checkable, concrete statements over ads/speculation when useful for the task.
- Do not invent facts. Let Coverage/Executability feedback decide format and rules.
"""


def make_systems(display_task: str) -> dict[str, str]:
    return {
        "observer": (
            f"You are DataEvolve's Data Quality Observer for {display_task} news facts. "
            "The host has already called read_experience() for you. Independently find NEW "
            "quality issues in the raw samples and return them so the host can write_experience(). "
            "Do not restate issues already in the experience list. Reply with JSON only."
        ),
        "designer": (
            f"You design cleaning strategies (prompts) for auxiliary news text for {display_task}. "
            "Evolve from experience-pool issues and Quality Judge Coverage/Executability gaps. "
            "Do not blindly keep a fixed output schema if feedback says it fails. "
            "Reply with JSON only."
        ),
        "cleaner": (
            f"You clean auxiliary news text for {display_task}. "
            "Follow the provided strategy strictly. Output the cleaned fact list only "
            "(bullet lines; no markdown fences, no commentary)."
        ),
        "judge": (
            f"You are DataEvolve's Quality Judge for {display_task} auxiliary news cleaning. "
            "Compare original vs cleaned pairs, score cleaning, give SHORT Coverage and "
            "Executability notes, and list only genuinely NEW residual issues. "
            "Keep every string field concise (Coverage/Executability ≤40 words each; "
            "pair comments ≤12 words; ≤4 new_issues). Reply with compact JSON only — no markdown."
        ),
    }


def apply_task_systems(display_task: str) -> None:
    """Mutate module-level system strings for this process (Stage1 run)."""
    global OBSERVER_SYSTEM, DESIGNER_SYSTEM, CLEANER_SYSTEM, JUDGE_SYSTEM, FACT_FORMAT_CONSTRAINTS
    sys = make_systems(display_task)
    OBSERVER_SYSTEM = sys["observer"]
    DESIGNER_SYSTEM = sys["designer"]
    CLEANER_SYSTEM = sys["cleaner"]
    JUDGE_SYSTEM = sys["judge"]
    FACT_FORMAT_CONSTRAINTS = (
        "Optional starting hint (NOT a hard lock — evolve from experience/Judge gaps):\n"
        f"- Prefer checkable, concrete statements useful for {display_task}.\n"
        "- Do not invent facts. Let Coverage/Executability feedback decide format and rules.\n"
    )


_TASK_CTX: dict[str, Any] = {
    "display_task": "U.S. monthly VMT forecasting",
    "useful_signal_types": [
        "fuel_price",
        "holiday_travel",
        "extreme_weather",
        "policy",
        "freight",
        "remote_work",
    ],
    "soft_start_hint": "",
}


# Past Stage1 failures that destroyed residual/Fact usefulness. Always inject into Designer.
PANEL_SAFETY_RULES = """## MANDATORY panel-safety (never violate; past failures broke Stage2)
1. LONG PANEL: each row is ONE calendar period. Clean ONLY that row's input text for THAT date.
2. NO GLOBAL YEAR CUTOFF: never "keep 2020+ only", "discard pre-YYYY", "drop older than N years",
   or "if no recent data → empty". Historical rows must keep their own dated facts when present.
3. NO DATE STUFFING: never copy a recent release (e.g. 2024 BLS/EIA) into empty/old rows.
   If the row's source text is empty/NA/useless, output EMPTY for that row — do not invent fill.
4. NO HEADER-ONLY SHELLS: do not emit section titles with no facts ("## Regime… ## Noise" alone).
5. DATE ALIGNMENT (HARD): keep a fact only if its printed event/release date is within ±1 calendar
   year of the ROW's year (prefer ±3 months). Drop bullets whose year is off by ≥2 years, and drop
   recycled encyclopedia blocks that mix many distant years. Do NOT delete an on-year U.S. driver
   just because the same paragraph also contains off-year noise — strip the noise, keep the driver.
6. NO OVER-DELETE: empty output is allowed only when THIS row's source has no usable on-period
   target drivers after stripping noise. If a clear on-period U.S. driver remains, KEEP it
   (short output is fine; total wipe is not "safer").
7. MSE must NOT justify breaking (1)–(6). Residual/Fact need full-history row-aligned text.
"""


def apply_task_context(profile: Any) -> None:
    """Apply systems + observer/designer user-template context from DomainProfile."""
    apply_task_systems(profile.display_task)
    _TASK_CTX["display_task"] = profile.display_task
    _TASK_CTX["useful_signal_types"] = list(
        profile.useful_signal_types
        or _TASK_CTX["useful_signal_types"]
    )
    _TASK_CTX["soft_start_hint"] = str(getattr(profile, "soft_start_hint", "") or "")


# Defaults = traffic/VMT (backward compatible)
_apply = make_systems("U.S. monthly VMT forecasting")
OBSERVER_SYSTEM = _apply["observer"]
DESIGNER_SYSTEM = _apply["designer"]
CLEANER_SYSTEM = _apply["cleaner"]
JUDGE_SYSTEM = _apply["judge"]


def build_observer_user(
    samples: Sequence[str],
    existing: Sequence[str],
    *,
    older_note: str = "",
) -> str:
    """Lightweight protocol equivalent of Observer tool loop.

    Host already executed read_experience(); model only proposes NEW issues;
    host will call write_experience() for each returned item.
    Visible list is typically the recent window; older_note covers compacted themes.
    """
    existing_txt = "\n".join(f"- {x}" for x in existing) or "(none yet — empty experience database)"
    older = f"\n{older_note}\n" if older_note else "\n"
    blocks = []
    for i, s in enumerate(samples, 1):
        blocks.append(f"**Raw Data Example {i}:**\n{s[:2500]}\n")
    task = _TASK_CTX["display_task"]
    signals = _TASK_CTX["useful_signal_types"]
    # Keep historical VMT wording when on the traffic default task (no prompt drift).
    if task == "U.S. monthly VMT forecasting":
        examine = (
            "Below are {n} randomly sampled texts. Look for problems that hurt U.S. monthly\n"
            "VMT forecasting usefulness (non-U.S./foreign digressions, ads, fluff, missing\n"
            "date/place/entity, invented VMT causality, direction/strength tags, encoding noise, etc.)."
        ).format(n=len(samples))
    else:
        examine = (
            f"Below are {len(samples)} randomly sampled texts. Look for problems that hurt "
            f"{task} usefulness (non-U.S./foreign digressions, ads, fluff, missing "
            "date/place/entity, invented causality, direction/strength tags, encoding noise, etc.)."
        )
    signal_json = json.dumps(list(signals))
    return f"""# Data Quality Observation Task (lightweight DataEvolve protocol)

## Step 1 — read_experience() [ALREADY DONE BY HOST]
Recent / visible experience issues (host may hide older near-duplicates):
{existing_txt}
{older}
Do NOT duplicate these or rephrase older themes. Build on them only by finding genuinely NEW issues.

## Step 2 — Independently examine raw samples
{examine}

{chr(10).join(blocks)}

## Step 3 — Propose issues for write_experience()
Return ONLY new, distinct, actionable issues (one problem per string).
Also list useful_signal_types that should be preserved as checkable facts.

Return JSON:
{{
  "new_issues": ["clear description of one new issue", "..."],
  "useful_signal_types": {signal_json}
}}
"""


def build_designer_user(
    experiences: Sequence[str],
    best: Optional[dict[str, Any]],
    *,
    older_note: str = "",
) -> str:
    exp = "\n".join(f"- {x}" for x in experiences) or "(none)"
    older = f"\n{older_note}\n" if older_note else "\n"
    task = _TASK_CTX["display_task"]
    if best:
        hist = f"""## Previous best strategy (id={best.get('id')}, score={best.get('score_objective')})
```
{best.get('prompt_content')}
```
Diagnosis:
{best.get('clean_analysis', 'N/A')}

Evolve incrementally: keep what worked, fix diagnosed failures.
"""
    else:
        if task == "U.S. monthly VMT forecasting":
            hist = "## First generation: create a baseline cleaning strategy for U.S. VMT news from experiences.\n"
        else:
            hist = f"## First generation: create a baseline cleaning strategy for {task} news from experiences.\n"

    hint = (_TASK_CTX.get("soft_start_hint") or "").strip()
    hint_block = f"\n## Domain soft-start / panel context (MUST honor)\n{hint}\n" if hint else "\n"
    return f"""# Design cleaning strategy prompt

{hist}
{hint_block}
{PANEL_SAFETY_RULES}

## Experience pool (visible recent issues) — evolve rules to cover these gaps
{exp}
{older}

{FACT_FORMAT_CONSTRAINTS}

Return JSON:
{{
  "cleaning_prompt": "full strategy text the cleaner will follow",
  "design_rationale": "short rationale tied to experience / judge gaps",
  "key_improvements": ["..."]
}}
"""


def build_cleaner_user(strategy: str, raw_text: str) -> str:
    return f"""# Cleaning strategy
{strategy}

# Original text
{raw_text}

# Output
Cleaned text only (follow the strategy; no markdown fences, no commentary):
"""


def build_judge_user(
    strategy: str,
    pairs: Sequence[dict[str, str]],
    experiences: Sequence[str],
    iteration: int,
    *,
    older_note: str = "",
    pair_char_limit: int = 700,
    strategy_char_limit: int = 1600,
) -> str:
    """DataEvolve-aligned Quality Judge (lightweight tool protocol).

    Host already called read_experience(); returned new_issues are written by
    host via write_experience(). `experiences` should be the visible window only;
    host near-dedups writes against the full DB.

    Keep the user prompt compact so multi-batch judging fits context and
    max_tokens without truncating JSON.
    """
    exp = "\n".join(f"- {x}" for x in experiences) or "(none yet)"
    older = f"\n{older_note}\n" if older_note else "\n"
    strat = (strategy or "")[: max(400, int(strategy_char_limit))]
    if len(strategy or "") > len(strat):
        strat += "\n...(truncated)..."
    blocks = []
    lim = max(300, int(pair_char_limit))
    for i, p in enumerate(pairs, 1):
        blocks.append(
            f"### Pair {i}\n"
            f"ORIG:\n{(p.get('text') or '')[:lim]}\n\n"
            f"CLEAN:\n{(p.get('refined_text') or '')[:lim]}\n"
        )
    return f"""# Quality Judge batch {iteration} (compact JSON)

## Known issues (visible; do NOT paraphrase)
{exp}
{older}
## Cleaning prompt
{strat}

## Task (be brief)
1) Score each pair 1-10 (short comment).
2) Coverage: what known issues are addressed vs missing (≤40 words).
3) Executability: clear vs vague / execution fails (≤40 words).
4) new_issues: ≤4 genuinely NEW residuals after cleaning (one short sentence each).

## Data pairs
{chr(10).join(blocks)}

Return JSON only:
{{
  "pair_evaluations": [{{"pair_id": 1, "score": 8.0, "comment": "brief"}}],
  "average_score": 7.5,
  "coverage_analysis": "≤40 words",
  "executability_analysis": "≤40 words",
  "clean_analysis": "Coverage: ...; Executability: ...",
  "new_issues": ["short NEW residual", "..."],
  "suggestions": ["HIGH/MEDIUM/LOW: one fix"]
}}
"""
