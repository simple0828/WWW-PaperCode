"""APO-inspired diverse Designer for DataEvolve-VMT cleaning strategies.

Upstream DataEvolve Prompt Designer explicitly says \"evolution, not a revolution\"
and mutates a single best parent — that encourages regurgitation.

APO (language/apo/beam_trainer.py) counters mode collapse with:
  - paradigm catalog injection + paradigm transfer
  - high mutation temperature
  - Jaccard diversity gate (drop near-duplicates)

This module ports that pattern to cleaning-strategy prompts.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from language import VLLMClient
from language.dataevolve_vmt import prompts as de_prompts
from language.dataevolve_vmt.clean_ops import AlternatingConfig, _extract_json
from language.dataevolve_vmt.strategy_families import (
    SEED_F0_LIGHT,
    SEED_F1_FILTER,
    SEED_F3_ENTITY,
    SEED_F4_WINDOW,
    StrategySeed,
)

CATALOG: dict[str, StrategySeed] = {
    "F0": SEED_F0_LIGHT,
    "F1": SEED_F1_FILTER,
    "F3": SEED_F3_ENTITY,
    "F4": SEED_F4_WINDOW,
}
CATALOG_CYCLE = ["F1", "F4", "F3", "F0"]

# Domain-templated snippets (traffic defaults preserve historical wording).
_SIGNAL_HINT = "useful VMT signal"
_PARADIGM_HINT = (
    "U.S. VMT movers (fuel, holiday, weather, policy, freight) over style edits."
)


_PANEL_SAFETY_SHORT = (
    "LONG panel: clean EACH row for ITS date only. "
    "FORBIDDEN: global year cutoffs; stuffing recent releases into empty/old rows; "
    "header-only empty shells; facts whose printed year is ≥2 years from the row year. "
    "Strip off-year noise but KEEP on-year U.S. drivers (do not over-delete). "
    "Empty source → empty output (do not invent fill)."
)


def apply_designer_domain(profile: Any) -> None:
    global _SIGNAL_HINT, _PARADIGM_HINT
    task = getattr(profile, "display_task", "") or ""
    hint = str(getattr(profile, "soft_start_hint", "") or "").strip()
    key = getattr(profile, "key", "") or ""
    if key == "traffic" or "VMT" in task:
        _SIGNAL_HINT = "useful VMT signal"
        _PARADIGM_HINT = (
            "U.S. VMT movers (fuel, holiday, weather, policy, freight) over style edits. "
            + _PANEL_SAFETY_SHORT
        )
    elif key in ("agriculture", "agri") or "broiler" in task.lower():
        _SIGNAL_HINT = f"useful signal for {task}"
        _PARADIGM_HINT = (
            "LONG monthly panel (~1983–2024): clean EACH row for its own calendar month. "
            "NEVER invent global year cutoffs (2020+ only / discard pre-2020 / empty if no 2020+). "
            "Keep broiler movers (feed, supply, stocks, disease, trade, demand, weather) "
            "aligned to the row date; drop ads/recipes. "
            + _PANEL_SAFETY_SHORT
            + (" " + hint[:200] if hint else "")
        )
    else:
        _SIGNAL_HINT = f"useful signal for {task}"
        _PARADIGM_HINT = (
            (hint[:280] if hint else f"task-relevant movers for {task}")
            + " "
            + _PANEL_SAFETY_SHORT
        )


def _panel_safety_block() -> str:
    """Full panel-safety text for Designer user prompts."""
    from language.dataevolve_vmt.prompts import PANEL_SAFETY_RULES

    hint = (de_prompts._TASK_CTX.get("soft_start_hint") or "").strip()
    parts = [PANEL_SAFETY_RULES]
    if hint:
        parts.append(f"## Domain soft-start / panel context (MUST honor)\n{hint}")
    return "\n".join(parts)


def _norm_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _norm_prompt(text)))


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def is_diverse(prompt: str, existing: Sequence[str], threshold: float) -> bool:
    """True if prompt is NOT too similar to any existing (APO-style)."""
    for other in existing:
        if jaccard(prompt, other) >= threshold:
            return False
    return True


def _ask_design_json(
    client: VLLMClient,
    *,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    raw = client.ask(
        user,
        system=de_prompts.DESIGNER_SYSTEM,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("designer did not return a JSON object")
    return payload


def _payload_to_seed(
    payload: dict[str, Any],
    *,
    round_id: int,
    cand_id: int,
    mode: str,
    fallback: StrategySeed,
) -> StrategySeed:
    family = str(payload.get("strategy_family", fallback.family)).strip() or fallback.family
    # Normalize family tag to catalog root when possible
    fam_root = family.split("_")[0].upper()
    if fam_root.startswith("F") and len(fam_root) <= 3:
        family = fam_root
    fmt = str(payload.get("output_format", fallback.output_format)).strip() or fallback.output_format
    if fmt not in ("plain", "bullets", "json"):
        fmt = fallback.output_format
    prompt = str(payload.get("cleaning_prompt", "")).strip()
    if not prompt:
        raise ValueError("empty cleaning_prompt")
    # Soft-guard only against known-bad tag formats; do not lock output schema.
    bad = ("direction=", "strength=")
    if any(b.lower() in prompt.lower() for b in bad):
        prompt = (
            prompt
            + "\n\nNOTE: Prefer not to use direction=/strength= tag formats unless "
            "experience feedback explicitly requires them."
        )
    return StrategySeed(
        family=f"{family}_r{round_id}c{cand_id}",
        name=mode,
        output_format=fmt,  # type: ignore[arg-type]
        cleaning_prompt=prompt,
    )


def design_exploit(
    client: VLLMClient,
    *,
    experiences: list[str],
    parent: dict[str, Any],
    cfg: AlternatingConfig,
    round_id: int,
    cand_id: int,
    existing_prompts: Sequence[str],
    diversity_jaccard: float,
    temperature: float,
    max_retries: int = 3,
) -> StrategySeed:
    """Conservative refine of parent; reject near-duplicates (APO protegi-like)."""
    from language.dataevolve_vmt.experience_store import (
        DEFAULT_VISIBLE_K,
        older_theme_summary,
        visible_experiences,
    )

    parent_prompt = parent.get("cleaning_prompt") or ""
    visible = visible_experiences(experiences, k=DEFAULT_VISIBLE_K)
    older_note = older_theme_summary(experiences, visible_k=DEFAULT_VISIBLE_K)
    exp = "\n".join(f"- {x}" for x in visible) or "(none)"
    older = f"\n{older_note}\n" if older_note else "\n"
    base_user = f"""# Design an IMPROVED cleaning strategy (exploit / refine)

## Parent strategy to evolve (MUST keep continuity, but MUST change rules)
family={parent.get('family')} score={parent.get('score_objective')}
```
{parent_prompt}
```
Parent diagnosis: {parent.get('clean_analysis', 'N/A')}

## Experience database (visible recent; host near-dedups full DB)
Use these known / residual issues; do not ignore Coverage/Executability feedback.
{exp}
{older}

{_panel_safety_block()}

## Hard requirements
1. Return JSON keys: strategy_family, output_format, cleaning_prompt, design_rationale, key_improvements
2. cleaning_prompt must NOT be a near-copy of the parent.
3. Change concrete rules to close Coverage/Executability gaps in the experience feedback.
4. Prefer light cleaning unless feedback says noise remains; avoid over-deletion if MSE/signal feedback warns.
5. output_format may be plain, bullets, or json — choose what experience/Judge gaps justify.
6. cleaning_prompt MUST restate panel-safety (no year cutoff / no date stuffing / no empty shells / row-date align).
"""
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            payload = _ask_design_json(
                client,
                user=base_user
                + (
                    f"\n## Retry {attempt+1}: previous draft was too similar to parent/peers; "
                    "make a LARGER rule change."
                    if attempt
                    else ""
                ),
                temperature=temperature + 0.1 * attempt,
                max_tokens=cfg.max_tokens,
            )
            seed = _payload_to_seed(
                payload,
                round_id=round_id,
                cand_id=cand_id,
                mode="exploit",
                fallback=SEED_F0_LIGHT,
            )
            if _norm_prompt(seed.cleaning_prompt) == _norm_prompt(parent_prompt):
                last_err = "identical to parent"
                continue
            if not is_diverse(
                seed.cleaning_prompt,
                list(existing_prompts) + [parent_prompt],
                diversity_jaccard,
            ):
                last_err = "not diverse enough"
                continue
            return seed
        except Exception as exc:
            last_err = str(exc)
    print(f"  exploit fallback after retries ({last_err})", flush=True)
    # Force a small deterministic tweak so we don't emit an identical prompt.
    tweaked = (
        parent_prompt.rstrip()
        + "\nAlso delete phone numbers, unsubscribe lines, and cookie/consent banners."
    )
    return StrategySeed(
        family=f"F0_r{round_id}c{cand_id}",
        name="exploit_fallback",
        output_format="plain",
        cleaning_prompt=tweaked,
    )


def design_explore(
    client: VLLMClient,
    *,
    experiences: list[str],
    parent: dict[str, Any],
    cfg: AlternatingConfig,
    round_id: int,
    cand_id: int,
    existing_prompts: Sequence[str],
    diversity_jaccard: float,
    temperature: float,
    forbidden_family: Optional[str] = None,
    max_retries: int = 3,
) -> StrategySeed:
    """Explore a different family/format than parent (APO explore)."""
    parent_prompt = parent.get("cleaning_prompt") or ""
    parent_fam = str(parent.get("family", "F0")).split("_")[0]
    forbid = forbidden_family or parent_fam
    targets = [f for f in list(CATALOG.keys()) if f != forbid]
    if not targets:
        targets = list(CATALOG.keys()) or ["F0"]
    from language.dataevolve_vmt.experience_store import (
        DEFAULT_VISIBLE_K,
        older_theme_summary,
        visible_experiences,
    )

    visible = visible_experiences(experiences, k=DEFAULT_VISIBLE_K)
    older_note = older_theme_summary(experiences, visible_k=DEFAULT_VISIBLE_K)
    exp = "\n".join(f"- {x}" for x in visible) or "(none)"
    older = f"\n{older_note}\n" if older_note else "\n"
    base_user = f"""# Design a DIVERSE cleaning strategy (explore / paradigm shift)

## Parent (for contrast only — do NOT copy)
family={parent.get('family')}
```
{parent_prompt[:1200]}
```

## Experience database (visible recent; host near-dedups full DB)
{exp}
{older}

{_panel_safety_block()}

## Hard requirements
1. strategy_family MUST be one of {targets} (NOT {forbid}).
2. output_format may be plain, bullets, or json.
3. cleaning_prompt must be clearly different in structure AND deletion/preserve policy
   (Jaccard-distinct from parent and other candidates this round).
4. Close Coverage/Executability gaps from experience; do not over-delete {_SIGNAL_HINT}.
5. Return JSON: strategy_family, output_format, cleaning_prompt, design_rationale, key_improvements
6. cleaning_prompt MUST restate panel-safety (no year cutoff / no date stuffing / no empty shells / row-date align).
"""
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            payload = _ask_design_json(
                client,
                user=base_user
                + (f"\n## Retry {attempt+1}: draft too similar; switch family/rules harder." if attempt else ""),
                temperature=temperature + 0.05 * attempt,
                max_tokens=cfg.max_tokens,
            )
            seed = _payload_to_seed(
                payload,
                round_id=round_id,
                cand_id=cand_id,
                mode="explore",
                fallback=SEED_F1_FILTER,
            )
            fam_root = seed.family.split("_")[0]
            if fam_root == forbid and attempt < max_retries - 1:
                last_err = f"still family {forbid}"
                continue
            if not is_diverse(
                seed.cleaning_prompt,
                list(existing_prompts) + [parent_prompt],
                diversity_jaccard,
            ):
                last_err = "not diverse enough"
                continue
            return seed
        except Exception as exc:
            last_err = str(exc)
    # Catalog fallback: pick first diverse seed family
    for fam in targets:
        s = CATALOG[fam]
        if is_diverse(s.cleaning_prompt, existing_prompts, diversity_jaccard):
            print(f"  explore fallback → catalog {fam} ({last_err})", flush=True)
            return StrategySeed(
                family=f"{fam}_r{round_id}c{cand_id}",
                name="explore_catalog_fallback",
                output_format=s.output_format,
                cleaning_prompt=s.cleaning_prompt,
            )
    s = CATALOG[targets[0]]
    return StrategySeed(
        family=f"{targets[0]}_r{round_id}c{cand_id}",
        name="explore_catalog_fallback",
        output_format=s.output_format,
        cleaning_prompt=s.cleaning_prompt + f"\n(Round {round_id} explore variant {cand_id})",
    )


def design_paradigm_catalog(
    *,
    target_family: str,
    round_id: int,
    cand_id: int,
    existing_prompts: Sequence[str],
    diversity_jaccard: float,
    parent: Optional[dict[str, Any]] = None,
) -> StrategySeed:
    """Inject a seed paradigm (APO paradigm_catalog). Optionally lightly annotate."""
    seed = CATALOG[target_family]
    prompt = seed.cleaning_prompt
    if parent and not is_diverse(prompt, [parent.get("cleaning_prompt") or ""], diversity_jaccard):
        prompt = (
            prompt
            + f"\nExtra emphasis for round {round_id}: prioritize forecast-relevant "
            f"{_PARADIGM_HINT}"
        )
    if not is_diverse(prompt, existing_prompts, diversity_jaccard):
        prompt = prompt + f"\n[paradigm={target_family} r{round_id}c{cand_id}]"
    return StrategySeed(
        family=f"{target_family}_r{round_id}c{cand_id}",
        name=f"paradigm_catalog_{target_family}",
        output_format=seed.output_format,
        cleaning_prompt=prompt,
    )


def design_paradigm_transfer(
    client: VLLMClient,
    *,
    parent: dict[str, Any],
    target_family: str,
    cfg: AlternatingConfig,
    round_id: int,
    cand_id: int,
    existing_prompts: Sequence[str],
    diversity_jaccard: float,
    temperature: float,
    max_retries: int = 3,
) -> StrategySeed:
    """Rewrite parent into another family's format (APO paradigm_transfer)."""
    target = CATALOG[target_family]
    parent_prompt = parent.get("cleaning_prompt") or ""
    base_user = f"""# Paradigm transfer of a cleaning strategy

## Parent prompt
```
{parent_prompt}
```

## TARGET paradigm ({target_family} / {target.name})
output_format={target.output_format}
Reference style:
```
{target.cleaning_prompt}
```

## Task
Rewrite the parent into the TARGET paradigm's output format and deletion style.
Keep the forecasting goal (lower CFA test MSE). Do not copy parent verbatim.
Return JSON: strategy_family (use {target_family}), output_format, cleaning_prompt,
design_rationale, key_improvements
"""
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            payload = _ask_design_json(
                client,
                user=base_user + (f"\nRetry {attempt+1}: increase format distance from parent." if attempt else ""),
                temperature=temperature,
                max_tokens=cfg.max_tokens,
            )
            payload["strategy_family"] = target_family
            payload["output_format"] = target.output_format
            seed = _payload_to_seed(
                payload,
                round_id=round_id,
                cand_id=cand_id,
                mode=f"paradigm_transfer_{target_family}",
                fallback=target,
            )
            if not is_diverse(
                seed.cleaning_prompt,
                list(existing_prompts) + [parent_prompt],
                diversity_jaccard,
            ):
                last_err = "not diverse enough"
                continue
            return seed
        except Exception as exc:
            last_err = str(exc)
    print(f"  paradigm_transfer fallback → catalog {target_family} ({last_err})", flush=True)
    return design_paradigm_catalog(
        target_family=target_family,
        round_id=round_id,
        cand_id=cand_id,
        existing_prompts=existing_prompts,
        diversity_jaccard=diversity_jaccard,
        parent=parent,
    )


def design_round_candidates(
    client: VLLMClient,
    *,
    experiences: list[str],
    parents: list[dict[str, Any]],
    cfg: AlternatingConfig,
    round_id: int,
    mutations_per_parent: int,
    diversity_jaccard: float = 0.78,
    paradigm_shift_ratio: float = 0.5,
    temperature_exploit: float = 0.55,
    temperature_explore: float = 0.9,
) -> list[StrategySeed]:
    """Build a diverse candidate set for one outer round.

    For each beam parent, allocate mutations like APO:
      paradigm_budget ≈ ratio * mutations_per_parent
      remainder = exploit/explore LLM edits
    Always continues from ``parents`` (previous beam), never reseeds from scratch
    after round 1 (callers pass catalog seeds only for round 1).
    """
    seeds: list[StrategySeed] = []
    existing: list[str] = [p.get("cleaning_prompt") or "" for p in parents]
    catalog_cycle = list(CATALOG_CYCLE) if CATALOG_CYCLE else list(CATALOG.keys())
    if not catalog_cycle:
        catalog_cycle = ["F0", "F1", "F3", "F4"]
    cat_i = 0
    cand_id = 0

    for parent in parents:
        n_mut = max(1, int(mutations_per_parent))
        n_paradigm = max(1, int(round(n_mut * paradigm_shift_ratio))) if n_mut > 1 else (
            1 if paradigm_shift_ratio >= 0.5 else 0
        )
        n_edit = n_mut - n_paradigm
        parent_fam = str(parent.get("family", "F0")).split("_")[0]

        # --- LLM edits (continue evolution from this parent) ---
        for j in range(n_edit):
            cand_id += 1
            # Alternate exploit/explore across parents & edit slots so a 1-edit
            # budget still explores (not all F0-refine).
            try:
                p_idx = parents.index(parent)
            except ValueError:
                p_idx = 0
            use_explore = ((p_idx + j) % 2 == 1)
            if not use_explore:
                seed = design_exploit(
                    client,
                    experiences=experiences,
                    parent=parent,
                    cfg=cfg,
                    round_id=round_id,
                    cand_id=cand_id,
                    existing_prompts=existing,
                    diversity_jaccard=diversity_jaccard,
                    temperature=temperature_exploit,
                )
            else:
                seed = design_explore(
                    client,
                    experiences=experiences,
                    parent=parent,
                    cfg=cfg,
                    round_id=round_id,
                    cand_id=cand_id,
                    existing_prompts=existing,
                    diversity_jaccard=diversity_jaccard,
                    temperature=temperature_explore,
                    forbidden_family=parent_fam,
                )
            seeds.append(seed)
            existing.append(seed.cleaning_prompt)
            print(
                f"  designed {seed.family} mode={seed.name} "
                f"from={parent.get('family')} chars={len(seed.cleaning_prompt)} "
                f"jacc_parent={jaccard(seed.cleaning_prompt, parent.get('cleaning_prompt') or ''):.3f}",
                flush=True,
            )

        # --- Paradigm shift slots ---
        for j in range(n_paradigm):
            cand_id += 1
            # Pick a catalog family different from parent when possible
            target = catalog_cycle[cat_i % len(catalog_cycle)]
            cat_i += 1
            if target == parent_fam:
                target = catalog_cycle[cat_i % len(catalog_cycle)]
                cat_i += 1
            if j % 2 == 0:
                seed = design_paradigm_catalog(
                    target_family=target,
                    round_id=round_id,
                    cand_id=cand_id,
                    existing_prompts=existing,
                    diversity_jaccard=diversity_jaccard,
                    parent=parent,
                )
                # If still colliding with existing, transfer via LLM
                if not is_diverse(seed.cleaning_prompt, existing, diversity_jaccard):
                    seed = design_paradigm_transfer(
                        client,
                        parent=parent,
                        target_family=target,
                        cfg=cfg,
                        round_id=round_id,
                        cand_id=cand_id,
                        existing_prompts=existing,
                        diversity_jaccard=diversity_jaccard,
                        temperature=temperature_explore,
                    )
            else:
                seed = design_paradigm_transfer(
                    client,
                    parent=parent,
                    target_family=target,
                    cfg=cfg,
                    round_id=round_id,
                    cand_id=cand_id,
                    existing_prompts=existing,
                    diversity_jaccard=diversity_jaccard,
                    temperature=temperature_explore,
                )
            seeds.append(seed)
            existing.append(seed.cleaning_prompt)
            print(
                f"  designed {seed.family} mode={seed.name} "
                f"from={parent.get('family')} chars={len(seed.cleaning_prompt)} "
                f"jacc_parent={jaccard(seed.cleaning_prompt, parent.get('cleaning_prompt') or ''):.3f}",
                flush=True,
            )

    return seeds
