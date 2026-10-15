"""Seed strategy families for Self-Evo VMT cleaning (weak R1 starting points).

Seeds are soft starting hints only. Later rounds should evolve rules from
experience / Coverage / Executability gaps — not re-lock a fixed schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OutputFormat = Literal["plain", "bullets", "json"]

# Soft R1 hint only — Designer mutations must NOT re-inject this as a hard lock.
SOFT_START_HINT = """\
Starting hint (soft): keep U.S. traffic/VMT-relevant concrete information;
drop obvious ads and non-U.S. digressions when clear; do not invent facts.
Let later experience/Judge feedback refine format and rules.
"""


@dataclass(frozen=True)
class StrategySeed:
    family: str
    name: str
    output_format: OutputFormat
    cleaning_prompt: str
    identity: bool = False


SEED_F0_LIGHT = StrategySeed(
    family="F0",
    name="light_clean",
    output_format="plain",
    cleaning_prompt=(
        "Lightly clean raw U.S. monthly traffic/VMT auxiliary news for forecasting.\n"
        "DEFAULT: keep source wording when already useful; strip only obvious noise.\n"
        f"{SOFT_START_HINT}"
    ),
)

SEED_F1_FILTER = StrategySeed(
    family="F1",
    name="strict_us_filter",
    output_format="plain",
    cleaning_prompt=(
        "Filter raw news toward U.S.-specific, concrete traffic/VMT-related content.\n"
        "Discard clear foreign digressions and product pitches when unrelated.\n"
        f"{SOFT_START_HINT}"
    ),
)

SEED_F4_WINDOW = StrategySeed(
    family="F4",
    name="window_brief",
    output_format="bullets",
    cleaning_prompt=(
        "Rewrite raw news into a short brief useful for U.S. monthly VMT forecasting.\n"
        "Prefer concrete observations (fuel, holiday, weather, policy, freight, etc.).\n"
        f"{SOFT_START_HINT}"
    ),
)

SEED_F3_ENTITY = StrategySeed(
    family="F3",
    name="entity_centric",
    output_format="bullets",
    cleaning_prompt=(
        "Extract entity-centric notes from raw U.S. traffic/VMT news "
        "(who/what, where, when, what happened) when available.\n"
        f"{SOFT_START_HINT}"
    ),
)

DEFAULT_SCREEN_SEEDS: tuple[StrategySeed, ...] = (
    SEED_F0_LIGHT,
    SEED_F1_FILTER,
    SEED_F4_WINDOW,
)


def seed_by_families(families: list[str], domain: str = "traffic") -> list[StrategySeed]:
    from language.dataevolve_vmt.domain_profiles import seeds_for_domain

    catalog = {s.family: s for s in seeds_for_domain(domain)}
    out = []
    for f in families:
        if f not in catalog:
            raise KeyError(f"Unknown family {f}; choose from {sorted(catalog)}")
        out.append(catalog[f])
    return out


def apply_domain_seeds(domain: str) -> dict[str, StrategySeed]:
    """Replace module-level seed constants + designer_diverse.CATALOG for this domain."""
    from language.dataevolve_vmt.domain_profiles import seeds_for_domain
    import language.dataevolve_vmt.designer_diverse as dd

    seed_list = list(seeds_for_domain(domain))
    seeds = {s.family: s for s in seed_list}
    global SEED_F0_LIGHT, SEED_F1_FILTER, SEED_F3_ENTITY, SEED_F4_WINDOW, DEFAULT_SCREEN_SEEDS

    def _get(*cands: str) -> StrategySeed:
        for c in cands:
            if c in seeds:
                return seeds[c]
        return seed_list[0]

    SEED_F0_LIGHT = _get("F0")
    SEED_F1_FILTER = _get("F1")
    SEED_F3_ENTITY = _get("F3", "F2")
    SEED_F4_WINDOW = _get("F4", "F5")
    DEFAULT_SCREEN_SEEDS = tuple(seed_list[:3])
    dd.CATALOG = dict(seeds)
    dd.SEED_F0_LIGHT = SEED_F0_LIGHT
    dd.SEED_F1_FILTER = SEED_F1_FILTER
    dd.SEED_F3_ENTITY = SEED_F3_ENTITY
    dd.SEED_F4_WINDOW = SEED_F4_WINDOW
    # explore cycle over all families present
    dd.CATALOG_CYCLE = [s.family for s in seed_list]
    return seeds
