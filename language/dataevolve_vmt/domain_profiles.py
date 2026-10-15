"""Domain profiles for Stage1 DataEvolve — loaded from shared seed prompt YAML.

Edit prompts / markers / defaults in: language/dataevolve_vmt/stage1_seed_prompts.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from language.dataevolve_vmt.strategy_families import StrategySeed

_YAML = Path(__file__).with_name("stage1_seed_prompts.yaml")
_YAML_EXTRA = Path(__file__).with_name("stage1_seed_prompts_extra.yaml")

# CLI sentinel defaults (must match scripts/run_dataevolve_cfa_strict.py argparse).
TRAFFIC_CLI_INIT_DEFAULT = (
    "artifacts/ctrl_raw_cfa_continue_gettext_fixed/"
    "cfa_bundle_raw_continue30_gettext_fixed.pt"
)
TRAFFIC_CLI_OUTPUT_DEFAULT = "artifacts/dataevolve_cfa_selfevo_emergent"

_TRAFFIC_MARKERS = (
    "vmt",
    "vehicle miles",
    "travel volume",
    "fhwa",
    "bts.gov",
    "highway",
    "congestion",
    "fuel price",
    "gasoline",
    "recession",
    "inflation",
)
_TRAFFIC_SIGNAL_TYPES = (
    "fuel_price",
    "holiday_travel",
    "extreme_weather",
    "policy",
    "freight",
    "remote_work",
)


@dataclass(frozen=True)
class DomainProfile:
    key: str
    display_task: str
    soft_start_hint: str
    raw_csv: str
    cleaned_csv_name: str
    text_col: str = "Final_Search_4"
    apo_domain: str = "traffic"
    beam_priority: tuple[str, ...] = ()
    default_init_bundle: str = ""
    default_output_dir: str = ""
    canonical_artifact_dir: str = ""
    useful_signal_types: tuple[str, ...] = ()
    useful_markers: tuple[str, ...] = ()


def _aliases(domain: str) -> str:
    key = (domain or "traffic").strip().lower().replace("-", "_")
    return {
        "vmt": "traffic",
        "agri": "agriculture",
        "algriculture": "agriculture",
        "health": "public_health",
        "publichealth": "public_health",
        "flu": "public_health",
        "social": "social_good",
        "socialgood": "social_good",
        "unemployment": "social_good",
        "fema": "security",
        "trade": "economy",
        "gas": "energy",
        "gasoline": "energy",
    }.get(key, key)


def _load_yaml() -> dict[str, Any]:
    data = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid seed prompt YAML: {_YAML}")
    if _YAML_EXTRA.is_file():
        extra = yaml.safe_load(_YAML_EXTRA.read_text(encoding="utf-8")) or {}
        if not isinstance(extra, dict):
            raise ValueError(f"Invalid extra YAML: {_YAML_EXTRA}")
        overlap = set(data) & set(extra)
        if overlap:
            raise ValueError(f"Domain keys duplicated in extra YAML: {sorted(overlap)}")
        data = {**data, **extra}
    return data


def _as_str_tuple(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return tuple(str(x).strip() for x in raw if str(x).strip())


def _profile_from_block(key: str, block: dict[str, Any]) -> DomainProfile:
    prio = block.get("beam_priority") or list((block.get("seeds") or {}).keys())
    markers = _as_str_tuple(block.get("useful_markers"))
    signals = _as_str_tuple(block.get("useful_signal_types"))
    if key == "traffic":
        markers = markers or _TRAFFIC_MARKERS
        signals = signals or _TRAFFIC_SIGNAL_TYPES
    return DomainProfile(
        key=key,
        display_task=str(block["display_task"]),
        soft_start_hint=str(block.get("soft_start_hint") or "").strip(),
        raw_csv=str(block["raw_csv"]),
        cleaned_csv_name=str(block["cleaned_csv_name"]),
        text_col=str(block.get("text_col") or "Final_Search_4"),
        apo_domain=str(block.get("apo_domain") or key),
        beam_priority=tuple(str(x) for x in prio),
        default_init_bundle=str(block.get("default_init_bundle") or ""),
        default_output_dir=str(block.get("default_output_dir") or ""),
        canonical_artifact_dir=str(block.get("canonical_artifact_dir") or ""),
        useful_signal_types=signals,
        useful_markers=markers,
    )


def _seeds_from_block(block: dict[str, Any]) -> tuple[StrategySeed, ...]:
    seeds_blk = block.get("seeds") or {}
    if not seeds_blk:
        raise ValueError(f"No seeds in {_YAML}")
    out: list[StrategySeed] = []
    for fam, s in seeds_blk.items():
        out.append(
            StrategySeed(
                family=str(fam),
                name=str(s.get("name") or str(fam).lower()),
                output_format=str(s.get("output_format") or "plain"),  # type: ignore[arg-type]
                cleaning_prompt=str(s["cleaning_prompt"]).strip() + "\n",
            )
        )
    if len(out) < 3:
        raise ValueError(f"Need ≥3 seeds in {_YAML}")
    return tuple(out)


def get_profile(domain: str) -> DomainProfile:
    key = _aliases(domain)
    data = _load_yaml()
    if key not in data:
        raise KeyError(f"Unknown domain={domain!r}; YAML domains={sorted(data)}")
    return _profile_from_block(key, data[key])


def seeds_for_domain(domain: str) -> tuple[StrategySeed, ...]:
    key = _aliases(domain)
    data = _load_yaml()
    if key not in data:
        raise KeyError(f"Unknown domain={domain!r}; YAML domains={sorted(data)}")
    block = data[key]
    seeds = {s.family: s for s in _seeds_from_block(block)}
    prio = [str(x) for x in (block.get("beam_priority") or list(seeds.keys()))]
    ordered: list[StrategySeed] = []
    for fam in prio:
        if fam in seeds:
            ordered.append(seeds[fam])
    for fam, s in seeds.items():
        if fam not in prio:
            ordered.append(s)
    return tuple(ordered)


def list_domains() -> list[str]:
    return sorted(_load_yaml().keys())


def resolve_stage1_paths(
    domain: str,
    *,
    init_bundle: Optional[str] = None,
    output_dir: Optional[str] = None,
    raw_csv: Optional[str] = None,
    cleaned_csv_name: Optional[str] = None,
) -> dict[str, str]:
    """Fill CLI/path defaults from profile without changing traffic CLI sentinels."""
    profile = get_profile(domain)
    init = (init_bundle or "").strip() or TRAFFIC_CLI_INIT_DEFAULT
    out = (output_dir or "").strip() or TRAFFIC_CLI_OUTPUT_DEFAULT
    if profile.key != "traffic":
        if init == TRAFFIC_CLI_INIT_DEFAULT:
            if not profile.default_init_bundle:
                raise SystemExit(
                    f"--domain={domain} needs --init-bundle "
                    f"(or default_init_bundle in stage1_seed_prompts.yaml)."
                )
            init = profile.default_init_bundle
        if out == TRAFFIC_CLI_OUTPUT_DEFAULT and profile.default_output_dir:
            out = profile.default_output_dir
    return {
        "domain": profile.key,
        "init_bundle": init,
        "output_dir": out,
        "raw_csv": (raw_csv or "").strip() or profile.raw_csv,
        "cleaned_csv_name": (cleaned_csv_name or "").strip() or profile.cleaned_csv_name,
        "text_col": profile.text_col,
        "apo_domain": profile.apo_domain,
        "display_task": profile.display_task,
    }


def apply_stage1_domain(domain_or_profile: str | DomainProfile) -> DomainProfile:
    """Wire systems / heuristics / designer hints for this process."""
    profile = (
        domain_or_profile
        if isinstance(domain_or_profile, DomainProfile)
        else get_profile(domain_or_profile)
    )
    from language.dataevolve_vmt import designer_diverse as dd
    from language.dataevolve_vmt import heuristics as hz
    from language.dataevolve_vmt.prompts import apply_task_context
    from language.dataevolve_vmt.strategy_families import apply_domain_seeds

    apply_task_context(profile)
    hz.apply_domain_heuristics(profile)
    dd.apply_designer_domain(profile)
    apply_domain_seeds(profile.key)
    return profile
