"""Time-MMD domain registry for Stage1/Stage2 CFA args.

Traffic + agriculture keep their dedicated builders; other domains use this table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TimeMMDDomainSpec:
    key: str
    data_subdir: str
    data_path: str
    model_id_prefix: str
    seasonal: str = "Monthly"  # or Weekly
    text_col: str = "Final_Search_4"
    lr_multiplier: float = 2.0


# Domains beyond traffic/agriculture (already wired separately).
TIMEMMD_EXTRA: dict[str, TimeMMDDomainSpec] = {
    "economy": TimeMMDDomainSpec(
        key="economy",
        data_subdir="Economy",
        data_path="US_TradeBalance_Month.csv",
        model_id_prefix="Economy",
    ),
    "security": TimeMMDDomainSpec(
        key="security",
        data_subdir="Security",
        data_path="US_FEMAGrant_Month.csv",
        model_id_prefix="Security",
    ),
    "energy": TimeMMDDomainSpec(
        key="energy",
        data_subdir="Energy",
        data_path="US_GasolinePrice_Week.csv",
        model_id_prefix="Energy",
        seasonal="Weekly",
    ),
    "public_health": TimeMMDDomainSpec(
        key="public_health",
        data_subdir="Public_Health",
        data_path="US_FLURATIO_Week.csv",
        model_id_prefix="PublicHealth",
        seasonal="Weekly",
    ),
    "social_good": TimeMMDDomainSpec(
        key="social_good",
        data_subdir="SocialGood",
        data_path="Unadj_UnemploymentRate_ALL_processed.csv",
        model_id_prefix="SocialGood",
    ),
}

ALIASES = {
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
}


def normalize_domain(domain: str) -> str:
    key = (domain or "").strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def get_timemmd_spec(domain: str) -> TimeMMDDomainSpec:
    key = normalize_domain(domain)
    if key not in TIMEMMD_EXTRA:
        raise KeyError(f"Unknown Time-MMD extra domain={domain!r}; known={sorted(TIMEMMD_EXTRA)}")
    return TIMEMMD_EXTRA[key]


def csv_path(spec: TimeMMDDomainSpec) -> Path:
    return ROOT / "data" / spec.data_subdir / spec.data_path


def default_bundle_path(spec: TimeMMDDomainSpec) -> Path:
    return ROOT / "artifacts" / f"apo_{spec.key}" / "runpy_cfa_bundle.pt"
