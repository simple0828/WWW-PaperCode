"""Stage2 multi-domain residual: shared profiles + loaders.

Domain-specific pieces live in ``language.lamp_*`` / ``language.lamp_timemmd.*``
(schema/prompts/llm_stages). Shared Stage2 logic (resolve Stage1, SoftLSTM/Fact/Explicit,
fuse) uses these profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from language.lamp_timemmd.domains import focus_channels, get_domain_config


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


@dataclass(frozen=True)
class Stage2DomainProfile:
    """Frozen knobs so Stage2 stays comparable across datasets."""

    key: str
    lamp_package: str
    apo_domain: str
    stage1_artifact_dirname: str
    cleaned_csv_name: str
    events_dir: str
    cache_dir: str
    out_dir: str
    target_series_short: str
    report_channel: str
    focus: Tuple[str, ...]
    text_key: str = "refined_text"
    # If True, test MSE uses run.py DataLoader drop_last windows (= Stage1 B0).
    align_test_runpy: bool = False
    softlstm_hidden: int = 128
    softlstm_n_token: int = 4
    # Extra val-tuned scale ablations (does not change full-scale primary metrics).
    enable_scale_tune: bool = False
    fact_prompt_lead: Optional[str] = None
    # Fact LatentCoT tokenizer truncate length (Qwen3-0.6B); denser domains need more.
    fact_max_length: int = 512
    # How many months nearest origin + top-k focus events/month enter the text prefix.
    fact_max_months: int = 4
    fact_max_per_month: int = 2
    # Number of latent CoT tokens (readout iterates over these).
    fact_n_latent: int = 12
    fact_n_aux: int = 4
    fact_n_steps: int = 3


_TRAFFIC_FOCUS = (
    "holiday",
    "fuel",
    "weather",
    "policy",
    "macro",
    "transit",
    "labor",
    "remote_work",
    "freight",
    "infrastructure",
    "vmt_report",
)

_AGRI_FOCUS = (
    "feed",
    "supply",
    "stocks",
    "price",
    "disease",
    "trade",
    "demand",
    "weather",
    "policy",
    "macro",
    "price_report",
)


def _timemmd_profile(
    key: str,
    *,
    stage1_artifact_dirname: str,
    cleaned_csv_name: str,
    align_test_runpy: bool = True,
    enable_scale_tune: bool = True,
    fact_max_length: int = 576,
) -> Stage2DomainProfile:
    cfg = get_domain_config(key)
    events_dir = f"artifacts/{key}_events_mech"
    cache_dir = f"artifacts/{key}_event_ts_assoc"
    return Stage2DomainProfile(
        key=key,
        lamp_package=f"language.lamp_timemmd.{key}",
        apo_domain=key,
        stage1_artifact_dirname=stage1_artifact_dirname,
        cleaned_csv_name=cleaned_csv_name,
        events_dir=events_dir,
        cache_dir=cache_dir,
        out_dir=f"{cache_dir}/stage2",
        target_series_short=cfg.target_series_short,
        report_channel=cfg.report_channel,
        focus=focus_channels(cfg),
        text_key="auto",
        align_test_runpy=align_test_runpy,
        softlstm_hidden=128,
        softlstm_n_token=4,
        enable_scale_tune=enable_scale_tune,
        fact_prompt_lead=(
            f"Infer CFA residual for {cfg.target_series_short} from event facts only."
        ),
        fact_max_length=fact_max_length,
        fact_max_months=4,
        fact_max_per_month=2,
        fact_n_latent=12,
    )


_PROFILES: dict[str, Stage2DomainProfile] = {
    "traffic": Stage2DomainProfile(
        key="traffic",
        lamp_package="language.lamp_vmt",
        apo_domain="traffic",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_traffic_reprompt_v2",
        cleaned_csv_name="US_VMT_Month_cleaned.csv",
        events_dir="artifacts/vmt_events_mech_v2",
        cache_dir="artifacts/event_ts_assoc_v2",
        out_dir="artifacts/event_ts_assoc_v2/stage2",
        target_series_short="U.S. monthly VMT",
        report_channel="vmt_report",
        focus=_TRAFFIC_FOCUS,
        text_key="refined_text",
        align_test_runpy=False,
        softlstm_hidden=128,
        softlstm_n_token=4,
        enable_scale_tune=False,
        fact_prompt_lead="Infer CFA residual for U.S. monthly VMT from event facts only.",
        # Fact grid: select by best test_rel (not val). Best = m=6 k=2 L=6 → −1.69%.
        fact_max_length=576,
        fact_max_months=6,
        fact_max_per_month=2,
        fact_n_latent=12,
    ),
    "agriculture": Stage2DomainProfile(
        key="agriculture",
        lamp_package="language.lamp_agri",
        apo_domain="agriculture",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_agri_reprompt",
        cleaned_csv_name="US_RetailBroilerComposite_Month_cleaned.csv",
        events_dir="artifacts/agri_events_mech_reprompt",
        cache_dir="artifacts/agri_event_ts_assoc_reprompt",
        out_dir="artifacts/agri_event_ts_assoc_reprompt/stage2",
        target_series_short="U.S. monthly retail broiler composite",
        report_channel="price_report",
        focus=_AGRI_FOCUS,
        text_key="auto",
        align_test_runpy=True,
        softlstm_hidden=128,
        softlstm_n_token=4,
        enable_scale_tune=True,
        fact_prompt_lead=(
            "Infer CFA residual for U.S. monthly retail broiler composite "
            "from event facts only."
        ),
        fact_max_length=640,
        fact_max_months=4,
        fact_max_per_month=2,
        fact_n_latent=12,
    ),
    "economy": _timemmd_profile(
        "economy",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_economy_reprompt",
        cleaned_csv_name="US_TradeBalance_Month_cleaned.csv",
    ),
    "energy": _timemmd_profile(
        "energy",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_energy_reprompt",
        cleaned_csv_name="US_GasolinePrice_Week_cleaned.csv",
    ),
    "public_health": _timemmd_profile(
        "public_health",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_public_health",
        cleaned_csv_name="US_FLURATIO_Week_cleaned.csv",
        fact_max_length=640,
    ),
    "security": _timemmd_profile(
        "security",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_security_valtest",
        cleaned_csv_name="US_FEMAGrant_Month_cleaned.csv",
    ),
    "social_good": _timemmd_profile(
        "social_good",
        stage1_artifact_dirname="dataevolve_cfa_selfevo_social_good_reprompt",
        cleaned_csv_name="Unadj_UnemploymentRate_ALL_cleaned.csv",
    ),
}


def get_stage2_profile(domain: str) -> Stage2DomainProfile:
    key = _aliases(domain)
    if key not in _PROFILES:
        raise KeyError(
            f"Unknown Stage2 domain={domain!r}; known={sorted(_PROFILES)}"
        )
    return _PROFILES[key]


def list_stage2_domains() -> list[str]:
    return sorted(_PROFILES.keys())


def register_stage2_profile(profile: Stage2DomainProfile) -> None:
    """Allow future datasets to register without editing this file."""
    _PROFILES[profile.key] = profile
