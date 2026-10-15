"""Per-domain channel sets + extract-prompt templates for Time-MMD Stage2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class DomainLampConfig:
    key: str
    target_series_short: str
    expert_role: str
    report_channel: str
    period_label: str  # "month" or "week"
    channels: Tuple[str, ...]
    channel_defs: Dict[str, str]
    channel_aliases: Dict[str, str]
    priority_ties: Tuple[str, ...]
    mechanism_examples: Tuple[str, ...]
    chain_end_hint: str  # e.g. "trade_balance↑/↓"
    what_counts: str
    extract_task_line: str


def _cfg(**kwargs) -> DomainLampConfig:
    return DomainLampConfig(**kwargs)


DOMAIN_CONFIGS: Dict[str, DomainLampConfig] = {
    "economy": _cfg(
        key="economy",
        target_series_short="U.S. monthly goods & services trade balance",
        expert_role="senior U.S. international-trade / balance-of-payments economist",
        report_channel="trade_report",
        period_label="month",
        channels=(
            "exports",
            "imports",
            "tariff",
            "fx",
            "bilateral",
            "commodity",
            "policy",
            "macro",
            "trade_report",
            "other",
        ),
        channel_defs={
            "exports": "U.S. goods/services export levels, shipments, or foreign demand",
            "imports": "U.S. goods/services import levels, domestic absorption of foreign goods",
            "tariff": "tariffs, quotas, sanctions, trade remedies with trade-flow impact",
            "fx": "USD FX moves tied to trade competitiveness (not pure finance fluff)",
            "bilateral": "bilateral/regional balances, USMCA/China/EU trade-flow news",
            "commodity": "commodity shipment / energy / ag trade shocks moving the balance",
            "policy": "USTR/Commerce/domestic trade policy (non-tariff) affecting flows",
            "macro": "growth/recession demand that shifts import or export volumes",
            "trade_report": "official Census FT900 / BEA International Transactions prints",
            "other": "only if none of the above fit",
        },
        channel_aliases={
            "export": "exports",
            "import": "imports",
            "tariffs": "tariff",
            "sanctions": "tariff",
            "quota": "tariff",
            "dollar": "fx",
            "usd": "fx",
            "exchange_rate": "fx",
            "china_trade": "bilateral",
            "usmca": "bilateral",
            "nafta": "bilateral",
            "ft900": "trade_report",
            "census": "trade_report",
            "bea": "trade_report",
            "trade_balance": "trade_report",
            "deficit": "trade_report",
            "surplus": "trade_report",
        },
        priority_ties=(
            "- Official FT900 / BEA trade-balance level print → trade_report",
            "- Tariff/quota/sanction with flow impact → tariff (not policy)",
            "- USD FX tied to competitiveness → fx (not macro)",
            "- Bilateral partner balance news → bilateral",
            "- Use other ONLY if none apply.",
        ),
        mechanism_examples=(
            "Exports: foreign_demand↑ → U.S._exports↑ → trade_balance↑",
            "Imports: U.S._absorption↑ → imports↑ → trade_balance↓",
            "Tariff: retaliatory_tariff↑ → export_volumes↓ → trade_balance↓",
            "FX: USD↑ → export_competitiveness↓ → exports↓ → trade_balance↓",
            "Report: FT900_deficit_widens → realized_balance↓ → trade_balance↓",
        ),
        chain_end_hint="trade_balance↑/↓",
        what_counts=(
            "One checkable claim that could plausibly affect the U.S. monthly "
            "goods & services trade balance. Skip ads, domestic retail fluff, "
            "and foreign politics without a U.S. trade-flow link."
        ),
        extract_task_line="Extract ALL atomic trade-balance-relevant events from this month.",
    ),
    "energy": _cfg(
        key="energy",
        target_series_short="U.S. weekly retail gasoline price",
        expert_role="senior U.S. petroleum / fuel-markets economist",
        report_channel="gas_report",
        period_label="week",
        channels=(
            "crude",
            "inventory",
            "refining",
            "opec",
            "demand",
            "retail",
            "geopolitics",
            "policy",
            "macro",
            "gas_report",
            "other",
        ),
        channel_defs={
            "crude": "WTI/Brent crude price or supply shocks transmitting to gasoline",
            "inventory": "EIA gasoline/distillate/crude inventory prints",
            "refining": "refinery outages, utilization, crack spreads, maintenance",
            "opec": "OPEC+/producer quota decisions affecting crude→pump pass-through",
            "demand": "seasonal/driving demand, PADD demand, holiday travel fuel use",
            "retail": "retail/rack national or PADD price moves (not only official print)",
            "geopolitics": "geopolitical supply risk with a clear U.S. pump link",
            "policy": "fuel taxes, RFS, biofuel, export bans affecting pump prices",
            "macro": "broad growth/recession demand for fuel",
            "gas_report": "official EIA / AAA / national retail gasoline price prints",
            "other": "only if none of the above fit",
        },
        channel_aliases={
            "wti": "crude",
            "brent": "crude",
            "oil": "crude",
            "stocks": "inventory",
            "eia_inventory": "inventory",
            "refinery": "refining",
            "crack": "refining",
            "opec_plus": "opec",
            "pump": "retail",
            "rack": "retail",
            "eia_price": "gas_report",
            "aaa": "gas_report",
            "gasoline_price": "gas_report",
        },
        priority_ties=(
            "- Official national/PADD retail gasoline price print → gas_report",
            "- EIA inventory print → inventory",
            "- Refinery outage / crack → refining",
            "- OPEC quota → opec (not geopolitics)",
            "- Use other ONLY if none apply.",
        ),
        mechanism_examples=(
            "Crude: WTI↑ → wholesale_gasoline↑ → retail_pump↑",
            "Inventory: gasoline_stocks↓ → prompt_supply_tight↑ → pump↑",
            "Refining: Gulf_outage↑ → supply↓ → rack↑ → pump↑",
            "Demand: summer_driving↑ → gasoline_demand↑ → pump↑",
            "Report: EIA_retail_print↑ → realized_pump↑ → gasoline_price↑",
        ),
        chain_end_hint="gasoline_price↑/↓",
        what_counts=(
            "One checkable claim that could plausibly affect U.S. weekly retail "
            "gasoline prices. Skip EV lifestyle/climate opinion without a pump link."
        ),
        extract_task_line="Extract ALL atomic retail-gasoline-relevant events from this week.",
    ),
    "public_health": _cfg(
        key="public_health",
        target_series_short="U.S. weekly influenza-like illness (ILI) ratio",
        expert_role="senior U.S. influenza / syndromic-surveillance epidemiologist",
        report_channel="flu_report",
        period_label="week",
        channels=(
            "ili",
            "hospitalization",
            "vaccine",
            "outbreak",
            "school",
            "variant",
            "season",
            "policy",
            "flu_report",
            "other",
        ),
        channel_defs={
            "ili": "outpatient ILI / ILINet activity, clinic visits, syndromic signals",
            "hospitalization": "flu hospitalization / FluSurv-NET burden",
            "vaccine": "vaccine coverage, mismatch, campaign effects on ILI burden",
            "outbreak": "regional/state outbreaks or pediatric clusters with ILI impact",
            "school": "school term, closures, or pediatric mixing affecting ILI",
            "variant": "dominant strain / antigenic drift with surveillance impact",
            "season": "season timing, weather/cold seasonality for flu transmission",
            "policy": "CDC guidance, testing/reporting changes distorting ILI ratio",
            "flu_report": "official CDC FluView / national ILI-ratio level prints",
            "other": "only if none of the above fit",
        },
        channel_aliases={
            "ilinet": "ili",
            "outpatient": "ili",
            "hosp": "hospitalization",
            "flu_hosp": "hospitalization",
            "vaccination": "vaccine",
            "mismatch": "vaccine",
            "regional_outbreak": "outbreak",
            "pediatric": "outbreak",
            "school_season": "school",
            "strain": "variant",
            "h3n2": "variant",
            "h1n1": "variant",
            "fluview": "flu_report",
            "cdc": "flu_report",
            "ili_ratio": "flu_report",
        },
        priority_ties=(
            "- Official FluView / national ILI% print → flu_report",
            "- Outpatient ILI activity (non-print) → ili",
            "- Hospital burden → hospitalization",
            "- School mixing/closures → school",
            "- Use other ONLY if none apply. COVID-only news without ILI link → drop/other.",
        ),
        mechanism_examples=(
            "ILI: outpatient_ILI↑ → measured_ILI_ratio↑",
            "Hosp: flu_hospitalizations↑ → severe_burden↑ → ILI_ratio↑",
            "Vaccine: coverage↑ → infections↓ → ILI_ratio↓",
            "School: schools_reopen↑ → pediatric_mixing↑ → ILI_ratio↑",
            "Report: FluView_ILI%↑ → realized_ILI_ratio↑",
        ),
        chain_end_hint="ILI_ratio↑/↓",
        what_counts=(
            "One checkable claim that could plausibly affect the U.S. weekly ILI "
            "ratio / FluView-like burden. Skip wellness fluff, foreign outbreaks "
            "without U.S. surveillance link, and COVID essays unless they distort ILI reporting."
        ),
        extract_task_line="Extract ALL atomic ILI-ratio-relevant events from this week.",
    ),
    "security": _cfg(
        key="security",
        target_series_short="U.S. monthly FEMA grant outlays",
        expert_role="senior U.S. disaster-finance / FEMA outlays analyst",
        report_channel="fema_report",
        period_label="month",
        channels=(
            "declaration",
            "hurricane",
            "wildfire",
            "flood",
            "tornado",
            "funding",
            "appropriation",
            "mitigation",
            "hazard",
            "fema_report",
            "other",
        ),
        channel_defs={
            "declaration": "presidential major disaster / emergency declarations, mission assignments",
            "hurricane": "hurricanes / tropical storms with federal response or damage scale",
            "wildfire": "wildfires with FEMA/federal fire or recovery funding link",
            "flood": "floods / severe storms with federal assistance",
            "tornado": "tornado outbreaks with federal disaster response",
            "funding": "Public/Individual Assistance obligations, mission spend (non-appropriation)",
            "appropriation": "supplemental / DHS appropriations for disaster relief",
            "mitigation": "hazard mitigation grants, pre-disaster resilience outlays",
            "hazard": "other U.S. hazards (quake, winter storm, hazmat) with FEMA link",
            "fema_report": "official FEMA grant-outlay / obligation level statistics",
            "other": "only if none of the above fit",
        },
        channel_aliases={
            "disaster_declaration": "declaration",
            "emergency_declaration": "declaration",
            "tropical_storm": "hurricane",
            "typhoon": "hurricane",
            "fire": "wildfire",
            "flooding": "flood",
            "public_assistance": "funding",
            "individual_assistance": "funding",
            "supplemental": "appropriation",
            "hmgp": "mitigation",
            "earthquake": "hazard",
            "winter_storm": "hazard",
            "fema_outlay": "fema_report",
            "obligation": "fema_report",
        },
        priority_ties=(
            "- Official FEMA outlay/obligation print → fema_report",
            "- Presidential declaration → declaration",
            "- Named hurricane/tropical cyclone → hurricane",
            "- PA/IA spending news → funding (not appropriation)",
            "- Congressional supplemental → appropriation",
            "- Use other ONLY if none apply. Crime / foreign wars → not FEMA outlays.",
        ),
        mechanism_examples=(
            "Declaration: major_disaster_declared↑ → PA/IA_obligations↑ → FEMA_outlays↑",
            "Hurricane: landfall_damage↑ → emergency_spend↑ → FEMA_outlays↑",
            "Appropriation: disaster_supplemental↑ → available_budget↑ → outlays↑",
            "Mitigation: HMGP_awards↑ → mitigation_outlays↑ → FEMA_outlays↑",
            "Report: FEMA_monthly_outlays↑ → realized_grants↑ → FEMA_outlays↑",
        ),
        chain_end_hint="FEMA_outlays↑/↓",
        what_counts=(
            "One checkable claim that could plausibly affect U.S. monthly FEMA "
            "grant outlays. Skip local crime, foreign wars, and generic homeland-security essays."
        ),
        extract_task_line="Extract ALL atomic FEMA-outlay-relevant events from this month.",
    ),
    "social_good": _cfg(
        key="social_good",
        target_series_short="U.S. monthly unemployment rate",
        expert_role="senior U.S. labor-market / macro labor economist",
        report_channel="bls_report",
        period_label="month",
        channels=(
            "payrolls",
            "claims",
            "layoffs",
            "participation",
            "recession",
            "policy",
            "macro",
            "labor",
            "bls_report",
            "other",
        ),
        channel_defs={
            "payrolls": "nonfarm payrolls, job gains/losses, hiring (non-BLS headline print)",
            "claims": "initial/continuing jobless claims",
            "layoffs": "layoff announcements, WARN notices, mass job cuts",
            "participation": "labor-force participation, employment-population ratio",
            "recession": "recession/expansion regime signals tied to unemployment",
            "policy": "UI extensions, job programs, labor regulation affecting the rate",
            "macro": "GDP/demand shocks transmitting to unemployment",
            "labor": "strikes, wages, hours, broader labor-market tightness (non-payroll)",
            "bls_report": "official BLS unemployment rate / Employment Situation prints",
            "other": "only if none of the above fit",
        },
        channel_aliases={
            "nfp": "payrolls",
            "nonfarm": "payrolls",
            "jobs": "payrolls",
            "jobless_claims": "claims",
            "ui_claims": "claims",
            "layoff": "layoffs",
            "warn": "layoffs",
            "lfpr": "participation",
            "epop": "participation",
            "recession_signal": "recession",
            "unemployment_rate": "bls_report",
            "bls": "bls_report",
            "employment_situation": "bls_report",
            "cps": "bls_report",
        },
        priority_ties=(
            "- Official BLS unemployment-rate print → bls_report",
            "- Nonfarm payrolls detail → payrolls",
            "- Jobless claims → claims",
            "- Mass layoffs → layoffs",
            "- Use other ONLY if none apply.",
        ),
        mechanism_examples=(
            "Payrolls: job_gains↑ → unemployment_rate↓",
            "Claims: initial_claims↑ → layoffs_flow↑ → unemployment_rate↑",
            "Layoffs: mass_layoffs↑ → job_losers↑ → unemployment_rate↑",
            "Participation: participation↑ → measured_unemployment↑ (composition)",
            "Report: BLS_u_rate↑ → realized_unemployment↑ → unemployment_rate↑",
        ),
        chain_end_hint="unemployment_rate↑/↓",
        what_counts=(
            "One checkable claim that could plausibly affect the U.S. monthly "
            "unemployment rate. Skip foreign labor news and lifestyle pieces without a labor-market link."
        ),
        extract_task_line="Extract ALL atomic unemployment-relevant events from this month.",
    ),
}


def get_domain_config(domain: str) -> DomainLampConfig:
    key = (domain or "").strip().lower().replace("-", "_")
    if key not in DOMAIN_CONFIGS:
        raise KeyError(
            f"Unknown lamp_timemmd domain={domain!r}; known={sorted(DOMAIN_CONFIGS)}"
        )
    return DOMAIN_CONFIGS[key]


def focus_channels(cfg: DomainLampConfig) -> Tuple[str, ...]:
    """Focus = all channels except ``other`` (includes report channel)."""
    return tuple(c for c in cfg.channels if c != "other")


def list_domains() -> Sequence[str]:
    return tuple(sorted(DOMAIN_CONFIGS))
