"""Per-dataset Scheme5 protocol seeds (rigorous, domain-conditioned).

Each Stage2 domain has its own residual semantics, standing rules, forbidden
cross-domain vocabulary, and per-channel prior IF–THEN lessons. Rewrite prompts
should be rendered from the same schema so pollution (e.g. VMT wording on
economy) is prevented upstream — not cleaned by post-hoc regex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class DomainSeedSpec:
    key: str
    target_series: str
    residual_def: str  # what (true − CFA) means in words
    sign_up: str  # residual > 0 means …
    sign_down: str
    period: str  # month | week
    expert_role: str
    forbidden_mentions: Tuple[str, ...]
    standing_rules: Tuple[str, ...]
    # channel -> (role_one_liner, prior_lessons…)
    channel_priors: Mapping[str, Tuple[str, Tuple[str, ...]]] = field(default_factory=dict)
    max_lessons: int = 20


def _spec_traffic() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="traffic",
        target_series="U.S. monthly vehicle miles traveled (VMT)",
        residual_def=(
            "Residual = true VMT − CFA forecast. Positive ⇒ actual travel higher than CFA; "
            "negative ⇒ actual travel lower than CFA."
        ),
        sign_up="true VMT > CFA (under-forecasted travel / missed upward shock)",
        sign_down="true VMT < CFA (over-forecasted travel / missed suppression)",
        period="month",
        expert_role="U.S. highway travel / mobility analyst",
        forbidden_mentions=(
            "trade balance",
            "gasoline pump price as the target",
            "ILI",
            "FEMA outlays",
            "unemployment rate as the target",
            "broiler price",
        ),
        standing_rules=(
            "Reason ONLY from THIS channel's events; do not import other channels' facts.",
            "Prefer exogenous / delayed drivers (weather, policy, fuel cost, mode shift) over restatements of VMT itself.",
            "Mechanism hypotheses are HINTS for CoT — synthesize a net residual direction; do not copy them verbatim.",
            "If events conflict, are local/non-national, or are meta-report noise → gate=0 (abstain).",
            "Output one residual direction for the forecast horizon (or abstain). Do not invent numeric VMT levels.",
            "National monthly VMT is dominated by passenger car travel; freight and transit matter only via clear substitution/scale.",
        ),
        channel_priors={
            "weather": (
                "Acute weather shocks affecting road usability / trip cancellation",
                (
                    "IF acute national-scale storms/floods/ice cause closures or trip suppression within lag≤4 THEN residual=down",
                    "IF weather text is seasonal baseline or historical mechanism without a current shock THEN gate=0",
                    "IF evacuation vs closure mechanisms conflict without clear dominance THEN gate=0",
                ),
            ),
            "fuel": (
                "Fuel / oil price or tax shocks changing the cost of driving",
                (
                    "IF a large, sustained fuel-cost spike is clear and CFA looks static THEN residual=down (trip suppression)",
                    "IF only moderate price moves or elasticities without timing THEN gate=0",
                ),
            ),
            "holiday": (
                "Calendar holiday / peak travel timing vs CFA seasonality",
                (
                    "IF a major holiday timing mismatch vs CFA's implied seasonality is explicit THEN residual follows the mismatch sign",
                    "IF holiday mentions lack lag alignment to the horizon THEN gate=0",
                ),
            ),
            "policy": (
                "Transport / pricing / access policy affecting miles driven",
                (
                    "IF binding cost/access restrictions (tolls, VMT fees, rationing) are enacted THEN residual=down",
                    "IF only proposals, studies, or goals without implementation timing THEN gate=0",
                ),
            ),
            "macro": (
                "Macro activity / income / trade intensity driving travel demand",
                (
                    "IF clear recession / income collapse / import slump signals suppress travel THEN residual=down",
                    "IF macro text is generic outlook without travel-demand link THEN gate=0",
                ),
            ),
            "transit": (
                "Public transit shocks that force auto substitution or the reverse",
                (
                    "IF confirmed transit capacity/ridership collapse forces auto substitution THEN residual=up",
                    "IF only minor incentives against high auto dependency THEN gate=0",
                ),
            ),
            "labor": (
                "Labor disruptions altering commuting / freight patterns",
                (
                    "IF a major strike forces mode shift onto roads THEN residual=up",
                    "IF labor news lacks a clear VMT transmission THEN gate=0",
                ),
            ),
            "remote_work": (
                "Telework / commute elimination vs induced demand",
                (
                    "IF structural telework adoption is confirmed THEN residual=down (trip elimination dominates early)",
                    "IF only speculative rebound/induced-demand claims THEN gate=0",
                ),
            ),
            "freight": (
                "Truck freight intensity / modal shift",
                (
                    "IF rail→truck modal shift or freight surge is confirmed THEN residual=up",
                    "IF freight news is local or mixed without net national direction THEN gate=0",
                ),
            ),
            "infrastructure": (
                "Capacity loss / expansion / induced demand",
                (
                    "IF active capacity loss (closures, failure) dominates THEN residual=down",
                    "IF capacity expansion / reliability gains dominate without offsetting closures THEN residual=up",
                ),
            ),
            "vmt_report": (
                "Official / structural VMT measurement and mandate context",
                (
                    "IF report is mainly a restatement of the target series level THEN gate=0",
                    "IF structural efficiency mandates conflict with state growth noise, prefer the structural mandate for residual sign",
                ),
            ),
        },
    )


def _spec_agriculture() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="agriculture",
        target_series="U.S. monthly retail broiler composite price",
        residual_def=(
            "Residual = true retail broiler price − CFA. Positive ⇒ price higher than CFA "
            "(tighter supply / stronger demand than priced); negative ⇒ price lower than CFA."
        ),
        sign_up="true price > CFA",
        sign_down="true price < CFA",
        period="month",
        expert_role="U.S. poultry / broiler market analyst",
        forbidden_mentions=("VMT", "vehicle miles", "ILI", "FEMA", "unemployment rate as target", "gasoline price as target"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is broiler retail price residual vs CFA — not volumes alone.",
            "Prefer exogenous supply/demand/trade/disease shocks over restating the price series.",
            "Mechanism hypotheses are HINTS; resolve net price pressure (supply↓ ⇒ price↑ residual usually).",
            "Lagged export / policy / feed-cost effects often matter more than same-month noise; if timing unclear → gate=0.",
            "Conflict, substitution ambiguity, or meta-report noise → gate=0.",
            "Do not invent numeric price levels; do not discuss unrelated series (VMT, gasoline, ILI, FEMA, unemployment).",
        ),
        channel_priors={
            "feed": (
                "Feed / grain cost shocks into broiler margins and placements",
                (
                    "IF acute feed-cost or feed-quality shocks with mortality/scarcity THEN residual=up",
                    "IF only generic 'cheap grain' narratives without acute constraint THEN gate=0",
                ),
            ),
            "supply": (
                "Production / placement / slaughter / capacity",
                (
                    "IF supply contraction (placements↓, mortality↑, slaughter cuts) THEN residual=up",
                    "IF clear oversupply / capacity surge CFA missed THEN residual=down",
                ),
            ),
            "stocks": (
                "Cold storage / inventory buffers",
                (
                    "IF stable stocks contradict a forecasted price collapse THEN residual=up",
                    "IF inventory news lacks price implication THEN gate=0",
                ),
            ),
            "price": (
                "Exogenous price anchors / international price transmission",
                (
                    "IF exogenous supply-linked price anchors contradict CFA THEN residual follows that pressure",
                    "IF event only restates historical price levels without a driver THEN gate=0",
                ),
            ),
            "disease": (
                "Poultry disease / food-safety shocks",
                (
                    "IF outbreak reduces near-term supply THEN residual=up",
                    "IF consumer aversion / bans hit demand for broilers specifically THEN residual=down",
                    "IF disease is chronic/stale without active transmission THEN gate=0",
                ),
            ),
            "trade": (
                "Export / import volumes shifting domestic availability",
                (
                    "IF lagged export surge (often ~3–7 months) tightens domestic supply THEN residual=up",
                    "IF lag falls outside the actionable window THEN gate=0",
                ),
            ),
            "demand": (
                "Substitution / safety / income demand for broilers",
                (
                    "IF substitute meats are sharply cheaper → broiler demand↓ THEN residual=down",
                    "IF competitor contamination shifts demand toward broilers THEN residual=up",
                ),
            ),
            "weather": (
                "Weather stress on birds / logistics",
                (
                    "IF heat/cold/storm clearly cuts placements or slaughter THEN residual=up",
                    "IF weather lacks poultry supply link THEN gate=0",
                ),
            ),
            "policy": (
                "USDA/FDA purchases, supports, labeling, inspections",
                (
                    "IF federal purchases / supports tighten marketable supply THEN residual=up",
                    "IF only vague policy mentions without instrument THEN gate=0",
                ),
            ),
            "macro": (
                "Macro / feed-crop policy cost-push into poultry",
                (
                    "IF explicit feed-crop support / cost-push is active THEN residual=up",
                    "IF only generic macro uncertainty THEN gate=0",
                ),
            ),
            "price_report": (
                "Official price prints / anchors vs CFA path",
                (
                    "IF multi-event price anchors contradict CFA's implied path with high coverage THEN residual follows anchors",
                    "IF report only restates the target without exogenous driver THEN gate=0",
                ),
            ),
        },
    )


def _spec_economy() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="economy",
        target_series="U.S. monthly goods & services trade balance",
        residual_def=(
            "Residual = true trade balance − CFA. Positive ⇒ balance higher (surplus larger / deficit narrower) "
            "than CFA; negative ⇒ balance lower (deficit wider) than CFA."
        ),
        sign_up="true balance > CFA (better than forecast)",
        sign_down="true balance < CFA (worse than forecast)",
        period="month",
        expert_role="U.S. international trade / BOP economist",
        forbidden_mentions=("VMT", "vehicle miles", "ILI", "FEMA outlays", "broiler", "gasoline pump as target"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is the trade balance residual — exports↑ helps, imports↑ hurts (ceteris paribus).",
            "Prefer exogenous policy/FX/partner demand shocks over restating the deficit/surplus print itself.",
            "Synthesize net balance pressure; conflicting export vs import signals without dominance → gate=0.",
            "Minor partner or tiny sector moves rarely drive the national residual — abstain unless systemic.",
            "Do not invent numeric balance levels; never mention VMT or other domains' target series.",
        ),
        channel_priors={
            "exports": (
                "U.S. export volumes / foreign demand",
                (
                    "IF clear export weakness (foreign demand↓, competitiveness↓) THEN residual=down",
                    "IF single ambiguous sector print without net mechanism THEN gate=0",
                ),
            ),
            "imports": (
                "U.S. import absorption",
                (
                    "IF domestic absorption / import surge widens the deficit beyond CFA THEN residual=down",
                    "IF import contraction improves the balance vs a static CFA widening THEN residual=up",
                ),
            ),
            "tariff": (
                "Tariffs / remedies / retaliation",
                (
                    "IF import-suppressing tariffs dominate short-run THEN residual=up",
                    "IF retaliation clearly collapses exports and dominates THEN residual=down",
                ),
            ),
            "fx": (
                "USD moves affecting competitiveness",
                (
                    "IF USD appreciation is a clear export headwind THEN residual=down",
                    "IF FX text lacks trade-competitiveness link THEN gate=0",
                ),
            ),
            "bilateral": (
                "Major-partner bilateral imbalances",
                (
                    "IF major partner (e.g. China/EU) structural imbalance dominates THEN residual follows that pressure",
                    "IF only minor-partner noise THEN gate=0",
                ),
            ),
            "commodity": (
                "Commodity / energy trade flows",
                (
                    "IF energy or commodity flow shift is large enough to move the aggregate THEN residual follows net balance impact",
                    "IF sectoral offsets cancel THEN gate=0",
                ),
            ),
            "policy": (
                "Non-tariff trade policy / liberalization",
                (
                    "IF import-side liberalization is immediate/binding THEN residual=down",
                    "IF export restrictions are binding and immediate THEN residual=up",
                ),
            ),
            "macro": (
                "Growth / absorption / global demand",
                (
                    "IF strong domestic absorption + weak foreign demand THEN residual=down",
                    "IF generic macro without trade-flow link THEN gate=0",
                ),
            ),
            "trade_report": (
                "Official FT900 / BEA prints",
                (
                    "IF print mainly restates the target balance THEN gate=0",
                    "IF multi-event structural deficit signals dominate CFA narrowing THEN residual=down",
                ),
            ),
        },
    )


def _spec_energy() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="energy",
        target_series="U.S. weekly retail gasoline price",
        residual_def=(
            "Residual = true retail gasoline price − CFA. Positive ⇒ pump price higher than CFA; "
            "negative ⇒ pump price lower than CFA."
        ),
        sign_up="true pump price > CFA",
        sign_down="true pump price < CFA",
        period="week",
        expert_role="U.S. petroleum / fuel-markets analyst",
        forbidden_mentions=("VMT as target", "vehicle miles as target", "ILI", "FEMA", "trade balance as target", "broiler"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is retail gasoline price residual (weekly).",
            "Cost shocks (crude, refining, inventory tightness) usually dominate demand-response narratives.",
            "Duplicate / static / already-priced rhetoric → gate=0; require new exogenous information.",
            "Pass-through to U.S. pump must be plausible; foreign noise without transmission → gate=0.",
            "Do not invent numeric $/gal; do not discuss unrelated target series (VMT, ILI, FEMA, trade balance).",
        ),
        channel_priors={
            "crude": (
                "Crude benchmark shocks transmitting to pump",
                (
                    "IF sustained crude up (down) is confirmed THEN residual=up (down)",
                    "IF crude noise is one-off without persistence THEN prefer gate=0 unless cluster confirms",
                ),
            ),
            "inventory": (
                "EIA gasoline-relevant inventory",
                (
                    "IF structural gasoline inventory tightness THEN residual=up",
                    "IF report is non-gasoline stocks only THEN gate=0",
                ),
            ),
            "refining": (
                "Refinery runs / outages / cracks",
                (
                    "IF outage / utilization drop cluster raises crack risk THEN residual=up",
                    "IF only static cost-saving anecdotes THEN gate=0",
                ),
            ),
            "opec": (
                "OPEC+ supply policy",
                (
                    "IF coherent OPEC supply constraint narrative THEN residual=up",
                    "IF contradictory quota rumors without dominance THEN gate=0",
                ),
            ),
            "demand": (
                "Driving / seasonal fuel demand",
                (
                    "IF demand is only a response to price (not an independent shock) THEN do not invert cost-driven residuals; often gate=0",
                    "IF independent demand surge is clear THEN residual=up",
                ),
            ),
            "retail": (
                "Retail/rack price prints and margins",
                (
                    "IF recent direct price prints contradict CFA direction THEN residual follows the prints",
                    "IF prints already align with CFA THEN gate=0",
                ),
            ),
            "geopolitics": (
                "Geopolitical supply risk with pump link",
                (
                    "IF new risk premium / supply shock intensity rises THEN residual=up",
                    "IF redundant reprints of a known priced-in event THEN gate=0",
                ),
            ),
            "policy": (
                "Fuel taxes / SPR / RFS / export rules",
                (
                    "IF confirmed SPR release or unpriced supply relief THEN residual=down",
                    "IF unenacted bills / rhetoric / static known rules THEN gate=0",
                ),
            ),
            "macro": (
                "Macro demand for fuel",
                (
                    "IF specific U.S.-linked demand shock is time-bound THEN residual follows that shock",
                    "IF generic inflation/GDP talk without fuel link THEN gate=0",
                ),
            ),
            "gas_report": (
                "Official / survey gasoline price prints",
                (
                    "IF direct prints contradict CFA THEN residual follows prints",
                    "IF report only restates known path already in CFA THEN gate=0",
                ),
            ),
        },
    )


def _spec_public_health() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="public_health",
        target_series="U.S. weekly influenza-like illness (ILI) ratio",
        residual_def=(
            "Residual = true ILI ratio − CFA. Positive ⇒ ILI higher than CFA (more illness / reporting intensity); "
            "negative ⇒ ILI lower than CFA."
        ),
        sign_up="true ILI > CFA",
        sign_down="true ILI < CFA",
        period="week",
        expert_role="U.S. influenza / ILI surveillance analyst",
        forbidden_mentions=("VMT", "vehicle miles", "FEMA outlays", "trade balance", "gasoline price", "broiler"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is U.S. ILI ratio residual (weekly).",
            "Prefer active U.S. transmission / surveillance / vaccine-mismatch shocks over animal-only or historical strain lists.",
            "Duplicate identical reports across lags are NOT independent evidence — collapse them.",
            "Generic seasonality CFA already prices → gate=0 unless a distinct deviation is shown.",
            "Do not invent numeric ILI%; never mention VMT or other domains' targets.",
        ),
        channel_priors={
            "ili": (
                "ILI clinical / syndromic intensity",
                (
                    "IF ≥3 distinct biological observations support rise THEN residual=up",
                    "IF duplicates dominate without new biology THEN gate=0",
                ),
            ),
            "hospitalization": (
                "Flu hospitalization / severity",
                (
                    "IF demographic severity surge compounds beyond static risk THEN residual=up",
                    "IF only static annual numbers THEN residual=down or gate=0 if non-informative",
                ),
            ),
            "vaccine": (
                "Vaccine effectiveness / mismatch / supply",
                (
                    "IF waning VE or mismatch is confirmed THEN residual=up",
                    "IF generic annual recommendation without mismatch THEN gate=0",
                ),
            ),
            "outbreak": (
                "Active outbreak / transmission events",
                (
                    "IF active U.S. human-to-human outbreak onset THEN residual=up",
                    "IF animal spillover alerts alone THEN gate=0",
                ),
            ),
            "school": (
                "School mixing / closures",
                (
                    "IF unmitigated school mixing drives pediatric spillover THEN residual=up",
                    "IF mitigation/closures clearly dominate THEN residual=down",
                ),
            ),
            "variant": (
                "Strain / antigenic drift",
                (
                    "IF current-season strain dominance / mismatch THEN residual=up",
                    "IF historical strain lists without current emergence THEN gate=0",
                ),
            ),
            "season": (
                "Seasonal timing deviations",
                (
                    "IF only generic cooler-weather / school-reopening baselines THEN gate=0",
                    "IF a distinct inter-season or off-calendar shift vs CFA THEN residual follows that shift",
                ),
            ),
            "policy": (
                "Surveillance definition / reporting policy",
                (
                    "IF reporting expansion / definition change inflates measured ILI THEN residual=up",
                    "IF repeated guidance without new rollout THEN gate=0",
                ),
            ),
            "flu_report": (
                "Official flu surveillance prints",
                (
                    "IF confirmed epidemic onset / mortality spike THEN residual=up",
                    "IF print is low-activity baseline CFA already expects THEN gate=0",
                ),
            ),
        },
    )


def _spec_security() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="security",
        target_series="U.S. monthly FEMA grant outlays",
        residual_def=(
            "Residual = true FEMA grant outlays − CFA. Positive ⇒ outlays higher than CFA "
            "(more obligations/disbursements); negative ⇒ outlays lower than CFA."
        ),
        sign_up="true outlays > CFA",
        sign_down="true outlays < CFA",
        period="month",
        expert_role="U.S. disaster finance / FEMA outlays analyst",
        forbidden_mentions=("VMT", "vehicle miles", "ILI", "trade balance", "gasoline", "broiler", "unemployment as target"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is FEMA grant outlays residual (monthly).",
            "Declarations and appropriations drive lagged PA/IA/HMGP spending — timing windows matter (often ~1–7 months).",
            "Pending/potential language without formal obligation triggers → gate=0.",
            "Clustered disasters compound; stale events outside the lag window → gate=0.",
            "Do not invent dollar outlays; never mention VMT or other domains' targets.",
        ),
        channel_priors={
            "declaration": (
                "Presidential / major disaster declarations",
                (
                    "IF confirmed declaration with lag in ~2–6 months to horizon THEN residual=up",
                    "IF lag outside window or only pending appeals THEN gate=0",
                ),
            ),
            "hurricane": (
                "Hurricane landfalls → lagged aid",
                (
                    "IF landfall/cluster implies 1–3 month aid lag into horizon THEN residual=up",
                    "IF horizon is beyond spent lag without new trigger THEN gate=0",
                ),
            ),
            "wildfire": (
                "Wildfire declarations / federal assistance",
                (
                    "IF confirmed declaration/assistance request THEN residual=up",
                    "IF no declaration and event stale THEN gate=0",
                ),
            ),
            "flood": (
                "Flood disasters → lagged cleanup/mitigation spend",
                (
                    "IF major flood with active declaration in lag window THEN residual=up",
                    "IF last event >6 months and no new declaration THEN gate=0",
                ),
            ),
            "tornado": (
                "Tornado outbreaks → lagged FEMA grants",
                (
                    "IF major tornado with ~3+ month grant lag into horizon THEN residual=up",
                    "IF timing misaligned THEN gate=0",
                ),
            ),
            "funding": (
                "PA/IA/SBA/grants and procurement triggers",
                (
                    "IF grant approvals / statutory spend deadlines land in window THEN residual=up",
                    "IF administrative chatter without funding authority THEN gate=0",
                ),
            ),
            "appropriation": (
                "Legislative appropriations / supplementals",
                (
                    "IF new appropriation/supplemental authority is clear THEN residual=up",
                    "IF purely administrative without new authority THEN gate=0",
                ),
            ),
            "mitigation": (
                "HMGP / mitigation policy mandates",
                (
                    "IF structural mitigation mandates increase awards beyond static CFA THEN residual=up",
                    "IF no implementation lag alignment THEN gate=0",
                ),
            ),
            "hazard": (
                "Multi-hazard major disasters",
                (
                    "IF major disaster cluster with 2–7 month obligation lag THEN residual=up",
                    "IF horizon outside all lag windows THEN gate=0",
                ),
            ),
            "fema_report": (
                "Federal Register / formal FEMA obligation notices",
                (
                    "IF formal declaration/obligation entry aligns to lag THEN residual=up",
                    "IF reviews/discussions without obligation THEN gate=0",
                ),
            ),
        },
    )


def _spec_social_good() -> DomainSeedSpec:
    return DomainSeedSpec(
        key="social_good",
        target_series="U.S. monthly unemployment rate",
        residual_def=(
            "Residual = true unemployment rate − CFA. Positive ⇒ unemployment higher than CFA; "
            "negative ⇒ unemployment lower than CFA."
        ),
        sign_up="true unemployment rate > CFA",
        sign_down="true unemployment rate < CFA",
        period="month",
        expert_role="U.S. labor-market economist",
        forbidden_mentions=("VMT", "vehicle miles", "ILI", "FEMA outlays", "trade balance as target", "gasoline", "broiler"),
        standing_rules=(
            "Reason ONLY from THIS channel; target is unemployment-rate residual (monthly).",
            "Distinguish payroll shocks, claims, participation (denominator), and policy — they can move the rate in opposite ways.",
            "Lags matter (payrolls often affect the rate with multi-month delay); mis-timed events → gate=0.",
            "Regional plant news rarely moves the national rate unless sectoral/national transmission is clear.",
            "Do not invent numeric unemployment levels; never mention VMT or other domains' targets.",
        ),
        channel_priors={
            "payrolls": (
                "Establishment payroll changes",
                (
                    "IF payroll weakness aligns to an actionable lag into the horizon THEN residual=up",
                    "IF payroll events are too recent or stale vs the lag window THEN gate=0",
                ),
            ),
            "claims": (
                "UI claims / layoff pressure",
                (
                    "IF severe claims spike signals recession onset THEN residual=up",
                    "IF single-month claims drop lacks structural confirmation THEN gate=0",
                ),
            ),
            "layoffs": (
                "Announced layoffs",
                (
                    "IF large national/sectoral layoffs THEN residual=up",
                    "IF single-plant regional news without national transmission THEN gate=0",
                ),
            ),
            "participation": (
                "Labor-force participation / denominator effects",
                (
                    "IF labor-force exit shrinks denominator THEN residual=down (rate↓ vs CFA)",
                    "IF returning workers raise unemployment without hiring THEN residual=up",
                ),
            ),
            "recession": (
                "Recession / unemployment momentum",
                (
                    "IF rising unemployment momentum is confirmed THEN residual=up",
                    "IF signal is a single noisy month without regime confirmation THEN gate=0",
                ),
            ),
            "policy": (
                "Labor / UI / wage policy shocks",
                (
                    "IF immediate wage/demand shock is enacted THEN residual follows labor-demand effect",
                    "IF lagged administrative feedback or expired benefits only THEN gate=0",
                ),
            ),
            "macro": (
                "Macro drag on labor demand",
                (
                    "IF structural drag (tightening, trade shock, stagflation pattern) THEN residual=up",
                    "IF CFA already prices the same drag explicitly THEN gate=0",
                ),
            ),
            "labor": (
                "Structural labor-market frictions",
                (
                    "IF structural mismatch/duration rise THEN residual=up",
                    "IF conflicting structural facts without catalyst THEN gate=0",
                ),
            ),
            "bls_report": (
                "BLS unemployment prints / persistence",
                (
                    "IF print implies persistence CFA underestimates THEN residual=up",
                    "IF report only restates the target without exogenous driver THEN gate=0",
                ),
            ),
        },
    )


_SPECS: Dict[str, DomainSeedSpec] = {
    "traffic": _spec_traffic(),
    "agriculture": _spec_agriculture(),
    "economy": _spec_economy(),
    "energy": _spec_energy(),
    "public_health": _spec_public_health(),
    "security": _spec_security(),
    "social_good": _spec_social_good(),
}


def get_seed_spec(domain: str) -> DomainSeedSpec:
    key = (domain or "").strip().lower().replace("-", "_")
    aliases = {
        "vmt": "traffic",
        "agri": "agriculture",
        "health": "public_health",
        "flu": "public_health",
        "social": "social_good",
        "fema": "security",
        "trade": "economy",
        "gas": "energy",
    }
    key = aliases.get(key, key)
    if key not in _SPECS:
        raise KeyError(f"No scheme5 seed spec for domain={domain!r}; known={sorted(_SPECS)}")
    return _SPECS[key]


def list_seed_domains() -> list[str]:
    return sorted(_SPECS.keys())


def seed_prompt(channel: str, target: str = "", *, domain: str) -> str:
    """Build a rigorous per-dataset protocol seed for ``channel``."""
    spec = get_seed_spec(domain)
    target = target or spec.target_series
    role, priors = spec.channel_priors.get(channel, ("(no specialized prior)", ()))
    forbid = ", ".join(spec.forbidden_mentions[:8])
    lines = [
        f"CHANNEL ASSISTANT PROTOCOL — {channel}",
        f"Domain: {spec.key} | Period: {spec.period}",
        f"Target series: {target}",
        f"Expert role: {spec.expert_role}",
        "",
        "Residual definition:",
        f"- {spec.residual_def}",
        f"- residual>0 means: {spec.sign_up}",
        f"- residual<0 means: {spec.sign_down}",
        "",
        f"This channel's role: {role}",
        "",
        "Standing rules:",
    ]
    for i, r in enumerate(spec.standing_rules, 1):
        lines.append(f"{i}) {r}")
    lines += [
        f"{len(spec.standing_rules)+1}) Forbidden in lessons/reasoning: {forbid}.",
        f"{len(spec.standing_rules)+2}) Accumulated lessons must stay ≤{spec.max_lessons} IF–THEN bullets "
        "(evidence → residual up/down OR abstain).",
        "",
        "Prior lessons (domain seed — refine/merge via outcomes; do not delete the residual definition):",
    ]
    if priors:
        for p in priors:
            lines.append(f"- {p}")
    else:
        lines.append(
            "- IF evidence is weak, conflicting, or only restates the target series THEN gate=0 (abstain)"
        )
    lines += [
        "",
        "Accumulated lessons (learned; rewrite over time — keep IF–THEN; merge with priors):",
        "- (none yet — learn from outcomes)",
    ]
    return "\n".join(lines)


def sanitize_rewritten_protocol(
    text: str,
    *,
    channel: str,
    domain: str,
    max_lessons: int | None = None,
) -> str:
    """Light schema-based sanitize after LLM rewrite (no traffic-specific regex rewrite).

    - Drop forbidden cross-domain phrases
    - Cap Accumulated lessons bullet count
    - Preserve Standing / Residual sections when present
    """
    import re

    spec = get_seed_spec(domain)
    n = int(max_lessons or spec.max_lessons)
    t = (text or "").strip()
    if not t:
        return seed_prompt(channel, domain=domain)
    # strip fences
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    low_forbid = [f.lower() for f in spec.forbidden_mentions]
    lines_out: list[str] = []
    in_lessons = False
    lesson_n = 0
    for line in t.splitlines():
        s = line.rstrip()
        if re.match(r"(?i)^Accumulated lessons", s):
            in_lessons = True
            lines_out.append(s if s.endswith(":") else s.rstrip(":") + ":")
            continue
        if in_lessons and re.match(r"(?i)^(Standing|Residual|Prior|CHANNEL|Domain:|Target)", s):
            in_lessons = False
        if in_lessons and s.strip().startswith("-"):
            blob = s.lower()
            if any(f in blob for f in low_forbid):
                continue
            if "improved=" in blob or "cov=" in blob or "wid=" in blob:
                continue
            lesson_n += 1
            if lesson_n > n:
                continue
        lines_out.append(s)
    out = "\n".join(lines_out).strip()
    # If rewrite destroyed residual definition, fall back to seed
    if "Residual definition" not in out and "residual" not in out.lower()[:800]:
        return seed_prompt(channel, domain=domain)
    return out


def protocol_has_forbidden(text: str, domain: str) -> bool:
    """True if Accumulated lessons (not Standing rules) mention forbidden phrases.

    Seeds intentionally list forbidden vocabulary in Standing rules; scanning the
    whole protocol would reject every compliant rewrite that keeps those rules.
    """
    import re

    spec = get_seed_spec(domain)
    t = text or ""
    m = re.search(r"(?im)^Accumulated lessons\s*:?\s*$", t)
    if m:
        blob = t[m.end() :].lower()
    else:
        # no lessons section → only check bullet lines (priors / freeform)
        blob = "\n".join(
            ln for ln in t.splitlines() if ln.strip().startswith("-")
        ).lower()
    return any(f.lower() in blob for f in spec.forbidden_mentions)


def prompt_rewrite_from_spec(
    *,
    channel: str,
    domain: str,
    old_protocol: str,
    outcomes_block: str,
    max_lessons: int | None = None,
) -> str:
    """Domain-conditioned rewrite prompt (no hardcoded VMT)."""
    spec = get_seed_spec(domain)
    n = int(max_lessons or spec.max_lessons)
    forbid = "; ".join(spec.forbidden_mentions)
    return "\n".join(
        [
            f"Rewrite the PROTOCOL for domain={spec.key} channel={channel}.",
            f"Target: {spec.target_series}.",
            f"Residual: {spec.residual_def}",
            "Goals: durable, paper-interpretable IF–THEN rules that improve residual decisions.",
            "Constraints:",
            f"- Keep Residual definition + Standing rules; rewrite mainly Accumulated lessons (≤{n} bullets).",
            "- Each lesson: IF <evidence> THEN residual=up|down OR gate=0 (abstain).",
            "- Merge duplicates; drop stale noise; drop blanket 'CFA always underestimates' lines.",
            f"- NEVER mention: {forbid}.",
            "- Do NOT write outcome log fields (hit/false_alarm/cov/wid) into the protocol.",
            "- Do NOT invent numeric target levels. Do NOT add retrieval instructions.",
            "",
            "CURRENT PROTOCOL:",
            old_protocol.strip(),
            "",
            "RECENT OUTCOMES (time-ordered):",
            outcomes_block,
            "",
            "Return the FULL updated PROTOCOL text only (plain text, no markdown fences).",
        ]
    )
