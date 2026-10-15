"""Event schema for per-month VMT news structuring + expert mechanism priors."""

from __future__ import annotations

from typing import Any

# Domain channel for mechanism prior (not open-loop polarity).
# Keep this closed set small enough for retrieval, but cover dominant VMT drivers.
CHANNELS = (
    "fuel",
    "holiday",
    "weather",
    "freight",
    "policy",
    "remote_work",
    "transit",
    "labor",
    "tourism",
    # Newly added (were collapsing into other)
    "vmt_report",       # official/statistical VMT, AADT, miles-traveled levels
    "infrastructure",   # roads, bridges, capacity expansion, construction
    "congestion",       # congestion, speed, travel time, bottlenecks
    "safety",           # crashes, fatalities, enforcement, NHTSA/FARS
    "macro",            # income, unemployment, recession, broad demand
    "fleet",            # vehicle sales, registrations, EV adoption, stock
    "other",
)

CHANNEL_DEFS = {
    "fuel": "fuel / gasoline / diesel prices or fuel supply shocks",
    "holiday": "holiday / calendar peak travel periods",
    "weather": "weather, storms, closures from weather",
    "freight": "truck freight / tonnage / logistics miles",
    "policy": "regulation, taxes, mandates, DOT policy rules (non-infra)",
    "remote_work": "work-from-home / telework commute effects",
    "transit": "transit, rail transit, bus, bikeshare mode shift",
    "labor": "strikes, labor disruptions affecting travel/freight",
    "tourism": "tourism / leisure destination travel (non-holiday)",
    "vmt_report": "reports of VMT/AADT/vehicle-miles levels or growth rates themselves",
    "infrastructure": "highway/road/bridge construction, expansion, capacity",
    "congestion": "congestion, speeds, delays, travel-time reliability",
    "safety": "crashes, fatalities, safety programs/enforcement",
    "macro": "macroeconomy: income, jobs, recession, broad activity demand",
    "fleet": "vehicle fleet, auto sales, registrations, EV penetration",
    "other": "only if none of the above fit",
}

# Tip of a mechanism hypothesis: pressure on VMT implied by domain prior.
# NOT residual sign, NOT news sentiment.
VMT_EFFECTS = ("up", "down", "mixed", "unclear")

PLAUSIBILITY = ("high", "medium", "low")

PRED_LEN = 6
SEQ_LEN = 8

# Default client-side parallelism for all vLLM batch calls in this module.
VLLM_CONCURRENCY = 48
MAX_MECHANISMS_PER_EVENT = 3


def normalize_channel(x: Any) -> str:
    s = str(x or "other").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "gas": "fuel",
        "gasoline": "fuel",
        "diesel": "fuel",
        "oil": "fuel",
        "fuel_price": "fuel",
        "holiday_travel": "holiday",
        "vacation": "holiday",
        "storm": "weather",
        "closure": "weather",
        "road_closure": "weather",
        "truck": "freight",
        "regulation": "policy",
        "wfh": "remote_work",
        "telework": "remote_work",
        "bus": "transit",
        "rail": "transit",
        "bikeshare": "transit",
        "bike_share": "transit",
        "strike": "labor",
        "travel": "tourism",
        # new
        "vmt": "vmt_report",
        "aadt": "vmt_report",
        "traffic_volume_report": "vmt_report",
        "miles_traveled": "vmt_report",
        "vehicle_miles": "vmt_report",
        "construction": "infrastructure",
        "highway": "infrastructure",
        "bridge": "infrastructure",
        "capacity": "infrastructure",
        "roadwork": "infrastructure",
        "traffic": "congestion",
        "speed": "congestion",
        "delay": "congestion",
        "travel_time": "congestion",
        "crash": "safety",
        "accident": "safety",
        "fatality": "safety",
        "nhtsa": "safety",
        "economy": "macro",
        "unemployment": "macro",
        "income": "macro",
        "recession": "macro",
        "ev": "fleet",
        "vehicle_sales": "fleet",
        "registration": "fleet",
        "auto_sales": "fleet",
    }
    s = aliases.get(s, s)
    return s if s in CHANNELS else "other"


def normalize_vmt_effect(x: Any) -> str:
    s = str(x or "unclear").strip().lower().replace(" ", "_")
    aliases = {
        "increase": "up",
        "rise": "up",
        "favorable": "up",
        "positive": "up",
        "boost": "up",
        "decrease": "down",
        "fall": "down",
        "adverse": "down",
        "negative": "down",
        "suppress": "down",
        "volatile": "mixed",
        "dual": "mixed",
        "both": "mixed",
        "neutral": "unclear",
        "none": "unclear",
    }
    s = aliases.get(s, s)
    return s if s in VMT_EFFECTS else "unclear"


def normalize_plausibility(x: Any) -> str:
    s = str(x or "medium").strip().lower()
    return s if s in PLAUSIBILITY else "medium"


def normalize_mechanism_chain(x: Any) -> str:
    """Keep an A → B → VMT↑/↓ style domain prior chain."""
    s = str(x or "").strip()
    s = " ".join(s.split())
    for a in ("->", "=>", "→", "⟶", "➜"):
        s = s.replace(a, "→")
    s = s.replace("→", " → ")
    s = " ".join(s.split())
    return s[:320]


def normalize_mechanism(m: Any) -> dict[str, Any] | None:
    if not isinstance(m, dict):
        return None
    chain = normalize_mechanism_chain(m.get("chain") or m.get("mechanism_chain"))
    if not chain:
        return None
    return {
        "chain": chain,
        "vmt_effect": normalize_vmt_effect(m.get("vmt_effect")),
        "plausibility": normalize_plausibility(m.get("plausibility")),
        "rationale": str(m.get("rationale") or "")[:400],
        "caveat": str(m.get("caveat") or "")[:240],
    }


def normalize_mechanisms(raw: Any) -> list[dict[str, Any]]:
    """Accept mechanisms list, or legacy single mechanism_chain + vmt_effect."""
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            nm = normalize_mechanism(item)
            if nm:
                out.append(nm)
    elif isinstance(raw, dict):
        nm = normalize_mechanism(raw)
        if nm:
            out.append(nm)
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for m in out:
        key = m["chain"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq[:MAX_MECHANISMS_PER_EVENT]


def primary_vmt_effect(mechanisms: list[dict[str, Any]]) -> str:
    if not mechanisms:
        return "unclear"
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(mechanisms, key=lambda m: rank.get(m.get("plausibility", "medium"), 1))
    top = [m for m in ordered if m.get("plausibility") == ordered[0].get("plausibility")]
    effects = {normalize_vmt_effect(m.get("vmt_effect")) for m in top}
    effects.discard("unclear")
    if not effects:
        return "unclear"
    if effects == {"up"}:
        return "up"
    if effects == {"down"}:
        return "down"
    if "mixed" in effects or effects == {"up", "down"}:
        return "mixed"
    return next(iter(effects))


def event_to_line(ev: dict[str, Any]) -> str:
    """Compact one-line form for fusion / logging."""
    mechs = ev.get("mechanisms") or []
    primary = mechs[0]["chain"] if mechs else ""
    return (
        f"[{ev.get('date', '')}] channel={ev.get('channel', 'other')} | "
        f"effect={ev.get('vmt_effect', 'unclear')} | "
        f"{primary} | "
        f"{ev.get('fact_text', '')}"
    )
