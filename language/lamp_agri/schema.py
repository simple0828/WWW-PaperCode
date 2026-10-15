"""Broiler / agriculture event schema for Stage2 (parallel to lamp_vmt)."""

from __future__ import annotations

from typing import Any

# Closed channel set for U.S. retail broiler composite drivers.
CHANNELS = (
    "feed",           # corn, soybean meal, feed costs
    "supply",         # placements, egg sets, flock, slaughter, production
    "stocks",         # cold storage
    "price",          # wholesale/retail/composite prints & spreads
    "disease",        # HPAI / avian influenza, culls
    "trade",          # exports/imports, trade policy
    "demand",         # foodservice, retail demand, cutout
    "weather",        # heat/cold stress in poultry regions
    "policy",         # USDA regs, subsidies, inspection (non-trade)
    "macro",          # income, inflation affecting meat demand
    "price_report",   # official broiler composite / chicken price statistics
    "other",
)

CHANNEL_DEFS = {
    "feed": "corn, soybean meal, or other feed-cost shocks",
    "supply": "chick placements, egg sets, flock size, slaughter, RTC production",
    "stocks": "cold-storage inventories of chicken / broiler products",
    "price": "wholesale/retail chicken prices, spreads, feature prices (not only composite print)",
    "disease": "HPAI or other disease culls, quarantine, flock loss",
    "trade": "chicken exports/imports, trade cases, foreign demand for U.S. chicken",
    "demand": "foodservice vs retail demand, cutout strength, substitution",
    "weather": "heat/cold/storm stress on major poultry states",
    "policy": "USDA/FDA/inspection or domestic poultry regulation (non-trade)",
    "macro": "income, CPI food, recession affecting meat demand",
    "price_report": "official broiler composite / AMS chicken price level prints themselves",
    "other": "only if none of the above fit",
}

# Keep field name vmt_effect for SoftLSTM soft() compatibility (= effect on target series).
VMT_EFFECTS = ("up", "down", "mixed", "unclear")
PLAUSIBILITY = ("high", "medium", "low")
PRED_LEN = 6
SEQ_LEN = 8
VLLM_CONCURRENCY = 48
MAX_MECHANISMS_PER_EVENT = 3


def normalize_channel(x: Any) -> str:
    s = str(x or "other").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "corn": "feed",
        "soy": "feed",
        "soybean": "feed",
        "soybean_meal": "feed",
        "feed_cost": "feed",
        "placement": "supply",
        "placements": "supply",
        "egg_set": "supply",
        "slaughter": "supply",
        "production": "supply",
        "flock": "supply",
        "cold_storage": "stocks",
        "inventory": "stocks",
        "wholesale": "price",
        "retail": "price",
        "spread": "price",
        "hpai": "disease",
        "avian": "disease",
        "avian_influenza": "disease",
        "export": "trade",
        "import": "trade",
        "foodservice": "demand",
        "heat": "weather",
        "cold": "weather",
        "broiler_composite": "price_report",
        "ams_price": "price_report",
        "composite": "price_report",
    }
    s = aliases.get(s, s)
    return s if s in CHANNELS else "other"


def normalize_vmt_effect(x: Any) -> str:
    s = str(x or "unclear").strip().lower().replace(" ", "_")
    aliases = {
        "increase": "up",
        "rise": "up",
        "bullish": "up",
        "decrease": "down",
        "fall": "down",
        "bearish": "down",
        "volatile": "mixed",
        "neutral": "unclear",
    }
    s = aliases.get(s, s)
    return s if s in VMT_EFFECTS else "unclear"


def normalize_plausibility(x: Any) -> str:
    s = str(x or "medium").strip().lower()
    return s if s in PLAUSIBILITY else "medium"


def normalize_mechanism_chain(x: Any) -> str:
    s = str(x or "").strip()
    s = " ".join(s.split())
    for a in ("->", "=>", "→", "⟶", "➜"):
        s = s.replace(a, "→")
    s = s.replace("→", " → ")
    return " ".join(s.split())[:320]


def normalize_mechanism(m: Any) -> dict[str, Any] | None:
    if not isinstance(m, dict):
        return None
    chain = normalize_mechanism_chain(m.get("chain") or m.get("mechanism_chain"))
    if not chain:
        return None
    return {
        "chain": chain,
        "vmt_effect": normalize_vmt_effect(m.get("vmt_effect") or m.get("price_effect")),
        "plausibility": normalize_plausibility(m.get("plausibility")),
        "rationale": str(m.get("rationale") or "")[:400],
        "caveat": str(m.get("caveat") or "")[:240],
    }


def normalize_mechanisms(raw: Any) -> list[dict[str, Any]]:
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
    return out[:MAX_MECHANISMS_PER_EVENT]


def primary_vmt_effect(mechanisms: list[dict[str, Any]]) -> str:
    if not mechanisms:
        return "unclear"
    # prefer high plausibility
    order = {"high": 0, "medium": 1, "low": 2}
    ms = sorted(mechanisms, key=lambda m: order.get(m.get("plausibility", "medium"), 9))
    return normalize_vmt_effect(ms[0].get("vmt_effect"))
