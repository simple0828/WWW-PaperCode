"""Prompts for per-month broiler event extraction + mechanism hypotheses."""

from __future__ import annotations

from language.lamp_agri.schema import (
    CHANNEL_DEFS,
    CHANNELS,
    MAX_MECHANISMS_PER_EVENT,
    VMT_EFFECTS,
)


def _channel_guide() -> str:
    lines = ["CHANNEL DEFINITIONS (pick the single best fit; avoid other unless necessary):"]
    for c in CHANNELS:
        lines.append(f"- {c}: {CHANNEL_DEFS.get(c, '')}")
    lines.extend(
        [
            "PRIORITY TIES:",
            "- Official broiler composite / AMS chicken price level print → price_report",
            "- Corn/soy/feed cost → feed (not macro)",
            "- Placements/slaughter/production → supply",
            "- Cold storage → stocks",
            "- HPAI/culls → disease",
            "- Exports/imports/trade cases → trade",
            "- Use other ONLY if none apply.",
        ]
    )
    return "\n".join(lines)


EXTRACT_SYSTEM = """You are a senior U.S. poultry / livestock market economist.

ROLE
- Structure ONE month of cleaned news into atomic events relevant to the
  U.S. monthly retail broiler composite price.
- For EACH event, write PROFESSIONAL mechanism hypotheses an expert would use
  in a broiler market briefing.
- You are NOT a numerical residual forecaster. Do not invent future prices.

FACT vs MECHANISM
- fact_text: grounded ONLY in THIS month's text. No invented numbers/entities.
- mechanisms: expert PRIOR reasoning about HOW the fact could transmit to the
  retail broiler composite (or near-term chicken price complex).

EXPERT MECHANISM STANDARD
- Name the economic channel (feed-cost pass-through, supply response, cold-storage
  buffer, disease shock, export demand, retail feature demand, etc.).
- Prefer 2–3 step chains ending at broiler composite / chicken price ↑/↓, e.g.
  corn_price↑ → feed_cost↑ → grower_margins↓ → placements↓ → future_supply↓ → price↑
- Competing hypotheses allowed (up to 3) with DIFFERENT pathways / directions.
- plausibility ∈ {high, medium, low}; add rationale + caveat (lags, region, seasonality).

WHAT COUNTS AS AN EVENT
- One checkable claim that could plausibly affect U.S. retail broiler composite.
- Extract ALL such claims (no fixed count). Skip ads, recipes, foreign fluff.

OUTPUT: JSON only. Field name for direction remains \"vmt_effect\" (legacy schema)
meaning effect on the TARGET SERIES (broiler composite), not vehicle-miles."""


def build_extract_month_prompt(*, date: str, text: str) -> str:
    txt = (text or "").strip()
    return "\n".join(
        [
            f"Month date: {date}",
            "",
            "TASK",
            "1) Extract ALL atomic broiler-price-relevant events from this month.",
            "2) For each event, propose up to "
            f"{MAX_MECHANISMS_PER_EVENT} expert mechanism hypotheses.",
            "If none, return {\"events\": []}.",
            "",
            "SCHEMA (each event):",
            f"- channel ∈ {list(CHANNELS)}",
            "- fact_text: one short grounded sentence from THIS month",
            "- mechanisms: array (1.."
            f"{MAX_MECHANISMS_PER_EVENT}) of objects:",
            "    - chain: 'A → B → C → broiler_composite↑/↓'",
            f"    - vmt_effect ∈ {list(VMT_EFFECTS)}  # direction on broiler composite",
            "    - plausibility ∈ ['high','medium','low']",
            "    - rationale: 1–2 expert sentences",
            "    - caveat: when this pathway may not dominate",
            "",
            _channel_guide(),
            "",
            "EXAMPLES (style; adapt to actual fact)",
            "Feed: corn↑ → feed_cost↑ → margins↓ → placements↓ → supply↓ → price↑",
            "Disease: HPAI_culls↑ → near-term_supply↓ → wholesale↑ → composite↑",
            "Trade: Mexico_export_demand↑ → disappearance↑ → price↑",
            "Stocks: cold_storage↑ → available_supply↑ → price↓ (near term)",
            "",
            "RETURN JSON:",
            '{ "events": [ { "channel": "feed", "fact_text": "...", "mechanisms": [',
            '  { "chain": "corn_price↑ → feed_cost↑ → broiler_composite↑",',
            '    "vmt_effect": "up", "plausibility": "high",',
            '    "rationale": "Feed is the dominant variable cost.",',
            '    "caveat": "Pass-through lags 1–3 months." } ] } ] }',
            "",
            "MONTH TEXT (only source of facts):",
            txt[:6000] if txt else "(empty)",
        ]
    )
