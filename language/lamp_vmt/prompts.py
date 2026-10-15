"""Prompts for per-month VMT event extraction + expert multi-hypothesis mechanisms."""

from __future__ import annotations

from language.lamp_vmt.schema import (
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
            "- If the fact is mainly a VMT/AADT/vehicle-miles level or growth statistic → vmt_report",
            "- Road/bridge/capacity construction → infrastructure (not policy)",
            "- Congestion/speed/delay → congestion",
            "- Crash/fatality/safety program → safety",
            "- Auto sales / EV / fleet stock → fleet",
            "- Jobs/income/recession → macro",
            "- Mode shift to transit/bike → transit",
            "- Use other ONLY if none of the above apply.",
        ]
    )
    return "\n".join(lines)


EXTRACT_SYSTEM = """You are a senior U.S. transportation economist / travel-demand analyst.

ROLE
- Structure ONE month of cleaned news into atomic VMT-relevant events.
- For EACH event, write PROFESSIONAL domain-mechanism hypotheses that an expert
  would use in travel-demand / VMT briefing notes.
- You are NOT a numerical residual forecaster. Do not invent future VMT levels
  or claim you know the realized forecast error.

FACT vs MECHANISM (strict separation)
- fact_text: grounded ONLY in THIS month's text. No invented numbers/entities.
- mechanisms: expert PRIOR reasoning about HOW the fact could transmit to national
  monthly vehicle-miles traveled (VMT). Use transportation economics, not sentiment.

EXPERT MECHANISM STANDARD (required)
Write like a domain expert, not a slogan:
- Name the behavioral / economic channel (price elasticity, trip generation,
  destination choice, mode shift, freight logistics, network capacity, etc.).
- Prefer 2–3 step causal chains ending at VMT, e.g.
  gasoline_price↑ → generalized_travel_cost↑ → discretionary_trip_rate↓ → VMT↓
- When scientifically plausible, give MULTIPLE competing hypotheses (up to 3)
  with DIFFERENT pathways and possibly DIFFERENT vmt_effect outcomes.
- Rank each hypothesis with plausibility ∈ {high, medium, low}.
- Add a short rationale (why this pathway) and caveat (when it may fail:
  lag, geography, income effects, rebound, already in seasonality, etc.).
- If effects compete, keep separate hypotheses rather than collapsing early;
  event-level net pressure can be mixed.

WHAT COUNTS AS AN EVENT
- One checkable claim that could plausibly affect U.S. monthly VMT.
- Extract ALL such claims in this month (NO fixed count / NO cap of 12).
- Skip ads, foreign digressions, boilerplate, and vague commentary.

CHANNELING
- Assign exactly one channel from the closed list.
- VMT/AADT level reports are first-class: use channel=vmt_report (NOT other).
- Prefer specific channels over other.

DO NOT
- Invent facts absent from the month text.
- Output actor / action / magnitude / polarity fields.
- Force mechanisms to match any residual you do not see.
- Dump important VMT statistics into channel=other.

OUTPUT: JSON only."""


RECLASS_CHANNEL_SYSTEM = """You are a senior transportation analyst.
Re-assign each event to exactly ONE channel from a closed vocabulary.
Prefer specific channels. VMT/AADT/vehicle-miles statistics → vmt_report (never other).
Output JSON only."""


def build_extract_month_prompt(*, date: str, text: str) -> str:
    txt = (text or "").strip()
    return "\n".join(
        [
            f"Month date: {date}",
            "",
            "TASK",
            "1) Extract ALL atomic VMT-relevant events from this month.",
            "2) For each event, propose up to "
            f"{MAX_MECHANISMS_PER_EVENT} expert mechanism hypotheses.",
            "   Competing pathways and opposite VMT directions are allowed when plausible.",
            "If none, return {\"events\": []}.",
            "",
            "SCHEMA (each event):",
            f"- channel ∈ {list(CHANNELS)}",
            "- fact_text: one short grounded sentence from THIS month",
            "- mechanisms: array (1.."
            f"{MAX_MECHANISMS_PER_EVENT}) of objects:",
            "    - chain: 'A → B → C → VMT↑/↓' (expert causal chain)",
            f"    - vmt_effect ∈ {list(VMT_EFFECTS)}",
            "    - plausibility ∈ ['high','medium','low']",
            "    - rationale: 1–2 expert sentences (elasticity / behavior / lag)",
            "    - caveat: when this pathway may not dominate",
            "",
            _channel_guide(),
            "",
            "EXPERT EXAMPLES (style to imitate; adapt to the actual fact)",
            "Fuel price rise:",
            "  high: gasoline_real_price↑ → operating_cost↑ → discretionary_VMT↓ → VMT↓",
            "Holiday/tourism surge:",
            "  high: holiday_activity↑ → long-distance_car_trips↑ → VMT↑",
            "VMT statistics report:",
            "  channel=vmt_report; high: reported_VMT↑ → realized_travel_activity↑ → VMT↑",
            "Highway capacity expansion:",
            "  channel=infrastructure; high: capacity↑ → induced_demand↑ → VMT↑",
            "Severe congestion:",
            "  channel=congestion; medium: delay↑ → trip_suppression↓ → VMT↓ (or detour miles↑)",
            "Freight tonnage rise:",
            "  channel=freight; high: truck_tonnage↑ → freight_VMT↑ → national VMT↑",
            "Remote work:",
            "  channel=remote_work; high: WFH_share↑ → commute_trips↓ → VMT↓",
            "",
            "RETURN JSON:",
            "{",
            '  "events": [',
            "    {",
            '      "channel": "vmt_report",',
            '      "fact_text": "...",',
            '      "mechanisms": [',
            "        {",
            '          "chain": "reported_national_VMT↑ → realized_activity↑ → VMT↑",',
            '          "vmt_effect": "up",',
            '          "plausibility": "high",',
            '          "rationale": "Official VMT prints are direct measures of travel activity.",',
            '          "caveat": "Revisions and seasonal adjustment can change the print later."',
            "        }",
            "      ]",
            "    }",
            "  ]",
            "}",
            "",
            "MONTH TEXT (only source of facts):",
            txt[:6000] if txt else "(empty)",
        ]
    )


def build_reclass_channel_prompt(*, items: list[dict]) -> str:
    """items: [{id, date, fact_text, mechanism_chain, old_channel}, ...]"""
    lines = [
        "TASK: Re-label channel for each event. Keep id unchanged.",
        f"channel ∈ {list(CHANNELS)}",
        "",
        _channel_guide(),
        "",
        "EVENTS:",
    ]
    for it in items:
        lines.append(
            f"- id={it['id']} | old={it.get('old_channel')} | date={it.get('date')} | "
            f"fact={it.get('fact_text', '')[:220]} | "
            f"chain={it.get('mechanism_chain', '')[:160]}"
        )
    lines.extend(
        [
            "",
            "RETURN JSON:",
            '{ "labels": [ {"id": 0, "channel": "vmt_report"}, {"id": 1, "channel": "infrastructure"} ] }',
        ]
    )
    return "\n".join(lines)
