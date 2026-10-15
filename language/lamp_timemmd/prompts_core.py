"""Build extract prompts from a DomainLampConfig (mirrors lamp_agri/lamp_vmt)."""

from __future__ import annotations

from typing import Any, Dict

from language.lamp_timemmd.domains import DomainLampConfig, get_domain_config
from language.lamp_timemmd.schema_core import MAX_MECHANISMS_PER_EVENT, VMT_EFFECTS


def _channel_guide(cfg: DomainLampConfig) -> str:
    lines = [
        "CHANNEL DEFINITIONS (pick the single best fit; avoid other unless necessary):"
    ]
    for c in cfg.channels:
        lines.append(f"- {c}: {cfg.channel_defs.get(c, '')}")
    lines.append("PRIORITY TIES:")
    lines.extend(cfg.priority_ties)
    return "\n".join(lines)


def build_extract_system(cfg: DomainLampConfig) -> str:
    period = cfg.period_label
    target = cfg.target_series_short
    return f"""You are a {cfg.expert_role}.

ROLE
- Structure ONE {period} of cleaned news into atomic events relevant to the
  {target}.
- For EACH event, write PROFESSIONAL mechanism hypotheses an expert would use
  in a domain briefing.
- You are NOT a numerical residual forecaster. Do not invent future values.

FACT vs MECHANISM
- fact_text: grounded ONLY in THIS {period}'s text. No invented numbers/entities.
- mechanisms: expert PRIOR reasoning about HOW the fact could transmit to the
  {target}.

EXPERT MECHANISM STANDARD
- Name the economic / domain channel.
- Prefer 2–3 step chains ending at {cfg.chain_end_hint}.
- Competing hypotheses allowed (up to 3) with DIFFERENT pathways / directions.
- plausibility ∈ {{high, medium, low}}; add rationale + caveat (lags, region, seasonality).

WHAT COUNTS AS AN EVENT
- {cfg.what_counts}
- Extract ALL such claims (no fixed count).

OUTPUT: JSON only. Field name for direction remains \"vmt_effect\" (legacy schema)
meaning effect on the TARGET SERIES ({target}), not vehicle-miles."""


def build_extract_month_prompt_for(
    cfg: DomainLampConfig, *, date: str, text: str
) -> str:
    txt = (text or "").strip()
    period = cfg.period_label
    return "\n".join(
        [
            f"{period.capitalize()} date: {date}",
            "",
            "TASK",
            f"1) {cfg.extract_task_line}",
            "2) For each event, propose up to "
            f"{MAX_MECHANISMS_PER_EVENT} expert mechanism hypotheses.",
            'If none, return {"events": []}.',
            "",
            "SCHEMA (each event):",
            f"- channel ∈ {list(cfg.channels)}",
            f"- fact_text: one short grounded sentence from THIS {period}",
            "- mechanisms: array (1.."
            f"{MAX_MECHANISMS_PER_EVENT}) of objects:",
            f"    - chain: 'A → B → C → {cfg.chain_end_hint}'",
            f"    - vmt_effect ∈ {list(VMT_EFFECTS)}  # direction on target series",
            "    - plausibility ∈ ['high','medium','low']",
            "    - rationale: 1–2 expert sentences",
            "    - caveat: when this pathway may not dominate",
            "",
            _channel_guide(cfg),
            "",
            "EXAMPLES (style; adapt to actual fact)",
            *cfg.mechanism_examples,
            "",
            "RETURN JSON:",
            '{ "events": [ { "channel": "'
            + cfg.report_channel
            + '", "fact_text": "...", "mechanisms": [',
            f'  {{ "chain": "signal↑ → {cfg.chain_end_hint.replace("↑/↓", "↑")}",',
            '    "vmt_effect": "up", "plausibility": "high",',
            '    "rationale": "Direct measure of the target series.",',
            '    "caveat": "Revisions and seasonal adjustment can change the print." } ] } ] }',
            "",
            f"{period.upper()} TEXT (only source of facts):",
            txt[:6000] if txt else "(empty)",
        ]
    )


def install_prompts(namespace: Dict[str, Any], domain: str) -> DomainLampConfig:
    cfg = get_domain_config(domain)
    extract_system = build_extract_system(cfg)

    def build_extract_month_prompt(*, date: str, text: str) -> str:
        return build_extract_month_prompt_for(cfg, date=date, text=text)

    namespace.update(
        {
            "DOMAIN_KEY": cfg.key,
            "EXTRACT_SYSTEM": extract_system,
            "build_extract_month_prompt": build_extract_month_prompt,
            "_channel_guide": lambda: _channel_guide(cfg),
        }
    )
    return cfg
