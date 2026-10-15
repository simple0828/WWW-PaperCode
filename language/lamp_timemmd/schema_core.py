"""Shared schema helpers; bind to a DomainLampConfig for module-level exports."""

from __future__ import annotations

from typing import Any, Dict

from language.lamp_timemmd.domains import DomainLampConfig, get_domain_config

VMT_EFFECTS = ("up", "down", "mixed", "unclear")
PLAUSIBILITY = ("high", "medium", "low")
PRED_LEN = 6
SEQ_LEN = 8
VLLM_CONCURRENCY = 48
MAX_MECHANISMS_PER_EVENT = 3


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


def make_normalize_channel(cfg: DomainLampConfig):
    channels = set(cfg.channels)
    aliases = dict(cfg.channel_aliases)

    def normalize_channel(x: Any) -> str:
        s = str(x or "other").strip().lower().replace(" ", "_").replace("-", "_")
        s = aliases.get(s, s)
        return s if s in channels else "other"

    return normalize_channel


def make_normalize_mechanism(normalize_channel_fn=None):
    """normalize_channel unused; kept for API symmetry with domain packages."""

    def normalize_mechanism(m: Any) -> dict[str, Any] | None:
        if not isinstance(m, dict):
            return None
        chain = normalize_mechanism_chain(m.get("chain") or m.get("mechanism_chain"))
        if not chain:
            return None
        return {
            "chain": chain,
            "vmt_effect": normalize_vmt_effect(
                m.get("vmt_effect") or m.get("price_effect")
            ),
            "plausibility": normalize_plausibility(m.get("plausibility")),
            "rationale": str(m.get("rationale") or "")[:400],
            "caveat": str(m.get("caveat") or "")[:240],
        }

    return normalize_mechanism


def make_normalize_mechanisms(normalize_mechanism_fn):
    def normalize_mechanisms(raw: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                nm = normalize_mechanism_fn(item)
                if nm:
                    out.append(nm)
        elif isinstance(raw, dict):
            nm = normalize_mechanism_fn(raw)
            if nm:
                out.append(nm)
        return out[:MAX_MECHANISMS_PER_EVENT]

    return normalize_mechanisms


def primary_vmt_effect(mechanisms: list[dict[str, Any]]) -> str:
    if not mechanisms:
        return "unclear"
    order = {"high": 0, "medium": 1, "low": 2}
    ms = sorted(
        mechanisms, key=lambda m: order.get(m.get("plausibility", "medium"), 9)
    )
    return normalize_vmt_effect(ms[0].get("vmt_effect"))


def install_schema(namespace: Dict[str, Any], domain: str) -> DomainLampConfig:
    """Populate a schema module namespace for ``domain`` (like lamp_agri.schema)."""
    cfg = get_domain_config(domain)
    normalize_channel = make_normalize_channel(cfg)
    normalize_mechanism = make_normalize_mechanism()
    normalize_mechanisms = make_normalize_mechanisms(normalize_mechanism)

    namespace.update(
        {
            "CHANNELS": cfg.channels,
            "CHANNEL_DEFS": cfg.channel_defs,
            "VMT_EFFECTS": VMT_EFFECTS,
            "PLAUSIBILITY": PLAUSIBILITY,
            "PRED_LEN": PRED_LEN,
            "SEQ_LEN": SEQ_LEN,
            "VLLM_CONCURRENCY": VLLM_CONCURRENCY,
            "MAX_MECHANISMS_PER_EVENT": MAX_MECHANISMS_PER_EVENT,
            "TARGET_SERIES_SHORT": cfg.target_series_short,
            "REPORT_CHANNEL": cfg.report_channel,
            "DOMAIN_KEY": cfg.key,
            "normalize_channel": normalize_channel,
            "normalize_vmt_effect": normalize_vmt_effect,
            "normalize_plausibility": normalize_plausibility,
            "normalize_mechanism_chain": normalize_mechanism_chain,
            "normalize_mechanism": normalize_mechanism,
            "normalize_mechanisms": normalize_mechanisms,
            "primary_vmt_effect": primary_vmt_effect,
        }
    )
    return cfg
