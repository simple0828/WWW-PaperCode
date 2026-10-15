"""Scheme 5 — per-channel accumulated prompt (no retrieval).

Each FOCUS channel keeps a **landing prompt text** ``prompt_c`` that is rewritten
over time from success/failure outcomes. Inference always feeds the *full*
current ``prompt_c`` (no similarity search).

Within a window, **all** events of that channel in the lookback are used
(mechanism hypotheses only as hints inside the CoT).

Optimization (this runner): walk windows in time order over **train+test**
(as requested) and periodically rewrite ``prompt_c``. LLM weights stay frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.lamp_vmt.json_utils import extract_json
from language.stage2.actions import BIAS

DIR_TOK = {
    "increase": "up",
    "higher": "up",
    "up": "up",
    "decrease": "down",
    "lower": "down",
    "down": "down",
    "flat": "flat",
    "mixed": "flat",
    "volatile": "flat",
    "abstain": "flat",
}


def _clip(s: Any, n: int) -> str:
    return str(s or "").replace("\n", " ")[:n]


def seed_prompt(channel: str, target: str = "", *, domain: str = "") -> str:
    """Cold-start protocol. With ``domain``, use per-dataset seeds from scheme5_seeds."""
    if domain:
        from language.stage2.scheme5_seeds import seed_prompt as _domain_seed

        return _domain_seed(channel, target, domain=domain)
    return "\n".join(
        [
            f"CHANNEL ASSISTANT PROTOCOL — {channel}",
            f"Target series residual: {target} vs CFA forecast (true − CFA).",
            "Role: reason from THIS channel's events only; write durable rules into your",
            "own protocol over time (the text below IS your memory — not a retrieval store).",
            "",
            "Standing rules:",
            "1) Prefer exogenous / delayed drivers over restatements of the target series.",
            "2) Use mechanism hypotheses as HINTS for further CoT; do not merely copy them.",
            "3) Try to explain as many listed events as possible when they share a coherent story.",
            "4) If evidence conflicts or is only meta-report noise → gate=0 (abstain).",
            "5) Output a single residual direction for THIS channel for the forecast horizon.",
            "",
            "Accumulated lessons (append/rewrite over training):",
            "- (none yet — learn from outcomes)",
        ]
    )


def format_channel_events(
    by_ym: Mapping[str, Sequence[dict]],
    months: Sequence[str],
    channel: str,
    *,
    max_mechs: int = 3,
) -> tuple[str, int]:
    """Serialize ALL events of ``channel`` in the window (no top-k drop)."""
    lines: list[str] = []
    n = 0
    for t, ym in enumerate(months):
        ym = str(ym)[:7]
        lag = max(0, len(months) - 1 - t)
        evs = [e for e in by_ym.get(ym, []) if str(e.get("channel") or "") == channel]
        for ei, e in enumerate(evs):
            n += 1
            fact = _clip(e.get("fact_text"), 140)
            lines.append(f"[{ym}/E{ei}] lag={lag} fact: {fact}")
            mechs = e.get("mechanisms") or []
            if not mechs and (e.get("mechanism_chain") or e.get("vmt_effect")):
                mechs = [
                    {
                        "chain": e.get("mechanism_chain") or "",
                        "vmt_effect": e.get("vmt_effect") or "unclear",
                        "plausibility": "medium",
                    }
                ]
            for mi, m in enumerate(mechs[:max_mechs]):
                lines.append(
                    f"  H{mi+1}[{m.get('plausibility')}→{m.get('vmt_effect')}]: "
                    f"{_clip(m.get('chain') or m.get('mechanism_chain'), 100)}"
                )
    return ("\n".join(lines) if lines else "(no events)"), n


INFER_SYSTEM = (
    "You are a per-channel residual reasoning assistant. "
    "Follow the CHANNEL PROTOCOL text exactly. JSON only."
)


def prompt_infer(
    *,
    target: str,
    channel: str,
    protocol: str,
    origin: str,
    cfa: Sequence[float],
    events_block: str,
    n_events: int,
    H: int,
) -> str:
    return "\n".join(
        [
            f"Target={target}  channel={channel}  origin={origin}  H={H}",
            f"CFA forecast: {np.round(np.asarray(cfa, float), 4).tolist()}",
            "",
            "=== CHANNEL PROTOCOL (accumulated experience — follow & apply) ===",
            protocol.strip(),
            "=== END PROTOCOL ===",
            "",
            f"ALL events for this channel in the lookback ({n_events} events; do not ignore casually):",
            events_block,
            "",
            "Task: further CoT beyond raw mechanism hypotheses. Explain as many events as you can",
            "when coherent; judge residual of TRUE future vs CFA for THIS channel.",
            "JSON ONLY:",
            "{"
            '"cot":"≤40 words",'
            '"explained_events":["YYYY-MM/E#", "..."],'
            '"residual_dir":"increase|decrease|flat",'
            '"strength":"weak|medium|strong",'
            '"gate":0|1,'
            '"confidence":0-1,'
            '"horizon_bias":["flat|up|down|strong_up|strong_down"]x'
            + str(H)
            + ","
            '"reason":"≤20 words"'
            "}",
            "gate=1 only for clear increase/decrease; else 0.",
        ]
    )


REWRITE_SYSTEM = (
    "You rewrite a channel assistant PROTOCOL text. "
    "Accumulate durable rules from outcomes. No retrieval store — the protocol IS the memory. "
    "Return the full new protocol as plain text only."
)


def prompt_rewrite(
    *,
    channel: str,
    target: str,
    old_protocol: str,
    outcomes_block: str,
) -> str:
    return "\n".join(
        [
            f"Rewrite the PROTOCOL for channel={channel} (target residual: {target}).",
            "Constraints:",
            "- Keep it as operational rules / lessons (bullet clauses), not a dump of raw events.",
            "- Incorporate successes (what worked for residual + event coverage).",
            "- Add failure constraints / abstain rules from failures.",
            "- Prefer ≤45 lines; merge redundant clauses; drop stale noise.",
            "- Do NOT invent numeric VMT levels. Do NOT add retrieval instructions.",
            "",
            "CURRENT PROTOCOL:",
            old_protocol.strip(),
            "",
            "RECENT OUTCOMES (time-ordered; use to accumulate experience):",
            outcomes_block,
            "",
            "Return the FULL updated PROTOCOL text only (no markdown fences, no JSON wrapper).",
        ]
    )


def _norm_dir(rd: str) -> str:
    rd = str(rd or "flat").lower().replace(" ", "_")
    if rd in {"up", "increase", "inc", "positive", "higher"}:
        return "increase"
    if rd in {"down", "decrease", "dec", "negative", "lower"}:
        return "decrease"
    return "flat"


def parse_infer(text: str, H: int = 6) -> dict[str, Any]:
    try:
        obj = extract_json(text) if text else {}
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    try:
        conf = float(obj.get("confidence") or 0.5)
    except Exception:
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    try:
        gate = int(obj.get("gate") or 0)
    except Exception:
        gate = 0
    if gate not in (0, 1):
        gate = 0
    rd = _norm_dir(obj.get("residual_dir") or "flat")
    if rd in ("flat", "mixed", "volatile", "abstain") or conf < 0.55:
        gate = 0
    hb = obj.get("horizon_bias")
    if isinstance(hb, list) and len(hb) >= H:
        hb = [str(x).lower().replace(" ", "_") for x in hb[:H]]
        hb = [x if x in BIAS else DIR_TOK.get(rd, "flat") for x in hb]
    else:
        tok = DIR_TOK.get(rd, "flat")
        hb = [tok] * H
    vec = np.array([BIAS.get(x, 0.0) for x in hb], dtype=np.float64)
    if gate == 0:
        vec[:] = 0.0
    else:
        vec *= 0.35 + 0.65 * conf
    explained = obj.get("explained_events") or []
    if not isinstance(explained, list):
        explained = []
    return {
        "gate": gate,
        "vec": vec,
        "conf": conf,
        "dir": rd,
        "explained": [str(x) for x in explained],
        "n_explained": len(explained),
        "cot": _clip(obj.get("cot") or obj.get("reason") or obj.get("why"), 160),
        "raw": obj,
        "ok": bool(obj),
    }


def fuse_channel_vecs(vecs: Sequence[np.ndarray], gates: Sequence[int]) -> np.ndarray:
    active = [v for v, g in zip(vecs, gates) if g and np.any(np.abs(v) > 1e-12)]
    if not active:
        return np.zeros_like(vecs[0]) if vecs else np.zeros(6)
    return np.mean(np.stack(active, 0), axis=0)


@dataclass
class Scheme5Config:
    rewrite_every: int = 24
    n_epochs: int = 2
    wave: int = 48
    infer_temp: float = 0.0
    rewrite_temp: float = 0.3
    max_outcome_chars: int = 3500
    max_protocol_chars: int = 4500
    soft_amp_scale: float = 1.0  # optional global scale after fuse
    cache_tag: str = "traffic_v1"
    max_tokens_infer: int = 1024
    max_tokens_rewrite: int = 2048


@dataclass
class ChannelState:
    channel: str
    protocol: str
    version: int = 0
    outcomes: list[str] = field(default_factory=list)
    n_infer: int = 0
    n_success: int = 0


def protocol_path(root: Path, channel: str) -> Path:
    return root / f"prompt_{channel}.txt"


def save_protocol(root: Path, st: ChannelState) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol_path(root, st.channel).write_text(st.protocol, encoding="utf-8")
    meta = root / f"meta_{st.channel}.json"
    meta.write_text(
        json.dumps(
            {
                "channel": st.channel,
                "version": st.version,
                "n_infer": st.n_infer,
                "n_success": st.n_success,
                "n_outcomes_buffered": len(st.outcomes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_or_init_states(
    root: Path,
    channels: Sequence[str],
    target: str,
    *,
    domain: str = "",
) -> dict[str, ChannelState]:
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, ChannelState] = {}
    for ch in channels:
        p = protocol_path(root, ch)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            ver = 0
            mp = root / f"meta_{ch}.json"
            if mp.exists():
                try:
                    ver = int(json.loads(mp.read_text()).get("version") or 0)
                except Exception:
                    ver = 0
            out[ch] = ChannelState(channel=ch, protocol=text, version=ver)
        else:
            st = ChannelState(
                channel=ch,
                protocol=seed_prompt(ch, target, domain=domain),
                version=0,
            )
            save_protocol(root, st)
            out[ch] = st
    return out


def outcome_line(
    *,
    wid: str,
    channel: str,
    n_events: int,
    act: dict,
    improved: bool,
    resid_sign: int,
) -> str:
    cov = act["n_explained"] / max(n_events, 1)
    return (
        f"- wid={wid} ch={channel} n_ev={n_events} explained={act['n_explained']} "
        f"cov={cov:.2f} gate={act['gate']} dir={act['dir']} "
        f"resid_sign={resid_sign:+d} improved={improved} cot={act['cot']}"
    )


def resid_sign_of(yt, yc) -> int:
    r = float(np.mean(np.asarray(yt, float) - np.asarray(yc, float)))
    if abs(r) < 1e-9:
        return 0
    return 1 if r > 0 else -1


def channel_improved(yt, yc, vec) -> bool:
    b0 = float(np.mean((np.asarray(yc, float) - np.asarray(yt, float)) ** 2))
    m1 = float(np.mean((np.asarray(yc, float) + np.asarray(vec, float) - np.asarray(yt, float)) ** 2))
    return m1 < b0 - 1e-12
