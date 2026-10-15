"""Explicit branch **Candidate 1** — hierarchical channel→window one-shot.

Replaces the old event→window two-level hierarchy.

Pipeline (LLM frozen; KEEP budgets are decode-time hyperparams):
  1) **Channel level** (within lookback L): all events of one channel → one-shot
     JSON. LLM returns a *ranked* ``keep_events`` list + channel residual dir/gate.
     Decode keeps only the top-``keep_events_per_channel`` entries.
  2) **Window level** (across channels): channel judgments → one-shot JSON.
     LLM returns a *ranked* ``keep_channels`` list + gate + horizon_bias.
     Decode keeps only the top-``keep_channels`` entries; conflicting channel
     signs or low confidence → gate=0 (abstain).

Hyperparameter search (dataset-specific) over
``keep_events_per_channel``, ``keep_channels``, ``soft_amp`` reuses the same
cached LLM responses (rank-then-truncate).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.lamp_vmt.json_utils import extract_json
from language.stage2.actions import BIAS
from language.stage2.mech_keep import _effect_score, _plaus
from language.stage2.profiles import Stage2DomainProfile

DIR_SIGN = {
    "increase": 1.0,
    "higher": 1.0,
    "up": 1.0,
    "decrease": -1.0,
    "lower": -1.0,
    "down": -1.0,
}
STRENGTH_W = {"weak": 0.55, "medium": 1.0, "strong": 1.45, "high": 1.35}


CHANNEL_PROMPT = """You aggregate ONE channel's events inside a lookback window for {target}.

Channel={channel}  Origin={origin}  CFA={cfa}

EVENTS in this channel across L months ({n_events}; use as many as coherent):
{events}

Equal-candidate Stage1 mechanisms are HINTS only (further CoT beyond them).

Rules:
- Decide the *net* residual effect of THIS channel on TRUE future vs CFA.
- Rank events you rely on in keep_events (most important first). Cite ids exactly.
- Prefer exogenous / delayed drivers; ignore pure target-series restatements.
- If events conflict or only meta-report noise → gate=0 and channel_dir=abstain.
- Do NOT invent events.

JSON ONLY:
{{"channel_dir":"increase|decrease|abstain","gate":0|1,"confidence":0-1,
 "keep_events":["YYYY-MM/E#", "..."],
 "strength":"weak|medium|strong","reason":"<=25 words"}}
"""


WINDOW_PROMPT = """You aggregate channel-level residual judgments for {target}.

CFA baseline already exists. Decide whether TRUE future may DEVIATE from CFA.

Origin={origin}  CFA={cfa}

CHANNEL JUDGMENTS ({n_channels}):
{channels}

Rules:
- Rank channels you rely on in keep_channels (most important first). Cite channel names exactly.
- Active (gate=1) kept channels should share ONE residual sign; else gate=0.
- If evidence weak / mostly abstain / conflicting → gate=0.
- Do NOT invent channels. You may ignore low-confidence rows.
- horizon_bias length must be {H} (tokens: strong_up|up|flat|down|strong_down).

JSON ONLY:
{{"residual_hypothesis":"increase|decrease|flat|mixed","gate":0|1,"confidence":0-1,
 "keep_channels":["channel_name", "..."],
 "horizon_bias":["flat","flat","flat","flat","flat","flat"],
 "reason":"<=25 words"}}
"""


@dataclass
class Scheme1Config:
    """Hierarchy + fill knobs. KEEP sizes used at *decode* are separate fields."""

    # prompt / bank caps (what the LLM sees)
    max_events_per_channel: int = 16
    max_mechs_per_event: int = 3
    # decode-time KEEP budgets (overridden by search)
    keep_events_per_channel: int = 4
    keep_channels: int = 4
    soft_amp: float = 0.05
    min_channel_conf: float = 0.45
    min_window_conf: float = 0.50
    cache_tag: str = "ch2w_v1"
    temperature: float = 0.0
    max_tokens_channel: int = 1024
    max_tokens_window: int = 1024
    wave: int = 48
    # legacy aliases so older callers don't crash
    max_per_month: int = 4
    max_mechs: int = 3
    max_events_window: int = 64


@dataclass
class EventLine:
    display_id: str  # YYYY-MM/E#
    ym: str
    fact: str
    lag: int
    mechs: list[dict[str, Any]] = field(default_factory=list)
    eid: str = ""


@dataclass
class ChannelPack:
    """All events of one channel inside one window's lookback L."""

    channel: str
    key: str  # cache key = wid|channel (window-specific)
    events: list[EventLine] = field(default_factory=list)


def _as_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "text"):
        return str(x.text)
    return str(x)


def stable_event_id(e: Mapping[str, Any]) -> str:
    ym = str(e.get("date") or "")[:7]
    ch = str(e.get("channel") or "other")
    rid = e.get("source_row_id")
    if rid is not None and str(rid) != "":
        return f"{ym}:{ch}:{rid}"
    fact = str(e.get("fact_text") or "").replace("\n", " ").strip()
    h = hashlib.md5(fact.encode("utf-8")).hexdigest()[:10]
    return f"{ym}:{ch}:{h}"


def _event_mechs(e: Mapping[str, Any], max_mechs: int) -> list[dict[str, Any]]:
    raw = list(e.get("mechanisms") or [])
    if not raw:
        raw = [
            {
                "chain": e.get("mechanism_chain") or "",
                "vmt_effect": e.get("vmt_effect") or "unclear",
                "plausibility": "medium",
            }
        ]
    scored = [(_effect_score(m) * _plaus(m), m) for m in raw]
    scored.sort(key=lambda t: -abs(t[0]))
    out = []
    for i, (_, m) in enumerate(scored[:max_mechs], 1):
        out.append(
            {
                "mid": f"M{i}",
                "chain": str(m.get("chain") or m.get("mechanism_chain") or "")[:110],
                "effect": str(m.get("vmt_effect") or "unclear").lower(),
                "plaus": str(m.get("plausibility") or "medium").lower(),
            }
        )
    return out


def build_channel_packs(
    by_ym: Mapping[str, Sequence[dict]],
    months: Sequence[str],
    focus: Sequence[str],
    *,
    wid: str,
    max_events_per_channel: int = 16,
    max_mechs_per_event: int = 3,
) -> list[ChannelPack]:
    """Group ALL focus events in L by channel (no cross-channel top-k drop)."""
    focus_set = set(focus)
    by_ch: dict[str, list[EventLine]] = {c: [] for c in focus}
    for t, ym in enumerate(months):
        ym = str(ym)[:7]
        lag = max(0, len(months) - 1 - t)
        evs = [e for e in by_ym.get(ym, []) if e.get("channel") in focus_set]
        # within-month: prefer non-report, stronger |effect|
        evs = sorted(
            evs,
            key=lambda e: (
                1 if "report" in str(e.get("channel") or "") else 0,
                -abs(
                    sum(
                        _effect_score(m) * _plaus(m)
                        for m in (e.get("mechanisms") or [])[:3]
                    )
                ),
            ),
        )
        counts: dict[str, int] = {}
        for e in evs:
            ch = str(e.get("channel") or "other")
            if ch not in by_ch:
                continue
            ei = counts.get(ch, 0)
            counts[ch] = ei + 1
            by_ch[ch].append(
                EventLine(
                    display_id=f"{ym}/E{ei}",
                    ym=ym,
                    fact=str(e.get("fact_text") or "").replace("\n", " ")[:150],
                    lag=lag,
                    mechs=_event_mechs(e, max_mechs_per_event),
                    eid=stable_event_id(e),
                )
            )
    packs: list[ChannelPack] = []
    for ch in focus:
        evs = by_ch.get(ch) or []
        if not evs:
            continue
        if len(evs) > max_events_per_channel:
            # keep nearest-to-origin first
            evs = sorted(evs, key=lambda x: (x.lag, x.ym, x.display_id))[
                :max_events_per_channel
            ]
            evs = sorted(evs, key=lambda x: (x.ym, x.display_id))
        packs.append(ChannelPack(channel=ch, key=f"{wid}|{ch}", events=evs))
    return packs


def render_channel_prompt(
    pack: ChannelPack,
    profile: Stage2DomainProfile,
    *,
    origin: str,
    cfa: Sequence[float],
) -> str:
    lines = []
    for ev in pack.events:
        mech_bits = "; ".join(
            f"{m['mid']}[{m['plaus']}→{m['effect']}] {m['chain']}" for m in ev.mechs[:3]
        )
        lines.append(
            f"  [{ev.display_id}] lag={ev.lag} fact: {ev.fact}"
            + (f"\n    mechs: {mech_bits}" if mech_bits else "")
        )
    return CHANNEL_PROMPT.format(
        target=profile.target_series_short,
        channel=pack.channel,
        origin=origin,
        cfa=np.round(np.asarray(cfa, float), 4).tolist(),
        n_events=len(pack.events),
        events="\n".join(lines) or "  (empty)",
    )


def render_window_prompt(
    *,
    profile: Stage2DomainProfile,
    origin: str,
    cfa: Sequence[float],
    packs: Sequence[ChannelPack],
    judgments: Sequence[Mapping[str, Any]],
    H: int,
) -> str:
    lines = []
    for p, j in zip(packs, judgments):
        keep = j.get("keep_events") or []
        lines.append(
            f"  [{p.channel}] n_ev={len(p.events)} dir={j.get('channel_dir')} "
            f"gate={j.get('gate')} conf={float(j.get('confidence') or 0):.2f} "
            f"str={j.get('strength')} keep={keep[:6]} "
            f"| why: {str(j.get('reason') or '')[:50]}"
        )
    return WINDOW_PROMPT.format(
        target=profile.target_series_short,
        origin=origin,
        cfa=np.round(np.asarray(cfa, float), 4).tolist(),
        n_channels=len(packs),
        channels="\n".join(lines) or "  (empty)",
        H=H,
    )


def _parse_json(text: str) -> dict:
    obj: Any = {}
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = extract_json(t) if t else {}
    except Exception:
        try:
            obj = json.loads(t[t.find("{") : t.rfind("}") + 1])
        except Exception:
            obj = {}
    return obj if isinstance(obj, dict) else {}


def parse_channel_response(text: str, *, min_conf: float = 0.45) -> dict[str, Any]:
    obj = _parse_json(text)
    raw_dir = str(obj.get("channel_dir") or obj.get("dir") or "abstain").lower().strip()
    if raw_dir in DIR_SIGN:
        channel_dir = "increase" if DIR_SIGN[raw_dir] > 0 else "decrease"
    else:
        channel_dir = "abstain"
    try:
        gate = int(obj.get("gate")) if obj.get("gate") is not None else 0
    except Exception:
        gate = 0
    try:
        conf = float(obj.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    conf = float(np.clip(conf, 0.0, 1.0))
    strength = str(obj.get("strength") or "medium").lower()
    if strength not in STRENGTH_W:
        strength = "medium"
    keep = obj.get("keep_events") or obj.get("keep") or []
    if not isinstance(keep, list):
        keep = []
    keep = [str(x) for x in keep]
    reason = str(obj.get("reason") or "")[:140]
    if channel_dir == "abstain" or conf < min_conf:
        gate = 0
        channel_dir = "abstain"
    if gate != 1:
        gate = 0
        channel_dir = "abstain"
    return {
        "channel_dir": channel_dir,
        "gate": gate,
        "confidence": conf,
        "strength": strength,
        "keep_events": keep,  # full ranked list from LLM
        "reason": reason,
        "sign": float(DIR_SIGN.get(channel_dir, 0.0)),
        "raw": obj,
    }


def parse_window_response(text: str, H: int, *, min_conf: float = 0.50) -> dict[str, Any]:
    obj = _parse_json(text)
    hyp = str(obj.get("residual_hypothesis") or "").lower()
    try:
        gate = int(obj.get("gate")) if obj.get("gate") is not None else 0
    except Exception:
        gate = 0
    try:
        conf = float(obj.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    conf = float(np.clip(conf, 0.0, 1.0))
    keep = obj.get("keep_channels") or obj.get("keep") or []
    if not isinstance(keep, list):
        keep = []
    keep = [str(x) for x in keep]
    hb = obj.get("horizon_bias")
    if isinstance(hb, list):
        hb = [str(x).lower().replace(" ", "_") for x in hb]
    else:
        hb = []
    while len(hb) < H:
        hb.append("flat")
    hb = [x if x in BIAS else "flat" for x in hb[:H]]
    reason = str(obj.get("reason") or "")[:160]
    if hyp not in ("increase", "decrease") or conf < min_conf:
        gate = 0
    if gate != 1:
        gate = 0
    return {
        "hyp": hyp,
        "gate": gate,
        "confidence": conf,
        "keep_channels": keep,  # full ranked list
        "horizon_bias": hb,
        "reason": reason,
        "raw": obj,
    }


def _ramp(H: int) -> np.ndarray:
    return np.linspace(0.85, 1.15, H, dtype=np.float32)


def _match_keep(ids: Sequence[str], valid: Sequence[str]) -> list[str]:
    """Preserve rank order; fuzzy-match against valid names/ids."""
    vset = list(valid)
    out: list[str] = []
    used = set()
    for kid in ids:
        hit = None
        if kid in vset and kid not in used:
            hit = kid
        else:
            for v in vset:
                if v in used:
                    continue
                if v == kid or v.endswith(kid) or kid.endswith(v) or kid in v or v in kid:
                    hit = v
                    break
        if hit is not None and hit not in used:
            out.append(hit)
            used.add(hit)
    return out


def apply_keep_budgets(
    packs: Sequence[ChannelPack],
    channel_judgments: Sequence[Mapping[str, Any]],
    window_sel: Mapping[str, Any],
    *,
    keep_events_per_channel: int,
    keep_channels: int,
) -> tuple[list[tuple[ChannelPack, dict[str, Any]]], dict[str, Any]]:
    """Truncate ranked KEEP lists; return active (pack, judgment) pairs."""
    by_ch = {p.channel: (p, dict(j)) for p, j in zip(packs, channel_judgments)}

    # truncate event keep inside each channel judgment (audit only; dir already set)
    for ch, (p, j) in by_ch.items():
        valid_eids = [e.display_id for e in p.events]
        ranked = _match_keep(j.get("keep_events") or [], valid_eids)
        # if LLM returned empty keep but gated, fall back to all / nearest
        if not ranked and int(j.get("gate") or 0) == 1:
            ranked = valid_eids[:]
        j["keep_events"] = ranked[: max(0, int(keep_events_per_channel))]
        # require at least one kept event when gated, else drop gate
        if int(j.get("gate") or 0) == 1 and not j["keep_events"]:
            j["gate"] = 0
            j["channel_dir"] = "abstain"
            j["sign"] = 0.0
        by_ch[ch] = (p, j)

    ranked_ch = _match_keep(
        list(window_sel.get("keep_channels") or []),
        [p.channel for p in packs],
    )
    if not ranked_ch:
        # fallback: all gated channels in pack order
        ranked_ch = [
            p.channel
            for p, j in zip(packs, channel_judgments)
            if int(j.get("gate") or 0) == 1
        ]
    ranked_ch = ranked_ch[: max(0, int(keep_channels))]

    active: list[tuple[ChannelPack, dict[str, Any]]] = []
    for ch in ranked_ch:
        hit = by_ch.get(ch)
        if hit is None:
            continue
        p, j = hit
        if int(j.get("gate") or 0) == 1 and abs(float(j.get("sign") or 0.0)) > 0:
            active.append((p, j))

    meta = {
        "keep_events_per_channel": int(keep_events_per_channel),
        "keep_channels_budget": int(keep_channels),
        "ranked_channels": ranked_ch,
        "n_active_channels": len(active),
    }
    return active, meta


def delta_from_hierarchy(
    packs: Sequence[ChannelPack],
    channel_judgments: Sequence[Mapping[str, Any]],
    window_sel: Mapping[str, Any],
    H: int,
    amp: float,
    *,
    keep_events_per_channel: int = 4,
    keep_channels: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build Δ from channel→window hierarchy with KEEP budgets."""
    active, keep_meta = apply_keep_budgets(
        packs,
        channel_judgments,
        window_sel,
        keep_events_per_channel=keep_events_per_channel,
        keep_channels=keep_channels,
    )

    signs = [float(j.get("sign") or 0.0) for _, j in active]
    unified = False
    sign = 0.0
    if signs and all(s > 0 for s in signs):
        unified, sign = True, 1.0
    elif signs and all(s < 0 for s in signs):
        unified, sign = True, -1.0

    gate = int(window_sel.get("gate") or 0) == 1 and unified and len(active) > 0
    meta = {
        "n_channel_packs": len(packs),
        "n_active": len(active),
        "unified": unified,
        "gate": int(gate),
        "hyp": window_sel.get("hyp"),
        "conf": float(window_sel.get("confidence") or 0.0),
        "keep_channels": [p.channel for p, _ in active],
        "hierarchy": "channel→window",
        **keep_meta,
    }
    if not gate:
        return np.zeros(H, np.float32), meta

    hb = window_sel.get("horizon_bias") or ["flat"] * H
    vec = np.array([BIAS.get(str(x), 0.0) for x in hb[:H]], dtype=np.float64)
    if float(np.max(np.abs(vec))) < 1e-12:
        vec = sign * np.array([0.05] * H, dtype=np.float64)
    elif float(np.mean(vec)) * sign < 0:
        vec = -vec

    wsum = 0.0
    for _, j in active:
        wsum += float(j.get("confidence") or 0.5) * STRENGTH_W.get(
            str(j.get("strength") or "medium"), 1.0
        )
    w_scale = float(np.clip(wsum / max(len(active), 1), 0.4, 1.6))
    conf = float(window_sel.get("confidence") or 0.65)
    out = (vec * amp * w_scale * (0.40 + 0.60 * conf) * _ramp(H)).astype(np.float32)
    meta["amp_eff"] = float(amp * w_scale * (0.40 + 0.60 * conf))
    return out, meta


def load_kv_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key") or row.get("window_id") or row.get("channel_key")
        if key is None:
            continue
        out[str(key)] = row.get("response") or ""
    return out


def append_kv_cache(path: Path, key: str, response: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "response": response, **extra}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def channel_cache_path(cache_dir: Path, cfg: Scheme1Config) -> Path:
    return cache_dir / f"scheme1_channel_{cfg.cache_tag}.jsonl"


def window_cache_path(cache_dir: Path, cfg: Scheme1Config) -> Path:
    return cache_dir / f"scheme1_window_{cfg.cache_tag}.jsonl"


# backward-compat aliases for old event-cache callers
def event_cache_path(cache_dir: Path, cfg: Scheme1Config) -> Path:
    return channel_cache_path(cache_dir, cfg)


def _llm_fill(
    client: Any,
    system: str,
    items: list[tuple[str, str]],
    cache: dict[str, str],
    path: Path,
    *,
    wave: int,
    temperature: float,
    max_tokens: int,
    extra_by_key: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> int:
    todo = [(k, p) for k, p in items if k not in cache or not cache.get(k)]
    if not todo:
        return 0
    filled = 0
    for s in range(0, len(todo), wave):
        chunk = todo[s : s + wave]
        print(f"  wave {s}:{s + len(chunk)} / {len(todo)}", flush=True)
        try:
            raws = client.ask_many(
                [p for _, p in chunk],
                system=system,
                concurrency=len(chunk),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            try:
                raws = client.ask_many(
                    [p for _, p in chunk], system=system, concurrency=len(chunk)
                )
            except Exception as e:
                print("  fail", e, flush=True)
                raws = []
                for _, p in chunk:
                    try:
                        raws.append(client.ask(p, system=system))
                    except Exception as e2:
                        print("  sfail", e2, flush=True)
                        raws.append("")
        except Exception as e:
            print("  fail", e, flush=True)
            raws = []
            for _, p in chunk:
                try:
                    raws.append(client.ask(p, system=system))
                except Exception as e2:
                    print("  sfail", e2, flush=True)
                    raws.append("")
        for (key, _), raw in zip(chunk, raws):
            text = _as_text(raw)
            cache[key] = text
            extra = dict(extra_by_key.get(key) or {}) if extra_by_key else {}
            append_kv_cache(path, key, text, **extra)
            filled += 1
    return filled


def attach_scheme1(
    rows: Sequence[dict[str, Any]],
    by_ym: Mapping[str, Sequence[dict]],
    profile: Stage2DomainProfile,
    H: int,
    *,
    cache_dir: Path,
    client: Any = None,
    cfg: Optional[Scheme1Config] = None,
    fill_missing: bool = True,
    channels: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Fill channel→window caches; write ``mech`` / ``mech_llm`` / ``scheme1_meta``."""
    cfg = cfg or Scheme1Config()
    focus = tuple(channels) if channels else tuple(profile.focus)
    cache_dir = Path(cache_dir)
    cpath = channel_cache_path(cache_dir, cfg)
    wpath = window_cache_path(cache_dir, cfg)
    ccache = load_kv_cache(cpath)
    wcache = load_kv_cache(wpath)

    row_packs: list[list[ChannelPack]] = []
    unique_prompts: dict[str, str] = {}
    unique_extra: dict[str, dict[str, Any]] = {}
    for r in rows:
        wid = r["wid"]
        cfa = r.get("yc") if r.get("yc") is not None else r.get("y_cfa")
        packs = build_channel_packs(
            by_ym,
            r.get("months") or [],
            focus,
            wid=wid,
            max_events_per_channel=cfg.max_events_per_channel,
            max_mechs_per_event=cfg.max_mechs_per_event or cfg.max_mechs,
        )
        row_packs.append(packs)
        for p in packs:
            if p.key not in unique_prompts:
                unique_prompts[p.key] = render_channel_prompt(
                    p,
                    profile,
                    origin=str(r.get("origin") or ""),
                    cfa=cfa if cfa is not None else [0.0] * H,
                )
                unique_extra[p.key] = {
                    "channel_key": p.key,
                    "wid": wid,
                    "channel": p.channel,
                    "n_events": len(p.events),
                }

    sys_c = (
        f"You aggregate per-channel residual judgments for {profile.target_series_short}. "
        "Compact JSON only. No markdown."
    )
    sys_w = (
        f"You aggregate channel residual judgments for {profile.target_series_short}. "
        "Compact JSON only. No markdown. No absolute level forecast."
    )

    n_fill_c = n_fill_w = 0
    if fill_missing and client is not None:
        items = list(unique_prompts.items())
        print(
            f"[scheme1] channel prompts unique={len(items)} "
            f"cached={sum(1 for k in unique_prompts if ccache.get(k))}",
            flush=True,
        )
        n_fill_c = _llm_fill(
            client,
            sys_c,
            items,
            ccache,
            cpath,
            wave=cfg.wave,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_channel,
            extra_by_key=unique_extra,
        )
        w_items: list[tuple[str, str]] = []
        w_extra: dict[str, dict[str, Any]] = {}
        for r, packs in zip(rows, row_packs):
            wid = r["wid"]
            if wcache.get(wid):
                continue
            judgments = [
                parse_channel_response(
                    ccache.get(p.key, ""), min_conf=cfg.min_channel_conf
                )
                for p in packs
            ]
            cfa = r.get("yc") if r.get("yc") is not None else r.get("y_cfa")
            prompt = render_window_prompt(
                profile=profile,
                origin=str(r.get("origin") or ""),
                cfa=cfa if cfa is not None else [0.0] * H,
                packs=packs,
                judgments=judgments,
                H=H,
            )
            w_items.append((wid, prompt))
            w_extra[wid] = {"window_id": wid, "n_channels": len(packs)}
        print(
            f"[scheme1] window todo={len(w_items)} cached={len(rows) - len(w_items)}",
            flush=True,
        )
        n_fill_w = _llm_fill(
            client,
            sys_w,
            w_items,
            wcache,
            wpath,
            wave=cfg.wave,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_window,
            extra_by_key=w_extra,
        )

    n_on = 0
    n_ch_used = 0
    n_ev_used = 0
    for r, packs in zip(rows, row_packs):
        judgments = [
            parse_channel_response(ccache.get(p.key, ""), min_conf=cfg.min_channel_conf)
            for p in packs
        ]
        wsel = parse_window_response(
            wcache.get(r["wid"], ""), H, min_conf=cfg.min_window_conf
        )
        vec, meta = delta_from_hierarchy(
            packs,
            judgments,
            wsel,
            H,
            cfg.soft_amp,
            keep_events_per_channel=cfg.keep_events_per_channel,
            keep_channels=cfg.keep_channels,
        )
        r["mech_llm"] = vec.astype(np.float32)
        r["mech"] = r["mech_llm"]
        r["scheme1_meta"] = {
            **meta,
            "n_channel_gate": int(sum(1 for j in judgments if j.get("gate") == 1)),
            "cache_tag": cfg.cache_tag,
        }
        r["scheme1_channel_judgments"] = [
            {
                "channel": p.channel,
                "key": p.key,
                "n_events": len(p.events),
                "channel_dir": j.get("channel_dir"),
                "gate": j.get("gate"),
                "confidence": j.get("confidence"),
                "keep_events": j.get("keep_events"),
            }
            for p, j in zip(packs, judgments)
        ]
        # stash packs+judgments+wsel for hyperparam re-decode without LLM
        r["_scheme1_packs"] = packs
        r["_scheme1_ch_judgments"] = judgments
        r["_scheme1_window"] = wsel
        if float(np.max(np.abs(vec))) > 1e-8:
            n_on += 1
        n_ch_used += len(packs)
        n_ev_used += sum(len(p.events) for p in packs)

    return {
        "stats": {
            "n_rows": len(rows),
            "n_nonzero": n_on,
            "frac_active": n_on / max(len(rows), 1),
            "mean_channels": n_ch_used / max(len(rows), 1),
            "mean_events": n_ev_used / max(len(rows), 1),
            "filled_channels": n_fill_c,
            "filled_windows": n_fill_w,
            "filled_events": n_fill_c,  # alias for old readers
            "channel_cache": str(cpath),
            "event_cache": str(cpath),  # alias
            "window_cache": str(wpath),
            "channel_cache_n": len(ccache),
            "event_cache_n": len(ccache),
            "window_cache_n": len(wcache),
            "hierarchy": "channel→window",
            "keep_events_per_channel": cfg.keep_events_per_channel,
            "keep_channels": cfg.keep_channels,
            "focus": list(focus),
        }
    }


def redecode_row(
    row: dict[str, Any],
    H: int,
    *,
    keep_events_per_channel: int,
    keep_channels: int,
    soft_amp: float,
) -> np.ndarray:
    """Re-apply KEEP budgets on cached judgments (no LLM)."""
    packs = row.get("_scheme1_packs") or []
    judgments = row.get("_scheme1_ch_judgments") or []
    wsel = row.get("_scheme1_window") or {}
    if not packs:
        return np.zeros(H, np.float32)
    vec, _ = delta_from_hierarchy(
        packs,
        judgments,
        wsel,
        H,
        soft_amp,
        keep_events_per_channel=keep_events_per_channel,
        keep_channels=keep_channels,
    )
    return vec


def search_keep_hyperparams(
    train_rows: Sequence[dict[str, Any]],
    H: int,
    *,
    keep_events_grid: Sequence[int] = (2, 3, 4, 6),
    keep_channels_grid: Sequence[int] = (2, 3, 4, 6),
    amp_grid: Sequence[float] = (0.03, 0.05, 0.08),
) -> dict[str, Any]:
    """Grid-search KEEP budgets + amp on train (minimize mean MSE)."""

    def mse(a, b) -> float:
        a = np.asarray(a, float).reshape(-1)
        b = np.asarray(b, float).reshape(-1)
        return float(np.mean((a - b) ** 2))

    best: dict[str, Any] = {"rel": 1e9}
    b0 = float(np.mean([mse(r["yc"], r["yt"]) for r in train_rows])) if train_rows else 1.0
    results = []
    for ke in keep_events_grid:
        for kc in keep_channels_grid:
            for amp in amp_grid:
                mses = []
                n_on = 0
                for r in train_rows:
                    d = redecode_row(
                        r,
                        H,
                        keep_events_per_channel=ke,
                        keep_channels=kc,
                        soft_amp=amp,
                    )
                    if float(np.max(np.abs(d))) > 1e-8:
                        n_on += 1
                    mses.append(mse(np.asarray(r["yc"], float) + d, r["yt"]))
                m1 = float(np.mean(mses)) if mses else b0
                rel = (m1 - b0) / b0 if b0 else 0.0
                row = {
                    "keep_events_per_channel": ke,
                    "keep_channels": kc,
                    "soft_amp": amp,
                    "mse": m1,
                    "B0": b0,
                    "rel": rel,
                    "frac_active": n_on / max(len(train_rows), 1),
                }
                results.append(row)
                if rel < best.get("rel", 1e9):
                    best = row
    return {"best": best, "grid": results, "B0": b0}
