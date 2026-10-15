"""LLM stages for agri event extraction (mirrors lamp_vmt.llm_stages)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from language import VLLMClient
from language.lamp_vmt.json_utils import as_list, extract_json
from language.lamp_agri.prompts import EXTRACT_SYSTEM, build_extract_month_prompt
from language.lamp_agri.schema import (
    VLLM_CONCURRENCY,
    normalize_channel,
    normalize_mechanisms,
    normalize_vmt_effect,
    primary_vmt_effect,
)


def _safe_parse(raw: str) -> Any:
    try:
        return extract_json(raw)
    except Exception:
        return None


def normalize_event(ev: dict[str, Any], *, date: str = "") -> dict[str, Any]:
    mechanisms = normalize_mechanisms(ev.get("mechanisms"))
    if not mechanisms:
        mechanisms = normalize_mechanisms(
            [
                {
                    "chain": ev.get("mechanism_chain"),
                    "vmt_effect": ev.get("vmt_effect") or ev.get("price_effect"),
                    "plausibility": "medium",
                }
            ]
        )
    effect = primary_vmt_effect(mechanisms)
    if ev.get("vmt_effect") and effect == "unclear":
        effect = normalize_vmt_effect(ev.get("vmt_effect"))
    primary_chain = mechanisms[0]["chain"] if mechanisms else ""
    return {
        "date": str(date or ev.get("date") or "")[:7],
        "channel": normalize_channel(ev.get("channel")),
        "fact_text": str(ev.get("fact_text") or "")[:400],
        "vmt_effect": effect,
        "mechanism_chain": primary_chain,
        "mechanisms": mechanisms,
    }


def _events_from_payload(payload: Any, *, date: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for ev in as_list(payload, "events" if isinstance(payload, dict) else None):
        if not isinstance(ev, dict):
            continue
        fact = str(ev.get("fact_text") or "").strip()
        if not fact:
            continue
        events.append(normalize_event(ev, date=date))
    return events


def load_done_row_ids(events_path: Path) -> set[int]:
    done: set[int] = set()
    if not events_path.is_file():
        return done
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "source_row_id" in row:
                done.add(int(row["source_row_id"]))
    return done


def dump_events_jsonl(path: Path, events: list[dict[str, Any]], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def extract_events_corpus(
    client: VLLMClient,
    months: Sequence[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    concurrency: int = VLLM_CONCURRENCY,
    on_wave: Optional[Callable[[list[dict[str, Any]], int, int], None]] = None,
) -> list[dict[str, Any]]:
    """months: [{date, text, source_row_id}, ...]"""
    jobs = []
    for m in months:
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        jobs.append(m)

    all_events: list[dict[str, Any]] = []
    wave = max(1, int(concurrency))
    for s in range(0, len(jobs), wave):
        chunk = jobs[s : s + wave]
        prompts = [
            build_extract_month_prompt(date=str(m.get("date") or "")[:10], text=str(m.get("text") or ""))
            for m in chunk
        ]
        t0 = time.time()
        raws = client.ask_many(
            prompts,
            system=EXTRACT_SYSTEM,
            concurrency=len(chunk),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        wave_events: list[dict[str, Any]] = []
        for m, raw in zip(chunk, raws):
            date = str(m.get("date") or "")[:7]
            payload = _safe_parse(raw if isinstance(raw, str) else str(raw))
            evs = _events_from_payload(payload, date=date)
            rid = m.get("source_row_id")
            for ev in evs:
                if rid is not None:
                    ev["source_row_id"] = int(rid)
                wave_events.append(ev)
        all_events.extend(wave_events)
        if on_wave:
            on_wave(wave_events, s, len(jobs))
        print(
            f"  wave {s}:{s+len(chunk)} events+={len(wave_events)} "
            f"elapsed={time.time()-t0:.1f}s total_events={len(all_events)}",
            flush=True,
        )
    return all_events
