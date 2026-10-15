"""Shared LLM extract stages; bind prompts/schema modules per domain package."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Sequence

from language import VLLMClient
from language.lamp_vmt.json_utils import as_list, extract_json


def _safe_parse(raw: str) -> Any:
    try:
        return extract_json(raw)
    except Exception:
        return None


def make_normalize_event(schema: ModuleType):
    def normalize_event(ev: dict[str, Any], *, date: str = "") -> dict[str, Any]:
        mechanisms = schema.normalize_mechanisms(ev.get("mechanisms"))
        if not mechanisms:
            mechanisms = schema.normalize_mechanisms(
                [
                    {
                        "chain": ev.get("mechanism_chain"),
                        "vmt_effect": ev.get("vmt_effect") or ev.get("price_effect"),
                        "plausibility": "medium",
                    }
                ]
            )
        effect = schema.primary_vmt_effect(mechanisms)
        if ev.get("vmt_effect") and effect == "unclear":
            effect = schema.normalize_vmt_effect(ev.get("vmt_effect"))
        primary_chain = mechanisms[0]["chain"] if mechanisms else ""
        return {
            "date": str(date or ev.get("date") or "")[:7],
            "channel": schema.normalize_channel(ev.get("channel")),
            "fact_text": str(ev.get("fact_text") or "")[:400],
            "vmt_effect": effect,
            "mechanism_chain": primary_chain,
            "mechanisms": mechanisms,
        }

    return normalize_event


def make_events_from_payload(normalize_event_fn):
    def _events_from_payload(payload: Any, *, date: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for ev in as_list(payload, "events" if isinstance(payload, dict) else None):
            if not isinstance(ev, dict):
                continue
            fact = str(ev.get("fact_text") or "").strip()
            if not fact:
                continue
            events.append(normalize_event_fn(ev, date=date))
        return events

    return _events_from_payload


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


def dump_events_jsonl(
    path: Path, events: list[dict[str, Any]], *, append: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def make_extract_events_corpus(prompts: ModuleType, schema: ModuleType):
    normalize_event = make_normalize_event(schema)
    events_from_payload = make_events_from_payload(normalize_event)
    concurrency_default = int(getattr(schema, "VLLM_CONCURRENCY", 48))

    def extract_events_corpus(
        client: VLLMClient,
        months: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        concurrency: int = 0,
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
        wave = max(1, int(concurrency or concurrency_default))
        for s in range(0, len(jobs), wave):
            chunk = jobs[s : s + wave]
            prompt_list = [
                prompts.build_extract_month_prompt(
                    date=str(m.get("date") or "")[:10],
                    text=str(m.get("text") or ""),
                )
                for m in chunk
            ]
            t0 = time.time()
            raws = client.ask_many(
                prompt_list,
                system=prompts.EXTRACT_SYSTEM,
                concurrency=len(chunk),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            wave_events: list[dict[str, Any]] = []
            for m, raw in zip(chunk, raws):
                date = str(m.get("date") or "")[:7]
                payload = _safe_parse(raw if isinstance(raw, str) else str(raw))
                evs = events_from_payload(payload, date=date)
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

    return extract_events_corpus


def install_llm_stages(
    namespace: dict[str, Any],
    *,
    prompts: ModuleType,
    schema: ModuleType,
) -> None:
    normalize_event = make_normalize_event(schema)
    namespace.update(
        {
            "normalize_event": normalize_event,
            "_events_from_payload": make_events_from_payload(normalize_event),
            "load_done_row_ids": load_done_row_ids,
            "dump_events_jsonl": dump_events_jsonl,
            "extract_events_corpus": make_extract_events_corpus(prompts, schema),
        }
    )
