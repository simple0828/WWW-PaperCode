"""LLM stages: per-month event extraction with expert multi-hypothesis mechanisms."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from language import VLLMClient
from language.lamp_vmt.json_utils import as_list, extract_json
from language.lamp_vmt.prompts import EXTRACT_SYSTEM, build_extract_month_prompt
from language.lamp_vmt.schema import (
    VLLM_CONCURRENCY,
    normalize_channel,
    normalize_mechanisms,
    primary_vmt_effect,
)


def _safe_parse(raw: str) -> Any:
    try:
        return extract_json(raw)
    except Exception:
        return None


def normalize_event(ev: dict[str, Any], *, date: str = "") -> dict[str, Any]:
    """Grounded fact + expert mechanism hypotheses (possibly competing)."""
    mechanisms = normalize_mechanisms(ev.get("mechanisms"))
    if not mechanisms:
        mechanisms = normalize_mechanisms(
            [
                {
                    "chain": ev.get("mechanism_chain"),
                    "vmt_effect": ev.get("vmt_effect"),
                    "plausibility": "medium",
                    "rationale": "",
                    "caveat": "",
                }
            ]
        )
    effect = primary_vmt_effect(mechanisms)
    if ev.get("vmt_effect") and effect == "unclear":
        from language.lamp_vmt.schema import normalize_vmt_effect

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


def extract_events_for_month(
    client: VLLMClient,
    *,
    date: str,
    text: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[list[dict[str, Any]], str]:
    """Structure ALL events in one month (no hard count cap)."""
    if not (text or "").strip():
        return [], ""
    prompt = build_extract_month_prompt(date=date, text=text)
    raw = client.ask(
        prompt, system=EXTRACT_SYSTEM, temperature=temperature, max_tokens=max_tokens
    )
    payload = _safe_parse(raw)
    return _events_from_payload(payload, date=str(date)[:7]), raw or ""


def extract_events_by_month(
    client: VLLMClient,
    month_blocks: Sequence[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    concurrency: int = VLLM_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Extract events for each month independently; concatenate in date order."""
    jobs: list[tuple[int, str, str]] = []
    for i, b in enumerate(month_blocks):
        date = str(b.get("date") or "")[:10]
        text = str(b.get("text") or "").strip()
        if text:
            jobs.append((i, date, text))

    if not jobs:
        return []

    prompts = [build_extract_month_prompt(date=d, text=t) for _, d, t in jobs]
    raws = client.ask_many(
        prompts,
        system=EXTRACT_SYSTEM,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    by_index: dict[int, list[dict[str, Any]]] = {}
    for (i, date, _), raw in zip(jobs, raws):
        payload = _safe_parse(raw)
        by_index[i] = _events_from_payload(payload, date=date[:7])

    out: list[dict[str, Any]] = []
    for i in range(len(month_blocks)):
        out.extend(by_index.get(i, []))
    return out


def extract_events_corpus(
    client: VLLMClient,
    rows: Sequence[dict[str, Any]],
    *,
    text_key: str = "refined_text",
    date_key: str = "date",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    concurrency: int = VLLM_CONCURRENCY,
    batch_size: Optional[int] = None,
    on_batch: Optional[Callable[[int, int, list[dict[str, Any]]], None]] = None,
    on_wave: Optional[Callable[[list[dict[str, Any]], int, int], None]] = None,
    skip_row_ids: Optional[set[Any]] = None,
) -> list[dict[str, Any]]:
    """Run per-row extraction in waves; optional incremental callback for resume.

    Accepts either Stage1 cleaned jsonl rows ({date, refined_text/text, row_id})
    or Stage2 month dicts ({date, text, source_row_id}) from ``extract_events.py``.

    ``batch_size`` defaults to ``concurrency`` so each wave finishes and can flush
    before the next wave starts (avoids 531-way all-or-nothing hangs).
    ``on_wave(events, start, n_jobs)`` matches lamp_timemmd; ``on_batch`` kept for
    older callers.
    """
    skip = skip_row_ids or set()
    jobs: list[tuple[int, str, str, Any]] = []
    for i, row in enumerate(rows):
        date = str(row.get(date_key) or "")[:10]
        text = str(
            row.get(text_key) or row.get("text") or row.get("refined_text") or ""
        ).strip()
        if not text:
            continue
        row_id = row.get("source_row_id", row.get("row_id", i))
        if row_id in skip:
            continue
        jobs.append((i, date, text, row_id))

    if not jobs:
        return []

    wave = batch_size if batch_size and batch_size > 0 else concurrency
    out: list[dict[str, Any]] = []
    n_jobs = len(jobs)
    t0 = time.time()
    for start in range(0, n_jobs, wave):
        chunk = jobs[start : start + wave]
        prompts = [build_extract_month_prompt(date=d, text=t) for _, d, t, _ in chunk]
        raws = client.ask_many(
            prompts,
            system=EXTRACT_SYSTEM,
            concurrency=min(concurrency, len(prompts)),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        batch_events: list[dict[str, Any]] = []
        for (_i, date, _t, row_id), raw in zip(chunk, raws):
            payload = _safe_parse(raw)
            for ev in _events_from_payload(payload, date=date[:7]):
                ev["source_row_id"] = row_id
                batch_events.append(ev)
        out.extend(batch_events)
        done = min(start + wave, n_jobs)
        elapsed = time.time() - t0
        print(
            f"[extract] months {done}/{n_jobs} "
            f"(+{len(batch_events)} events this wave, "
            f"total_events={len(out)}, elapsed={elapsed:.1f}s)",
            flush=True,
        )
        if on_wave is not None:
            on_wave(batch_events, start, n_jobs)
        if on_batch is not None:
            on_batch(done, n_jobs, batch_events)
    return out


def dump_events_jsonl(path: str, events: Sequence[dict[str, Any]], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def load_done_row_ids(path: str | Path) -> set[Any]:
    p = Path(path)
    if not p.is_file():
        return set()
    done: set[Any] = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "source_row_id" in row:
                done.add(row["source_row_id"])
    return done
