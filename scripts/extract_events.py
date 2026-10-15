#!/usr/bin/env python3
"""Unified per-month event extraction for any Stage2 domain.

  python scripts/extract_events.py --domain traffic
  python scripts/extract_events.py --domain agriculture --text-key auto
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from language import VLLMClient, VLLMConfig  # noqa: E402
from language.stage2.lamp_api import load_lamp_sub  # noqa: E402
from language.stage2.profiles import get_stage2_profile, list_stage2_domains  # noqa: E402
from language.stage2.resolve import resolve_stage1_candidate  # noqa: E402


def _pick_month_text(row: dict, text_key: str) -> tuple[str, str]:
    rt = str(row.get("refined_text") or "").strip()
    tx = str(row.get("text") or "").strip()
    if text_key == "refined_text":
        return "refined_text", rt or tx
    if text_key == "text":
        return "text", tx or rt
    # auto: skip empty Stage1 shells
    emptyish = (
        (not rt)
        or rt.startswith("Empty output")
        or (len(rt) < 120 and "## Regime" in rt)
    )
    if emptyish:
        return "text", tx
    if len(tx) > max(len(rt) * 1.5, 200) and len(rt) < 400:
        return "text", tx
    return "refined_text", rt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="traffic", choices=list_stage2_domains())
    p.add_argument("--cleaned-jsonl", default="", help="default: Stage1 best cleaned_texts.jsonl")
    p.add_argument("--output-dir", default="", help="default: profile.events_dir")
    p.add_argument(
        "--text-key",
        default="",
        help="refined_text|text|auto (default: profile.text_key)",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="0 = llm_runtime.yaml extract_concurrency (default 48).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 = llm_runtime.yaml max_tokens_extract (default 4096).",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://192.168.32.115:8000"),
    )
    p.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", "Qwen3-VL-30B-A3B-Instruct-AWQ"),
    )
    p.add_argument(
        "--llm-provider",
        default=os.getenv("LLM_PROVIDER", ""),
        choices=["", "vllm", "zhipu", "zai", "glm"],
        help="vllm | zhipu (glm-5.3-flash). Empty = auto from base-url / env.",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_stage2_profile(args.domain)
    schema = load_lamp_sub(profile, "schema")
    stages = load_lamp_sub(profile, "llm_stages")

    if args.cleaned_jsonl:
        cleaned_jsonl = Path(args.cleaned_jsonl)
    else:
        stage1 = resolve_stage1_candidate(ROOT, profile)
        cleaned_jsonl = stage1["cleaned_texts"]
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / profile.events_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    text_key = args.text_key or profile.text_key
    from language.dataevolve_vmt.llm_runtime import get_llm_runtime

    rt = get_llm_runtime(profile.key)
    concurrency = int(args.concurrency) or int(rt["extract_concurrency"])
    max_tokens = int(args.max_tokens) or int(rt["max_tokens_extract"])
    args.max_tokens = max_tokens

    rows = []
    with open(cleaned_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    done = stages.load_done_row_ids(events_path) if args.resume else set()
    months = []
    src_counts: dict[str, int] = {}
    for r in rows:
        rid = int(r.get("row_id", len(months)))
        if rid in done:
            continue
        src, text = _pick_month_text(r, text_key)
        src_counts[src] = src_counts.get(src, 0) + 1
        if not text.strip():
            continue
        months.append(
            {
                "date": str(r.get("date") or "")[:10],
                "text": text,
                "source_row_id": rid,
                "text_source": src,
            }
        )
    print(
        f"domain={profile.key} months todo={len(months)} skipped={len(done)} "
        f"total_rows={len(rows)} text_key={text_key} text_src={src_counts}",
        flush=True,
    )

    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    cfg = VLLMConfig.from_env()
    if args.llm_provider in {"zhipu", "zai", "glm"}:
        cfg.provider = "zhipu"
        cfg.base_url = (
            args.base_url
            if "bigmodel.cn" in (args.base_url or "")
            else "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/")
        if not args.model or "Qwen" in args.model:
            cfg.model = os.getenv("VLLM_MODEL", "glm-5.3-flash")
        else:
            cfg.model = args.model
        if not cfg.api_key:
            cfg.api_key = (
                os.getenv("ZAI_API_KEY")
                or os.getenv("ZHIPU_API_KEY")
                or os.getenv("VLLM_API_KEY")
            )
        zcap = int(rt["zhipu_concurrency_cap"])
        if concurrency > zcap:
            concurrency = zcap
        # Event JSON can be long; free pack still needs room.
        cfg.default_max_tokens = max(int(max_tokens or 0), 2048)
    else:
        cfg.base_url = args.base_url.rstrip("/")
        if args.model:
            cfg.model = args.model
        vcap = int(rt["vllm_concurrency_cap"])
        if concurrency > vcap:
            print(
                f"vllm: lowering extract concurrency {concurrency} → {vcap}",
                flush=True,
            )
            concurrency = vcap
    cfg.enable_thinking = False
    if cfg.max_model_len is None:
        cfg.max_model_len = int(rt["max_model_len"])
    client = VLLMClient(cfg)
    print(
        f"LLM provider={cfg.provider} url={cfg.base_url} model={cfg.model} "
        f"concurrency={concurrency} max_tokens={max_tokens}",
        flush=True,
    )

    def on_wave(wave_events, start, n_jobs):
        stages.dump_events_jsonl(events_path, wave_events, append=True)

    stages.extract_events_corpus(
        client,
        months,
        temperature=args.temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        on_wave=on_wave,
    )

    events = [json.loads(l) for l in events_path.open() if l.strip()]
    by_ch = Counter(e.get("channel") for e in events)
    summary = {
        "domain": profile.key,
        "n_events": len(events),
        "n_months": len({e.get("date") for e in events}),
        "channel_counts": dict(by_ch.most_common()),
        "cleaned_jsonl": str(cleaned_jsonl),
        "text_key": text_key,
        "events_path": str(events_path),
        "concurrency": concurrency,
        "max_tokens": args.max_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", events_path, "n_events=", len(events), flush=True)
    print("channels", summary["channel_counts"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
