#!/usr/bin/env python3
"""Fill Explicit-branch caches for scheme1 (channel→window hierarchy).

Replaces legacy event→window / S5 dual + LAMP fills.

  python scripts/fill_explicit.py --domain traffic
  python scripts/fill_explicit.py --domain agriculture
  python scripts/fill_explicit.py --domain climate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from language.apo.baseline import UnimodalBaselinePredictor  # noqa: E402
from language.apo.cfa_scorer import CFADownstreamScorer  # noqa: E402
from language.apo.config import build_domain_args  # noqa: E402
from language.apo.eval_windows import loader_window_indices  # noqa: E402
from language.lamp_vmt.windows import _norm_date  # noqa: E402
from language.stage2.profiles import get_stage2_profile, list_stage2_domains  # noqa: E402
from language.stage2.resolve import resolve_stage1_candidate  # noqa: E402
from language.stage2.scheme1_hier import Scheme1Config, attach_scheme1  # noqa: E402
from language.vllm_client import VLLMClient, VLLMConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="traffic", choices=list_stage2_domains())
    ap.add_argument("--events", default="", help="default: profile.events_dir/events.jsonl")
    ap.add_argument("--out-dir", default="", help="default: profile.cache_dir")
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--wave",
        type=int,
        default=0,
        help="0 = llm_runtime.yaml scheme_wave (default 48).",
    )
    ap.add_argument("--max-events-per-channel", type=int, default=16)
    ap.add_argument("--max-mechs", type=int, default=3)
    ap.add_argument("--keep-events-per-channel", type=int, default=4)
    ap.add_argument("--keep-channels", type=int, default=4)
    ap.add_argument("--cache-tag", default="ch2w_v1")
    # legacy no-ops kept so old CLI flags don't break
    ap.add_argument("--max-per-month", type=int, default=4)
    ap.add_argument("--max-events-window", type=int, default=64)
    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://192.168.32.115:8000"))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen3-VL-30B-A3B-Instruct-AWQ"))
    ap.add_argument(
        "--llm-provider",
        default=os.getenv("LLM_PROVIDER", ""),
        choices=["", "vllm", "zhipu", "zai", "glm"],
    )
    args = ap.parse_args()

    from language.dataevolve_vmt.llm_runtime import get_llm_runtime

    rt = get_llm_runtime(args.domain)
    if not args.wave:
        args.wave = int(rt["scheme_wave"])
    mt = int(rt["max_tokens_scheme"])

    profile = get_stage2_profile(args.domain)
    stage1 = resolve_stage1_candidate(ROOT, profile)
    print("Stage1", stage1["dir"], flush=True)

    events_path = Path(args.events) if args.events else ROOT / profile.events_dir / "events.jsonl"
    if not events_path.exists() or events_path.stat().st_size == 0:
        print("ERROR: events missing/empty:", events_path, flush=True)
        return 1
    events = [json.loads(l) for l in events_path.open() if l.strip()]
    by_ym = defaultdict(list)
    for e in events:
        by_ym[str(e.get("date") or "")[:7]].append(e)
    print(f"events={len(events)} months_with_events={len(by_ym)}", flush=True)

    cleaned = stage1["csv"]
    bundle = stage1["bundle"]
    cfa_args = build_domain_args(
        profile.apo_domain, mode="cfa", cfa_bundle_path=str(bundle)
    )
    cfa_args.root_path = str(cleaned.parent)
    cfa_args.data_path = cleaned.name
    cfa_args.text_emb = cfa_args.pred_len
    cfa_args.text_injection_mode = "cfa"
    uni_args = build_domain_args(profile.apo_domain, mode="unimodal")
    uni_args.root_path = str(cleaned.parent)
    uni_args.data_path = cleaned.name
    uni = UnimodalBaselinePredictor(uni_args)
    scorer = CFADownstreamScorer(cfa_args, bundle_path=bundle, baseline=uni)
    H = int(cfa_args.pred_len)

    def collect(split: str, *, align_runpy: bool):
        ds = uni.get_dataset(split)
        if align_runpy:
            wis = loader_window_indices(cfa_args, split)
        else:
            wis = list(range(len(ds)))
        samples, texts, meta = [], [], []
        for wi in wis:
            sample = uni.get_sample(split, wi)
            months = [_norm_date(ds.date[wi + off])[:7] for off in range(ds.seq_len)]
            samples.append(sample)
            texts.append(sample.raw_text)
            meta.append(
                {
                    "wi": wi,
                    "months": months,
                    "origin": months[-1],
                    "history_stats": sample.history_stats,
                }
            )
        print(f"CFA {split} n={len(samples)} align_runpy={align_runpy}", flush=True)
        y_cfa = scorer.predict_fusion_batch(samples, texts, batch_size=args.batch_size)
        rows = []
        for sample, yc, m in zip(samples, y_cfa, meta):
            rows.append(
                {
                    "wid": f"{split}:{m['wi']}",
                    "split": split,
                    "months": m["months"],
                    "origin": m["origin"],
                    "history_stats": m["history_stats"],
                    "yc": np.asarray(yc, float).reshape(-1),
                    "y_cfa": np.asarray(yc, float).reshape(-1),
                    "yt": np.asarray(sample.y_true, float).reshape(-1),
                }
            )
        return rows

    train = collect("train", align_runpy=False)
    test = collect("test", align_runpy=bool(profile.align_test_runpy))
    llm_rows = list(test) + [r for i, r in enumerate(train) if i % args.train_stride == 0]
    print(f"llm_rows={len(llm_rows)} (test={len(test)} + stride train)", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / profile.cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
    else:
        cfg.base_url = args.base_url.rstrip("/")
        if args.model:
            cfg.model = args.model
    cfg.enable_thinking = False
    cfg.default_temperature = 0.0
    cfg.default_max_tokens = mt
    client = VLLMClient(cfg)
    print(
        f"LLM provider={cfg.provider} {client.base_url} model={cfg.model} "
        f"wave={args.wave} max_tokens={mt}",
        flush=True,
    )

    s1_cfg = Scheme1Config(
        max_events_per_channel=args.max_events_per_channel,
        max_mechs_per_event=args.max_mechs,
        max_mechs=args.max_mechs,
        keep_events_per_channel=args.keep_events_per_channel,
        keep_channels=args.keep_channels,
        cache_tag=args.cache_tag,
        wave=args.wave,
        max_tokens_channel=mt,
        max_tokens_window=mt,
    )
    info = attach_scheme1(
        llm_rows,
        by_ym,
        profile,
        H,
        cache_dir=out_dir,
        client=client,
        cfg=s1_cfg,
        fill_missing=True,
    )
    stats = info["stats"]
    meta = {
        "domain": profile.key,
        "stage1": str(stage1["dir"]),
        "n_events": len(events),
        "n_llm_rows": len(llm_rows),
        "scheme": "scheme1_hier",
        "cache_tag": args.cache_tag,
        **stats,
    }
    (out_dir / "explicit_fill_meta.json").write_text(json.dumps(meta, indent=2))
    print("scheme1 stats=", json.dumps(stats), flush=True)
    print("Wrote", stats["event_cache"], "n=", stats["event_cache_n"], flush=True)
    print("Wrote", stats["window_cache"], "n=", stats["window_cache_n"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
