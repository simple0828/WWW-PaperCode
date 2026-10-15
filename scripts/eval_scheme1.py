#!/usr/bin/env python3
"""Scheme 1 eval — channel→window hierarchy + KEEP hyperparam search.

  PYTHONPATH=. python scripts/eval_scheme1.py --domain traffic
  PYTHONPATH=. python scripts/eval_scheme1.py --domain traffic --replay-only

Fill LLM caches once (ranked KEEP lists); search
``keep_events_per_channel`` × ``keep_channels`` × ``soft_amp`` on train
without re-calling the LLM; report test with the best budgets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from language import VLLMClient, VLLMConfig  # noqa: E402
from language.stage2.collect import collect_rows  # noqa: E402
from language.stage2.profiles import get_stage2_profile, list_stage2_domains  # noqa: E402
from language.stage2.scheme1_hier import (  # noqa: E402
    Scheme1Config,
    attach_scheme1,
    redecode_row,
    search_keep_hyperparams,
)


def mse(a, b) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def score_rows(rows, *, ke: int, kc: int, amp: float, H: int) -> dict:
    b0s, m1s, n_on = [], [], 0
    for r in rows:
        yt = np.asarray(r["yt"], float)
        yc = np.asarray(r["yc"], float)
        d = redecode_row(
            r, H, keep_events_per_channel=ke, keep_channels=kc, soft_amp=amp
        )
        b0s.append(mse(yc, yt))
        m1s.append(mse(yc + d, yt))
        if float(np.max(np.abs(d))) > 1e-8:
            n_on += 1
    b0 = float(np.mean(b0s)) if b0s else float("nan")
    m1 = float(np.mean(m1s)) if m1s else float("nan")
    rel = (m1 - b0) / b0 if b0 and b0 == b0 else float("nan")
    return {
        "mse": m1,
        "B0": b0,
        "rel": rel,
        "rel_pct": rel * 100.0 if rel == rel else float("nan"),
        "n": len(rows),
        "n_nonzero": n_on,
        "frac_active": n_on / max(len(rows), 1),
        "keep_events_per_channel": ke,
        "keep_channels": kc,
        "soft_amp": amp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Scheme1 channel→window + KEEP search")
    ap.add_argument("--domain", default="traffic", choices=list_stage2_domains())
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument(
        "--wave",
        type=int,
        default=0,
        help="0 = llm_runtime.yaml scheme_wave (default 48).",
    )
    ap.add_argument("--cache-tag", default="ch2w_v1")
    ap.add_argument("--max-events-per-channel", type=int, default=16)
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://192.168.32.115:8000"),
    )
    ap.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", "Qwen3-VL-30B-A3B-Instruct-AWQ"),
    )
    ap.add_argument(
        "--llm-provider",
        default=os.getenv("LLM_PROVIDER", ""),
        choices=["", "vllm", "zhipu", "zai", "glm"],
    )
    ap.add_argument(
        "--out",
        default="",
        help="default: {cache_dir}/scheme1_ch2w_eval.json",
    )
    ap.add_argument(
        "--keep-events-grid",
        default="2,3,4,6",
        help="comma list: channel-level KEEP budget",
    )
    ap.add_argument(
        "--keep-channels-grid",
        default="2,3,4,6",
        help="comma list: window-level KEEP budget",
    )
    ap.add_argument("--amp-grid", default="0.03,0.05,0.08")
    args = ap.parse_args()

    ke_grid = [int(x) for x in args.keep_events_grid.split(",") if x.strip()]
    kc_grid = [int(x) for x in args.keep_channels_grid.split(",") if x.strip()]
    amp_grid = [float(x) for x in args.amp_grid.split(",") if x.strip()]

    from language.dataevolve_vmt.llm_runtime import get_llm_runtime

    rt = get_llm_runtime(args.domain)
    if not args.wave:
        args.wave = int(rt["scheme_wave"])
    mt = int(rt["max_tokens_scheme"])

    profile = get_stage2_profile(args.domain)
    print(f"domain={profile.key} focus={list(profile.focus)}", flush=True)
    train_rows, test_rows, H, meta = collect_rows(ROOT, profile)
    by_ym = meta["by_ym"]
    print(f"rows train={len(train_rows)} test={len(test_rows)} H={H}", flush=True)

    # LLM fill on test + strided train (same pattern as fill_explicit)
    llm_rows = list(test_rows) + [
        r for i, r in enumerate(train_rows) if i % args.train_stride == 0
    ]
    print(f"llm_rows={len(llm_rows)}", flush=True)

    cfg = Scheme1Config(
        max_events_per_channel=args.max_events_per_channel,
        cache_tag=args.cache_tag,
        wave=args.wave,
        max_tokens_channel=mt,
        max_tokens_window=mt,
        # provisional; overwritten after search
        keep_events_per_channel=ke_grid[0] if ke_grid else 4,
        keep_channels=kc_grid[0] if kc_grid else 4,
        soft_amp=amp_grid[0] if amp_grid else 0.05,
    )

    client = None
    if not args.replay_only:
        if args.llm_provider:
            os.environ["LLM_PROVIDER"] = args.llm_provider
        vcfg = VLLMConfig.from_env()
        if args.llm_provider in {"zhipu", "zai", "glm"}:
            vcfg.provider = "zhipu"
            vcfg.base_url = (
                args.base_url
                if "bigmodel.cn" in (args.base_url or "")
                else "https://open.bigmodel.cn/api/paas/v4"
            ).rstrip("/")
            if not args.model or "Qwen" in args.model:
                vcfg.model = os.getenv("VLLM_MODEL", "glm-5.3-flash")
            else:
                vcfg.model = args.model
            if not vcfg.api_key:
                vcfg.api_key = (
                    os.getenv("ZAI_API_KEY")
                    or os.getenv("ZHIPU_API_KEY")
                    or os.getenv("VLLM_API_KEY")
                )
        else:
            vcfg.base_url = args.base_url.rstrip("/")
            vcfg.model = args.model
        vcfg.enable_thinking = False
        vcfg.default_temperature = 0.0
        vcfg.default_max_tokens = mt
        client = VLLMClient(vcfg)
        print(
            f"LLM provider={vcfg.provider} {vcfg.base_url} model={client.resolve_model()} "
            f"wave={args.wave} max_tokens={mt}",
            flush=True,
        )

    fill_info = attach_scheme1(
        llm_rows,
        by_ym,
        profile,
        H,
        cache_dir=ROOT / profile.cache_dir,
        client=client,
        cfg=cfg,
        fill_missing=not args.replay_only,
    )
    print("fill stats", fill_info["stats"], flush=True)

    # attach on ALL train+test so search/score have _scheme1_* fields
    # (decode from cache only)
    attach_scheme1(
        train_rows + test_rows,
        by_ym,
        profile,
        H,
        cache_dir=ROOT / profile.cache_dir,
        client=None,
        cfg=cfg,
        fill_missing=False,
    )
    print("attach all rows done", flush=True)

    # search only on train windows that were LLM-filled (strided)
    llm_train_ids = {r["wid"] for r in llm_rows if str(r["wid"]).startswith("train:")}
    search_train = [r for r in train_rows if r["wid"] in llm_train_ids]
    if not search_train:
        search_train = [r for r in train_rows if r.get("_scheme1_packs")]
    print(f"search on train n={len(search_train)} (strided fill)", flush=True)

    search = search_keep_hyperparams(
        search_train,
        H,
        keep_events_grid=ke_grid,
        keep_channels_grid=kc_grid,
        amp_grid=amp_grid,
    )
    best = search["best"]
    print(
        f"best train: ke={best.get('keep_events_per_channel')} "
        f"kc={best.get('keep_channels')} amp={best.get('soft_amp')} "
        f"rel={best.get('rel', 0):+.3%}",
        flush=True,
    )

    ke = int(best["keep_events_per_channel"])
    kc = int(best["keep_channels"])
    amp = float(best["soft_amp"])

    # score all train / all test with best
    train_score = score_rows(train_rows, ke=ke, kc=kc, amp=amp, H=H)
    test_score = score_rows(test_rows, ke=ke, kc=kc, amp=amp, H=H)
    print(
        f"scheme1 channel→window test rel={test_score['rel_pct']:+.3f}% "
        f"mse={test_score['mse']:.6f} B0={test_score['B0']:.6f} "
        f"active={test_score['frac_active']:.2f} "
        f"({test_score['n_nonzero']}/{test_score['n']})",
        flush=True,
    )

    out = {
        "domain": profile.key,
        "scheme": "scheme1_channel_window",
        "hierarchy": "channel→window",
        "stage1": meta.get("stage1"),
        "cache_tag": args.cache_tag,
        "focus": list(profile.focus),
        "fill": fill_info["stats"],
        "search": {
            "best": best,
            "n_train_search": len(search_train),
            "keep_events_grid": ke_grid,
            "keep_channels_grid": kc_grid,
            "amp_grid": amp_grid,
            "top5": sorted(search["grid"], key=lambda x: x["rel"])[:5],
        },
        "train": train_score,
        "test": test_score,
    }
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / profile.cache_dir / "scheme1_ch2w_eval.json"
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
