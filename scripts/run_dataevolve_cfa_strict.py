#!/usr/bin/env python3
"""Self-Evo DataEvolve aligned with original structure + emergent cleaning rules.

DataEvolve core:
  Observer ONCE on raw → each round Designer → Cleaner → Quality Judge
  Judge: original vs cleaned, Coverage + Executability, write residual experiences

Extensions:
  APO beam diversity; CFA continue-train + MSE for beam/global selection

Designer no longer hard-locks FACT_FORMAT_CONSTRAINTS each round — rules should
emerge from experience / Coverage / Executability gaps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from language import VLLMClient, VLLMConfig  # noqa: E402
from language.dataevolve_vmt.strict_loop import StrictLoopConfig, run_strict_loop  # noqa: E402


def parse_args() -> argparse.Namespace:
    from language.dataevolve_vmt.domain_profiles import (
        TRAFFIC_CLI_INIT_DEFAULT,
        TRAFFIC_CLI_OUTPUT_DEFAULT,
        list_domains,
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--domain",
        default="traffic",
        choices=list_domains(),
        help="Stage1 domain profile (YAML: language/dataevolve_vmt/stage1_seed_prompts.yaml).",
    )
    p.add_argument("--raw-csv", default="", help="Override domain raw CSV path.")
    p.add_argument(
        "--cleaned-csv-name",
        default="",
        help="Filename written under each candidate dir (domain default if empty).",
    )
    p.add_argument("--init-bundle", default=TRAFFIC_CLI_INIT_DEFAULT)
    p.add_argument("--output-dir", default=TRAFFIC_CLI_OUTPUT_DEFAULT)
    p.add_argument("--n-outer-rounds", type=int, default=5)
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--mutations-per-parent", type=int, default=2)
    p.add_argument(
        "--candidates-per-round",
        type=int,
        default=None,
        help="Legacy: maps to beam_size=1 × N mutations.",
    )
    p.add_argument(
        "--stagnation-patience",
        type=int,
        default=0,
        help="0 disables early-stop (run all outer rounds).",
    )
    p.add_argument(
        "--select-split",
        choices=("test", "val", "val_test"),
        default="test",
        help=(
            "CFA MSE for beam/global selection. "
            "val_test = mean(val, test) when both finite (security default)."
        ),
    )
    p.add_argument("--cfa-epochs", type=int, default=30)
    p.add_argument("--cfa-patience", type=int, default=10)
    p.add_argument(
        "--cfa-batch-size",
        type=int,
        default=48,
        help="CFA DataLoader batch_size for continue-train/eval (default 48).",
    )
    p.add_argument(
        "--clean-concurrency",
        type=int,
        default=None,
        help="Override llm_runtime.yaml clean_concurrency (default 48).",
    )
    p.add_argument(
        "--parallel-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--temperature-clean", type=float, default=0.0)
    p.add_argument("--temperature-design", type=float, default=0.55)
    p.add_argument("--temperature-design-explore", type=float, default=0.9)
    p.add_argument("--temperature-judge", type=float, default=0.2)
    p.add_argument(
        "--judge-batches",
        type=int,
        default=4,
        help="Quality Judge iterations per candidate (disjoint pair subsets).",
    )
    p.add_argument(
        "--judge-pairs",
        type=int,
        default=3,
        help="Pairs per Quality Judge batch (keep small; use more batches).",
    )
    p.add_argument(
        "--max-tokens-clean",
        type=int,
        default=None,
        help="Cleaner/Designer max_tokens; default from llm_runtime.yaml (4096).",
    )
    p.add_argument(
        "--max-tokens-judge",
        type=int,
        default=None,
        help="Quality Judge max_tokens; default from llm_runtime.yaml (4096).",
    )
    p.add_argument(
        "--experience-visible-k",
        type=int,
        default=20,
        help="Recent experience issues shown in Judge/Designer prompts.",
    )
    p.add_argument(
        "--near-dup-jaccard",
        type=float,
        default=0.65,
        help="Token-Jaccard threshold for full-DB near-duplicate rejection/compact.",
    )
    p.add_argument("--diversity-jaccard", type=float, default=0.78)
    p.add_argument("--paradigm-shift-ratio", type=float, default=0.25)
    p.add_argument("--reuse-raw-control-bundle", default="")
    p.add_argument(
        "--resume-from-round",
        type=int,
        default=1,
        help="1-based outer round to continue from (loads beam from round_{n-1}).",
    )
    p.add_argument(
        "--llm-provider",
        choices=("vllm", "zhipu"),
        default=os.getenv("LLM_PROVIDER", "vllm"),
        help="vllm = local OpenAI-compat server; zhipu = 智谱 glm-*-flash API",
    )
    p.add_argument(
        "--base-url",
        default="",
        help="Override LLM base URL (default: vLLM host or Zhipu paas/v4).",
    )
    p.add_argument(
        "--model",
        default="",
        help="Override model id (default: local Qwen or glm-5.3-flash for zhipu).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    beam_size = args.beam_size
    mut = args.mutations_per_parent
    if args.candidates_per_round is not None and args.candidates_per_round > 0:
        print(
            f"note: --candidates-per-round={args.candidates_per_round} "
            f"(legacy) → beam_size=1, mutations_per_parent={args.candidates_per_round}",
            flush=True,
        )
        beam_size = 1
        mut = int(args.candidates_per_round)

    from language.dataevolve_vmt.domain_profiles import get_profile, resolve_stage1_paths
    from language.dataevolve_vmt.llm_runtime import get_llm_runtime

    rt = get_llm_runtime(args.domain)
    if args.clean_concurrency is None:
        args.clean_concurrency = int(rt["clean_concurrency"])
    if args.max_tokens_clean is None:
        args.max_tokens_clean = int(rt["max_tokens_clean"])
    if args.max_tokens_judge is None:
        args.max_tokens_judge = int(rt["max_tokens_judge"])

    os.environ["LLM_PROVIDER"] = args.llm_provider
    vcfg = VLLMConfig.from_env()
    vcfg.provider = args.llm_provider
    if args.base_url:
        vcfg.base_url = args.base_url.rstrip("/")
    elif args.llm_provider == "zhipu":
        vcfg.base_url = "https://open.bigmodel.cn/api/paas/v4"
    else:
        vcfg.base_url = (
            os.getenv("VLLM_BASE_URL", "http://10.30.1.2:8001").rstrip("/")
        )
    if args.model:
        vcfg.model = args.model
    elif args.llm_provider == "zhipu":
        vcfg.model = os.getenv("VLLM_MODEL", "glm-5.3-flash")
    else:
        # Served model id on the lab vLLM (Qwen3.5-35B-A3B-FP8).
        vcfg.model = os.getenv("VLLM_MODEL", "/model")
    vcfg.enable_thinking = False
    if vcfg.max_model_len is None:
        vcfg.max_model_len = (
            128000 if args.llm_provider == "zhipu" else int(rt["max_model_len"])
        )
    if args.llm_provider == "zhipu":
        zcap = int(rt["zhipu_concurrency_cap"])
        if args.clean_concurrency > zcap:
            print(
                f"zhipu: lowering --clean-concurrency {args.clean_concurrency} → {zcap}",
                flush=True,
            )
            args.clean_concurrency = zcap
        if not (vcfg.api_key or "").strip():
            raise SystemExit(
                "zhipu provider needs API key via env ZAI_API_KEY "
                "(or ZHIPU_API_KEY / VLLM_API_KEY). Do not put the key in the command line."
            )
        if vcfg.default_max_tokens is None:
            vcfg.default_max_tokens = 1024
    else:
        vcap = int(rt["vllm_concurrency_cap"])
        if args.clean_concurrency > vcap:
            print(
                f"vllm: lowering --clean-concurrency {args.clean_concurrency} → {vcap} "
                f"(from llm_runtime.yaml vllm_concurrency_cap)",
                flush=True,
            )
            args.clean_concurrency = vcap
    client = VLLMClient(vcfg)
    n_cand = beam_size * mut
    print(
        f"LLM provider={vcfg.provider} url={vcfg.base_url} model={vcfg.model}",
        flush=True,
    )
    print(
        f"llm_runtime domain={args.domain} max_tokens_clean={args.max_tokens_clean} "
        f"max_tokens_judge={args.max_tokens_judge} clean_concurrency={args.clean_concurrency} "
        f"max_model_len={vcfg.max_model_len}",
        flush=True,
    )
    print(
        f"Emergent DataEvolve: Observer(ONCE)→"
        f"[Designer→Cleaner→QualityJudge→CFA]×{args.n_outer_rounds}; "
        f"beam={beam_size}×mut={mut}={n_cand}/round; "
        f"select_CFA={args.select_split}; judge={args.judge_batches}×{args.judge_pairs}; "
        f"max_tokens_judge={args.max_tokens_judge}; "
        f"exp_visible={args.experience_visible_k}; near_dup={args.near_dup_jaccard}; "
        f"paradigm_shift_ratio={args.paradigm_shift_ratio}; "
        f"no hard FACT_FORMAT lock in Designer",
        flush=True,
    )

    paths = resolve_stage1_paths(
        args.domain,
        init_bundle=args.init_bundle,
        output_dir=args.output_dir,
        raw_csv=args.raw_csv,
        cleaned_csv_name=args.cleaned_csv_name,
    )
    profile = get_profile(args.domain)
    init_bundle = paths["init_bundle"]
    output_dir = paths["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Security-only: empty val loader under drop_last → select on mean(val, test).
    select_split = args.select_split
    if profile.key == "security" and select_split == "test":
        select_split = "val_test"
        print(
            "security: overriding --select-split test → val_test "
            "(mean of finite val+test CFA MSE)",
            flush=True,
        )
    print(
        f"Stage1 paths domain={profile.key} init={init_bundle} out={output_dir}",
        flush=True,
    )

    cfg = StrictLoopConfig(
        output_dir=output_dir,
        init_bundle=init_bundle,
        domain=args.domain,
        raw_csv=paths["raw_csv"],
        cleaned_csv_name=paths["cleaned_csv_name"],
        text_col=profile.text_col,
        n_outer_rounds=args.n_outer_rounds,
        beam_size=beam_size,
        mutations_per_parent=mut,
        stagnation_patience=args.stagnation_patience,
        select_split=select_split,
        cfa_epochs=args.cfa_epochs,
        cfa_patience=args.cfa_patience,
        cfa_batch_size=int(args.cfa_batch_size),
        clean_concurrency=args.clean_concurrency,
        parallel_clean=bool(args.parallel_clean),
        reuse_raw_control_bundle=args.reuse_raw_control_bundle,
        temperature_clean=args.temperature_clean,
        temperature_design=args.temperature_design,
        temperature_design_explore=args.temperature_design_explore,
        temperature_judge=args.temperature_judge,
        judge_batches=args.judge_batches,
        judge_pairs=args.judge_pairs,
        max_tokens=int(args.max_tokens_clean),
        max_tokens_judge=int(args.max_tokens_judge),
        experience_visible_k=args.experience_visible_k,
        near_dup_jaccard=args.near_dup_jaccard,
        diversity_jaccard=args.diversity_jaccard,
        paradigm_shift_ratio=args.paradigm_shift_ratio,
        resume_from_round=args.resume_from_round,
    )
    run_strict_loop(client, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
