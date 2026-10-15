#!/usr/bin/env python3
"""Train unimodal+CFA (run.py) and export init bundle for a Time-MMD domain."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("DELETE_CHECKPOINTS", "0")

from language.apo.config import build_domain_args, build_setting_name  # noqa: E402
from language.apo.runpy_cfa import export_runpy_cfa_bundle  # noqa: E402
from language.apo.timemmd_domains import (  # noqa: E402
    TIMEMMD_EXTRA,
    default_bundle_path,
    get_timemmd_spec,
    normalize_domain,
)


def _run_repro(domain: str, mode: str, lr: float) -> None:
    spec = get_timemmd_spec(domain)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(ROOT / "data" / spec.data_subdir),
        "--data_path",
        spec.data_path,
        "--model_id",
        f"{spec.model_id_prefix}_2021_8_6_PatchTST_BERT_{mode}_lr{lr}",
        "--model",
        "PatchTST",
        "--data",
        "custom",
        "--features",
        "S",
        "--seq_len",
        "8",
        "--label_len",
        "4",
        "--pred_len",
        "6",
        "--des",
        "Exp",
        "--seed",
        "2021",
        "--type_tag",
        "#F#",
        "--text_len",
        "4",
        "--prompt_weight",
        "0.1",
        "--pool_type",
        "avg",
        "--save_name",
        f"repro_PatchTST_{spec.model_id_prefix}_{mode}_H6_lr{lr}",
        "--llm_model",
        "BERT",
        "--huggingface_token",
        "NA",
        "--use_text_integrated",
        "1",
        "--text_injection_mode",
        mode,
        "--lr_multiplier",
        str(lr),
        "--train_epochs",
        "10",
        "--batch_size",
        "32",
        "--num_workers",
        "0",
        "--patience",
        "5",
        "--learning_rate",
        "0.0001",
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})


def ensure_domain_ready(domain: str, *, force: bool = False) -> Path:
    domain = normalize_domain(domain)
    if domain in ("traffic", "agriculture"):
        raise SystemExit("traffic/agriculture already have Stage1; use their existing bundles")
    spec = get_timemmd_spec(domain)
    out_bundle = default_bundle_path(spec)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)

    uni = build_domain_args(domain, mode="unimodal")
    uni_setting = build_setting_name(uni, 0)
    uni_ckpt = ROOT / "checkpoints" / uni_setting / "checkpoint.pth"
    if force or not uni_ckpt.exists():
        print(f"=== train unimodal {domain} ===", flush=True)
        _run_repro(domain, "unimodal", spec.lr_multiplier)
    else:
        print(f"unimodal ok: {uni_ckpt}", flush=True)

    cfa = build_domain_args(domain, mode="cfa")
    cfa_setting = build_setting_name(cfa, 0)
    cfa_ckpt = ROOT / "checkpoints" / cfa_setting / "checkpoint.pth"
    cfa_bundle_in_ckpt = ROOT / "checkpoints" / cfa_setting / "cfa_bundle.pt"
    if force or not cfa_ckpt.exists() or not cfa_bundle_in_ckpt.exists():
        print(f"=== train cfa {domain} ===", flush=True)
        _run_repro(domain, "cfa", spec.lr_multiplier)
    else:
        print(f"cfa ckpt ok: {cfa_ckpt}", flush=True)

    print(f"=== export bundle → {out_bundle} ===", flush=True)
    export_runpy_cfa_bundle(cfa, out_bundle, force_retrain=False)
    if not out_bundle.exists():
        # fallback copy from checkpoint dir
        import shutil

        src = cfa_bundle_in_ckpt
        if src.exists():
            shutil.copy2(src, out_bundle)
    assert out_bundle.exists(), f"missing {out_bundle}"
    print("READY", out_bundle, flush=True)
    return out_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(TIMEMMD_EXTRA))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_domain_ready(args.domain, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
