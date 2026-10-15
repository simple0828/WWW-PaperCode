#!/usr/bin/env python3
"""Leak upper-bound sweep: progressively stack LatentCoT tricks 1 → 1+2 → 1+2+4.

  L0 baseline: plain leak (use results from stage2_fact_train_test_leak if present)
  L1: --coconut-recycle
  L2: --coconut-recycle --adaptive-halt
  L3: --coconut-recycle --adaptive-halt --n-thought-groups 3

All stages: train∪test + val=test, LoRA, BS=48 (OOM fallback), all domains.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HOME", "/tmp/hf_qwen06")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PY = os.environ.get("MMD_PY", "/root/anaconda3/envs/mmd-tfs/bin/python")
OUT = ROOT / "artifacts" / "stage2_fact_leak_tricks"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    d.strip()
    for d in os.environ.get(
        "FACT_DOMAINS",
        "agriculture,economy,energy,security,social_good,public_health,traffic",
    ).split(",")
    if d.strip()
]
EPOCHS = os.environ.get("FACT_EPOCHS", "8")
BS = os.environ.get("FACT_BS", "48")
BS_FALLBACKS = [
    int(x)
    for x in os.environ.get("FACT_BS_FALLBACKS", "48,40,32,24,16,8").split(",")
    if x.strip()
]
N_LATENT = os.environ.get("FACT_N_LATENT", "12")
LR = os.environ.get("FACT_LR", "1e-4")

STAGES = [
    {
        "name": "L1_coconut",
        "desc": "trick1 coconut recycle",
        "extra": ["--coconut-recycle"],
    },
    {
        "name": "L2_coconut_halt",
        "desc": "trick1+2 coconut + adaptive halt",
        "extra": ["--coconut-recycle", "--adaptive-halt"],
    },
    {
        "name": "L3_coconut_halt_mix",
        "desc": "trick1+2+4 coconut + halt + mixture(K=3)",
        "extra": [
            "--coconut-recycle",
            "--adaptive-halt",
            "--n-thought-groups",
            "3",
        ],
    },
]


def events_ok(domain: str) -> bool:
    from language.stage2.profiles import get_stage2_profile

    p = get_stage2_profile(domain)
    path = ROOT / p.events_dir / "events.jsonl"
    return path.is_file() and path.stat().st_size > 100


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env
        )
    print(f"exit={proc.returncode} log={log_path}", flush=True)
    return int(proc.returncode)


def load_baseline_leak() -> dict:
    """Pull prior leak (use_aux=True) gated rel as L0 reference."""
    out = {}
    base = ROOT / "artifacts" / "stage2_fact_train_test_leak"
    for d in DOMAINS:
        p = base / d / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        fact = s.get("B_fact_latentcot_alone") or {}
        out[d] = {
            "fact_gated_rel": fact.get("rel"),
            "amp_scale": fact.get("amp_scale"),
            "B0_test": s.get("B0_test"),
            "source": str(p),
        }
    vals = [v["fact_gated_rel"] for v in out.values() if isinstance(v.get("fact_gated_rel"), (int, float))]
    return {
        "domains": out,
        "mean_rel": float(sum(vals) / len(vals)) if vals else None,
        "n": len(vals),
        "note": "prior leak use_aux=True (no tricks)",
    }


def run_stage(stage: dict) -> dict:
    tag = stage["name"]
    per = {}
    print(f"\n######## {tag}: {stage['desc']} ########", flush=True)
    for domain in DOMAINS:
        if not events_ok(domain):
            per[domain] = {"error": "no events"}
            print(f"[{tag}/{domain}] skip — no events", flush=True)
            continue
        log = OUT / tag / domain / "fair_branch.log"
        base = [
            PY,
            "-u",
            "scripts/fair_branch_resid.py",
            "--domain",
            domain,
            "--tag",
            tag,
            "--epochs",
            str(EPOCHS),
            "--lr-fact",
            str(LR),
            "--n-latent",
            str(N_LATENT),
            "--skip-soft",
            "--lora",
            "--train-includes-test",
            "--val-on-test",
            "--amp-select",
            "val",
        ] + list(stage["extra"])

        tried = []
        code = 1
        bs_list = [int(BS)] + [b for b in BS_FALLBACKS if b != int(BS)]
        for bs in bs_list:
            if bs in tried:
                continue
            tried.append(bs)
            cmd = base + ["--batch-size", str(bs)]
            print(f"[{tag}/{domain}] try batch-size={bs}", flush=True)
            code = run_cmd(cmd, log)
            if code == 0:
                break
            print(f"[{tag}/{domain}] bs={bs} exit={code}; fallback", flush=True)

        row: dict = {"fair_exit": code, "batch_sizes_tried": tried}
        if code == 0:
            row["batch_size"] = tried[-1] if tried else None
        summ = ROOT / "artifacts" / "stage2_fact_improve_ablate" / tag / domain / "summary.json"
        if summ.is_file():
            s = json.loads(summ.read_text(encoding="utf-8"))
            fact = s.get("B_fact_latentcot_alone") or {}
            ung = s.get("B_fact_latentcot_ungated") or {}
            dest = OUT / tag / domain
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "summary.json").write_text(json.dumps(s, indent=2))
            row.update(
                {
                    "rel": fact.get("rel"),
                    "mse": fact.get("mse"),
                    "amp_scale": fact.get("amp_scale"),
                    "ungated_rel": ung.get("rel"),
                    "B0_test": s.get("B0_test"),
                    "fact_cfg": s.get("fact_cfg"),
                }
            )
        per[domain] = row
        print(f"[{tag}/{domain}] rel={row.get('rel')} exit={code}", flush=True)

    rels = [v["rel"] for v in per.values() if isinstance(v.get("rel"), (int, float))]
    return {
        "desc": stage["desc"],
        "extra": stage["extra"],
        "domains": per,
        "mean_rel": float(sum(rels) / len(rels)) if rels else None,
        "n": len(rels),
    }


def main() -> int:
    status = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "ORACLE LEAK progressive tricks 1→1+2→1+2+4; BS=48",
        "config": {
            "epochs": EPOCHS,
            "batch_size": BS,
            "n_latent": N_LATENT,
            "lr_fact": LR,
            "lora": True,
            "train_includes_test": True,
            "val_on_test": True,
        },
        "L0_baseline_leak": load_baseline_leak(),
        "stages": {},
    }
    sp = OUT / "status.json"

    def dump():
        sp.write_text(json.dumps(status, indent=2), encoding="utf-8")

    dump()
    print(
        f"OUT={OUT} domains={DOMAINS} L0_mean={status['L0_baseline_leak'].get('mean_rel')}",
        flush=True,
    )

    for stage in STAGES:
        tag = stage["name"]
        status["stages"][tag] = {"running": True, "desc": stage["desc"]}
        dump()
        out = run_stage(stage)
        status["stages"][tag] = out
        dump()
        print(f"==> {tag} mean_rel={out['mean_rel']}", flush=True)

    # Rollup table
    table = {}
    for d in DOMAINS:
        table[d] = {
            "L0_leak": (status["L0_baseline_leak"]["domains"].get(d) or {}).get(
                "fact_gated_rel"
            )
        }
        for tag, payload in status["stages"].items():
            table[d][tag] = (payload.get("domains") or {}).get(d, {}).get("rel")

    means = {
        "L0_leak": status["L0_baseline_leak"].get("mean_rel"),
    }
    for tag, payload in status["stages"].items():
        means[tag] = payload.get("mean_rel")

    rollup = {
        "means": means,
        "table_rel": table,
        "note": "ORACLE LEAK — not a valid score; progressive tricks 1/2/4",
    }
    (OUT / "summary_leak_tricks.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["rollup"] = str(OUT / "summary_leak_tricks.json")
    dump()

    print("\n=== LEAK TRICKS RESULT ===", flush=True)
    for k, v in means.items():
        print(f"  {k}: mean_rel={v}", flush=True)
    print(f"done → {OUT / 'summary_leak_tricks.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
