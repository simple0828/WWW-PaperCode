#!/usr/bin/env python3
"""Train-only + val=test (α/ckpt oracle) progressive tricks: L0 → L1 → L2 → L3.

Unlike leak (train∪test), here Fact is trained on train only; only val=test
for checkpoint / amp selection (partial oracle upper bound).

  L0: baseline (no tricks)
  L1: --coconut-recycle
  L2: --coconut-recycle --adaptive-halt
  L3: --coconut-recycle --adaptive-halt --n-thought-groups 3

BS=48 (OOM fallback), all domains, LoRA.
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
OUT = ROOT / "artifacts" / "stage2_fact_valtest_tricks"
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
        "name": "VT0_baseline",
        "desc": "train-only + val=test, no tricks",
        "extra": [],
    },
    {
        "name": "VT1_coconut",
        "desc": "trick1 coconut recycle",
        "extra": ["--coconut-recycle"],
    },
    {
        "name": "VT2_coconut_halt",
        "desc": "trick1+2 coconut + adaptive halt",
        "extra": ["--coconut-recycle", "--adaptive-halt"],
    },
    {
        "name": "VT3_coconut_halt_mix",
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


def load_leak_tricks_means() -> dict:
    p = ROOT / "artifacts" / "stage2_fact_leak_tricks" / "summary_leak_tricks.json"
    if not p.exists():
        return {}
    o = json.loads(p.read_text(encoding="utf-8"))
    return {
        "means": o.get("means"),
        "table_rel": o.get("table_rel"),
        "note": "full leak train∪test for comparison",
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
            "--val-on-test",  # partial oracle: train-only, val=test
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
        summ = (
            ROOT / "artifacts" / "stage2_fact_improve_ablate" / tag / domain / "summary.json"
        )
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
        "note": (
            "PARTIAL ORACLE: train-only Fact + val=test for ckpt/α; "
            "progressive tricks L0→L1→L2→L3; BS=48"
        ),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BS,
            "n_latent": N_LATENT,
            "lr_fact": LR,
            "lora": True,
            "train_includes_test": False,
            "val_on_test": True,
        },
        "leak_tricks_ref": load_leak_tricks_means(),
        "stages": {},
    }
    sp = OUT / "status.json"

    def dump():
        sp.write_text(json.dumps(status, indent=2), encoding="utf-8")

    dump()
    print(f"OUT={OUT} domains={DOMAINS} val=test only (no train∪test)", flush=True)

    for stage in STAGES:
        tag = stage["name"]
        status["stages"][tag] = {"running": True, "desc": stage["desc"]}
        dump()
        out = run_stage(stage)
        status["stages"][tag] = out
        dump()
        print(f"==> {tag} mean_rel={out['mean_rel']}", flush=True)

    table = {}
    means = {}
    for d in DOMAINS:
        table[d] = {}
        for tag, payload in status["stages"].items():
            table[d][tag] = (payload.get("domains") or {}).get(d, {}).get("rel")
    for tag, payload in status["stages"].items():
        means[tag] = payload.get("mean_rel")

    # attach leak means for side-by-side
    leak_means = (status.get("leak_tricks_ref") or {}).get("means") or {}
    compare = {
        "valtest_means": means,
        "leak_means": leak_means,
        "delta_valtest_minus_leak": {},
    }
    # map VT* → L*
    mapping = {
        "VT0_baseline": "L0_leak",
        "VT1_coconut": "L1_coconut",
        "VT2_coconut_halt": "L2_coconut_halt",
        "VT3_coconut_halt_mix": "L3_coconut_halt_mix",
    }
    for vt, lk in mapping.items():
        a, b = means.get(vt), leak_means.get(lk)
        compare["delta_valtest_minus_leak"][vt] = (
            None
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float))
            else float(a) - float(b)
        )

    rollup = {
        "means": means,
        "table_rel": table,
        "compare_to_leak": compare,
        "note": "PARTIAL ORACLE val=test only — not a fully valid score",
    }
    (OUT / "summary_valtest_tricks.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["rollup"] = str(OUT / "summary_valtest_tricks.json")
    dump()

    print("\n=== VAL=TEST TRICKS RESULT ===", flush=True)
    for k, v in means.items():
        print(f"  {k}: mean_rel={v}", flush=True)
    print("vs leak means:", leak_means, flush=True)
    print(f"done → {OUT / 'summary_valtest_tricks.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
