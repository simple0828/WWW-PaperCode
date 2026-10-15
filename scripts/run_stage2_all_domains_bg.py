#!/usr/bin/env python3
"""Background queue: extract events → schemes 1/2/5 → fair_branch (no SoftLSTM).

Covers all cleaned Time-MMD domains. Defaults to local vLLM at 10.30.1.2:8001.

Usage:
  STAGE2_FORCE_EXTRACT=1 STAGE2_SKIP_SOFT=1 \\
    nohup python -u scripts/run_stage2_all_domains_bg.py \\
      > artifacts/stage2_all_domains/run.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

PY = os.environ.get("MMD_PY", "/root/anaconda3/envs/mmd-tfs/bin/python")
OUT = ROOT / "artifacts" / "stage2_all_domains"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    d.strip()
    for d in os.environ.get(
        "STAGE2_DOMAINS",
        "traffic,agriculture,economy,energy,security,social_good,public_health",
    ).split(",")
    if d.strip()
]
FORCE_EXTRACT = os.environ.get("STAGE2_FORCE_EXTRACT", "1").strip() not in (
    "0",
    "false",
    "False",
    "",
)
SKIP_SOFT = os.environ.get("STAGE2_SKIP_SOFT", "1").strip() not in (
    "0",
    "false",
    "False",
    "",
)
WAIT_TRAFFIC = os.environ.get("STAGE2_WAIT_TRAFFIC", "1").strip() not in (
    "0",
    "false",
    "False",
    "",
)

# Prefer local Qwen vLLM
os.environ.setdefault("LLM_PROVIDER", "vllm")
os.environ.setdefault("VLLM_MODEL", "/model")
os.environ.setdefault("VLLM_BASE_URL", "http://10.30.1.2:8001")
os.environ.setdefault("VLLM_REASONING_EFFORT", "low")
os.environ.setdefault("VLLM_ENABLE_THINKING", "0")

# Obsolete caches/events from prior Stage2 runs (force-fresh extract path).
OBSOLETE_PATHS = [
    "artifacts/vmt_events_mech",
    "artifacts/event_ts_assoc",
    "artifacts/agri_events_mech_panelfix",
    "artifacts/agri_event_ts_assoc_panelfix",
    "artifacts/agri_events_mech_reprompt",
    "artifacts/agri_event_ts_assoc_reprompt",
    "artifacts/economy_events_mech",
    "artifacts/economy_event_ts_assoc",
    "artifacts/energy_events_mech",
    "artifacts/energy_event_ts_assoc",
    "artifacts/security_events_mech",
    "artifacts/security_event_ts_assoc",
    "artifacts/social_good_events_mech",
    "artifacts/social_good_event_ts_assoc",
    "artifacts/public_health_events_mech",
    "artifacts/public_health_event_ts_assoc",
    "artifacts/dataevolve_cfa_selfevo_security_reprompt",  # superseded by valtest
]


def _vllm_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url.rstrip("/") + "/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status) < 500
    except Exception:
        return False


def resolve_llm() -> dict:
    for url in (
        os.getenv("VLLM_BASE_URL", ""),
        "http://10.30.1.2:8001",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8000",
    ):
        if url and _vllm_ok(url):
            return {
                "provider": "vllm",
                "base_url": url.rstrip("/"),
                "model": os.getenv("VLLM_MODEL", "/model"),
                "ready": True,
            }
    return {"provider": "none", "base_url": "", "model": "", "ready": False}


def wait_llm(poll_s: float = 30.0, max_wait_s: float = 0) -> dict:
    t0 = time.time()
    while True:
        llm = resolve_llm()
        if llm["ready"]:
            print(f"LLM ready: {llm}", flush=True)
            return llm
        if max_wait_s > 0 and (time.time() - t0) >= max_wait_s:
            print("LLM wait timed out", flush=True)
            return llm
        print(
            f"[{time.strftime('%H:%M:%S')}] waiting vLLM at "
            f"{os.getenv('VLLM_BASE_URL', 'http://10.30.1.2:8001')} …",
            flush=True,
        )
        time.sleep(poll_s)


def traffic_stage1_done() -> bool:
    summary = ROOT / "artifacts/dataevolve_cfa_selfevo_traffic_reprompt/summary.json"
    if not summary.is_file():
        return False
    try:
        s = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    gb = s.get("global_best") or {}
    path = gb.get("cleaned_csv") or gb.get("path") or ""
    if path:
        pp = Path(path)
        if not pp.is_absolute():
            pp = ROOT / pp
        if pp.is_file():
            return True
    # finished marker: n_rounds_ran + global_best present
    if gb and int(s.get("n_rounds_ran") or 0) >= 1:
        art = ROOT / "artifacts/dataevolve_cfa_selfevo_traffic_reprompt"
        return any(art.rglob("US_VMT_Month_cleaned.csv"))
    return False


def wait_traffic_stage1(poll_s: float = 60.0) -> None:
    if not WAIT_TRAFFIC:
        return
    while not traffic_stage1_done():
        print(
            f"[{time.strftime('%H:%M:%S')}] waiting traffic Stage1 "
            f"(dataevolve_cfa_selfevo_traffic_reprompt/summary.json) …",
            flush=True,
        )
        time.sleep(poll_s)
    print("traffic Stage1 ready", flush=True)


def cleanup_obsolete() -> None:
    for rel in OBSOLETE_PATHS:
        p = ROOT / rel
        if not p.exists():
            continue
        print(f"remove obsolete {rel}", flush=True)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def events_ok(domain: str) -> bool:
    from language.stage2.profiles import get_stage2_profile

    p = get_stage2_profile(domain)
    path = ROOT / p.events_dir / "events.jsonl"
    return path.is_file() and path.stat().st_size > 100


def scheme1_cache_ok(domain: str) -> bool:
    from language.stage2.profiles import get_stage2_profile

    p = get_stage2_profile(domain)
    d = ROOT / p.cache_dir
    ch = list(d.glob("scheme1_channel_ch2w*.jsonl"))
    win = list(d.glob("scheme1_window_ch2w*.jsonl"))
    return bool(ch and win and ch[0].stat().st_size > 0 and win[0].stat().st_size > 0)


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
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(f"exit={proc.returncode} log={log_path}", flush=True)
    return int(proc.returncode)


def extract_domain(domain: str, llm: dict) -> int:
    from language.stage2.profiles import get_stage2_profile

    prof = get_stage2_profile(domain)
    ev_dir = ROOT / prof.events_dir
    if FORCE_EXTRACT and ev_dir.exists():
        print(f"[{domain}] force-extract: remove {ev_dir}", flush=True)
        shutil.rmtree(ev_dir, ignore_errors=True)
    elif events_ok(domain):
        print(f"[{domain}] events already present", flush=True)
        return 0

    log = OUT / domain / "extract.log"
    cmd = [
        PY,
        "-u",
        "scripts/extract_events.py",
        "--domain",
        domain,
        "--llm-provider",
        llm["provider"],
        "--base-url",
        llm["base_url"],
        "--model",
        llm["model"],
        "--concurrency",
        os.environ.get("STAGE2_EXTRACT_CONCURRENCY", "0"),
        "--max-tokens",
        os.environ.get("STAGE2_EXTRACT_MAX_TOKENS", "0"),
        "--resume",
    ]
    return run_cmd(cmd, log)


def _llm_flags(llm: dict) -> list[str]:
    return [
        "--llm-provider",
        llm["provider"],
        "--base-url",
        llm["base_url"],
        "--model",
        llm["model"],
    ]


def eval_scheme1(domain: str, llm: dict) -> int:
    log = OUT / domain / "eval_scheme1.log"
    cmd = [
        PY,
        "-u",
        "scripts/eval_scheme1.py",
        "--domain",
        domain,
        *_llm_flags(llm),
    ]
    if not FORCE_EXTRACT and scheme1_cache_ok(domain):
        cmd.append("--replay-only")
        print(f"[{domain}] scheme1 replay-only (cache hit)", flush=True)
    return run_cmd(cmd, log)


def eval_scheme2(domain: str, llm: dict) -> int:
    log = OUT / domain / "eval_scheme2.log"
    cmd = [
        PY,
        "-u",
        "scripts/eval_scheme2.py",
        "--domain",
        domain,
        "--n-rounds",
        "3",
        *_llm_flags(llm),
    ]
    return run_cmd(cmd, log)


def eval_scheme5(domain: str, llm: dict) -> int:
    log = OUT / domain / "eval_scheme5.log"
    cmd = [
        PY,
        "-u",
        "scripts/eval_scheme5.py",
        "--domain",
        domain,
        "--n-epochs",
        "2",
        "--wave",
        "0",
        "--cache-tag",
        f"{domain}_v1",
        *_llm_flags(llm),
    ]
    return run_cmd(cmd, log)


def fair_domain(domain: str) -> int:
    log = OUT / domain / "fair_branch.log"
    cmd = [
        PY,
        "-u",
        "scripts/fair_branch_resid.py",
        "--domain",
        domain,
        "--epochs",
        "8",
        "--batch-size",
        "2",
    ]
    if SKIP_SOFT:
        cmd.append("--skip-soft")
    return run_cmd(cmd, log)


def main() -> int:
    status: dict = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domains": {},
        "force_extract": FORCE_EXTRACT,
        "skip_soft": SKIP_SOFT,
        "domain_list": DOMAINS,
    }
    status_path = OUT / "status.json"

    def dump():
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    dump()
    print(f"OUT={OUT} domains={DOMAINS} force_extract={FORCE_EXTRACT} skip_soft={SKIP_SOFT}", flush=True)

    wait_traffic_stage1()
    cleanup_obsolete()

    llm = wait_llm(poll_s=30.0, max_wait_s=0)
    if not llm["ready"]:
        print("LLM not ready", flush=True)
        return 1

    for domain in DOMAINS:
        dstat: dict = {"domain": domain}
        status["domains"][domain] = dstat
        dump()
        try:
            print(f"[{domain}] === extract ===", flush=True)
            dstat["extract"] = extract_domain(domain, llm)
            if not events_ok(domain):
                dstat["error"] = "no events after extract"
                dump()
                continue

            print(f"[{domain}] === explicit scheme1 ===", flush=True)
            dstat["scheme1"] = eval_scheme1(domain, llm)
            print(f"[{domain}] === explicit scheme2 ===", flush=True)
            dstat["scheme2"] = eval_scheme2(domain, llm)
            print(f"[{domain}] === explicit scheme5 ===", flush=True)
            dstat["scheme5"] = eval_scheme5(domain, llm)

            print(f"[{domain}] === fair_branch Fact/Explicit (skip SoftLSTM={SKIP_SOFT}) ===", flush=True)
            dstat["fair"] = fair_domain(domain)

            from language.stage2.profiles import get_stage2_profile

            prof = get_stage2_profile(domain)
            cache = ROOT / prof.cache_dir
            out_dir = ROOT / prof.out_dir
            dstat["artifacts"] = {
                "scheme1": str(cache / "scheme1_ch2w_eval.json"),
                "scheme2": str(cache / "scheme2_eval.json"),
                "scheme5": str(cache / "scheme5_eval.json"),
                "fair": str(out_dir / "summary_fair_branches.json"),
            }
            summ = out_dir / "summary_fair_branches.json"
            if summ.is_file():
                dstat["summary"] = json.loads(summ.read_text(encoding="utf-8"))
            for name, path in (
                ("scheme1_eval", cache / "scheme1_ch2w_eval.json"),
                ("scheme2_eval", cache / "scheme2_eval.json"),
                ("scheme5_eval", cache / "scheme5_eval.json"),
            ):
                if path.is_file():
                    try:
                        dstat[name] = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        dstat[name] = str(path)
        except Exception as e:
            dstat["error"] = repr(e)
            print(f"[{domain}] ERROR {e}", flush=True)
        dump()

    combined = {}
    for k, v in status["domains"].items():
        combined[k] = {
            "fair": v.get("summary"),
            "scheme1": v.get("scheme1_eval"),
            "scheme2": v.get("scheme2_eval"),
            "scheme5": v.get("scheme5_eval"),
            "exits": {
                "extract": v.get("extract"),
                "scheme1": v.get("scheme1"),
                "scheme2": v.get("scheme2"),
                "scheme5": v.get("scheme5"),
                "fair": v.get("fair"),
            },
            "error": v.get("error"),
        }
    rollup = OUT / "summary_explicit_125_all.json"
    rollup.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    fair_rollup = OUT / "summary_fair_branches_all.json"
    fair_rollup.write_text(
        json.dumps({k: v.get("fair") for k, v in combined.items()}, indent=2),
        encoding="utf-8",
    )
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["rollup"] = str(rollup)
    dump()
    print(f"done → {rollup}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
