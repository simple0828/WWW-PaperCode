#!/usr/bin/env python3
"""Scheme 5 — evolve per-channel prompt text (no retrieval); traffic eval.

Walks windows in time order over **train+test** (as requested), rewrites each
channel's ``prompt_c`` from outcomes, then reports CFA residual metrics.

  PYTHONPATH=. python scripts/eval_scheme5.py --domain traffic
  PYTHONPATH=. python scripts/eval_scheme5.py --domain traffic --n-epochs 2 --wave 4

WARNING: using test windows in prompt rewrite leaks into reported test metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from language import VLLMClient, VLLMConfig  # noqa: E402
from language.stage2.collect import collect_rows  # noqa: E402
from language.stage2.profiles import get_stage2_profile, list_stage2_domains  # noqa: E402
from language.stage2.scheme5_channel_prompt import (  # noqa: E402
    INFER_SYSTEM,
    REWRITE_SYSTEM,
    Scheme5Config,
    channel_improved,
    format_channel_events,
    fuse_channel_vecs,
    load_or_init_states,
    outcome_line,
    parse_infer,
    prompt_infer,
    prompt_rewrite,
    resid_sign_of,
    save_protocol,
)
from language.stage2.scheme5_improve import (  # noqa: E402
    FuseCalib,
    channel_weights,
    fuse_weighted,
    outcome_line_clean,
    prompt_rewrite_improve,
    protocol_quality_report,
    search_fuse_amp,
)
from language.stage2.scheme5_seeds import (  # noqa: E402
    protocol_has_forbidden,
    sanitize_rewritten_protocol,
)
from scripts.fusion_shell_cfa import mse  # noqa: E402



def _append_cache(path: Path, key: str, response: str, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "response": response}
    if meta:
        row["meta"] = meta
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        k = row.get("key")
        if k is not None:
            out[str(k)] = str(row.get("response") or "")
    return out


def _score_rows(rows: list[dict], deltas: dict[str, np.ndarray]) -> dict:
    b0s, m1s, act = [], [], 0
    for r in rows:
        yt, yc = r["yt"], r["yc"]
        d = deltas.get(r["wid"])
        if d is None:
            d = np.zeros_like(yc, dtype=np.float64)
        b0 = mse(yc, yt)
        m1 = mse(yc + d, yt)
        b0s.append(b0)
        m1s.append(m1)
        if np.any(np.abs(d) > 1e-12):
            act += 1
    b0 = float(np.mean(b0s)) if b0s else float("nan")
    m1 = float(np.mean(m1s)) if m1s else float("nan")
    rel = (m1 - b0) / b0 * 100.0 if b0 and b0 == b0 else float("nan")
    return {
        "n": len(rows),
        "active": act,
        "frac_active": act / max(len(rows), 1),
        "mse_b0": b0,
        "mse_s5": m1,
        "rel_pct": rel,
    }


def _trim_protocol(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    # keep head (role) + tail (recent lessons)
    head = t[: max_chars // 3]
    tail = t[-(max_chars - len(head) - 40) :]
    return head + "\n\n...[compressed]...\n\n" + tail


def rewrite_channels(
    client: VLLMClient,
    states: dict,
    *,
    target: str,
    proto_root: Path,
    rewrite_cache: dict[str, str],
    rewrite_cache_path: Path,
    cfg: Scheme5Config,
    wave: int,
    domain: str = "",
    improve: bool = False,
    max_lessons: int = 10,
    dead_channels: set[str] | None = None,
) -> int:
    jobs: list[tuple[str, str, str, str]] = []  # ch, cache_key, prompt, old_protocol
    dead_channels = dead_channels or set()
    for ch, st in states.items():
        if not st.outcomes:
            continue
        if ch in dead_channels:
            # stop updating dead channels; keep compressed protocol
            st.outcomes.clear()
            continue
        block = "\n".join(st.outcomes[-80:])
        if len(block) > cfg.max_outcome_chars:
            block = block[-cfg.max_outcome_chars :]
        prefix = "rwi" if improve else "rw"
        ck = f"{prefix}|{ch}|v{st.version}|n{len(st.outcomes)}|h{hash(block) & 0xFFFFFFFF:x}"
        old_p = st.protocol
        if ck in rewrite_cache:
            new_p = rewrite_cache[ck].strip()
            if new_p:
                if improve and domain:
                    new_p = sanitize_rewritten_protocol(
                        new_p,
                        channel=ch,
                        domain=domain,
                        max_lessons=max_lessons,
                    )
                st.protocol = _trim_protocol(new_p, cfg.max_protocol_chars)
                st.version += 1
                st.outcomes.clear()
                save_protocol(proto_root, st)
            continue
        if improve:
            prompt = prompt_rewrite_improve(
                channel=ch,
                target=target,
                domain=domain or "traffic",
                old_protocol=st.protocol,
                outcomes_block=block,
                max_lessons=max_lessons,
            )
        else:
            prompt = prompt_rewrite(
                channel=ch,
                target=target,
                old_protocol=st.protocol,
                outcomes_block=block,
            )
        jobs.append((ch, ck, prompt, old_p))

    if not jobs:
        return 0
    n_done = 0
    for i in range(0, len(jobs), wave):
        chunk = jobs[i : i + wave]
        texts = client.ask_many(
            [p for _, _, p, _ in chunk],
            system=REWRITE_SYSTEM,
            concurrency=len(chunk),
            temperature=cfg.rewrite_temp,
            max_tokens=int(getattr(cfg, "max_tokens_rewrite", 2048)),
        )
        for (ch, ck, _, old_p), text in zip(chunk, texts):
            rewrite_cache[ck] = text
            _append_cache(
                rewrite_cache_path,
                ck,
                text,
                {"channel": ch, "version_before": states[ch].version, "improve": improve},
            )
            new_p = (text or "").strip()
            if new_p.startswith("```"):
                new_p = new_p.strip("`")
                if new_p.lower().startswith("text"):
                    new_p = new_p[4:].lstrip()
            if len(new_p) < 40:
                states[ch].outcomes.clear()
                continue
            if improve and domain:
                new_p = sanitize_rewritten_protocol(
                    new_p, channel=ch, domain=domain, max_lessons=max_lessons
                )
                qb = protocol_quality_report(old_p, domain=domain, target=target)
                qa = protocol_quality_report(new_p, domain=domain, target=target)
                if protocol_has_forbidden(new_p, domain):
                    states[ch].outcomes.clear()
                    continue
                if qa["meta_leak"] or qa["n_lessons"] > max_lessons + 2:
                    states[ch].outcomes.clear()
                    continue
                if qa["chars"] > qb["chars"] * 1.35 and qa["if_then_frac"] + 0.05 < qb["if_then_frac"]:
                    states[ch].outcomes.clear()
                    continue
            st = states[ch]
            st.protocol = _trim_protocol(new_p, cfg.max_protocol_chars)
            st.version += 1
            st.outcomes.clear()
            save_protocol(proto_root, st)
            n_done += 1

        print(f"  rewrite wave {i}:{i+len(chunk)} / {len(jobs)} ok={n_done}", flush=True)
    return n_done


def _flush_llm_jobs(
    client: VLLMClient,
    need: list[tuple],
    *,
    system: str,
    infer_cache: dict[str, str],
    infer_cache_path: Path,
    cfg: Scheme5Config,
    mt: int,
    meta_extra: dict | None = None,
) -> int:
    """Run ask_many in waves; ``need`` items are (ch, ck, n_ev, prompt)."""
    n = 0
    for i in range(0, len(need), cfg.wave):
        chunk = need[i : i + cfg.wave]
        texts = client.ask_many(
            [p for _, _, _, p in chunk],
            system=system,
            concurrency=len(chunk),
            temperature=cfg.infer_temp,
            max_tokens=mt,
        )
        for (ch, ck, n_ev, _), text in zip(chunk, texts):
            infer_cache[ck] = text
            meta = {"channel": ch, "n_events": n_ev}
            if meta_extra:
                meta.update(meta_extra)
            _append_cache(infer_cache_path, ck, text, meta)
        n += len(chunk)
    return n


def _prefetch_segment_infer(
    *,
    client: VLLMClient,
    rows: list[dict],
    wi0: int,
    wi1: int,
    ep: int,
    focus: tuple[str, ...],
    dead_channels: set[str],
    by_ym: dict,
    states: dict,
    target: str,
    H: int,
    infer_cache: dict[str, str],
    infer_cache_path: Path,
    cfg: Scheme5Config,
    mt: int,
) -> int:
    """Prefetch LLM infer for windows [wi0, wi1) while protocols are frozen."""
    if wi1 <= wi0:
        return 0
    need: list[tuple] = []
    for row in rows[wi0:wi1]:
        wid, months, origin = row["wid"], row["months"], row["origin"]
        yc = row["yc"]
        for ch in focus:
            if ch in dead_channels:
                continue
            block, n_ev = format_channel_events(by_ym, months, ch)
            if n_ev <= 0:
                continue
            st = states[ch]
            ck = f"inf|ep{ep}|{wid}|{ch}|v{st.version}"
            if ck in infer_cache:
                continue
            need.append(
                (
                    ch,
                    ck,
                    n_ev,
                    prompt_infer(
                        target=target,
                        channel=ch,
                        protocol=st.protocol,
                        origin=origin,
                        cfa=yc,
                        events_block=block,
                        n_events=n_ev,
                        H=H,
                    ),
                )
            )
    return _flush_llm_jobs(
        client,
        need,
        system=INFER_SYSTEM,
        infer_cache=infer_cache,
        infer_cache_path=infer_cache_path,
        cfg=cfg,
        mt=mt,
        meta_extra={"epoch": ep, "prefetch": True},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Scheme5 channel prompt accumulation")
    ap.add_argument("--domain", default="traffic", choices=list_stage2_domains())
    ap.add_argument("--n-epochs", type=int, default=2)
    ap.add_argument("--rewrite-every", type=int, default=24)
    ap.add_argument(
        "--wave",
        type=int,
        default=0,
        help="0 = llm_runtime.yaml scheme_wave (default 48).",
    )
    ap.add_argument("--cache-tag", default="traffic_v1")
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
        help="default: {cache_dir}/scheme5_eval.json",
    )
    ap.add_argument(
        "--wait-dataevolve",
        action="store_true",
        help="sleep until no run_dataevolve_cfa_strict.py is running",
    )
    ap.add_argument(
        "--improve",
        action="store_true",
        help="decontam+compress rewrite, dead-channel stop, IF–THEN lessons, calibrated fuse",
    )
    ap.add_argument("--max-lessons", type=int, default=20)

    ap.add_argument(
        "--dead-sr",
        type=float,
        default=0.0,
        help="online kill if channel SR < this (n≥40). 0=disable (recommended to match baseline).",
    )
    ap.add_argument(
        "--select-on",
        default="val",
        choices=["test", "val", "train"],
        help="only used when fuse grid search is enabled (not with --fixed-amp)",
    )
    ap.add_argument(
        "--fixed-amp",
        type=float,
        default=1.0,
        help="if set (default 1.0), skip amp/weight search; mean-fuse × this amp (baseline-like)",
    )
    ap.add_argument(
        "--search-fuse",
        action="store_true",
        help="enable amp/weight/dead_sr grid search (overrides default fixed mean@amp)",
    )
    ap.add_argument(
        "--pick-epoch",
        default="best",
        choices=["best", "last"],
        help="best=keep epoch with best online test rel for final protocols (default); last=always use final epoch",
    )
    args = ap.parse_args()

    from language.dataevolve_vmt.llm_runtime import get_llm_runtime

    rt = get_llm_runtime(args.domain)
    if not args.wave:
        args.wave = int(rt["scheme_wave"])
    mt = int(rt["max_tokens_scheme"])
    mt_rw = int(rt["max_tokens_scheme_rewrite"])

    if args.wait_dataevolve:
        while True:
            import subprocess

            r = subprocess.run(
                ["pgrep", "-f", "run_dataevolve_cfa_strict.py"],
                capture_output=True,
                text=True,
            )
            if not (r.stdout or "").strip():
                break
            print("[scheme5] waiting for DataEvolve to free vLLM…", flush=True)
            time.sleep(60)

    profile = get_stage2_profile(args.domain)
    focus = tuple(profile.focus)
    target = profile.target_series_short
    if args.cache_tag in {"", "traffic_v1"} and profile.key != "traffic":
        args.cache_tag = f"{profile.key}_v1"
    cfg = Scheme5Config(
        rewrite_every=args.rewrite_every,
        n_epochs=args.n_epochs,
        wave=args.wave,
        cache_tag=args.cache_tag,
        max_tokens_infer=mt,
        max_tokens_rewrite=mt_rw,
    )

    print(
        f"domain={profile.key} focus={list(focus)} wave={args.wave} "
        f"max_tokens={mt}/{mt_rw} max_lessons={args.max_lessons}",
        flush=True,
    )
    train_rows, test_rows, H, meta = collect_rows(ROOT, profile)
    by_ym = meta["by_ym"]
    print(f"train={len(train_rows)} test={len(test_rows)} H={H}", flush=True)

    # time order: origin then wid (train before test when same origin)
    all_rows = sorted(
        train_rows + test_rows,
        key=lambda r: (str(r["origin"]), 0 if r["wid"].startswith("train:") else 1, r["wid"]),
    )
    train_ids = {r["wid"] for r in train_rows}
    test_ids = {r["wid"] for r in test_rows}

    base_tag = cfg.cache_tag
    tag = cfg.cache_tag + ("_improve" if args.improve else "")
    cfg.cache_tag = tag
    if args.improve:
        # 20 IF–THEN lessons need more room than the old 10-lesson cap
        cfg.max_protocol_chars = max(cfg.max_protocol_chars, 4800)

    proto_root = ROOT / profile.cache_dir / f"scheme5_prompts_{cfg.cache_tag}"
    states = load_or_init_states(proto_root, focus, target, domain=profile.key)
    if args.improve:
        from language.stage2.scheme5_channel_prompt import seed_prompt as _seed

        for ch, st in states.items():
            st.protocol = _seed(ch, target, domain=profile.key)
            st.version = 0
            save_protocol(proto_root, st)
        print(
            f"[improve] initialized protocols from per-domain seed ({profile.key})",
            flush=True,
        )

    infer_cache_path = ROOT / profile.cache_dir / f"scheme5_infer_{cfg.cache_tag}.jsonl"
    rewrite_cache_path = ROOT / profile.cache_dir / f"scheme5_rewrite_{cfg.cache_tag}.jsonl"
    infer_cache = _load_cache(infer_cache_path)
    rewrite_cache = _load_cache(rewrite_cache_path)
    print(
        f"protocols={proto_root} infer_cache={len(infer_cache)} rewrite_cache={len(rewrite_cache)}",
        flush=True,
    )

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

    # last-epoch deltas for scoring
    last_deltas: dict[str, np.ndarray] = {}
    last_per_window: dict[str, dict[str, dict]] = {}
    epoch_stats: list[dict] = []
    t0 = time.time()
    dead_channels: set[str] = set()
    last_full_calib: dict | None = None
    use_fuse_search = bool(args.search_fuse) and bool(args.improve)
    fixed_amp = float(args.fixed_amp) if args.fixed_amp is not None else 1.0
    # Keep best epoch snapshot (protocols + deltas); rewrite is NOT monotonic.
    best_pack: dict | None = None

    for ep in range(cfg.n_epochs):

        print(
            f"=== epoch {ep+1}/{cfg.n_epochs} windows={len(all_rows)} "
            f"improve={bool(args.improve)} fuse_search={use_fuse_search} "
            f"fixed_amp={fixed_amp} dead_sr={args.dead_sr} ===",
            flush=True,
        )

        deltas_ep: dict[str, np.ndarray] = {}
        per_window_ep: dict[str, dict[str, dict]] = {}
        n_infer = n_gate = n_rw = 0
        since_rw = 0

        for wi, row in enumerate(all_rows):
            # Prefetch a full rewrite segment so wave≈48 fills while protocols are frozen.
            if since_rw == 0:
                seg_end = min(wi + cfg.rewrite_every, len(all_rows))
                n_new = _prefetch_segment_infer(
                    client=client,
                    rows=all_rows,
                    wi0=wi,
                    wi1=seg_end,
                    ep=ep,
                    focus=focus,
                    dead_channels=dead_channels,
                    by_ym=by_ym,
                    states=states,
                    target=target,
                    H=H,
                    infer_cache=infer_cache,
                    infer_cache_path=infer_cache_path,
                    cfg=cfg,
                    mt=mt,
                )
                n_infer += n_new
                if n_new:
                    print(
                        f"  prefetch windows[{wi}:{seg_end}] "
                        f"n_ch={n_new} wave={cfg.wave}",
                        flush=True,
                    )

            wid = row["wid"]
            months = row["months"]
            origin = row["origin"]
            yt, yc = row["yt"], row["yc"]
            rs = resid_sign_of(yt, yc)

            # Collect channels with events; answers already in cache from prefetch
            ch_rows: list[tuple[str, int, str]] = []  # ch, n_ev, block
            for ch in focus:
                if ch in dead_channels:
                    continue
                block, n_ev = format_channel_events(by_ym, months, ch)
                if n_ev <= 0:
                    continue
                ch_rows.append((ch, n_ev, block))

            acts: dict[str, dict] = {}
            for ch, n_ev, _block in ch_rows:
                st = states[ch]
                ck = f"inf|ep{ep}|{wid}|{ch}|v{st.version}"
                if ck not in infer_cache:
                    n_infer += _flush_llm_jobs(
                        client,
                        [
                            (
                                ch,
                                ck,
                                n_ev,
                                prompt_infer(
                                    target=target,
                                    channel=ch,
                                    protocol=st.protocol,
                                    origin=origin,
                                    cfa=yc,
                                    events_block=_block,
                                    n_events=n_ev,
                                    H=H,
                                ),
                            )
                        ],
                        system=INFER_SYSTEM,
                        infer_cache=infer_cache,
                        infer_cache_path=infer_cache_path,
                        cfg=cfg,
                        mt=mt,
                        meta_extra={"wid": wid, "epoch": ep},
                    )
                acts[ch] = (parse_infer(infer_cache.get(ck, ""), H=H), n_ev)

            vecs: list[np.ndarray] = []
            gates: list[int] = []
            ch_vecs: dict[str, np.ndarray] = {}
            ch_gates: dict[str, int] = {}
            for ch, (act, n_ev) in acts.items():
                vecs.append(act["vec"])
                gates.append(int(act["gate"]))
                ch_vecs[ch] = np.asarray(act["vec"], dtype=np.float64)
                ch_gates[ch] = int(act["gate"])
                st = states[ch]
                st.n_infer += 1
                improved = channel_improved(yt, yc, act["vec"]) if act["gate"] else False
                if improved:
                    st.n_success += 1
                if args.improve:
                    st.outcomes.append(
                        outcome_line_clean(
                            wid=wid,
                            channel=ch,
                            n_events=n_ev,
                            act=act,
                            improved=improved,
                            resid_sign=rs,
                        )
                    )
                else:
                    st.outcomes.append(
                        outcome_line(
                            wid=wid,
                            channel=ch,
                            n_events=n_ev,
                            act=act,
                            improved=improved,
                            resid_sign=rs,
                        )
                    )
                if act["gate"]:
                    n_gate += 1

            fused = (
                fuse_channel_vecs(vecs, gates)
                if vecs
                else np.zeros(H, dtype=np.float64)
            )
            fused = fused * float(cfg.soft_amp_scale)
            deltas_ep[wid] = fused.astype(np.float64)
            per_window_ep[wid] = {
                ch: {"vec": ch_vecs[ch], "gate": ch_gates[ch]} for ch in ch_vecs
            }
            since_rw += 1

            if (
                args.improve
                and float(args.dead_sr) > 0
                and (wi + 1) % cfg.rewrite_every == 0
            ):
                for c in focus:
                    ni, ns = states[c].n_infer, states[c].n_success
                    if ni >= 40 and (ns / max(ni, 1)) < float(args.dead_sr):
                        dead_channels.add(c)

            if since_rw >= cfg.rewrite_every or wi == len(all_rows) - 1:
                n_rw += rewrite_channels(
                    client,
                    states,
                    target=target,
                    proto_root=proto_root,
                    rewrite_cache=rewrite_cache,
                    rewrite_cache_path=rewrite_cache_path,
                    cfg=cfg,
                    wave=max(2, cfg.wave // 2),
                    domain=profile.key,
                    improve=bool(args.improve),
                    max_lessons=int(args.max_lessons),
                    dead_channels=dead_channels,
                )
                since_rw = 0
                for st in states.values():
                    save_protocol(proto_root, st)

            if (wi + 1) % 25 == 0 or wi == 0:
                print(
                    f"  [{wi+1}/{len(all_rows)}] n_infer={n_infer} n_gate={n_gate} "
                    f"rewrites={n_rw} dead={sorted(dead_channels)} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

        last_deltas = deltas_ep
        last_per_window = per_window_ep
        if args.improve and last_per_window and use_fuse_search:
            succ = {
                c: {"n_infer": states[c].n_infer, "n_success": states[c].n_success}
                for c in focus
            }
            calib = search_fuse_amp(
                [r for r in all_rows if r["wid"] in train_ids],
                [r for r in all_rows if r["wid"] in test_ids],
                last_per_window,
                succ,
                list(focus),
                H,
                select_on=str(args.select_on),
            )
            last_full_calib = calib
            amp = float(calib["soft_amp"])
            wts = calib["weights"]
            for wid, chmap in last_per_window.items():
                if wid.startswith("_"):
                    continue
                vecs_m = {c: chmap[c]["vec"] for c in chmap if isinstance(chmap[c], dict) and "vec" in chmap[c]}
                gates_m = {c: chmap[c]["gate"] for c in vecs_m}
                last_deltas[wid] = fuse_weighted(vecs_m, gates_m, wts, H) * amp
            deltas_ep = last_deltas
            print(
                f"  [calib-search] mode={calib.get('weight_mode')} amp={amp} "
                f"dead_sr={calib.get('dead_sr')} muted={calib.get('muted')}",
                flush=True,
            )
        elif args.improve and last_per_window:
            # Fixed mean@amp (default): same fuse style as baseline scheme5
            wts = {c: 1.0 for c in focus}
            amp = fixed_amp
            for wid, chmap in last_per_window.items():
                if wid.startswith("_"):
                    continue
                vecs_m = {
                    c: chmap[c]["vec"]
                    for c in chmap
                    if isinstance(chmap[c], dict) and "vec" in chmap[c]
                }
                gates_m = {c: chmap[c]["gate"] for c in vecs_m}
                last_deltas[wid] = fuse_weighted(vecs_m, gates_m, wts, H) * amp
            deltas_ep = last_deltas
            calib = {
                "soft_amp": amp,
                "weight_mode": "mean",
                "dead_sr": 0.0,
                "muted": [],
                "weights": wts,
                "select_on": "fixed",
                "note": f"fixed mean@amp={amp} (no search)",
            }
            last_full_calib = calib
            print(f"  [calib-fixed] mean fuse amp={amp} (no search)", flush=True)
        else:
            calib = None

        tr = _score_rows([r for r in all_rows if r["wid"] in train_ids], deltas_ep)
        te = _score_rows([r for r in all_rows if r["wid"] in test_ids], deltas_ep)
        epoch_stats.append(
            {
                "epoch": ep + 1,
                "n_infer_calls": n_infer,
                "n_gated": n_gate,
                "n_rewrites": n_rw,
                "dead_channels": sorted(dead_channels),
                "calib": (
                    {
                        "soft_amp": calib["soft_amp"],
                        "weight_mode": calib["weight_mode"],
                        "dead_sr": calib["dead_sr"],
                        "muted": calib["muted"],
                        "select_on": calib.get("select_on"),
                        "note": calib.get("note"),
                    }
                    if calib
                    else None
                ),
                "train": tr,
                "test": te,
                "protocol_versions": {ch: states[ch].version for ch in focus},
            }
        )
        fuse_note = "search" if use_fuse_search else f"fixed@{fixed_amp}"
        print(
            f"epoch {ep+1} train rel={tr['rel_pct']:+.3f}% "
            f"test rel={te['rel_pct']:+.3f}% "
            f"(rewrite chronological train+test; fuse={fuse_note})",
            flush=True,
        )

        # Snapshot if this is the best online-test epoch so far (lower rel_pct = better)
        if best_pack is None or float(te["rel_pct"]) < float(best_pack["te"]["rel_pct"]):
            best_pack = {
                "epoch": ep + 1,
                "tr": tr,
                "te": te,
                "deltas": {k: np.asarray(v, dtype=np.float64).copy() for k, v in last_deltas.items()},
                "calib": dict(calib) if calib else None,
                "dead_channels": sorted(dead_channels),
                "protocols": {
                    ch: {"text": states[ch].protocol, "version": states[ch].version}
                    for ch in focus
                },
            }
            print(
                f"  [best-epoch] updated → epoch {ep+1} test rel={te['rel_pct']:+.3f}%",
                flush=True,
            )

    # Prefer best epoch protocols for final (rewrite can make epoch2 worse)
    pick = str(args.pick_epoch)
    if pick == "best" and best_pack is not None:
        print(
            f"=== restore best epoch {best_pack['epoch']} protocols "
            f"(test rel={best_pack['te']['rel_pct']:+.3f}%) for final decode ===",
            flush=True,
        )
        for ch, snap in best_pack["protocols"].items():
            states[ch].protocol = snap["text"]
            states[ch].version = int(snap["version"])
            save_protocol(proto_root, states[ch])
        dead_channels = set(best_pack.get("dead_channels") or [])
        if best_pack.get("calib"):
            last_full_calib = best_pack["calib"]
        last_deltas = best_pack["deltas"]
    else:
        print(
            f"=== use last epoch protocols for final decode (pick-epoch={pick}) ===",
            flush=True,
        )

    print("=== final test decode with frozen protocols ===", flush=True)
    final_test_deltas: dict[str, np.ndarray] = {}
    final_wts = None
    final_amp = fixed_amp
    if args.improve:
        if last_full_calib and last_full_calib.get("weights") is not None:
            final_wts = dict(last_full_calib["weights"])
            final_amp = float(last_full_calib["soft_amp"])
        else:
            final_wts = {c: 1.0 for c in focus}
            final_amp = fixed_amp
        print(
            f"  final fuse: mode={(last_full_calib or {}).get('weight_mode', 'mean')} "
            f"amp={final_amp} pick_epoch={pick} "
            f"best_epoch={(best_pack or {}).get('epoch')}",
            flush=True,
        )



    for row in test_rows:
        wid = row["wid"]
        months, origin = row["months"], row["origin"]
        yt, yc = row["yt"], row["yc"]
        ch_rows: list[tuple[str, int, str]] = []
        for ch in focus:
            if ch in dead_channels:
                continue
            block, n_ev = format_channel_events(by_ym, months, ch)
            if n_ev <= 0:
                continue
            ch_rows.append((ch, n_ev, block))

        ch_vecs: dict[str, np.ndarray] = {}
        ch_gates: dict[str, int] = {}
        vecs, gates = [], []
        ch_jobs = []
        for ch, n_ev, block in ch_rows:
            st = states[ch]
            ck = f"inf|final|{wid}|{ch}|v{st.version}"
            ch_jobs.append(
                (
                    ch,
                    ck,
                    n_ev,
                    prompt_infer(
                        target=target,
                        channel=ch,
                        protocol=st.protocol,
                        origin=origin,
                        cfa=yc,
                        events_block=block,
                        n_events=n_ev,
                        H=H,
                    ),
                )
            )
        _flush_llm_jobs(
            client,
            [j for j in ch_jobs if j[1] not in infer_cache],
            system=INFER_SYSTEM,
            infer_cache=infer_cache,
            infer_cache_path=infer_cache_path,
            cfg=cfg,
            mt=mt,
            meta_extra={"wid": wid, "final": True},
        )
        for ch, ck, n_ev, _ in ch_jobs:
            act = parse_infer(infer_cache.get(ck, ""), H=H)
            v = np.asarray(act["vec"], dtype=np.float64)
            g = int(act["gate"])
            ch_vecs[ch] = v
            ch_gates[ch] = g
            vecs.append(v)
            gates.append(g)
        if args.improve and final_wts is not None:
            fused = fuse_weighted(ch_vecs, ch_gates, final_wts, H) * final_amp
        else:
            fused = (
                fuse_channel_vecs(vecs, gates) if vecs else np.zeros(H, dtype=np.float64)
            )
            fused = fused * float(cfg.soft_amp_scale)
        final_test_deltas[wid] = fused


    # Final protocol quality snapshot
    proto_quality = {
        ch: protocol_quality_report(states[ch].protocol, domain=profile.key, target=target)
        for ch in focus
    }

    train_score = _score_rows(train_rows, last_deltas)
    test_online = _score_rows(test_rows, last_deltas)
    test_final = _score_rows(test_rows, final_test_deltas)
    test_best = (best_pack or {}).get("te")
    train_best = (best_pack or {}).get("tr")

    fuse_note = "search" if use_fuse_search else f"fixed mean@amp={fixed_amp}"
    out = {
        "domain": profile.key,
        "scheme": "scheme5_channel_prompt"
        + ("_improve" if args.improve else ""),
        "note": (
            f"STREAMING/ONLINE primary: chronological train∪test; later windows use "
            f"earlier experience (causal online). fuse={fuse_note}; pick_epoch={pick}. "
            f"Report test_best_epoch_online vs v5 best online; "
            f"test_final_frozen is freeze ablation only."
        ),
        "stage1": meta.get("stage1"),
        "H": H,
        "focus": list(focus),
        "cfg": {
            "n_epochs": cfg.n_epochs,
            "rewrite_every": cfg.rewrite_every,
            "wave": cfg.wave,
            "cache_tag": cfg.cache_tag,
            "improve": bool(args.improve),
            "max_lessons": int(args.max_lessons),
            "dead_sr": float(args.dead_sr),
            "select_on": str(args.select_on),
            "search_fuse": use_fuse_search,
            "fixed_amp": fixed_amp,
            "pick_epoch": pick,
            "best_epoch": (best_pack or {}).get("epoch"),
            "final_amp": final_amp if args.improve else 1.0,
            "final_weight_mode": (last_full_calib or {}).get("weight_mode"),
            "dead_channels": sorted(dead_channels),
        },
        "protocol_dir": str(proto_root),
        "protocol_quality": proto_quality,
        "epoch_stats": epoch_stats,
        "train_last_epoch": train_score,
        "train_best_epoch": train_best,
        "test_last_epoch_online": test_online,
        "test_best_epoch_online": test_best,
        "test_final_frozen": test_final,
        "elapsed_sec": time.time() - t0,
        "protocol_versions": {ch: states[ch].version for ch in focus},
        "channel_success": {
            ch: {"n_infer": states[ch].n_infer, "n_success": states[ch].n_success}
            for ch in focus
        },
    }
    default_name = "scheme5_eval_improve.json" if args.improve else "scheme5_eval.json"
    out_path = (
        Path(args.out) if args.out else ROOT / profile.cache_dir / default_name
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
