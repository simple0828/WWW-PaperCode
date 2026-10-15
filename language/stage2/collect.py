"""Collect CFA windows + soft/presence panels + explicit mech for a domain."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from language.apo.baseline import UnimodalBaselinePredictor
from language.apo.cfa_scorer import CFADownstreamScorer
from language.apo.config import build_domain_args
from language.apo.eval_windows import loader_window_indices
from language.lamp_vmt.windows import _norm_date
from language.stage2.actions import soft
from language.stage2.explicit_prompts import build_fact_prompt
from language.stage2.lamp_api import load_lamp_sub
from language.stage2.profiles import Stage2DomainProfile, get_stage2_profile
from language.stage2.resolve import resolve_stage1_candidate
from language.stage2.scheme1_hier import Scheme1Config, attach_scheme1


def soft_matrix(by_ym, months, channels: tuple[str, ...]):
    C = len(channels)
    ch2i = {c: i for i, c in enumerate(channels)}
    L = len(months)
    signed = np.zeros((L, C), np.float32)
    presence = np.zeros((L, C), np.float32)
    for t, ym in enumerate(months):
        counts = np.zeros(C, np.float32)
        for e in by_ym.get(ym, []):
            ch = e.get("channel")
            if ch not in ch2i:
                continue
            i = ch2i[ch]
            signed[t, i] += soft(e)
            counts[i] += 1.0
        presence[t] = (counts > 0).astype(np.float32)
        signed[t] = np.where(counts > 0, signed[t] / np.maximum(counts, 1.0), 0.0)
    return signed, presence


def collect_rows(
    root: Path,
    domain_or_profile: str | Stage2DomainProfile,
    *,
    batch_size: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict]:
    """Return train_rows, test_rows, H, meta."""
    profile = (
        domain_or_profile
        if isinstance(domain_or_profile, Stage2DomainProfile)
        else get_stage2_profile(domain_or_profile)
    )
    schema = load_lamp_sub(profile, "schema")
    channels = tuple(schema.CHANNELS)
    stage1 = resolve_stage1_candidate(root, profile)
    events_path = root / profile.events_dir / "events.jsonl"
    if not events_path.is_file() or events_path.stat().st_size == 0:
        raise FileNotFoundError(f"Events missing/empty: {events_path}")
    events = [json.loads(l) for l in events_path.open() if l.strip()]
    by_ym: dict[str, list] = defaultdict(list)
    for e in events:
        by_ym[str(e.get("date") or "")[:7]].append(e)

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

    def one(split: str, *, align_runpy: bool):
        ds = uni.get_dataset(split)
        if align_runpy:
            wis = loader_window_indices(cfa_args, split)
        else:
            wis = list(range(len(ds)))
        samples, texts, meta = [], [], []
        for wi in wis:
            s = uni.get_sample(split, wi)
            months = [_norm_date(ds.date[wi + off])[:7] for off in range(ds.seq_len)]
            samples.append(s)
            texts.append(s.raw_text)
            meta.append({"wi": wi, "months": months, "origin": months[-1]})
        print(
            f"CFA {split} n={len(samples)} align_runpy={align_runpy} domain={profile.key}",
            flush=True,
        )
        y_cfa = scorer.predict_fusion_batch(samples, texts, batch_size=batch_size)
        rows = []
        for s, yc, m in zip(samples, y_cfa, meta):
            yt = np.asarray(s.y_true, float).reshape(-1).astype(np.float32)
            yc = np.asarray(yc, float).reshape(-1).astype(np.float32)
            signed, presence = soft_matrix(by_ym, m["months"], channels)
            ev = np.stack([signed, presence], axis=-1).astype(np.float32)
            wid = f"{split}:{m['wi']}"
            rows.append(
                {
                    "wid": wid,
                    "yt": yt,
                    "yc": yc,
                    "ev": ev,
                    "months": list(m["months"]),
                    "origin": m["origin"],
                    "prompt": build_fact_prompt(
                        by_ym,
                        m["months"],
                        m["origin"],
                        profile,
                        max_months=int(profile.fact_max_months),
                        max_per_month=int(profile.fact_max_per_month),
                    ),
                    # filled below by scheme1 (channel→window); zeros if cache miss
                    "mech_llm": np.zeros(H, np.float32),
                    "mech": np.zeros(H, np.float32),
                }
            )
        return rows

    train = one("train", align_runpy=False)
    test = one("test", align_runpy=bool(profile.align_test_runpy))
    all_rows = train + test
    s1_info = attach_scheme1(
        all_rows,
        by_ym,
        profile,
        H,
        cache_dir=root / profile.cache_dir,
        client=None,
        cfg=Scheme1Config(),
        fill_missing=False,
        channels=channels,
    )
    print(f"scheme1 attach stats={s1_info['stats']}", flush=True)
    meta = {
        "profile": profile,
        "stage1": str(stage1["dir"]),
        "family": stage1["family"],
        "channels": channels,
        "C": len(channels),
        "n_events": len(events),
        "H": H,
        "by_ym": by_ym,
        "events_path": str(events_path),
        "scheme1": s1_info["stats"],
    }
    return train, test, H, meta
