#!/usr/bin/env python3
"""Fair Stage2 three-branch residual (SoftLSTM / Fact LatentCoT / Explicit).

Usage:
  python scripts/fair_branch_resid.py --domain traffic
  python scripts/fair_branch_resid.py --domain agriculture
  python scripts/fair_branch_resid.py --domain climate
  python scripts/fair_branch_resid.py --domain all

Domain knobs (paths, channels, eval windows) live in ``language.stage2.profiles``.
Dataset-specific schema/prompts live in ``language.lamp_*``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/tmp/hf_qwen06")

from language.stage2.collect import collect_rows as collect_rows_domain  # noqa: E402
from language.stage2.lamp_api import load_lamp_sub  # noqa: E402
from language.stage2.models import FactLatentCoT, SoftLSTMResid  # noqa: E402
from language.stage2.profiles import get_stage2_profile, list_stage2_domains  # noqa: E402

_spec_fs = importlib.util.spec_from_file_location(
    "fusion_shell_cfa", ROOT / "scripts/fusion_shell_cfa.py"
)
fs = importlib.util.module_from_spec(_spec_fs)
_spec_fs.loader.exec_module(fs)

# Module-level defaults for improve_latent_fuse import compatibility (traffic).
_DOMAIN = os.environ.get("STAGE2_DOMAIN", "traffic")
_profile0 = get_stage2_profile(_DOMAIN)
_schema0 = load_lamp_sub(_profile0, "schema")
CHANNELS = tuple(_schema0.CHANNELS)
C = len(CHANNELS)
FOCUS = _profile0.focus
OUT = ROOT / _profile0.out_dir


def mse_np(a, b) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def collect_rows(batch_size=8):
    """Back-compat for improve_latent_fuse: uses STAGE2_DOMAIN env (default traffic)."""
    profile = get_stage2_profile(os.environ.get("STAGE2_DOMAIN", "traffic"))
    train, test, H, meta = collect_rows_domain(ROOT, profile, batch_size=batch_size)
    global C, CHANNELS, FOCUS, OUT
    C = meta["C"]
    CHANNELS = meta["channels"]
    FOCUS = profile.focus
    OUT = ROOT / profile.out_dir
    return train, test, H


Z_CLIP = 3.0
# Val-selected amplitude gate on decoded Δ (primary Fact metric uses this).
AMP_GATE_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]


def attach_resid_norm(train_rows, *other_splits) -> dict:
    """Global residual z-score from TRAIN only; same (μ, σ) on train/val/test.

    What this does (and does *not* do):
      - Fit μ, σ on train residuals r = y_true − y_cfa only (no val/test leak).
      - Attach the *same* σ (and μ) to every window in train/val/test.
      - Model predicts ẑ; decode Δ = ẑ · σ + μ  (z clipped via tanh).
      - This is ordinary residual standardization, not per-window re-scaling.

    Previous bug: scale = σ_train × (‖y_cfa‖ / median), i.e. per-window amplitude
    tied to CFA size. That is *not* "normalize residual on all splits"; it made
    effective targets differ by window and mixed amplitude with normalization.

    Note: raw residual *MSE* can still differ across splits (data property).
    Normalization only puts the *learning target* on a shared z-scale; reporting
    and checkpointing use original-space MSE; amp gate further caps |Δ|.
    """
    res = np.concatenate(
        [
            (np.asarray(r["yt"], float) - np.asarray(r["yc"], float)).reshape(-1)
            for r in train_rows
        ]
    )
    mu = float(np.mean(res))
    std = float(np.std(res))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    b0_train = float(np.mean(res * res))

    n_attached = 0
    for split in (train_rows, *other_splits):
        for r in split:
            r["resid_scale"] = np.float32(std)
            r["resid_mean"] = np.float32(mu)
            n_attached += 1

    # Diagnostics: how B0 differs by split under the *same* global σ.
    def _b0(rows):
        if not rows:
            return None
        return float(
            np.mean(
                [
                    float(
                        np.mean(
                            (np.asarray(r["yt"], float) - np.asarray(r["yc"], float)) ** 2
                        )
                    )
                    for r in rows
                ]
            )
        )

    extras = list(other_splits)
    b0_val = _b0(extras[0]) if len(extras) >= 1 else None
    b0_test = _b0(extras[1]) if len(extras) >= 2 else None
    stats = {
        "mode": "global_train_mu_sigma",
        "resid_mean_train": mu,
        "resid_std_train": std,
        "resid_mse_train": b0_train,
        "resid_mse_val": b0_val,
        "resid_mse_test": b0_test,
        "n": n_attached,
        "z_clip": Z_CLIP,
        "note": (
            "Same μ,σ on train/val/test; loss+ckpt=original MSE; "
            "primary Fact metric applies val amp gate α·Δ"
        ),
    }
    print(
        f"resid_norm GLOBAL train μ={mu:.6f} σ={std:.6f} "
        f"B0_tr={b0_train:.6f} B0_va={b0_val} B0_te={b0_test}",
        flush=True,
    )
    return stats


class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        scale = float(r.get("resid_scale", 1.0) or 1.0)
        mean = float(r.get("resid_mean", 0.0) or 0.0)
        return {
            "ev": torch.from_numpy(r["ev"]),
            "yc": torch.from_numpy(r["yc"]),
            "yt": torch.from_numpy(r["yt"]),
            "mech": torch.from_numpy(r["mech"]),
            "resid_scale": torch.tensor([scale], dtype=torch.float32),
            "resid_mean": torch.tensor([mean], dtype=torch.float32),
            "prompt": r["prompt"],
            "wid": r["wid"],
        }


def collate_ev(batch):
    return {
        "ev": torch.stack([b["ev"] for b in batch]),
        "yc": torch.stack([b["yc"] for b in batch]),
        "yt": torch.stack([b["yt"] for b in batch]),
        "mech": torch.stack([b["mech"] for b in batch]),
        "resid_scale": torch.stack([b["resid_scale"] for b in batch]),
        "resid_mean": torch.stack([b["resid_mean"] for b in batch]),
        "wid": [b["wid"] for b in batch],
    }


def collate_text(batch, tokenizer, max_length):
    enc = tokenizer(
        [b["prompt"] for b in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "ev": torch.stack([b["ev"] for b in batch]),
        "yc": torch.stack([b["yc"] for b in batch]),
        "yt": torch.stack([b["yt"] for b in batch]),
        "mech": torch.stack([b["mech"] for b in batch]),
        "resid_scale": torch.stack([b["resid_scale"] for b in batch]),
        "resid_mean": torch.stack([b["resid_mean"] for b in batch]),
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "wid": [b["wid"] for b in batch],
    }


def _decode_delta(raw, scale, mean=None):
    """raw: (B, H); scale/mean: (B, 1). z clipped; Δ = z·σ + μ."""
    z = Z_CLIP * torch.tanh(raw / Z_CLIP)
    delta = z * scale
    if mean is not None:
        delta = delta + mean
    return z, delta


def split_tv(train_all):
    """Legacy: every 5th window by wid (in-distribution with train)."""
    order = sorted(range(len(train_all)), key=lambda i: train_all[i]["wid"])
    val_idx = set(order[::5])
    tr = [train_all[i] for i in order if i not in val_idx]
    va = [train_all[i] for i in order if i in val_idx]
    return tr, va


def split_tv_chrono(train_all, val_frac: float = 0.2):
    """Time-ordered hold-out: last val_frac of train (by wid) = val."""
    order = sorted(range(len(train_all)), key=lambda i: train_all[i]["wid"])
    n = len(order)
    n_val = max(1, int(round(n * float(val_frac))))
    n_val = min(n_val, n - 1) if n > 1 else n
    va_ids = set(order[-n_val:])
    tr = [train_all[i] for i in order if i not in va_ids]
    va = [train_all[i] for i in order if i in va_ids]
    return tr, va


def _prompt_has_facts(prompt: str) -> bool:
    """True if Fact prompt contains at least one event bullet line."""
    if not prompt:
        return False
    return any(ln.startswith("  ") and not ln.startswith("    ") for ln in prompt.splitlines())


def train_model(
    model,
    train_rows,
    val_rows,
    device,
    epochs,
    bs,
    lr,
    mode: str,
    max_length=256,
    *,
    weight_decay: float = 1e-4,
    delta_l2: float = 0.0,
    clip_target_z: float = 0.0,
):
    """Train residual head; loss + checkpoint both use original-space MSE."""
    if mode == "soft":

        def make_ld(rows, shuffle):
            return DataLoader(
                DS(rows), batch_size=bs, shuffle=shuffle, drop_last=shuffle, collate_fn=collate_ev
            )
    else:
        tok = model.tokenizer

        def make_ld(rows, shuffle):
            return DataLoader(
                DS(rows),
                batch_size=bs,
                shuffle=shuffle,
                drop_last=shuffle,
                collate_fn=lambda b: collate_text(b, tok, max_length),
            )

    tr_ld, va_ld = make_ld(train_rows, True), make_ld(val_rows, False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(weight_decay))
    best = None
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        trs_orig, trs_z = [], []
        for batch in tr_ld:
            yc = batch["yc"].to(device)
            yt = batch["yt"].to(device)
            scale = batch["resid_scale"].to(device)
            mean = batch["resid_mean"].to(device)
            if mode == "soft":
                raw = model(batch["ev"].to(device))
            else:
                raw = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["ev"].to(device),
                )
            z, delta = _decode_delta(raw, scale, mean)
            target_z = (yt - yc - mean) / scale
            if clip_target_z and float(clip_target_z) > 0:
                c = float(clip_target_z)
                target_z = target_z.clamp(-c, c)
                # Match decode path: supervise on clipped residual in original space.
                target_delta = target_z * scale + mean
                loss = ((yc + delta - (yc + target_delta)) ** 2).mean()
                # ≡ MSE(delta, target_delta)
                loss = ((delta - target_delta) ** 2).mean()
            else:
                loss = ((yc + delta - yt) ** 2).mean()
            if delta_l2 and float(delta_l2) > 0:
                loss = loss + float(delta_l2) * (delta ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            trs_orig.append(float(((yc + delta - yt) ** 2).mean().item()))
            trs_z.append(float(((z - target_z) ** 2).mean().item()))
        model.eval()
        vas_orig, vas_z = [], []
        with torch.no_grad():
            for batch in va_ld:
                yc = batch["yc"].to(device)
                yt = batch["yt"].to(device)
                scale = batch["resid_scale"].to(device)
                mean = batch["resid_mean"].to(device)
                if mode == "soft":
                    raw = model(batch["ev"].to(device))
                else:
                    raw = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["ev"].to(device),
                    )
                z, delta = _decode_delta(raw, scale, mean)
                target_z = (yt - yc - mean) / scale
                mse = ((yc + delta - yt) ** 2).mean()
                vas_orig.append(float(mse.item()))
                vas_z.append(float(((z - target_z) ** 2).mean().item()))
        tr_o, va_o = float(np.mean(trs_orig)), float(np.mean(vas_orig))
        tr_z, va_z = float(np.mean(trs_z)), float(np.mean(vas_z))
        hist.append(
            {
                "epoch": ep,
                "train": tr_o,
                "val": va_o,
                "train_z": tr_z,
                "val_z": va_z,
            }
        )
        print(
            f"  [{mode}] ep{ep:02d} train_mse={tr_o:.6f} val_mse={va_o:.6f} "
            f"train_z={tr_z:.4f} val_z={va_z:.4f}",
            flush=True,
        )
        if best is None or va_o < best[0]:
            best = (va_o, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1], strict=False)
    return hist, best[0]


@torch.no_grad()
def pred_soft(model, rows, device, bs=16):
    ld = DataLoader(DS(rows), batch_size=bs, shuffle=False, collate_fn=collate_ev)
    model.eval()
    out = {}
    for batch in ld:
        raw = model(batch["ev"].to(device))
        _z, delta = _decode_delta(
            raw,
            batch["resid_scale"].to(device),
            batch["resid_mean"].to(device),
        )
        for i, wid in enumerate(batch["wid"]):
            out[wid] = {
                "d_soft": delta[i].detach().cpu().numpy(),
                "mech": batch["mech"][i].numpy(),
                "yc": batch["yc"][i].numpy(),
                "yt": batch["yt"][i].numpy(),
            }
    return out


@torch.no_grad()
def pred_fact(model, rows, device, bs, max_length, *, empty_fact_zero: bool = False):
    ld = DataLoader(
        DS(rows),
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_text(b, model.tokenizer, max_length),
    )
    model.eval()
    # wid -> has_facts for empty-fact zeroing
    has_facts = {r["wid"]: _prompt_has_facts(r.get("prompt") or "") for r in rows}
    out = {}
    for batch in ld:
        raw = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["ev"].to(device),
        )
        _z, delta = _decode_delta(
            raw,
            batch["resid_scale"].to(device),
            batch["resid_mean"].to(device),
        )
        for i, wid in enumerate(batch["wid"]):
            d = delta[i].detach().cpu().numpy()
            if empty_fact_zero and not has_facts.get(wid, True):
                d = np.zeros_like(d)
            out[wid] = {"d_fact": d}
    return out


def apply_amp_gate(pred_map: dict, key: str, scale: float) -> dict:
    """Return copy with key scaled by val-selected amplitude gate."""
    out = {}
    for wid, v in pred_map.items():
        nv = dict(v)
        nv[key] = float(scale) * np.asarray(v[key], np.float32)
        out[wid] = nv
    return out


def score_delta(rows_map, key):
    ms, b0s = [], []
    for v in rows_map.values():
        b0s.append(mse_np(v["yc"], v["yt"]))
        ms.append(mse_np(v["yc"] + v[key], v["yt"]))
    b0 = float(np.mean(b0s))
    m = float(np.mean(ms))
    return {"mse": m, "rel": m / b0 - 1, "B0": b0}


def fuse_eval(mp, keys_weights: Dict[str, float]):
    ms, b0s = [], []
    for v in mp.values():
        delta = 0.0
        for k, w in keys_weights.items():
            delta = delta + w * v[k]
        b0s.append(mse_np(v["yc"], v["yt"]))
        ms.append(mse_np(v["yc"] + delta, v["yt"]))
    b0 = float(np.mean(b0s))
    m = float(np.mean(ms))
    return {"mse": m, "rel": m / b0 - 1}


def tune_fuse(val_mp, keys: List[str], grid=None):
    best = None
    grid = grid or [0.0, 0.25, 0.5, 0.75, 1.0]
    for ws in product(grid, repeat=len(keys)):
        if sum(ws) < 1e-9:
            continue
        kw = {k: float(w) for k, w in zip(keys, ws)}
        ms = [mse_np(v["yc"] + sum(kw[k] * v[k] for k in keys), v["yt"]) for v in val_mp.values()]
        m = float(np.mean(ms))
        if best is None or m < best[0]:
            best = (m, kw)
    return best


def tune_scale(val_mp, key: str, grid=None):
    grid = grid or [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
    best = None
    for s in grid:
        ms = [mse_np(v["yc"] + s * v[key], v["yt"]) for v in val_mp.values()]
        m = float(np.mean(ms))
        if best is None or m < best[0]:
            best = (m, float(s))
    return best


def heuristic_soft_exp_fact(mp):
    ms, b0s = [], []
    for v in mp.values():
        s, f, e = v["d_soft"], v["d_fact"], v["d_exp"]
        d = s + 0.4 * e
        if np.sign(np.mean(s)) == np.sign(np.mean(f)) and abs(np.mean(f)) > 0.01:
            d = d + 0.25 * f
        b0s.append(mse_np(v["yc"], v["yt"]))
        ms.append(mse_np(v["yc"] + d, v["yt"]))
    b0 = float(np.mean(b0s))
    m = float(np.mean(ms))
    return {"mse": m, "rel": m / b0 - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        default="traffic",
        choices=list_stage2_domains() + ["all", "both"],
        help="Single domain, or all/both = every registered Stage2 profile",
    )
    ap.add_argument("--model", default="/tmp/hf_qwen06/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-soft", type=float, default=1e-3)
    ap.add_argument("--lr-fact", type=float, default=1e-4)
    ap.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Fact LatentCoT truncate length; 0 = use profile.fact_max_length",
    )
    ap.add_argument(
        "--n-latent",
        type=int,
        default=0,
        help="Fact latent token count; 0 = profile.fact_n_latent (default 12).",
    )
    ap.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA on frozen Qwen (q/k/v/o_proj).",
    )
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument(
        "--no-amp-gate",
        action="store_true",
        help="Disable amplitude gate (primary Fact = ungated).",
    )
    ap.add_argument(
        "--amp-select",
        choices=["val", "test"],
        default="val",
        help="Split used to pick α for primary Fact metric. "
        "'test' is an oracle leak (diagnostic only).",
    )
    ap.add_argument(
        "--val-mode",
        choices=["stride", "chrono"],
        default="stride",
        help="stride=every 5th wid; chrono=last val-frac of train by time.",
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument(
        "--amp-cap",
        type=float,
        default=0.0,
        help="If >0, α_primary = min(α_selected, amp_cap).",
    )
    ap.add_argument(
        "--fixed-amp",
        type=float,
        default=-1.0,
        help="If >=0, use this α directly (skip val tune for primary).",
    )
    ap.add_argument("--delta-l2", type=float, default=0.0, help="L2 penalty on Δ during train.")
    ap.add_argument(
        "--clip-target-z",
        type=float,
        default=0.0,
        help="If >0, clip train target z to ±this (Stage1 residual mismatch mitigation).",
    )
    ap.add_argument(
        "--clip-target-z-q",
        type=float,
        default=0.0,
        help="If in (0,1], set clip_target_z = quantile(|train z|, q) (data-adaptive; overrides --clip-target-z if >0).",
    )
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Latent CoT steps; 0 = profile.fact_n_steps.",
    )
    ap.add_argument(
        "--learnable-amp",
        action="store_true",
        help="Trainable sigmoid amplitude inside FactLatentCoT.",
    )
    ap.add_argument(
        "--empty-fact-zero",
        action="store_true",
        help="Force Δ=0 when prompt has no event fact bullets.",
    )
    ap.add_argument(
        "--fuse-explicit",
        action="store_true",
        help="Primary Fact metric = val-tuned fuse of gated Fact + explicit mech.",
    )
    ap.add_argument("--tag", default="", help="Optional run tag written into summary.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-fact", action="store_true")
    ap.add_argument(
        "--skip-soft",
        action="store_true",
        help="Skip SoftLSTM (Fact LatentCoT + Explicit only).",
    )
    ap.add_argument(
        "--scale-tune",
        action="store_true",
        help="Also report extra soft/exp scale ablations.",
    )
    ap.add_argument(
        "--train-includes-test",
        action="store_true",
        help="ORACLE LEAK: append test rows into Fact train set (diagnostic upper bound).",
    )
    ap.add_argument(
        "--val-on-test",
        action="store_true",
        help="ORACLE LEAK: use test as val for ckpt/α (no train holdout).",
    )
    ap.add_argument(
        "--no-aux",
        action="store_true",
        help="Disable soft numeric aux tokens in Fact LatentCoT (text + latents only).",
    )
    ap.add_argument(
        "--coconut-recycle",
        action="store_true",
        help="Trick1: Coconut-style continuous thought recycle (no residual lat blend).",
    )
    ap.add_argument(
        "--adaptive-halt",
        action="store_true",
        help="Trick2: soft ACT-style halt weighted by fact density + halt head.",
    )
    ap.add_argument(
        "--n-thought-groups",
        type=int,
        default=1,
        help="Trick4: split latents into K parallel thought groups and attn-fuse (1=off).",
    )
    args = ap.parse_args()

    domains = (
        list_stage2_domains()
        if args.domain in ("all", "both")
        else [args.domain]
    )
    combined = {}
    for domain in domains:
        args.domain = domain
        combined[domain] = run_one_domain(args)

    if len(domains) > 1:
        path = ROOT / "artifacts/event_ts_assoc/summary_fair_branches_all.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\nwrote combined {path}", flush=True)


def run_one_domain(args) -> dict:
    os.environ["STAGE2_DOMAIN"] = args.domain
    profile = get_stage2_profile(args.domain)
    max_length = int(args.max_length) if int(args.max_length or 0) > 0 else int(profile.fact_max_length)
    n_latent = int(args.n_latent) if int(args.n_latent or 0) > 0 else int(profile.fact_n_latent)
    n_steps = int(args.n_steps) if int(getattr(args, "n_steps", 0) or 0) > 0 else int(profile.fact_n_steps)
    use_lora = bool(getattr(args, "lora", False))
    use_amp_gate = not bool(getattr(args, "no_amp_gate", False))
    amp_select = str(getattr(args, "amp_select", "val") or "val")
    val_mode = str(getattr(args, "val_mode", "stride") or "stride")
    val_frac = float(getattr(args, "val_frac", 0.2) or 0.2)
    amp_cap = float(getattr(args, "amp_cap", 0.0) or 0.0)
    fixed_amp = float(getattr(args, "fixed_amp", -1.0))
    delta_l2 = float(getattr(args, "delta_l2", 0.0) or 0.0)
    clip_target_z = float(getattr(args, "clip_target_z", 0.0) or 0.0)
    clip_target_z_q = float(getattr(args, "clip_target_z_q", 0.0) or 0.0)
    weight_decay = float(getattr(args, "weight_decay", 1e-4) or 1e-4)
    learnable_amp = bool(getattr(args, "learnable_amp", False))
    empty_fact_zero = bool(getattr(args, "empty_fact_zero", False))
    fuse_explicit = bool(getattr(args, "fuse_explicit", False))
    use_aux = not bool(getattr(args, "no_aux", False))
    coconut_recycle = bool(getattr(args, "coconut_recycle", False))
    adaptive_halt = bool(getattr(args, "adaptive_halt", False))
    n_thought_groups = int(getattr(args, "n_thought_groups", 1) or 1)
    run_tag = str(getattr(args, "tag", "") or "")
    print(
        f"fact_max_length={max_length} n_latent={n_latent} n_steps={n_steps} "
        f"lora={use_lora} amp_gate={use_amp_gate} amp_select={amp_select} "
        f"val_mode={val_mode} amp_cap={amp_cap} fixed_amp={fixed_amp} "
        f"delta_l2={delta_l2} clip_z={clip_target_z} learnable_amp={learnable_amp} "
        f"empty_zero={empty_fact_zero} fuse_exp={fuse_explicit} use_aux={use_aux} "
        f"coconut={coconut_recycle} halt={adaptive_halt} groups={n_thought_groups} "
        f"tag={run_tag}",
        flush=True,
    )
    do_scale = bool(args.scale_tune or profile.enable_scale_tune)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / profile.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_all, test_rows, H, meta = collect_rows_domain(ROOT, profile)
    global C
    C = meta["C"]
    train_includes_test = bool(getattr(args, "train_includes_test", False))
    val_on_test = bool(getattr(args, "val_on_test", False))
    if val_on_test:
        # No train holdout; ckpt/α selected on test (oracle).
        train_rows = list(train_all)
        val_rows = list(test_rows)
        print(
            f"ORACLE LEAK: val=test (n_val={len(val_rows)}); "
            f"no train holdout (n_train_base={len(train_rows)})",
            flush=True,
        )
    elif val_mode == "chrono":
        train_rows, val_rows = split_tv_chrono(train_all, val_frac=val_frac)
    else:
        train_rows, val_rows = split_tv(train_all)
    if train_includes_test:
        n_base = len(train_rows)
        train_rows = list(train_rows) + list(test_rows)
        print(
            f"ORACLE LEAK: train includes test "
            f"(base {n_base}+{len(test_rows)}={len(train_rows)}; "
            f"val_on_test={val_on_test} n_val={len(val_rows)})",
            flush=True,
        )
    resid_norm = attach_resid_norm(train_rows, val_rows, test_rows)
    resid_norm["train_includes_test"] = train_includes_test
    resid_norm["val_on_test"] = val_on_test
    if train_includes_test or val_on_test:
        resid_norm["leak_note"] = (
            "ORACLE diagnostic: Fact trained on train∪test; val=test for ckpt/α; "
            "test eval is an upper bound, not a valid score."
            if train_includes_test and val_on_test
            else (
                "ORACLE diagnostic: Fact trained on train∪test; test eval is an upper bound."
                if train_includes_test
                else "ORACLE diagnostic: val=test for ckpt/α; test eval is an upper bound."
            )
        )
    # Data-adaptive residual clip: quantile of |train z| (z already uses global μ,σ).
    if clip_target_z_q and 0.0 < clip_target_z_q <= 1.0:
        abs_z = []
        for r in train_rows:
            s = float(r.get("resid_scale", 1.0) or 1.0)
            m = float(r.get("resid_mean", 0.0) or 0.0)
            zz = (np.asarray(r["yt"], float) - np.asarray(r["yc"], float) - m) / max(s, 1e-8)
            abs_z.append(np.abs(zz).reshape(-1))
        abs_z = np.concatenate(abs_z) if abs_z else np.asarray([2.0])
        clip_target_z = float(np.quantile(abs_z, clip_target_z_q))
        clip_target_z = float(max(clip_target_z, 0.5))  # floor for stability
        resid_norm["clip_target_z_q"] = clip_target_z_q
        resid_norm["clip_target_z_adaptive"] = clip_target_z
        print(
            f"adaptive clip_target_z=q{clip_target_z_q} → ±{clip_target_z:.4f}",
            flush=True,
        )
    print(f"domain={profile.key} stage1={meta['stage1']} family={meta['family']}")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} H={H} C={C}")
    print(f"align_test_runpy={profile.align_test_runpy} n_events={meta['n_events']}")

    if getattr(args, "skip_soft", False):
        print("\n=== A) SoftLSTM SKIPPED ===", flush=True)

        def _stub_soft(rows):
            out = {}
            for r in rows:
                mech = r.get("mech")
                if mech is None:
                    mech = np.zeros(H, np.float32)
                else:
                    mech = np.asarray(mech, np.float32).reshape(-1)
                    if mech.shape[0] != H:
                        mech = np.zeros(H, np.float32)
                out[r["wid"]] = {
                    "yt": np.asarray(r["yt"], np.float32).reshape(-1),
                    "yc": np.asarray(r["yc"], np.float32).reshape(-1),
                    "d_soft": np.zeros(H, np.float32),
                    "mech": mech,
                }
            return out

        te_s = _stub_soft(test_rows)
        va_s = _stub_soft(val_rows)
        b0 = float(
            np.mean(
                [
                    float(np.mean((np.asarray(r["yc"], float) - np.asarray(r["yt"], float)) ** 2))
                    for r in test_rows
                ]
            )
        )
        hist_s, res_soft = [], {"mse": b0, "rel": 0.0, "B0": b0}
        print(f"SoftLSTM skipped B0={res_soft['B0']:.6f}", flush=True)
    else:
        print("\n=== A) SoftLSTM alone ===", flush=True)
        soft_m = SoftLSTMResid(
            H, C, hidden=profile.softlstm_hidden, n_token=profile.softlstm_n_token
        ).to(device)
        hist_s, _ = train_model(
            soft_m, train_rows, val_rows, device, args.epochs, 16, args.lr_soft, "soft"
        )
        te_s = pred_soft(soft_m, test_rows, device)
        va_s = pred_soft(soft_m, val_rows, device)
        res_soft = score_delta(te_s, "d_soft")
        print(f"SoftLSTM test rel={res_soft['rel']:+.3%}", flush=True)
        del soft_m

    if args.skip_fact:
        te_f = {wid: {"d_fact": np.zeros(H, np.float32)} for wid in te_s}
        va_f = {wid: {"d_fact": np.zeros(H, np.float32)} for wid in va_s}
        hist_f = []
        res_fact_ungated = {"mse": res_soft["B0"], "rel": 0.0, "B0": res_soft["B0"]}
        res_fact = {**res_fact_ungated, "amp_scale": 1.0}
        fact_cfg = {}
    else:
        print("\n=== B) Fact LatentCoT alone ===", flush=True)
        model_path = args.model if Path(args.model).exists() else "Qwen/Qwen3-0.6B"
        n_aux = int(profile.fact_n_aux) if use_aux else 0
        fact_m = FactLatentCoT(
            model_path,
            H,
            C,
            n_aux=n_aux,
            n_latent=n_latent,
            n_steps=n_steps,
            max_length=max_length,
            use_aux=use_aux,
            use_lora=use_lora,
            lora_r=int(getattr(args, "lora_r", 8)),
            lora_alpha=int(getattr(args, "lora_alpha", 16)),
            learnable_amp=learnable_amp,
            coconut_recycle=coconut_recycle,
            adaptive_halt=adaptive_halt,
            n_thought_groups=n_thought_groups,
        ).to(device)
        fact_cfg = {
            "n_latent": n_latent,
            "n_aux": n_aux,
            "use_aux": use_aux,
            "n_steps": n_steps,
            "use_lora": use_lora,
            "lora_r": int(getattr(args, "lora_r", 8)),
            "lora_alpha": int(getattr(args, "lora_alpha", 16)),
            "learnable_amp": learnable_amp,
            "coconut_recycle": coconut_recycle,
            "adaptive_halt": adaptive_halt,
            "n_thought_groups": n_thought_groups,
            "params": fact_m.trainable_param_count(),
            "lr": float(args.lr_fact),
            "delta_l2": delta_l2,
            "clip_target_z": clip_target_z,
            "weight_decay": weight_decay,
            "val_mode": val_mode,
            "val_frac": val_frac,
            "amp_cap": amp_cap,
            "fixed_amp": fixed_amp,
            "empty_fact_zero": empty_fact_zero,
            "fuse_explicit": fuse_explicit,
            "tag": run_tag,
            "train_includes_test": train_includes_test,
            "val_on_test": val_on_test,
        }
        print(f"Fact trainable params={fact_cfg['params']}", flush=True)
        hist_f, best_val_mse = train_model(
            fact_m,
            train_rows,
            val_rows,
            device,
            args.epochs,
            args.batch_size,
            args.lr_fact,
            "fact",
            max_length,
            weight_decay=weight_decay,
            delta_l2=delta_l2,
            clip_target_z=clip_target_z,
        )
        fact_cfg["best_val_mse"] = float(best_val_mse)
        if learnable_amp:
            fact_cfg["learned_amp"] = float(fact_m.amp_value())
            print(f"Fact learned_amp={fact_cfg['learned_amp']:.4f}", flush=True)
        te_f = pred_fact(
            fact_m,
            test_rows,
            device,
            args.batch_size,
            max_length,
            empty_fact_zero=empty_fact_zero,
        )
        va_f = pred_fact(
            fact_m,
            val_rows,
            device,
            args.batch_size,
            max_length,
            empty_fact_zero=empty_fact_zero,
        )
        for wid in te_f:
            te_f[wid].update({k: te_s[wid][k] for k in ("yc", "yt", "mech")})
        for wid in va_f:
            va_f[wid].update({k: va_s[wid][k] for k in ("yc", "yt", "mech")})
        res_fact_ungated = score_delta(
            {wid: {**te_s[wid], "d_fact": te_f[wid]["d_fact"]} for wid in te_s},
            "d_fact",
        )
        print(
            f"FactLatentCoT ungated test rel={res_fact_ungated['rel']:+.3%}",
            flush=True,
        )
        del fact_m
        torch.cuda.empty_cache()

    # Build maps; Fact primary path applies val amp gate on Δ.
    te_e_raw = {
        wid: {
            "d_exp": te_s[wid]["mech"],
            "yc": te_s[wid]["yc"],
            "yt": te_s[wid]["yt"],
            "d_soft": te_s[wid]["d_soft"],
            "d_fact": te_f[wid]["d_fact"],
        }
        for wid in te_s
    }
    va_e_raw = {
        wid: {
            "d_exp": va_s[wid]["mech"],
            "yc": va_s[wid]["yc"],
            "yt": va_s[wid]["yt"],
            "d_soft": va_s[wid]["d_soft"],
            "d_fact": va_f[wid]["d_fact"],
        }
        for wid in va_s
    }

    # Amplitude gates: always score both val-α and test-α (oracle) when Fact ran.
    res_fact_val_gated = None
    res_fact_test_oracle = None
    amp_scale_val = 1.0
    amp_scale_test = 1.0
    if use_amp_gate and not args.skip_fact:
        sc_val = tune_scale(va_e_raw, "d_fact", grid=AMP_GATE_GRID)
        sc_te = tune_scale(te_e_raw, "d_fact", grid=AMP_GATE_GRID)  # oracle leak
        amp_scale_val = float(sc_val[1])
        amp_scale_test = float(sc_te[1])
        if fixed_amp >= 0:
            amp_scale = float(fixed_amp)
            amp_src = "fixed"
        elif amp_select == "test":
            amp_scale = amp_scale_test
            amp_src = "test_oracle"
        else:
            amp_scale = amp_scale_val
            amp_src = "val"
        if amp_cap and amp_cap > 0:
            amp_scale = float(min(amp_scale, amp_cap))
            amp_src = f"{amp_src}+cap{amp_cap}"

        te_val = apply_amp_gate(te_e_raw, "d_fact", amp_scale_val)
        te_ora = apply_amp_gate(te_e_raw, "d_fact", amp_scale_test)
        te_pri = apply_amp_gate(te_e_raw, "d_fact", amp_scale)
        res_fact_val_gated = {
            **score_delta(te_val, "d_fact"),
            "amp_scale": amp_scale_val,
            "val_mse_gated": float(sc_val[0]),
            "select": "val",
            "ungated": res_fact_ungated,
        }
        res_fact_test_oracle = {
            **score_delta(te_ora, "d_fact"),
            "amp_scale": amp_scale_test,
            "test_mse_gated": float(sc_te[0]),
            "select": "test",
            "note": "ORACLE: α chosen on test (leaked); diagnostic upper bound",
            "ungated": res_fact_ungated,
        }
        te_e = apply_amp_gate(te_e_raw, "d_fact", amp_scale)
        va_e = apply_amp_gate(va_e_raw, "d_fact", amp_scale)
        res_fact = {
            **score_delta(te_pri, "d_fact"),
            "amp_scale": amp_scale,
            "amp_select": amp_src,
            "ungated": res_fact_ungated,
        }
        if fuse_explicit:
            # Val-tune fuse of gated Fact + explicit mech (scheme1 mech proxy / s5-ready).
            best_fe_g = tune_fuse(va_e, ["d_fact", "d_exp"])
            res_fuse = fuse_eval(te_e, best_fe_g[1])
            res_fact = {
                **res_fuse,
                "amp_scale": amp_scale,
                "amp_select": amp_src,
                "fuse_weights": best_fe_g[1],
                "fact_alone": dict(res_fact),
                "ungated": res_fact_ungated,
                "note": "primary = val-tuned fuse(gated Fact, explicit mech)",
            }
        print(
            f"FactLatentCoT α_val={amp_scale_val:.2f} → test rel="
            f"{res_fact_val_gated['rel']:+.3%} | "
            f"α_test(oracle)={amp_scale_test:.2f} → test rel="
            f"{res_fact_test_oracle['rel']:+.3%} | "
            f"primary α={amp_scale:.2f} ({amp_src}) rel={res_fact['rel']:+.3%} "
            f"ungated={res_fact_ungated['rel']:+.3%}",
            flush=True,
        )
    else:
        amp_scale = 1.0
        te_e, va_e = te_e_raw, va_e_raw
        res_fact = {
            **res_fact_ungated,
            "amp_scale": 1.0,
            "amp_select": "none",
            "ungated": res_fact_ungated,
        }

    res_exp = score_delta(te_e, "d_exp")
    print(f"Explicit scheme1_hier test rel={res_exp['rel']:+.3%}", flush=True)

    best_sf = tune_fuse(va_e, ["d_soft", "d_fact"])
    best_se = tune_fuse(va_e, ["d_soft", "d_exp"])
    best_fe = tune_fuse(va_e, ["d_fact", "d_exp"])
    best_all = tune_fuse(va_e, ["d_soft", "d_fact", "d_exp"])
    res_sf = fuse_eval(te_e, best_sf[1])
    res_se = fuse_eval(te_e, best_se[1])
    res_fe = fuse_eval(te_e, best_fe[1])
    res_all = fuse_eval(te_e, best_all[1])
    res_heur = heuristic_soft_exp_fact(te_e)

    out = {
        "domain": profile.key,
        "stage1": meta["stage1"],
        "family": meta["family"],
        "eval_note": (
            "test=runpy drop_last" if profile.align_test_runpy else "test=all windows"
        ),
        "B0_test": res_soft["B0"],
        "resid_norm": resid_norm,
        "fact_cfg": fact_cfg,
        "n_test": len(test_rows),
        "n_events": meta["n_events"],
        "C": C,
        "fact_prompt": {
            "keep_frac": None,
            "max_months": int(profile.fact_max_months),
            "max_per_month": int(profile.fact_max_per_month),
            "n_latent": n_latent,
            "n_aux": int(profile.fact_n_aux) if use_aux else 0,
            "use_aux": use_aux,
            "n_steps": int(profile.fact_n_steps),
            "max_length": int(max_length),
            "use_lora": use_lora,
            "amp_gate": use_amp_gate,
            "amp_select": amp_select,
            "amp_scale": amp_scale,
            "profile_fact_max_length": int(profile.fact_max_length),
            "note": (
                "global train μ/σ resid norm; loss+ckpt=raw MSE; "
                "primary Fact amp gate via --amp-select; "
                "always also logs B_fact_amp_test_oracle (leaked)"
            ),
        },
        "A_soft_lstm_alone": res_soft,
        "B_fact_latentcot_alone": res_fact,
        "B_fact_latentcot_ungated": res_fact_ungated,
        "B_fact_amp_val": res_fact_val_gated,
        "B_fact_amp_test_oracle": res_fact_test_oracle,
        "C_explicit_scheme1_hier": res_exp,
        "C_explicit_soft_mech": res_exp,
        "fuse_soft_fact": {**res_sf, "weights": best_sf[1], "val_mse": best_sf[0]},
        "fuse_soft_exp": {**res_se, "weights": best_se[1], "val_mse": best_se[0]},
        "fuse_fact_exp": {**res_fe, "weights": best_fe[1], "val_mse": best_fe[0]},
        "fuse_all_three": {**res_all, "weights": best_all[1], "val_mse": best_all[0]},
        "heuristic_soft+0.4exp+gatedFact": res_heur,
        "hist_soft": hist_s,
        "hist_fact": hist_f,
    }

    if do_scale:
        sc_s = tune_scale(
            va_e_raw, "d_soft", grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35]
        )
        sc_e = tune_scale(va_e_raw, "d_exp", grid=[0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0])
        # Fact already gated; keep B_fact_val_scaled as alias of primary.
        out["A_soft_lstm_val_scaled"] = {
            **score_delta(apply_amp_gate(te_e_raw, "d_soft", sc_s[1]), "d_soft"),
            "scale": sc_s[1],
            "val_mse": sc_s[0],
        }
        out["B_fact_val_scaled"] = {
            "mse": res_fact["mse"],
            "rel": res_fact["rel"],
            "B0": res_fact.get("B0", res_soft["B0"]),
            "scale": amp_scale,
            "val_mse": res_fact.get("val_mse_gated"),
            "note": "alias of B_fact_latentcot_alone (amp gate always on)",
        }
        out["C_explicit_val_scaled"] = {
            **score_delta(apply_amp_gate(te_e_raw, "d_exp", sc_e[1]), "d_exp"),
            "scale": sc_e[1],
            "val_mse": sc_e[0],
        }

    summary_name = "summary_fair_branches_fact40.json"
    (out_dir / summary_name).write_text(json.dumps(out, indent=2))
    (out_dir / "summary_fair_branches.json").write_text(json.dumps(out, indent=2))
    if run_tag:
        tag_dir = ROOT / "artifacts" / "stage2_fact_improve_ablate" / run_tag / profile.key
        tag_dir.mkdir(parents=True, exist_ok=True)
        (tag_dir / "summary.json").write_text(json.dumps(out, indent=2))
        print(f"Wrote tagged {tag_dir / 'summary.json'}", flush=True)
    print(f"\n=== FAIR SUMMARY ({profile.key}) ===", flush=True)
    for k, v in out.items():
        if isinstance(v, dict) and "rel" in v:
            extra = ""
            if k == "B_fact_latentcot_alone" and "amp_scale" in v:
                extra = f" α={v['amp_scale']:.2f}"
            print(
                f"  {k:36s} rel={v['rel']:+.3%} mse={v['mse']:.6f}{extra}",
                flush=True,
            )
    print("Wrote", out_dir / summary_name, flush=True)
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
