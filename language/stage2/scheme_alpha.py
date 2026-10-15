"""Explicit branch **Candidate 3** — scheme α: vLLM KEEP + learnable readout.

Pipeline:
  1) Remote vLLM produces explicit residual CoT / KEEP over Stage1 mechanism bank
     (reuse Readable Mech prompts + caches when present).
  2) Pack φ(KEEP, channel, plaus, soft prior, CFA).
  3) Ridge head φ → Δ∈R^H on train residual; val picks soft/readout blend + scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language.stage2.mech_features import pack_keep_features, soft_delta_from_selection
from language.stage2.mech_keep import (
    SEED_PROMPTS,
    append_jsonl_cache,
    build_mech_bank,
    load_jsonl_cache,
    mse,
    parse_keep_response,
    render_actor_prompt,
    reward_delta,
)


@dataclass
class SchemeAlphaConfig:
    arm: str = "select_residual_v1"
    soft_amp: float = 0.05
    ridge_alpha: float = 2.0
    temperature: float = 0.2
    max_tokens: int = 400
    # Δ = (1-λ)*soft + λ*readout ; λ and global scale chosen on val
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5)
    scale_grid: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0, 1.25)
    # kept for CLI compat; unused (Ridge has no epochs)
    epochs: int = 0


def _as_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "text"):
        return str(x.text)
    return str(x)


def ensure_keep_cache(
    rows: Sequence[dict[str, Any]],
    packs: Sequence[dict[str, Any]],
    *,
    profile,
    arm_name: str,
    template: str,
    client,
    cache_path: Path,
    fill_missing: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, str]:
    bucket = load_jsonl_cache(cache_path) if cache_path.exists() else {}
    missing = [r["wid"] for r in rows if r["wid"] not in bucket]
    if missing and fill_missing and client is not None:
        by_wid = {p["row"]["wid"]: p for p in packs}
        prompts, order = [], []
        for wid in missing:
            p = by_wid.get(wid)
            if not p:
                continue
            prompts.append(
                render_actor_prompt(
                    template,
                    target=profile.target_series_short,
                    origin=p["row"].get("origin") or "",
                    cfa=p["row"]["yc"],
                    bank=p["bank"],
                )
            )
            order.append(wid)
        sys = (
            "You select human-readable residual mechanisms. "
            "JSON only. No markdown. Cite bank ids exactly."
        )
        chunk = 24
        for i in range(0, len(prompts), chunk):
            pr = prompts[i : i + chunk]
            wids = order[i : i + chunk]
            print(f"  [alpha/{arm_name}] LLM fill {i}/{len(prompts)}", flush=True)
            resps = client.ask_many(
                pr,
                system=sys,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=min(16, len(pr)),
            )
            for wid, text in zip(wids, resps):
                text = _as_text(text)
                bucket[wid] = text
                append_jsonl_cache(cache_path, wid, text)
    return bucket


def attach_scheme_alpha(
    rows: list[dict[str, Any]],
    by_ym: Mapping[str, Sequence[dict]],
    profile,
    H: int,
    *,
    client=None,
    cache_dir: Optional[Path] = None,
    train_wids: Optional[set[str]] = None,
    val_wids: Optional[set[str]] = None,
    channels: Sequence[str] = (),
    cfg: Optional[SchemeAlphaConfig] = None,
    fill_missing: bool = True,
    reuse_readable_cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    cfg = cfg or SchemeAlphaConfig()
    train_set = train_wids or {r["wid"] for r in rows if str(r["wid"]).startswith("train:")}
    val_set = val_wids or set()
    focus = profile.focus
    chans = tuple(channels) if channels else tuple(focus)

    packs = []
    for r in rows:
        packs.append({"row": r, "bank": build_mech_bank(by_ym, r.get("months") or [], focus)})

    template = SEED_PROMPTS.get(cfg.arm) or SEED_PROMPTS["select_residual_v1"]
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        primary = cache_dir / f"alpha_keep_{cfg.arm}.jsonl"
    else:
        primary = Path("/tmp/alpha_keep.jsonl")

    bucket: dict[str, str] = {}
    if reuse_readable_cache_dir:
        legacy = Path(reuse_readable_cache_dir) / f"readable_keep_{cfg.arm}.jsonl"
        if legacy.is_file():
            bucket.update(load_jsonl_cache(legacy))
            print(f"  [alpha] seeded {len(bucket)} KEEP from {legacy}", flush=True)
    if primary.exists():
        bucket.update(load_jsonl_cache(primary))
    if cache_dir and not primary.exists() and bucket:
        for wid, text in bucket.items():
            append_jsonl_cache(primary, wid, text)

    bucket = ensure_keep_cache(
        rows,
        packs,
        profile=profile,
        arm_name=cfg.arm,
        template=template,
        client=client,
        cache_path=primary,
        fill_missing=fill_missing,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    feats, resid, softs, metas = {}, {}, {}, {}
    for p in packs:
        r = p["row"]
        wid = r["wid"]
        sel = parse_keep_response(bucket.get(wid, ""))
        x = pack_keep_features(
            p["bank"], sel, r["yc"], H, channels=chans, soft_amp=cfg.soft_amp
        )
        y = (np.asarray(r["yt"], float) - np.asarray(r["yc"], float)).astype(np.float32)
        soft = soft_delta_from_selection(p["bank"], sel, H, cfg.soft_amp)
        feats[wid] = x
        resid[wid] = y
        softs[wid] = soft
        metas[wid] = {"sel": sel, "n_keep": len(sel.get("keep") or []), "gate": sel.get("gate")}

    train_ids = [w for w in feats if w in train_set]
    val_ids = [w for w in feats if w in val_set] or train_ids[::5]
    Xtr = np.stack([feats[w] for w in train_ids]).astype(np.float32)
    Ytr = np.stack([resid[w] for w in train_ids]).astype(np.float32)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(Ridge(alpha=cfg.ridge_alpha))),
        ]
    )
    # Learn correction on top of soft prior (reduces Agri overfit vs raw residual fit).
    Soft_tr = np.stack([softs[w] for w in train_ids]).astype(np.float32)
    Y_corr = Ytr - Soft_tr
    model.fit(Xtr, Y_corr)

    def predict_raw(wid: str) -> np.ndarray:
        corr = model.predict(feats[wid].reshape(1, -1)).reshape(-1).astype(np.float32)
        return (softs[wid] + corr).astype(np.float32)

    by_wid_row = {p["row"]["wid"]: p["row"] for p in packs}
    best_lam, best_scale, best_vm = 0.0, 1.0, 1e18
    soft_only_best_s, soft_only_vm = 1.0, 1e18
    for lam in cfg.blend_grid:
        for s in cfg.scale_grid:
            ms = []
            for wid in val_ids:
                row = by_wid_row[wid]
                soft = softs[wid]
                pred = predict_raw(wid)
                # lam=0 → soft only; lam=1 → soft+correction
                d = ((1.0 - lam) * soft + lam * pred) * float(s)
                ms.append(mse(row["yc"] + d, row["yt"]))
            vm = float(np.mean(ms)) if ms else 1e18
            if lam == 0.0 and vm < soft_only_vm:
                soft_only_vm, soft_only_best_s = vm, float(s)
            if vm < best_vm:
                best_vm, best_lam, best_scale = vm, float(lam), float(s)
    # Require clear val gain vs soft-only; else stay on soft (anti-overfit).
    if best_lam > 0 and best_vm > soft_only_vm * 0.95:
        best_lam, best_scale, best_vm = 0.0, soft_only_best_s, soft_only_vm

    for p in packs:
        r = p["row"]
        wid = r["wid"]
        soft = softs[wid]
        pred = predict_raw(wid)
        d = (((1.0 - best_lam) * soft + best_lam * pred) * best_scale).astype(np.float32)
        r["mech_alpha"] = d
        r["mech_alpha_raw"] = pred
        r["mech_alpha_soft"] = soft
        r["mech_alpha_meta"] = {
            **metas[wid],
            "scale": best_scale,
            "blend_lambda": best_lam,
            "arm": cfg.arm,
            "head": "ridge",
        }

    tr_r = []
    for wid in train_ids[: min(80, len(train_ids))]:
        row = by_wid_row[wid]
        tr_r.append(reward_delta(row["yc"], row["yt"], row["mech_alpha"]))

    return {
        "stats": {
            "arm": cfg.arm,
            "head": "ridge",
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "feat_dim": int(Xtr.shape[1]),
            "blend_lambda": best_lam,
            "best_scale": best_scale,
            "best_val_mse": best_vm,
            "train_reward_mean": float(np.mean(tr_r)) if tr_r else 0.0,
            "cache": str(primary),
        }
    }
