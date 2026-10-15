"""Scheme5 v2 helpers: dead-channel mute, calibrated fuse/amp, rewrite glue.

Protocol seeds live in ``scheme5_seeds.py`` (per-dataset). This module only
keeps fusion/calibration and thin rewrite wrappers used by ``eval_scheme5.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_META_LEAK = re.compile(
    r"(?i)improved\s*=\s*(True|False)|wid=|n_ev=|cov=|resid_sign="
)
_VMT_LEAK = re.compile(
    r"(?i)\bVMT\b|vehicle miles|Do not invent numeric VMT|numeric VMT levels"
)


def extract_lesson_bullets(protocol: str) -> list[str]:
    text = protocol or ""
    m = re.search(r"(?is)Accumulated lessons[^\n]*:\n(.*)$", text)
    body = m.group(1) if m else text
    bullets: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("-", "•", "*")):
            bullets.append(re.sub(r"^[-•*]\s*", "", s).strip())
        elif re.match(r"^\d+[.)]\s+", s):
            bullets.append(re.sub(r"^\d+[.)]\s+", "", s).strip())
        elif bullets and not s.startswith(("Standing", "CHANNEL", "Role:", "Target")):
            bullets[-1] = (bullets[-1] + " " + s).strip()
    return [b for b in bullets if b and "none yet" not in b.lower()]


def prompt_rewrite_improve(
    *,
    channel: str,
    target: str,
    domain: str,
    old_protocol: str,
    outcomes_block: str,
    max_lessons: int = 20,
) -> str:
    """Delegate to per-dataset rewrite schema (no hardcoded VMT)."""
    from language.stage2.scheme5_seeds import prompt_rewrite_from_spec

    _ = target  # target is embedded in DomainSeedSpec
    return prompt_rewrite_from_spec(
        channel=channel,
        domain=domain,
        old_protocol=old_protocol,
        outcomes_block=outcomes_block,
        max_lessons=max_lessons,
    )


def outcome_line_clean(
    *,
    wid: str,
    channel: str,
    n_events: int,
    act: dict,
    improved: bool,
    resid_sign: int,
) -> str:
    """Outcome string for rewrite — no raw improved= leakage into lessons."""
    cov = act["n_explained"] / max(n_events, 1)
    if act["gate"] == 0:
        tag = "abstain"
    elif improved:
        tag = "hit"
    else:
        tag = "false_alarm"
    # resid_sign vs predicted direction agreement
    pred = str(act.get("dir") or "flat")
    return (
        f"- wid={wid} ch={channel} tag={tag} n_ev={n_events} cov={cov:.2f} "
        f"gate={act['gate']} dir={pred} true_resid_sign={resid_sign:+d} "
        f"cot={act.get('cot', '')}"
    )


# ---------------------------------------------------------------------------
# Dead channel + calibrated fusion / amplitude
# ---------------------------------------------------------------------------


@dataclass
class FuseCalib:
    soft_amp: float = 1.0
    dead_sr: float = 0.08  # mute channels below this success rate
    weight_mode: str = "success"  # mean | success | success_sqrt
    min_weight: float = 0.15


def channel_weights(
    success: Mapping[str, Mapping[str, int]],
    channels: Sequence[str],
    cfg: FuseCalib,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in channels:
        st = success.get(ch) or {}
        ni = int(st.get("n_infer") or 0)
        ns = int(st.get("n_success") or 0)
        sr = (ns / ni) if ni > 0 else 0.0
        # dead_sr<=0 ⇒ never mute by success-rate (per-dataset search may pick 0.0)
        if cfg.dead_sr > 0 and ni >= 40 and sr < cfg.dead_sr:
            out[ch] = 0.0
            continue
        if cfg.weight_mode == "mean":
            w = 1.0
        elif cfg.weight_mode == "success_sqrt":
            w = max(cfg.min_weight, float(np.sqrt(max(sr, 0.0))))
        else:  # success
            w = max(cfg.min_weight, float(sr))
        out[ch] = w
    return out


def fuse_weighted(
    ch_vecs: Mapping[str, np.ndarray],
    ch_gates: Mapping[str, int],
    weights: Mapping[str, float],
    H: int,
) -> np.ndarray:
    num = np.zeros(H, dtype=np.float64)
    den = 0.0
    for ch, v in ch_vecs.items():
        g = int(ch_gates.get(ch, 0))
        w = float(weights.get(ch, 1.0))
        if not g or w <= 0 or not np.any(np.abs(v) > 1e-12):
            continue
        num += w * np.asarray(v, dtype=np.float64)
        den += w
    if den <= 0:
        return num
    return num / den


def score_deltas(
    rows: Sequence[dict],
    deltas: Mapping[str, np.ndarray],
) -> dict[str, float]:
    from scripts.fusion_shell_cfa import mse

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
        "n": float(len(rows)),
        "active": float(act),
        "frac_active": act / max(len(rows), 1),
        "mse_b0": b0,
        "mse_s5": m1,
        "rel_pct": rel,
    }


def _chrono_split(
    rows: Sequence[dict], val_frac: float = 0.2
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda r: (str(r.get("origin") or ""), r["wid"]))
    n = len(ordered)
    n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
    if n_val <= 0:
        return list(ordered), []
    return list(ordered[:-n_val]), list(ordered[-n_val:])


def search_fuse_amp(
    train_rows: Sequence[dict],
    test_rows: Sequence[dict],
    # wid -> ch -> vec, gate
    per_window: Mapping[str, Mapping[str, dict]],
    success: Mapping[str, Mapping[str, int]],
    focus: Sequence[str],
    H: int,
    *,
    amp_grid: Sequence[float] = (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    weight_modes: Sequence[str] = ("mean", "success", "success_sqrt"),
    # 0.0 = no SR-mute; higher = more aggressive mute (searched per dataset on val)
    dead_srs: Sequence[float] = (0.0, 0.05, 0.08, 0.12, 0.15, 0.20),
    select_on: str = "val",
    val_frac: float = 0.2,
) -> dict[str, Any]:
    """Grid-search fusion weights + soft_amp + dead_sr (per dataset call).

    ``dead_sr``: mute channels with success-rate below threshold (and n_infer≥40).
    Searched independently for each domain on ``select_on`` split.

    ``select_on``:
      - ``val`` (default): chrono hold-out from train (last ``val_frac``).
      - ``test``: pick by test ``rel_pct`` — leaked / optimistic (debug only).
      - ``train``: pick by train ``rel_pct`` (overfits; not recommended).
    """
    select_on = (select_on or "val").lower().strip()
    if select_on not in {"test", "val", "train"}:
        select_on = "val"

    fit_rows, val_rows = _chrono_split(train_rows, val_frac=val_frac)
    if not val_rows:
        fit_rows, val_rows = list(train_rows), list(train_rows)

    def _fuse_rows(rows, wts, amp):
        out: dict[str, np.ndarray] = {}
        for r in rows:
            wid = r["wid"]
            chmap = per_window.get(wid) or {}
            vecs = {ch: chmap[ch]["vec"] for ch in chmap}
            gates = {ch: chmap[ch]["gate"] for ch in chmap}
            out[wid] = fuse_weighted(vecs, gates, wts, H) * float(amp)
        return out

    base_w = {ch: 1.0 for ch in focus}
    base_te = score_deltas(test_rows, _fuse_rows(test_rows, base_w, 1.0))
    base_tr = score_deltas(train_rows, _fuse_rows(train_rows, base_w, 1.0))
    base_val = score_deltas(val_rows, _fuse_rows(val_rows, base_w, 1.0))

    def _sel_score(tr_sc, te_sc, val_sc) -> float:
        if select_on == "train":
            return float(tr_sc["rel_pct"])
        if select_on == "val":
            return float(val_sc["rel_pct"])
        return float(te_sc["rel_pct"])

    best: Optional[dict] = None
    for wm in weight_modes:
        for dsr in dead_srs:
            cfg = FuseCalib(weight_mode=wm, dead_sr=dsr)
            wts = channel_weights(success, focus, cfg)
            for amp in amp_grid:
                tr_sc = score_deltas(train_rows, _fuse_rows(train_rows, wts, amp))
                te_sc = score_deltas(test_rows, _fuse_rows(test_rows, wts, amp))
                val_sc = score_deltas(val_rows, _fuse_rows(val_rows, wts, amp))
                cand = {
                    "weight_mode": wm,
                    "dead_sr": dsr,
                    "soft_amp": amp,
                    "weights": wts,
                    "fit_n": len(fit_rows),
                    "select_on": select_on,
                    "val": val_sc,
                    "val_baseline_mean": base_val,
                    "train": tr_sc,
                    "train_baseline_mean": base_tr,
                    "test": te_sc,
                    "test_baseline_mean": base_te,
                    "muted": [ch for ch, w in wts.items() if w <= 0],
                }
                key = _sel_score(tr_sc, te_sc, val_sc)
                if best is None or key < _sel_score(best["train"], best["test"], best["val"]):
                    best = cand
    if best is None:
        best = {
            "weight_mode": "mean",
            "dead_sr": 0.0,
            "soft_amp": 1.0,
            "weights": base_w,
            "fit_n": len(fit_rows),
            "select_on": select_on,
            "val": base_val,
            "val_baseline_mean": base_val,
            "train": base_tr,
            "train_baseline_mean": base_tr,
            "test": base_te,
            "test_baseline_mean": base_te,
            "muted": [],
            "note": "fallback mean@1",
        }
    else:
        best["note"] = (
            f"selected_on={select_on} (test metrics are optimistic / leaked)"
            if select_on == "test"
            else f"selected_on={select_on}"
        )
    return best


def protocol_quality_report(protocol: str, *, domain: str, target: str) -> dict[str, Any]:
    bullets = extract_lesson_bullets(protocol)
    forbidden = False
    try:
        from language.stage2.scheme5_seeds import protocol_has_forbidden

        forbidden = protocol_has_forbidden(protocol, domain)
    except Exception:
        allow_vmt = domain == "traffic" or "VMT" in (target or "")
        forbidden = bool((not allow_vmt) and _VMT_LEAK.search(protocol or ""))
    return {
        "n_lessons": len(bullets),
        "chars": len(protocol or ""),
        "forbidden_leak": forbidden,
        "vmt_leak": forbidden,  # backward-compatible alias
        "meta_leak": bool(_META_LEAK.search(protocol or "")),
        "if_then_frac": (
            sum(1 for b in bullets if re.match(r"(?i)^if\b", b) or "IF " in b) / max(len(bullets), 1)
        ),
    }
