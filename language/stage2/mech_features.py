"""Shared KEEP feature packing for scheme α readout and Q-agent."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.stage2.mech_keep import (
    DIR_SIGN,
    STRENGTH_W,
    MechCard,
    delta_from_selection,
    soft_signed_from_cards,
)

HYP_OH = ("increase", "decrease", "flat", "mixed")


def _cfa_feats(yc: Sequence[float]) -> np.ndarray:
    y = np.asarray(yc, float).reshape(-1)
    if y.size == 0:
        return np.zeros(5, np.float32)
    slope = float(y[-1] - y[0]) / max(len(y) - 1, 1)
    return np.asarray(
        [float(y.mean()), float(y.std() + 1e-8), float(y[0]), float(y[-1]), slope],
        np.float32,
    )


def selection_keep_cards(
    bank: Sequence[MechCard],
    selection: Mapping[str, Any],
) -> list[MechCard]:
    by_id = {c.mid: c for c in bank}
    out: list[MechCard] = []
    for it in selection.get("keep") or []:
        mid = str(it.get("id") or "").strip()
        if not mid:
            continue
        if mid not in by_id:
            hit = next((k for k in by_id if k.endswith(mid) or mid.endswith(k)), None)
            if not hit:
                continue
            mid = hit
        out.append(by_id[mid])
    return out


def pack_keep_features(
    bank: Sequence[MechCard],
    selection: Mapping[str, Any],
    yc: Sequence[float],
    H: int,
    *,
    channels: Sequence[str] = (),
    soft_amp: float = 0.05,
) -> np.ndarray:
    """Compact φ(KEEP, CFA) for learnable readout / Q head."""
    gate = float(int(selection.get("gate") or 0))
    conf = float(selection.get("conf") or 0.0)
    hyp = str(selection.get("hyp") or "").lower()
    hyp_v = np.asarray([1.0 if hyp == h else 0.0 for h in HYP_OH], np.float32)

    keep = selection.get("keep") or []
    n_keep = float(len(keep))

    effs, plas, hops, lags, strs, dirs = [], [], [], [], [], []
    strength_mul: dict[str, float] = {}
    dir_override: dict[str, float] = {}
    by_id = {c.mid: c for c in bank}
    matched_cards: list[MechCard] = []
    for it in keep:
        mid = str(it.get("id") or "").strip()
        if mid not in by_id:
            hit = next((k for k in by_id if k.endswith(mid) or mid.endswith(k)), None)
            if not hit:
                continue
            mid = hit
        c = by_id[mid]
        matched_cards.append(c)
        st = STRENGTH_W.get(str(it.get("strength") or "medium").lower(), 1.0)
        strength_mul[mid] = st
        d = DIR_SIGN.get(str(it.get("residual_dir") or "").lower(), None)
        if d is not None:
            dir_override[mid] = d
        effs.append(float(dir_override.get(mid, c.effect)))
        plas.append(float(c.plaus))
        hops.append(float(c.hops))
        lags.append(float(c.lag))
        strs.append(float(st))
        dirs.append(float(dir_override.get(mid, c.effect)))

    def _stats(xs: list[float]) -> np.ndarray:
        if not xs:
            return np.zeros(4, np.float32)
        a = np.asarray(xs, float)
        return np.asarray(
            [a.mean(), a.std() + 1e-8, a.min(), a.max()], np.float32
        )

    ch_vec = np.zeros(max(len(channels), 1), np.float32)
    if channels and matched_cards:
        ch2i = {c: i for i, c in enumerate(channels)}
        for c in matched_cards:
            i = ch2i.get(c.channel)
            if i is not None:
                ch_vec[i] += 1.0
        ch_vec = ch_vec / max(ch_vec.sum(), 1.0)

    soft = soft_signed_from_cards(
        matched_cards,
        H,
        soft_amp,
        strength_mul=strength_mul,
        dir_override=dir_override,
    )
    soft = soft * (0.40 + 0.60 * conf) * gate
    soft_stats = np.asarray(
        [
            float(soft.mean()),
            float(soft.std() + 1e-8),
            float(soft[0]) if soft.size else 0.0,
            float(soft[-1]) if soft.size else 0.0,
            float(np.max(np.abs(soft))),
        ],
        np.float32,
    )

    bank_n = float(len(bank))
    feats = np.concatenate(
        [
            np.asarray([gate, conf, n_keep, bank_n, n_keep / max(bank_n, 1.0)], np.float32),
            hyp_v,
            _stats(effs),
            _stats(plas),
            _stats(hops),
            _stats(lags),
            _stats(strs),
            _stats(dirs),
            ch_vec.astype(np.float32),
            soft_stats,
            _cfa_feats(yc),
            soft.astype(np.float32),  # soft template prior in φ
        ]
    )
    return feats.astype(np.float32)


def pack_step_features(
    bank: Sequence[MechCard],
    partial_keep: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any],
    yc: Sequence[float],
    H: int,
    *,
    channels: Sequence[str] = (),
) -> np.ndarray:
    """State+action features for step-level Q (prefix KEEP + current op)."""
    sel = {
        "gate": 1 if partial_keep else 0,
        "keep": list(partial_keep),
        "conf": 0.6,
        "hyp": "",
    }
    base = pack_keep_features(bank, sel, yc, H, channels=channels, soft_amp=0.05)
    op = str(action.get("op") or action.get("action") or "").lower()
    op_oh = np.asarray(
        [
            1.0 if op in ("select", "select_mech", "keep") else 0.0,
            1.0 if op in ("select_event", "event") else 0.0,
            1.0 if op in ("abstain", "skip", "drop") else 0.0,
            1.0 if op in ("done", "stop", "end") else 0.0,
            1.0 if op in ("set_gate", "gate") else 0.0,
            1.0 if op in ("set_amp", "amp", "amplitude") else 0.0,
        ],
        np.float32,
    )
    mid = str(action.get("id") or "").strip()
    if not mid and action.get("event") and action.get("mech"):
        mid = f"{action.get('event')}/{str(action.get('mech')).upper()}"
    by_id = {c.mid: c for c in bank}
    c = by_id.get(mid)
    if c is None and mid:
        hit = next((k for k in by_id if k.endswith(mid) or mid.endswith(k)), None)
        c = by_id.get(hit) if hit else None
    amp_v = 0.0
    if op in ("set_amp", "amp", "amplitude"):
        _amp = {"off": 0.0, "weak": 0.03, "medium": 0.05, "strong": 0.08, "high": 0.10}
        amp_v = float(_amp.get(str(action.get("amp") or "").lower(), 0.0))
    mech_f = np.asarray(
        [
            float(c.effect) if c else 0.0,
            float(c.plaus) if c else 0.0,
            float(c.hops) if c else 0.0,
            float(c.lag) if c else 0.0,
            float(STRENGTH_W.get(str(action.get("strength") or "medium").lower(), 1.0)),
            float(
                DIR_SIGN.get(
                    str(action.get("residual_dir") or action.get("dir") or "").lower(), 0.0
                )
            ),
            float(action.get("gate") or 0.0) if op in ("set_gate", "gate") else 0.0,
            amp_v,
        ],
        np.float32,
    )
    return np.concatenate([base, op_oh, mech_f]).astype(np.float32)


def soft_delta_from_selection(
    bank: Sequence[MechCard],
    selection: Mapping[str, Any],
    H: int,
    amp: float,
) -> np.ndarray:
    d, _, _ = delta_from_selection(bank, selection, H, amp)
    return d
