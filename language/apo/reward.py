from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

RewardMode = Literal["cfa_text_delta"]

REWARD_MODE_LABELS = {
    "cfa_text_delta": "cfa_raw_text_mse - cfa_rewrite_mse (vs run.py CFA text)",
}


@dataclass
class RewardBreakdown:
    baseline_mse: float
    downstream_mse: float
    residual_mse: float
    delta_mse: float
    reward: float
    residual_true: list[float]
    residual_pred: list[float]
    y_true: list[float]
    y_baseline: list[float]
    y_corrected: list[float]
    scorer: str = "cfa"
    reward_mode: str = "cfa_text_delta"

    def residual_diagnosis(self) -> str:
        y_true = np.asarray(self.y_true, dtype=float)
        y_base = np.asarray(self.y_baseline, dtype=float)
        y_fuse = np.asarray(self.y_corrected, dtype=float)
        e_base = y_true - y_base
        e_fuse = y_true - y_fuse
        fuse_minus_base = y_fuse - y_base
        worse = [
            int(i)
            for i in range(len(y_true))
            if abs(float(e_fuse[i])) > abs(float(e_base[i])) + 1e-8
        ]
        need_up = [int(i) for i, v in enumerate(e_base) if float(v) > 0.05]
        need_down = [int(i) for i, v in enumerate(e_base) if float(v) < -0.05]
        fuse_direction = []
        for i, v in enumerate(fuse_minus_base):
            if abs(float(v)) < 0.02:
                fuse_direction.append(f"h{i}=flat")
            elif float(v) > 0:
                fuse_direction.append(f"h{i}=up")
            else:
                fuse_direction.append(f"h{i}=down")
        return (
            f"horizons_where_fusion_worse_than_baseline={worse}; "
            f"baseline_needs_up_horizons={need_up}; "
            f"baseline_needs_down_horizons={need_down}; "
            f"fusion_vs_baseline={fuse_direction}; "
            f"residual_true={np.round(e_base, 4).tolist()}; "
            f"residual_fusion={np.round(fuse_minus_base, 4).tolist()}"
        )

    def critique(self) -> str:
        return (
            f"scorer={self.scorer}; "
            f"baseline_mse={self.baseline_mse:.6f}; "
            f"downstream_mse={self.downstream_mse:.6f}; "
            f"delta_mse={self.delta_mse:.6f}; "
            f"reward={self.reward:.6f}; "
            f"y_true={np.round(self.y_true, 4).tolist()}; "
            f"y_baseline={np.round(self.y_baseline, 4).tolist()}; "
            f"y_fusion={np.round(self.y_corrected, 4).tolist()}; "
            f"{self.residual_diagnosis()}"
        )


def compute_reward(
    y_true: np.ndarray,
    y_reference: np.ndarray,
    y_fusion: np.ndarray,
    *,
    scorer: str = "cfa",
    reward_mode: RewardMode = "cfa_text_delta",
) -> RewardBreakdown:
    """reward = cfa_raw_mse - cfa_rewrite_mse (per window: ref MSE - fusion MSE)."""
    if reward_mode != "cfa_text_delta":
        raise ValueError(f"Unsupported reward_mode={reward_mode!r}")

    y_true = np.asarray(y_true, dtype=float)
    y_reference = np.asarray(y_reference, dtype=float)
    y_fusion = np.asarray(y_fusion, dtype=float)
    residual_true = y_true - y_reference
    residual_pred = y_fusion - y_reference

    reference_mse = float(np.mean((y_reference - y_true) ** 2))
    downstream_mse = float(np.mean((y_fusion - y_true) ** 2))
    residual_mse = float(np.mean((residual_pred - residual_true) ** 2))
    delta_mse = reference_mse - downstream_mse

    return RewardBreakdown(
        baseline_mse=reference_mse,
        downstream_mse=downstream_mse,
        residual_mse=residual_mse,
        delta_mse=delta_mse,
        reward=delta_mse,
        residual_true=residual_true.tolist(),
        residual_pred=residual_pred.tolist(),
        y_true=y_true.tolist(),
        y_baseline=y_reference.tolist(),
        y_corrected=y_fusion.tolist(),
        scorer=scorer,
        reward_mode=reward_mode,
    )
