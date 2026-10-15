from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from data_provider.data_factory import data_provider
from models import PatchTST


def _coerce_text_cell(value) -> str:
    """Normalize dataset text/date cells to a plain Python str.

    Handles legacy (N,2) duplicate-column rows and 0-d / 1-d numpy cells.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        # Prefer first non-null scalar if row was duplicated.
        flat = value.reshape(-1)
        for item in flat:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                continue
            s = str(item)
            if s and s.lower() != "nan":
                return s
        return str(flat[0])
    if isinstance(value, (np.generic,)):
        s = str(value.item())
        return "" if s.lower() == "nan" else s
    s = str(value)
    return "" if s.lower() == "nan" else s


@dataclass
class BaselineSample:
    split: str
    window_index: int
    dataset_index: int
    raw_text: str
    y_true: np.ndarray
    y_baseline: np.ndarray
    residual_true: np.ndarray
    start_date: str
    end_date: str
    history_stats: str
    baseline_forecast: str


class UnimodalBaselinePredictor:
    """Load the trained unimodal PatchTST checkpoint and score one window."""

    def __init__(self, args, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.args = args
        self.device = torch.device(
            device or (f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
        )
        self.model = PatchTST.Model(args).float().to(self.device)
        ckpt = checkpoint_path or str(
            Path(args.checkpoints) / args.checkpoint_setting / "checkpoint.pth"
        )
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._datasets = {}
        self._raw_tables = {}

    def _predict_window(self, dataset, window_index: int) -> tuple[np.ndarray, np.ndarray]:
        seq_x, seq_y, seq_x_mark, seq_y_mark, _ = dataset[window_index]
        batch_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(0).to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                text_context=None,
            )
        pred = outputs[:, -self.args.pred_len :, :].detach().cpu().numpy()
        true = batch_y[:, -self.args.pred_len :, :].detach().cpu().numpy()

        # Match Time-MMD / CFA reporting: MSE on StandardScaler space unless --inverse.
        if dataset.scale and getattr(self.args, "inverse", False):
            pred = dataset.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            true = dataset.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

        return pred.squeeze(), true.squeeze()

    def get_dataset(self, split: str):
        if split not in self._datasets:
            dataset, _ = data_provider(self.args, split)
            self._datasets[split] = dataset
        return self._datasets[split]

    def _raw_table(self) -> pd.DataFrame:
        key = os.path.join(self.args.root_path, self.args.data_path)
        if key not in self._raw_tables:
            self._raw_tables[key] = pd.read_csv(key)
        return self._raw_tables[key]

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _history_from_row(self, row: pd.Series) -> str:
        parts = []
        for key in ["his_avg_1", "his_avg_2", "his_avg_3", "his_avg_4", "prior_history_avg"]:
            val = self._safe_float(row.get(key))
            if val is not None:
                parts.append(f"{key}={val:.4f}")
        anomaly = self._safe_float(row.get("Anomaly"))
        if anomaly is not None:
            parts.append(f"anomaly={anomaly:.4f}")
        return "; ".join(parts)

    def get_sample(self, split: str, window_index: int) -> BaselineSample:
        dataset = self.get_dataset(split)
        if window_index < 0 or window_index >= len(dataset):
            raise IndexError(f"window_index {window_index} out of range for split={split}")

        y_baseline, y_true = self._predict_window(dataset, window_index)
        residual_true = y_true - y_baseline
        raw_text_arr = dataset.get_text(np.array([window_index]))
        raw_text = _coerce_text_cell(raw_text_arr[0])
        text_row = window_index + dataset.seq_len
        start_date = _coerce_text_cell(dataset.start_date[text_row])
        end_date = _coerce_text_cell(dataset.end_date[text_row])

        df = self._raw_table()
        row = df.iloc[text_row]
        history_stats = self._history_from_row(row)
        baseline_forecast = f"y_baseline={np.round(y_baseline, 4).tolist()}"

        return BaselineSample(
            split=split,
            window_index=window_index,
            dataset_index=window_index,
            raw_text=raw_text,
            y_true=y_true.astype(float),
            y_baseline=y_baseline.astype(float),
            residual_true=residual_true.astype(float),
            start_date=start_date,
            end_date=end_date,
            history_stats=history_stats,
            baseline_forecast=baseline_forecast,
        )
