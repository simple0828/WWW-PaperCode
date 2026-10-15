"""Temporal alignment QA for Time-MMD Final_Search text vs row dates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass
class AlignmentReport:
    n_rows: int
    n_with_facts: int
    frac_within_days: float
    median_day_offset: float
    n_unique_texts: int
    ok: bool
    detail: str


def parse_fact_dates(text: str) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for y, m, d in _DATE_RE.findall(str(text)):
        try:
            out.append(pd.Timestamp(int(y), int(m), int(d)))
        except Exception:
            continue
    return out


def row_window(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(str(row.get("start_date", row["date"])))
    end = pd.Timestamp(str(row.get("end_date", row["date"])))
    return start, end


def fact_offsets_days(row: pd.Series, text_col: str = "Final_Search_4") -> list[float]:
    """Day offset of each fact date relative to row start_date (negative = before)."""
    start, _ = row_window(row)
    facts = parse_fact_dates(row[text_col])
    return [(f - start).days for f in facts]


def audit_dataframe(
    df: pd.DataFrame,
    text_col: str = "Final_Search_4",
    *,
    max_abs_days: int = 120,
    min_frac_ok: float = 0.85,
) -> AlignmentReport:
    """Check that fact dates fall near the row's calendar month/window."""
    if text_col not in df.columns:
        raise ValueError(f"Missing text column {text_col}")
    offsets: list[float] = []
    n_ok = 0
    n_with = 0
    for _, row in df.iterrows():
        offs = fact_offsets_days(row, text_col)
        if not offs:
            continue
        n_with += 1
        med = float(np.median(offs))
        offsets.append(med)
        if abs(med) <= max_abs_days:
            n_ok += 1
    frac = (n_ok / n_with) if n_with else 0.0
    med_off = float(np.median(offsets)) if offsets else float("nan")
    ok = frac >= min_frac_ok
    detail = (
        f"text={text_col}; with_facts={n_with}/{len(df)}; "
        f"frac_|median_offset|<= {max_abs_days}d = {frac:.3f}; "
        f"median_offset_days={med_off:.1f}; "
        f"unique_texts={df[text_col].astype(str).nunique()}"
    )
    return AlignmentReport(
        n_rows=len(df),
        n_with_facts=n_with,
        frac_within_days=frac,
        median_day_offset=med_off,
        n_unique_texts=int(df[text_col].astype(str).nunique()),
        ok=ok,
        detail=detail,
    )


def assert_domain_aligned(
    csv_path: str,
    text_col: str = "Final_Search_4",
    *,
    max_abs_days: int = 120,
    min_frac_ok: float = 0.85,
) -> AlignmentReport:
    df = pd.read_csv(csv_path)
    report = audit_dataframe(
        df, text_col, max_abs_days=max_abs_days, min_frac_ok=min_frac_ok
    )
    if not report.ok:
        raise ValueError(
            "Temporal text alignment check failed for "
            f"{csv_path}: {report.detail}"
        )
    return report


def sample_is_aligned(
    start_date: str,
    raw_text: str,
    *,
    max_abs_days: int = 120,
) -> bool:
    """Per-window gate used when loading APO samples."""
    try:
        start = pd.Timestamp(start_date)
    except Exception:
        return False
    facts = parse_fact_dates(raw_text)
    if not facts:
        return False
    med = float(np.median([(f - start).days for f in facts]))
    return abs(med) <= max_abs_days
