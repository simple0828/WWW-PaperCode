"""Shared cleaning helpers for Self-Evo (strict CFA loop)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from language import VLLMClient
from language.dataevolve_vmt.pipeline import EvolveConfig, run_clean


@dataclass
class CleanRuntimeConfig:
    """Runtime knobs shared by Designer + full-corpus Cleaner."""

    output_dir: str = "artifacts/dataevolve_cfa_selfevo_facts"
    init_bundle: str = (
        "artifacts/ctrl_raw_cfa_continue_gettext_fixed/"
        "cfa_bundle_raw_continue30_gettext_fixed.pt"
    )
    domain: str = "traffic"
    clean_concurrency: int = 48
    max_tokens: int = 4096
    temperature_clean: float = 0.0
    temperature_design: float = 0.55
    seed: int = 2021
    raw_csv: str = "data/Traffic/US_VMT_Month.csv"
    text_col: str = "Final_Search_4"


# Backward-compatible alias used by designer_diverse / strict_loop.
AlternatingConfig = CleanRuntimeConfig


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _full_clean_csv(
    client: VLLMClient,
    *,
    prompt: str,
    raw_csv: Path,
    text_col: str,
    out_csv: Path,
    out_jsonl: Path,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    semaphore=None,
    prompt_id: str = "full",
) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(raw_csv)
    samples = []
    empty_raw_ids: set[int] = set()
    for i in range(len(df)):
        raw = df.iloc[i][text_col]
        text = "" if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else str(raw)
        text_stripped = text.strip()
        # Hard guard: empty/NA/near-empty source must stay empty (blocks social_good 2024-stuffing).
        if (not text_stripped) or text_stripped.lower() in {"nan", "none", "na", "n/a"} or len(text_stripped) < 40:
            empty_raw_ids.add(i)
            text = ""
        samples.append({"row_id": i, "date": str(df.iloc[i].get("date", "")), "text": text})

    cfg = EvolveConfig(
        sample_jsonl="",
        output_dir=str(out_csv.parent),
        clean_concurrency=concurrency,
        temperature_clean=temperature,
        max_tokens=max_tokens,
    )
    # Skip LLM for empty rows: still call run_clean only on non-empty to save cost/errors,
    # then merge. If run_clean requires all samples, post-overwrite empties.
    pairs = run_clean(
        client,
        samples,
        {"id": prompt_id, "prompt_content": prompt},
        out_jsonl,
        cfg,
        semaphore=semaphore,
    )
    cleaned = []
    for p in pairs:
        rid = int(p.get("row_id", len(cleaned)))
        refined = p.get("refined_text") or ""
        if rid in empty_raw_ids or (samples[rid]["text"] == ""):
            cleaned.append("")
        else:
            cleaned.append(refined)
    # length safety if pairs order != samples
    if len(cleaned) != len(samples):
        cleaned = []
        by_id = {int(p.get("row_id", i)): p for i, p in enumerate(pairs)}
        for i in range(len(samples)):
            if i in empty_raw_ids or samples[i]["text"] == "":
                cleaned.append("")
            else:
                cleaned.append((by_id.get(i) or {}).get("refined_text") or "")
    out = df.copy()
    out[text_col] = cleaned
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    raw_lens = [len(s["text"]) for s in samples]
    cln_lens = [len(c) for c in cleaned]
    return {
        "n_rows": len(df),
        "n_forced_empty": len(empty_raw_ids),
        "raw_chars_mean": float(np.mean(raw_lens)),
        "cleaned_chars_mean": float(np.mean(cln_lens)),
        "retention_mean": float(np.mean([c / max(1, r) for c, r in zip(cln_lens, raw_lens)])),
    }
