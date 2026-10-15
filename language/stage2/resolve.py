"""Resolve Stage1 DataEvolve candidate for a Stage2 domain."""

from __future__ import annotations

import json
from pathlib import Path

from language.stage2.profiles import Stage2DomainProfile, get_stage2_profile


def resolve_stage1_candidate(
    root: Path,
    domain_or_profile: str | Stage2DomainProfile,
) -> dict:
    """Return ``{dir, bundle, csv, family, best}`` for the Stage1 best family."""
    profile = (
        domain_or_profile
        if isinstance(domain_or_profile, Stage2DomainProfile)
        else get_stage2_profile(domain_or_profile)
    )
    base = root / "artifacts" / profile.stage1_artifact_dirname
    if not base.is_dir():
        raise FileNotFoundError(f"Stage1 artifact dir missing: {base}")
    best_path = base / "best_strategy.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    family = best["family"]
    cand = next(
        p
        for p in base.glob(f"round_*/{family}")
        if (p / "cfa_bundle.pt").exists() and (p / profile.cleaned_csv_name).exists()
    )
    return {
        "dir": cand,
        "bundle": cand / "cfa_bundle.pt",
        "csv": cand / profile.cleaned_csv_name,
        "cleaned_texts": cand / "cleaned_texts.jsonl",
        "family": family,
        "best": best,
        "profile": profile,
    }
