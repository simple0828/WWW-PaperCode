"""Scheme 2 — sampled dual + sparse driver gate (open-loop Explicit).

Pipeline:
  1) n rounds: LLM CoT-selects ≤2 FOCUS events/month from L-month lookback
     (residual-vs-CFA rationale; diversify across rounds).
  2) Each round: on that subsample only, write M1 direct + M2 indirect
     → 2n window-level hypotheses.
  3) Decode Δ with a **sparse driver gate** (LAMP-style vote + margin),
     not soft averaging. Default: ``sparse_pair_m2``.

Traffic CFA residual (test 101, n=3): sparse_pair_m2 ≈ **−1.92%** vs B0
(classic dual⊕LAMP soft_mech ≈ −1.23%). Cross-domain replay later via profile.

Caches: ``{cache_dir}/scheme2_sampled_dual_cache.jsonl``
Manifest: ``artifacts/event_ts_assoc/scheme2_recipe.json``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from language.stage2.profiles import Stage2DomainProfile, get_stage2_profile
from language.stage2.sampled_dual import (
    DEFAULT_N,
    SYS_DUAL,
    SYS_SELECT,
    build_candidates,
    cache_key,
    decode_sparse_driver,
    format_subsample,
    parse_selected_eids,
    prompt_dual_on_sample,
    prompt_select,
)

# Locked decode for Scheme 2 comparisons across domains.
SCHEME2_DECODE = "sparse_pair_m2"
SCHEME2_DECODE_KW: dict[str, Any] = {
    "mode": "sparse_pair_m2",
    "min_margin": 2,
    "min_items": 2,
    "conf_floor": 0.55,
}


@dataclass(frozen=True)
class Scheme2Config:
    n_rounds: int = DEFAULT_N
    max_per_month: int = 2
    select_temp: float = 0.4
    dual_temp: float = 0.0
    decode: str = SCHEME2_DECODE
    cache_tag: str = "default"
    wave: int = 48

    def cache_name(self) -> str:
        return f"scheme2_sampled_dual_{self.cache_tag}.jsonl"


def scheme2_cache_path(profile: Stage2DomainProfile, cfg: Scheme2Config, root: Path) -> Path:
    return root / profile.cache_dir / cfg.cache_name()


def load_scheme2_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.open():
        row = __import__("json").loads(line)
        key = row.get("key") or row.get("window_id")
        if key is not None:
            out[str(key)] = row["response"]
    return out


def decode_scheme2(
    pair_texts: Sequence[str],
    *,
    H: int = 6,
    cfg: Optional[Scheme2Config] = None,
) -> dict:
    """Canonical Scheme2 Δ from n dual JSON strings."""
    cfg = cfg or Scheme2Config()
    kw = dict(SCHEME2_DECODE_KW)
    if cfg.decode != SCHEME2_DECODE:
        kw["mode"] = cfg.decode
    return decode_sparse_driver(pair_texts, H=H, **kw)


def pair_texts_from_cache(
    cache: Mapping[str, str],
    wid: str,
    n_rounds: int,
) -> list[str]:
    return [cache.get(cache_key(wid, ri, "dual"), "") for ri in range(n_rounds)]


def recipe_dict(
    *,
    domain: str = "traffic",
    cfg: Optional[Scheme2Config] = None,
    traffic_baseline: Optional[dict] = None,
) -> dict:
    cfg = cfg or Scheme2Config()
    return {
        "scheme": 2,
        "name": "sampled_dual_sparse_driver_gate",
        "domain_default": domain,
        "config": asdict(cfg),
        "decode_locked": SCHEME2_DECODE_KW,
        "modules": {
            "api": "language.stage2.scheme2_sampled_dual",
            "impl": "language.stage2.sampled_dual",
            "eval": "scripts/eval_scheme2.py",
        },
        "pipeline": [
            "n× CoT-select ≤2 FOCUS events/month from L-month lookback",
            "n× dual M1 direct + M2 indirect on subsample only",
            "2n hypotheses → sparse driver gate (pair collapse + vote margin)",
            "ŷ = y_CFA + Δ  (open-loop, no learnable params)",
        ],
        "traffic_baseline": traffic_baseline,
        "note": (
            "Use sparse_pair_m2 for cross-domain comparisons. "
            "Legacy soft aggregate of 2n hyps is ablation-only, not Scheme2."
        ),
    }


def write_recipe(root: Path, traffic_baseline: Optional[dict] = None) -> Path:
    import json

    path = root / "artifacts/event_ts_assoc/scheme2_recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            recipe_dict(traffic_baseline=traffic_baseline),
            indent=2,
            ensure_ascii=False,
        )
    )
    return path


# Re-export helpers used by eval / multi-domain runners
__all__ = [
    "SCHEME2_DECODE",
    "SCHEME2_DECODE_KW",
    "Scheme2Config",
    "SYS_DUAL",
    "SYS_SELECT",
    "build_candidates",
    "cache_key",
    "decode_scheme2",
    "format_subsample",
    "get_stage2_profile",
    "load_scheme2_cache",
    "pair_texts_from_cache",
    "parse_selected_eids",
    "prompt_dual_on_sample",
    "prompt_select",
    "recipe_dict",
    "scheme2_cache_path",
    "write_recipe",
]
