"""Load shared LLM runtime knobs from llm_runtime.yaml (defaults + per-domain)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from language.dataevolve_vmt.domain_profiles import _aliases

_YAML = Path(__file__).with_name("llm_runtime.yaml")

_DEFAULTS: dict[str, Any] = {
    "max_model_len": 10240,
    "max_tokens_clean": 4096,
    "max_tokens_design": 4096,
    "max_tokens_judge": 4096,
    "clean_concurrency": 48,
    "vllm_concurrency_cap": 48,
    "zhipu_concurrency_cap": 8,
    "extract_concurrency": 48,
    "max_tokens_extract": 4096,
    "scheme_wave": 48,
    "max_tokens_scheme": 1024,
    "max_tokens_scheme_rewrite": 2048,
}


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    if not _YAML.is_file():
        return {"defaults": dict(_DEFAULTS), "domains": {}}
    data = yaml.safe_load(_YAML.read_text(encoding="utf-8")) or {}
    defaults = {**_DEFAULTS, **(data.get("defaults") or {})}
    domains = data.get("domains") or {}
    return {"defaults": defaults, "domains": domains}


def get_llm_runtime(domain: str | None = None) -> dict[str, Any]:
    """Return merged runtime dict for ``domain`` (aliases resolved)."""
    blob = _raw()
    out = dict(blob["defaults"])
    if domain:
        key = _aliases(domain)
        override = blob["domains"].get(key) or {}
        if isinstance(override, dict):
            out.update({k: v for k, v in override.items() if v is not None})
    return out


def clear_llm_runtime_cache() -> None:
    _raw.cache_clear()
