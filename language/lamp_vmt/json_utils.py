"""JSON helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Prefer object / array spans.
        for left, right in (("{", "}"), ("[", "]")):
            start, end = text.find(left), text.rfind(right)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise


def as_list(payload: Any, key: str | None = None) -> list:
    if key is not None and isinstance(payload, dict):
        payload = payload.get(key, [])
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return [payload]
