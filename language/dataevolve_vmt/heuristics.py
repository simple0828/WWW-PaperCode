"""Cheap heuristic flags/scores for Stage1 cleaning (anchors LLM judge).

Domain markers come from ``DomainProfile.useful_markers`` via ``apply_domain_heuristics``.
Traffic defaults match the historical VMT marker list exactly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from language.dataevolve_vmt.domain_profiles import DomainProfile

NON_US_MARKERS = [
    "afghanistan",
    "australia",
    "canada",
    "china",
    "india",
    "japan",
    "mexico",
    "pakistan",
    "russia",
    "soviet",
    "united kingdom",
    "u.k.",
    "yemen",
    "ireland",
    "france",
    "germany",
]

AD_MARKERS = [
    "ecoflow",
    "for sale",
    "buy now",
    "mobile app",
    "smartphone",
    "advertisement",
    "sponsored",
]

# Module defaults = traffic/VMT (bitwise same as historical list).
USEFUL_MARKERS = [
    "vmt",
    "vehicle miles",
    "travel volume",
    "fhwa",
    "bts.gov",
    "highway",
    "congestion",
    "fuel price",
    "gasoline",
    "recession",
    "inflation",
]

_DISPLAY_TASK = "U.S. monthly VMT forecasting"
_IS_TRAFFIC_VMT = True


def apply_domain_heuristics(profile: "DomainProfile") -> None:
    global USEFUL_MARKERS, _DISPLAY_TASK, _IS_TRAFFIC_VMT
    markers = list(profile.useful_markers) if profile.useful_markers else list(USEFUL_MARKERS)
    USEFUL_MARKERS = markers
    _DISPLAY_TASK = profile.display_task
    _IS_TRAFFIC_VMT = profile.key == "traffic" or "VMT" in profile.display_task


def _count_any(text: str, words: list[str]) -> int:
    low = text.lower()
    return sum(1 for w in words if w in low)


def heuristic_issue_flags(text: str, *, display_task: Optional[str] = None) -> list[str]:
    flags = []
    low = text.lower()
    task = display_task or _DISPLAY_TASK
    if "direction=" in low or "strength=" in low:
        flags.append("Contains deprecated direction/strength effect tags")
    if _count_any(low, NON_US_MARKERS) >= 1:
        if _IS_TRAFFIC_VMT and task == "U.S. monthly VMT forecasting":
            flags.append(
                "Contains non-U.S. geography or foreign events unrelated to U.S. VMT"
            )
        else:
            flags.append(
                f"Contains non-U.S. geography or foreign events unrelated to {task}"
            )
    if _count_any(low, AD_MARKERS) >= 1:
        flags.append("Contains product/ad/app marketing content")
    if not re.search(r"\d{4}-\d{2}-\d{2}", text) and not re.search(
        r"\b19\d{2}\b|\b20\d{2}\b", text
    ):
        flags.append("Lacks clear time anchors (dates/years)")
    if not re.search(r"\d", text):
        flags.append("Lacks numeric anchors")
    if len(text) > 3500 and _count_any(low, USEFUL_MARKERS) < 2:
        if _IS_TRAFFIC_VMT and task == "U.S. monthly VMT forecasting":
            flags.append(
                "Long text with little VMT-specific substance (generic traffic fluff)"
            )
        else:
            flags.append(
                f"Long text with little task-specific substance for {task}"
            )
    return flags


def score_clean_pair(raw: str, cleaned: str) -> float:
    """Return 0-1 heuristic quality of cleaning."""
    raw_l = raw.lower()
    cl_l = (cleaned or "").lower()
    if not cl_l or cl_l in {"no text.", "n/a", "none"}:
        return 0.05

    raw_noise = _count_any(raw_l, NON_US_MARKERS) + _count_any(raw_l, AD_MARKERS)
    cl_noise = _count_any(cl_l, NON_US_MARKERS) + _count_any(cl_l, AD_MARKERS)
    noise_reduction = 1.0 if raw_noise == 0 else max(0.0, 1.0 - cl_noise / max(1, raw_noise))

    raw_useful = _count_any(raw_l, USEFUL_MARKERS)
    cl_useful = _count_any(cl_l, USEFUL_MARKERS)
    if raw_useful == 0:
        preserve = 0.7 if len(cleaned) > 40 else 0.3
    else:
        preserve = min(1.0, cl_useful / raw_useful)

    ratio = len(cleaned) / max(1, len(raw))
    length_pen = 1.0
    if ratio < 0.05:
        length_pen = 0.2
    elif ratio < 0.15:
        length_pen = 0.5
    elif ratio > 1.2:
        length_pen = 0.7

    dated_raw = bool(re.search(r"\d{4}-\d{2}-\d{2}", raw))
    dated_cl = bool(re.search(r"\d{4}-\d{2}-\d{2}", cleaned))
    date_keep = 1.0 if (not dated_raw or dated_cl) else 0.4

    return float(0.35 * noise_reduction + 0.35 * preserve + 0.15 * length_pen + 0.15 * date_keep)
