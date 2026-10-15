"""Experience database helpers (DataEvolve-style lightweight protocol).

Semantic equivalent of third_party/DataEvolve tools:
  read_experience()  — load issues from disk
  write_experience() — append one new issue (exact + near-dup deduped)

No Agents SDK tool-calling; the host enforces the call order.

Prompt visibility is capped (e.g. last 20) while write/compact operate on the
full DB so paraphrased residuals do not re-inflate the pool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence


DEFAULT_VISIBLE_K = 20
DEFAULT_NEAR_DUP_JACCARD = 0.65


def experience_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / "experience_database.jsonl"


def _norm_key(issue: str) -> str:
    """Normalize for exact-ish dedupe (case/punctuation/whitespace)."""
    s = (issue or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    return s


_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "by",
        "at",
        "it",
        "its",
        "into",
        "over",
        "under",
        "not",
        "no",
        "vs",
        "via",
    }
)


def _tokens(issue: str) -> set[str]:
    s = _norm_key(issue)
    toks = re.findall(r"[a-z0-9]+", s)
    keep_short = {"us", "uk", "vmt", "mse", "ads"}
    return {t for t in toks if (len(t) > 2 or t in keep_short) and t not in _STOP}


def token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(union) if union else 0.0


def token_containment(a: str, b: str) -> float:
    """Fraction of the smaller token set covered by the larger (paraphrase soft match)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return float(inter) / float(min(len(ta), len(tb)))


def is_near_duplicate(
    issue: str,
    known: Sequence[str],
    *,
    threshold: float = DEFAULT_NEAR_DUP_JACCARD,
    containment: float = 0.82,
) -> bool:
    """True if exact-norm match, high Jaccard, or high containment vs any known issue."""
    key = _norm_key(issue)
    if not key:
        return True
    for other in known:
        if _norm_key(other) == key:
            return True
        if token_jaccard(issue, other) >= threshold:
            return True
        # Synonym-heavy paraphrases often keep most content words of the shorter side.
        ta, tb = _tokens(issue), _tokens(other)
        if min(len(ta), len(tb)) >= 4 and token_containment(issue, other) >= containment:
            return True
    return False


def visible_experiences(
    experiences: Sequence[str],
    *,
    k: int = DEFAULT_VISIBLE_K,
) -> list[str]:
    """Most recent k issues for Judge/Designer prompts (full DB stays on disk)."""
    xs = [str(x).strip() for x in experiences if str(x).strip()]
    if k <= 0 or len(xs) <= k:
        return list(xs)
    return list(xs[-k:])


def older_theme_summary(
    experiences: Sequence[str],
    *,
    visible_k: int = DEFAULT_VISIBLE_K,
    max_themes: int = 12,
) -> str:
    """Compact one-liner of older (non-visible) issues so Judge does not 'forget' themes."""
    xs = [str(x).strip() for x in experiences if str(x).strip()]
    if len(xs) <= visible_k:
        return ""
    older = xs[:-visible_k]
    # Prefer distinctive content words from older pool.
    bag: dict[str, int] = {}
    for issue in older:
        for t in _tokens(issue):
            bag[t] = bag.get(t, 0) + 1
    ranked = sorted(bag.items(), key=lambda kv: (-kv[1], kv[0]))
    themes = [w for w, _ in ranked[:max_themes]]
    if not themes:
        return f"(+{len(older)} older issues compacted; not listed)"
    return (
        f"(+{len(older)} older issues not listed; recurring themes: "
        + ", ".join(themes)
        + ")"
    )


def compact_experiences(
    experiences: Sequence[str],
    *,
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
) -> tuple[list[str], int]:
    """Keep chronological first of each near-dup cluster; return (kept, n_dropped)."""
    kept: list[str] = []
    dropped = 0
    for x in experiences:
        s = str(x).strip()
        if not s:
            continue
        if is_near_duplicate(s, kept, threshold=near_dup_jaccard):
            dropped += 1
            continue
        kept.append(s)
    return kept, dropped


def read_experience(out_dir: Path | str) -> list[str]:
    """Lightweight read_experience(): return all issues from the experience DB."""
    path = experience_path(out_dir)
    if not path.is_file():
        return []
    issues: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        issue = str(row.get("issue", "")).strip()
        if issue:
            issues.append(issue)
    return issues


def write_experience(
    out_dir: Path | str,
    issue: str,
    *,
    known: set[str] | None = None,
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
) -> bool:
    """Lightweight write_experience(issue): append if not exact/near duplicate of full DB."""
    issue = (issue or "").strip()
    if not issue:
        return False
    path = experience_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if known is None:
        known_list = read_experience(out_dir)
        known_set = set(known_list)
    else:
        known_list = list(known)
        known_set = known

    if is_near_duplicate(issue, known_list, threshold=near_dup_jaccard):
        return False

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"issue": issue}, ensure_ascii=False) + "\n")
    known_set.add(issue)
    return True


def write_experiences(
    out_dir: Path | str,
    issues: list[str],
    *,
    state_experiences: list[str] | None = None,
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
) -> int:
    """Write many issues via write_experience; sync optional in-memory list."""
    known = set(state_experiences if state_experiences is not None else read_experience(out_dir))
    added = 0
    for issue in issues:
        s = (issue or "").strip()
        if not s:
            continue
        if write_experience(
            out_dir, s, known=known, near_dup_jaccard=near_dup_jaccard
        ):
            added += 1
            if state_experiences is not None and s not in state_experiences:
                state_experiences.append(s)
    return added


def persist_experience_snapshot(out_dir: Path | str, experiences: list[str]) -> None:
    """Rewrite full experience file from in-memory list (canonical dump)."""
    path = experience_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for x in experiences:
        s = str(x).strip()
        if not s:
            continue
        k = _norm_key(s)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(s)
    path.write_text(
        "\n".join(json.dumps({"issue": x}, ensure_ascii=False) for x in ordered)
        + ("\n" if ordered else ""),
        encoding="utf-8",
    )


def compact_and_persist(
    out_dir: Path | str,
    experiences: list[str],
    *,
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
) -> tuple[list[str], int]:
    """Near-dedup full list in-place, rewrite disk, return (kept, n_dropped)."""
    kept, dropped = compact_experiences(experiences, near_dup_jaccard=near_dup_jaccard)
    experiences.clear()
    experiences.extend(kept)
    persist_experience_snapshot(out_dir, experiences)
    return kept, dropped


def load_or_init_experiences(
    out_dir: Path | str,
    *,
    bootstrap: list[str] | None = None,
) -> list[str]:
    existing = read_experience(out_dir)
    if existing:
        return existing
    boot = [x for x in (bootstrap or []) if str(x).strip()]
    if boot:
        persist_experience_snapshot(out_dir, boot)
    return list(boot)


def sync_state_from_disk(out_dir: Path | str, state_experiences: list[str]) -> list[str]:
    """read_experience() then refresh in-memory list (disk as source of truth)."""
    disk = read_experience(out_dir)
    if not disk:
        return list(state_experiences)
    state_experiences.clear()
    state_experiences.extend(disk)
    return state_experiences
