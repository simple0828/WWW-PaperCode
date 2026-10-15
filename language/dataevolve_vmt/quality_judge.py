"""DataEvolve-aligned Quality Judge (vLLM, no Agents SDK).

Flow mirrors third_party/DataEvolve/quality_judge:
  1) read_experience (full DB); prompt gets visible recent window only
  2) score original vs cleaned pairs
  3) Coverage + Executability analysis of the cleaning prompt
  4) write_experience for NEW residual issues (near-dedup vs full DB)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from language import VLLMClient
from language.dataevolve_vmt.experience_store import (
    DEFAULT_NEAR_DUP_JACCARD,
    DEFAULT_VISIBLE_K,
    older_theme_summary,
    read_experience,
    sync_state_from_disk,
    visible_experiences,
    write_experiences,
)
from language.dataevolve_vmt import prompts as de_prompts
from language.dataevolve_vmt.prompts import build_judge_user
from language.dataevolve_vmt.heuristics import score_clean_pair
from language.dataevolve_vmt.pipeline import EvolveState, _ask_json


def load_clean_pairs(jsonl_path: Path | str) -> list[dict[str, str]]:
    path = Path(jsonl_path)
    pairs: list[dict[str, str]] = []
    if not path.is_file():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text")
        refined = row.get("refined_text")
        if text is None or refined is None:
            continue
        pairs.append({"text": str(text), "refined_text": str(refined)})
    return pairs


def run_quality_judge(
    client: VLLMClient,
    *,
    out_dir: Path | str,
    strategy: str,
    pairs: list[dict[str, str]],
    state: EvolveState,
    family: str,
    iteration: int,
    judge_batches: int = 4,
    judge_pairs: int = 3,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    experience_visible_k: int = DEFAULT_VISIBLE_K,
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD,
    pair_char_limit: int = 700,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Judge one cleaning strategy; write residual issues into experience DB."""
    out_dir = Path(out_dir)
    rng = rng or np.random.default_rng(0)
    print(
        f"=== Quality Judge [{family}] "
        f"(visible≤{experience_visible_k}; near_dup≥{near_dup_jaccard:.2f}; "
        f"batches={judge_batches}×pairs={judge_pairs}; max_tokens={max_tokens}) ===",
        flush=True,
    )

    if not pairs:
        return {
            "average_score": 0.0,
            "heuristic_score": 0.0,
            "quality_score": 0.0,
            "clean_analysis": "no cleaned pairs",
            "coverage_analysis": "",
            "executability_analysis": "",
            "new_issues_written": 0,
            "pair_evaluations": [],
            "suggestions": [],
        }

    h_scores = [score_clean_pair(p["text"], p["refined_text"]) for p in pairs]
    h_mean = float(np.mean(h_scores)) if h_scores else 0.0

    llm_scores: list[float] = []
    analyses: list[str] = []
    coverage_bits: list[str] = []
    exec_bits: list[str] = []
    suggestions: list[str] = []
    all_pair_evals: list[dict[str, Any]] = []
    new_issues_total = 0
    n_batches = max(1, int(judge_batches))
    k = min(int(judge_pairs), len(pairs))

    # Disjoint sampling across batches (reshuffle when pool exhausted).
    pool = rng.permutation(len(pairs)).tolist()
    pool_pos = 0

    for b in range(n_batches):
        if pool_pos + k > len(pool):
            pool = rng.permutation(len(pairs)).tolist()
            pool_pos = 0
        idxs = pool[pool_pos : pool_pos + k]
        pool_pos += k
        subset = [pairs[int(i)] for i in idxs]
        # Step 1: read full experience DB; prompt only recent visible window.
        sync_state_from_disk(out_dir, state.experiences)
        if not state.experiences:
            state.experiences.extend(read_experience(out_dir))
        full = list(state.experiences)
        visible = visible_experiences(full, k=experience_visible_k)
        older_note = older_theme_summary(full, visible_k=experience_visible_k)
        print(
            f"  judge batch {b + 1}/{n_batches}: read_experience() → "
            f"{len(full)} total, visible={len(visible)}",
            flush=True,
        )
        try:
            payload, _raw = _ask_json(
                client,
                de_prompts.JUDGE_SYSTEM,
                build_judge_user(
                    strategy,
                    subset,
                    visible,
                    iteration=b + 1,
                    older_note=older_note,
                    pair_char_limit=pair_char_limit,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            print(f"  quality judge batch {b + 1} failed: {exc}", flush=True)
            continue

        if not isinstance(payload, dict):
            continue

        score = float(payload.get("average_score", 5.0))
        llm_scores.append(score)
        analysis = str(payload.get("clean_analysis", "")).strip()
        if analysis:
            analyses.append(analysis)
        cov = str(payload.get("coverage_analysis", "")).strip()
        exe = str(payload.get("executability_analysis", "")).strip()
        if cov:
            coverage_bits.append(cov)
        if exe:
            exec_bits.append(exe)
        for s in payload.get("suggestions", []) or []:
            s = str(s).strip()
            if s and s not in suggestions:
                suggestions.append(s)
        for pe in payload.get("pair_evaluations", []) or []:
            if isinstance(pe, dict):
                all_pair_evals.append(pe)

        residual = []
        for issue in payload.get("new_issues", []) or []:
            s = str(issue).strip()
            if s:
                residual.append(s)
        # Write against full DB with near-dup filter (not just visible window).
        added = write_experiences(
            out_dir,
            residual,
            state_experiences=state.experiences,
            near_dup_jaccard=near_dup_jaccard,
        )
        new_issues_total += added
        print(
            f"  judge batch {b + 1}/{n_batches}: llm={score:.2f} "
            f"new_issues_written=+{added} (exp_total={len(state.experiences)})",
            flush=True,
        )

    llm_mean = float(np.mean(llm_scores)) if llm_scores else 5.0
    # Blend heuristic with LLM like lightweight pipeline; keep 1-10 scale.
    quality_score = 0.5 * (h_mean * 10.0) + 0.5 * llm_mean
    coverage = " | ".join(coverage_bits) if coverage_bits else ""
    executability = " | ".join(exec_bits) if exec_bits else ""
    clean_analysis = (
        f"Coverage: {coverage or '(n/a)'}. "
        f"Executability: {executability or '(n/a)'}. "
        f"Detail: {' || '.join(analyses) if analyses else '(n/a)'}"
    )

    result = {
        "average_score": llm_mean,
        "heuristic_score": h_mean,
        "quality_score": float(quality_score),
        "clean_analysis": clean_analysis,
        "coverage_analysis": coverage,
        "executability_analysis": executability,
        "new_issues_written": new_issues_total,
        "pair_evaluations": all_pair_evals,
        "suggestions": suggestions,
        "judge_batches_ok": len(llm_scores),
        "judge_batches_total": n_batches,
    }
    print(
        f"  Quality Judge [{family}] done: quality={quality_score:.2f} "
        f"(heur={h_mean:.3f}, llm={llm_mean:.2f}) residual_writes={new_issues_total} "
        f"batches_ok={len(llm_scores)}/{n_batches}",
        flush=True,
    )
    return result
