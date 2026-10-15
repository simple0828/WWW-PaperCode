"""Lightweight DataEvolve loop for Traffic VMT news cleaning (isolated from APO).

Reuses the four-role design from third_party/DataEvolve, but runs on our
VLLMClient without the OpenAI Agents SDK dependency.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from language import VLLMClient
from language.dataevolve_vmt import prompts as de_prompts
from language.dataevolve_vmt.prompts import (
    build_cleaner_user,
    build_designer_user,
    build_judge_user,
    build_observer_user,
)
from language.dataevolve_vmt.heuristics import heuristic_issue_flags, score_clean_pair


@dataclass
class EvolveConfig:
    sample_jsonl: str
    output_dir: str
    n_observe_batches: int = 2
    observe_batch_size: int = 5
    n_iterations: int = 3
    judge_pairs: int = 4
    clean_concurrency: int = 48
    temperature_observe: float = 0.3
    temperature_design: float = 0.5
    temperature_clean: float = 0.2
    temperature_judge: float = 0.2
    max_tokens: int = 4096
    seed: int = 2021


@dataclass
class EvolveState:
    experiences: list[str] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)

    def best_prompt(self) -> Optional[dict[str, Any]]:
        scored = [p for p in self.prompts if p.get("score_objective") is not None]
        if not scored:
            return None
        return max(scored, key=lambda p: float(p["score_objective"]))


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ask_json(client: VLLMClient, system: str, user: str, *, temperature: float, max_tokens: int) -> Any:
    raw = client.ask(
        user,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_json(raw), raw


def run_observe(
    client: VLLMClient,
    samples: list[dict[str, Any]],
    state: EvolveState,
    cfg: EvolveConfig,
    rng: np.random.Generator,
    *,
    out_dir: str | Path | None = None,
) -> None:
    """DataEvolve Observer with lightweight protocol (no Agents SDK tools).

    Per batch (same order as original):
      1) read_experience()          — host loads DB
      2) independently analyze raw  — LLM
      3) write_experience(issue)    — host writes each NEW issue
    """
    from language.dataevolve_vmt.experience_store import (
        read_experience,
        sync_state_from_disk,
        write_experiences,
        visible_experiences,
        older_theme_summary,
        DEFAULT_VISIBLE_K,
    )

    out = Path(out_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(
        "=== Phase 1: Observer (lightweight protocol: "
        "read_experience → discover NEW → write_experience) ===",
        flush=True,
    )
    n = len(samples)
    for b in range(cfg.n_observe_batches):
        # Step 1: read_experience()
        disk_issues = read_experience(out)
        if disk_issues:
            sync_state_from_disk(out, state.experiences)
        elif state.experiences:
            # First batch may only have in-memory bootstrap; flush so read sees it.
            from language.dataevolve_vmt.experience_store import persist_experience_snapshot

            persist_experience_snapshot(out, state.experiences)
        known_before = list(state.experiences)
        print(
            f"  observe batch {b + 1}/{cfg.n_observe_batches}: "
            f"read_experience() → {len(known_before)} known issues",
            flush=True,
        )

        idxs = rng.choice(n, size=min(cfg.observe_batch_size, n), replace=False)
        batch = [samples[int(i)]["text"] for i in idxs]

        # Cheap heuristic discoveries also go through write_experience.
        heuristic_new: list[str] = []
        for t in batch:
            heuristic_new.extend(heuristic_issue_flags(t))
        n_heur = write_experiences(out, heuristic_new, state_experiences=state.experiences)

        # Step 2–3: LLM proposes new_issues; host write_experience each.
        try:
            visible = visible_experiences(state.experiences, k=DEFAULT_VISIBLE_K)
            older_note = older_theme_summary(
                state.experiences, visible_k=DEFAULT_VISIBLE_K
            )
            payload, _raw = _ask_json(
                client,
                de_prompts.OBSERVER_SYSTEM,
                build_observer_user(batch, visible, older_note=older_note),
                temperature=cfg.temperature_observe,
                max_tokens=cfg.max_tokens,
            )
            if isinstance(payload, dict):
                issues = payload.get("new_issues") or payload.get("issues") or []
                useful = payload.get("useful_signal_types", []) or []
            else:
                issues = payload if isinstance(payload, list) else []
                useful = []

            to_write = [str(x).strip() for x in issues if str(x).strip()]
            for sig in useful:
                s = str(sig).strip()
                if s:
                    to_write.append(f"PRESERVE useful_signal: {s}")

            n_llm = write_experiences(out, to_write, state_experiences=state.experiences)
            print(
                f"  observe batch {b + 1}: write_experience "
                f"+{n_heur} heuristic +{n_llm} llm "
                f"(total={len(state.experiences)})",
                flush=True,
            )
        except Exception as exc:
            print(f"  observe batch {b + 1} JSON parse failed: {exc}", flush=True)
            if n_heur:
                print(
                    f"  (heuristics still wrote +{n_heur}; total={len(state.experiences)})",
                    flush=True,
                )



def run_design(
    client: VLLMClient,
    state: EvolveState,
    cfg: EvolveConfig,
    iteration: int,
) -> dict[str, Any]:
    from language.dataevolve_vmt.experience_store import (
        DEFAULT_VISIBLE_K,
        older_theme_summary,
        visible_experiences,
    )

    print(f"=== Phase 2: Strategy Designer (iter {iteration}) ===", flush=True)
    best = state.best_prompt()
    try:
        payload, raw = _ask_json(
            client,
            de_prompts.DESIGNER_SYSTEM,
            build_designer_user(
                visible_experiences(state.experiences, k=DEFAULT_VISIBLE_K),
                best,
                older_note=older_theme_summary(
                    state.experiences, visible_k=DEFAULT_VISIBLE_K
                ),
            ),
            temperature=cfg.temperature_design,
            max_tokens=cfg.max_tokens,
        )
        content = str(payload.get("cleaning_prompt", "")).strip()
        if not content:
            raise ValueError("empty cleaning_prompt")
        entry = {
            "id": len(state.prompts) + 1,
            "iteration": iteration,
            "prompt_content": content,
            "design_rationale": payload.get("design_rationale"),
            "key_improvements": payload.get("key_improvements"),
            "source_id": best["id"] if best else None,
            "score_objective": None,
            "clean_analysis": None,
            "heuristic_score": None,
        }
    except Exception as exc:
        print(f"  designer failed ({exc}); using fallback incremental prompt", flush=True)
        base = (
            best["prompt_content"]
            if best
            else "Remove content irrelevant to U.S. monthly VMT forecasting."
        )
        extras = "; ".join(state.experiences[-5:]) if state.experiences else "non-US geography, ads"
        content = (
            f"{base}\n\nAdditional rules based on recent failures: {extras}. "
            "Preserve dated U.S. facts with numbers when present. Output cleaned text only."
        )
        entry = {
            "id": len(state.prompts) + 1,
            "iteration": iteration,
            "prompt_content": content,
            "design_rationale": f"fallback after designer error: {exc}",
            "key_improvements": [],
            "source_id": best["id"] if best else None,
            "score_objective": None,
            "clean_analysis": None,
            "heuristic_score": None,
        }
    state.prompts.append(entry)
    print(f"  designed prompt id={entry['id']} chars={len(entry['prompt_content'])}", flush=True)
    return entry


def run_clean(
    client: VLLMClient,
    samples: list[dict[str, Any]],
    prompt_entry: dict[str, Any],
    out_path: Path,
    cfg: EvolveConfig,
    *,
    semaphore=None,
) -> list[dict[str, Any]]:
    print(f"=== Phase 3: Cleaner (prompt {prompt_entry['id']}) ===", flush=True)
    strategy = prompt_entry["prompt_content"]
    users = [build_cleaner_user(strategy, s["text"]) for s in samples]
    texts = client.ask_many(
        users,
        system=de_prompts.CLEANER_SYSTEM,
        concurrency=cfg.clean_concurrency,
        semaphore=semaphore,
        temperature=cfg.temperature_clean,
        max_tokens=cfg.max_tokens,
    )
    pairs = []
    for s, cleaned in zip(samples, texts):
        cleaned = (cleaned or "").strip() or "No text."
        pairs.append(
            {
                "row_id": s.get("row_id"),
                "date": s.get("date"),
                "text": s["text"],
                "refined_text": cleaned,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  cleaned {len(pairs)} docs → {out_path}", flush=True)
    return pairs


def run_judge(
    client: VLLMClient,
    pairs: list[dict[str, Any]],
    prompt_entry: dict[str, Any],
    state: EvolveState,
    cfg: EvolveConfig,
    rng: np.random.Generator,
) -> float:
    """Legacy sample-loop judge; delegates to DataEvolve-aligned Quality Judge."""
    from language.dataevolve_vmt.quality_judge import run_quality_judge

    out_dir = Path(cfg.output_dir)
    qj = run_quality_judge(
        client,
        out_dir=out_dir,
        strategy=prompt_entry["prompt_content"],
        pairs=[{"text": p["text"], "refined_text": p["refined_text"]} for p in pairs],
        state=state,
        family=str(prompt_entry.get("id", "prompt")),
        iteration=int(prompt_entry.get("iteration", 1)),
        judge_batches=1,
        judge_pairs=min(cfg.judge_pairs, len(pairs)),
        temperature=cfg.temperature_judge,
        max_tokens=cfg.max_tokens,
        rng=rng,
    )
    prompt_entry["heuristic_score"] = float(qj.get("heuristic_score", 0.0))
    prompt_entry["score_objective"] = float(qj.get("quality_score", 0.0))
    prompt_entry["clean_analysis"] = str(qj.get("clean_analysis", ""))
    return float(qj.get("quality_score", 0.0))


def run_evolve(client: VLLMClient, cfg: EvolveConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(cfg.sample_jsonl)
    rng = np.random.default_rng(cfg.seed)
    state = EvolveState()

    print(
        f"DataEvolve-VMT: n_samples={len(samples)} iters={cfg.n_iterations} "
        f"out={out_dir}",
        flush=True,
    )
    run_observe(client, samples, state, cfg, rng, out_dir=out_dir)
    (out_dir / "experience_database.jsonl").write_text(
        "\n".join(json.dumps({"issue": x}, ensure_ascii=False) for x in state.experiences)
        + ("\n" if state.experiences else ""),
        encoding="utf-8",
    )

    for it in range(1, cfg.n_iterations + 1):
        print(f"\n========== Iteration {it}/{cfg.n_iterations} ==========", flush=True)
        prompt_entry = run_design(client, state, cfg, it)
        cleaned_path = out_dir / "sample_cleaned_data" / f"{prompt_entry['id']}.jsonl"
        pairs = run_clean(client, samples, prompt_entry, cleaned_path, cfg)
        run_judge(client, pairs, prompt_entry, state, cfg, rng)
        # Persist after each iter
        with open(out_dir / "prompt.jsonl", "w", encoding="utf-8") as f:
            for p in state.prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        (out_dir / "experience_database.jsonl").write_text(
            "\n".join(
                json.dumps({"issue": x}, ensure_ascii=False) for x in state.experiences
            )
            + ("\n" if state.experiences else ""),
            encoding="utf-8",
        )

    best = state.best_prompt()
    summary = {
        "n_samples": len(samples),
        "n_iterations": cfg.n_iterations,
        "n_experiences": len(state.experiences),
        "prompts": state.prompts,
        "best_prompt_id": best["id"] if best else None,
        "best_score": best.get("score_objective") if best else None,
        "best_prompt_preview": (best["prompt_content"][:400] if best else None),
        "config": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if best:
        (out_dir / "best_cleaning_prompt.txt").write_text(
            best["prompt_content"], encoding="utf-8"
        )
    print("\n=== DataEvolve-VMT done ===", flush=True)
    print(json.dumps({k: summary[k] for k in ("best_prompt_id", "best_score", "n_experiences")}, indent=2))
    return summary
