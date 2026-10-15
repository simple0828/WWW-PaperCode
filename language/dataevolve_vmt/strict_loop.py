"""Strict DataEvolve loop aligned with third_party/DataEvolve structure.

Original DataEvolve:
  Phase1 Observer (ONCE on raw samples)
  then each iteration: Designer → Cleaner → Quality Judge
  Judge compares original vs cleaned, writes residual issues,
  and analyzes Coverage + Executability.

Our extensions (kept):
  - APO-style beam + paradigm/Jaccard diversity in Designer
  - After Quality Judge: CFA continue-train on cleaned corpus + MSE
    for beam / global selection (downstream utility).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from language import VLLMClient
from language.apo.cfa_finetune import finetune_cfa_bundle
from language.apo.cfa_trainer import (
    evaluate_cfa_split_mse,
    evaluate_cfa_test_mse,
    evaluate_cfa_val_test_mse,
)
from language.apo.config import build_domain_args
from language.dataevolve_vmt.clean_ops import AlternatingConfig, _full_clean_csv
from language.dataevolve_vmt.designer_diverse import design_round_candidates, jaccard
from language.dataevolve_vmt.experience_store import (
    DEFAULT_NEAR_DUP_JACCARD,
    DEFAULT_VISIBLE_K,
    compact_and_persist,
    load_or_init_experiences,
    persist_experience_snapshot,
)
from language.dataevolve_vmt.pipeline import EvolveConfig, EvolveState, run_observe
from language.dataevolve_vmt.quality_judge import load_clean_pairs, run_quality_judge
from language.dataevolve_vmt.strategy_families import (
    SEED_F0_LIGHT,
    SEED_F1_FILTER,
    SEED_F3_ENTITY,
    SEED_F4_WINDOW,
)


@dataclass
class StrictLoopConfig:
    output_dir: str
    init_bundle: str = (
        "artifacts/ctrl_raw_cfa_continue_gettext_fixed/"
        "cfa_bundle_raw_continue30_gettext_fixed.pt"
    )
    domain: str = "traffic"
    cleaned_csv_name: str = "US_VMT_Month_cleaned.csv"
    n_outer_rounds: int = 5
    # APO-style beam search over cleaning strategies.
    beam_size: int = 3
    mutations_per_parent: int = 2
    # 0 = disable early-stop and always run all n_outer_rounds.
    stagnation_patience: int = 0
    # Beam/global selection uses CFA MSE (our extension).
    select_split: str = "test"
    n_observe_batches: int = 2
    observe_batch_size: int = 5
    # Quality Judge (DataEvolve Phase 4).
    judge_batches: int = 4
    judge_pairs: int = 3
    temperature_judge: float = 0.2
    max_tokens_judge: int = 4096
    experience_visible_k: int = DEFAULT_VISIBLE_K
    near_dup_jaccard: float = DEFAULT_NEAR_DUP_JACCARD
    clean_concurrency: int = 48
    parallel_clean: bool = True
    reuse_raw_control_bundle: str = ""
    max_tokens: int = 4096
    temperature_clean: float = 0.0
    temperature_design: float = 0.55
    temperature_design_explore: float = 0.9
    diversity_jaccard: float = 0.78
    paradigm_shift_ratio: float = 0.5
    cfa_epochs: int = 30
    cfa_patience: int = 10
    # CFA train/eval DataLoader batch (paper default 32; 48 for throughput).
    cfa_batch_size: int = 48
    seed: int = 2021
    raw_csv: str = "data/Traffic/US_VMT_Month.csv"
    text_col: str = "Final_Search_4"
    observe_n: int = 24
    resume_from_round: int = 1

    @property
    def candidates_per_round(self) -> int:
        return int(self.beam_size) * int(self.mutations_per_parent)

    def select_mse(self, row: dict[str, Any]) -> float:
        """CFA MSE used for beam/global selection.

        ``val_test`` (security): mean of finite(val, test). If val loader is
        empty (NaN), falls back to test only — still records both splits.
        """
        split = (self.select_split or "test").strip().lower().replace("+", "_")
        if split in {"val_test", "valtest"}:
            parts = []
            for k in ("val_mse", "test_mse"):
                try:
                    v = float(row[k])
                except Exception:
                    continue
                if v == v:  # not NaN
                    parts.append(v)
            return float(np.mean(parts)) if parts else float("nan")
        if split == "test":
            return float(row["test_mse"])
        return float(row["val_mse"])


def _sample_observe_rows(csv_path: Path, text_col: str, n: int, seed: int) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    idxs = sorted(int(i) for i in rng.choice(len(df), size=min(n, len(df)), replace=False))
    rows = []
    for i in idxs:
        raw = df.iloc[i][text_col]
        text = "" if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else str(raw)
        rows.append({"row_id": i, "date": str(df.iloc[i].get("date", "")), "text": text})
    return rows


def run_strict_loop(client: VLLMClient, cfg: StrictLoopConfig) -> dict[str, Any]:
    from language.dataevolve_vmt.domain_profiles import apply_stage1_domain, get_profile

    profile = get_profile(cfg.domain)
    # Fill defaults from profile when caller left traffic paths on a new domain
    if cfg.domain not in ("traffic", "vmt"):
        if cfg.raw_csv == "data/Traffic/US_VMT_Month.csv":
            cfg.raw_csv = profile.raw_csv
        if cfg.cleaned_csv_name == "US_VMT_Month_cleaned.csv":
            cfg.cleaned_csv_name = profile.cleaned_csv_name
    apo_domain = profile.apo_domain
    apply_stage1_domain(profile)
    print(
        f"Stage1 domain={cfg.domain} apo={apo_domain} task={profile.display_task}\n"
        f"  raw_csv={cfg.raw_csv} cleaned_name={cfg.cleaned_csv_name}",
        flush=True,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfa_args = build_domain_args(apo_domain, mode="cfa", cfa_bundle_path=cfg.init_bundle)
    cfa_args.text_emb = cfa_args.pred_len
    cfa_args.batch_size = int(cfg.cfa_batch_size)
    # Large batch on small Time-MMD val/test: keep leftover rows.
    cfa_args.dataloader_drop_last = 0 if int(cfg.cfa_batch_size) > 32 else 1
    print(
        f"CFA loader batch_size={cfa_args.batch_size} "
        f"drop_last={cfa_args.dataloader_drop_last} "
        f"seq_len={cfa_args.seq_len} pred_len={cfa_args.pred_len}",
        flush=True,
    )
    init_bundle = Path(cfg.init_bundle)

    # Matched raw control once (same init / epochs) for reference.
    raw_ctrl_bundle = out_dir / "cfa_bundle_Traw_control.pt"
    if cfg.reuse_raw_control_bundle and Path(cfg.reuse_raw_control_bundle).is_file():
        src = Path(cfg.reuse_raw_control_bundle)
        print(f"=== Reuse raw control bundle: {src} ===", flush=True)
        if src.resolve() != raw_ctrl_bundle.resolve():
            import shutil

            shutil.copy2(src, raw_ctrl_bundle)
        if (cfg.select_split or "").replace("+", "_") in {"val_test", "valtest"}:
            raw_ctrl_select, raw_ctrl_val, raw_ctrl_test = evaluate_cfa_val_test_mse(
                cfa_args, raw_ctrl_bundle
            )
        else:
            raw_ctrl_val = evaluate_cfa_split_mse(cfa_args, raw_ctrl_bundle, split="val")
            raw_ctrl_test = evaluate_cfa_test_mse(cfa_args, raw_ctrl_bundle)
            raw_ctrl_select = cfg.select_mse(
                {"val_mse": raw_ctrl_val, "test_mse": raw_ctrl_test}
            )
        init_test = evaluate_cfa_test_mse(cfa_args, init_bundle)
    else:
        print("=== Raw control CFA continue (once) ===", flush=True)
        finetune_cfa_bundle(
            cfa_args,
            init_bundle_path=init_bundle,
            out_bundle_path=raw_ctrl_bundle,
            train_epochs=cfg.cfa_epochs,
            patience=cfg.cfa_patience,
        )
        if (cfg.select_split or "").replace("+", "_") in {"val_test", "valtest"}:
            raw_ctrl_select, raw_ctrl_val, raw_ctrl_test = evaluate_cfa_val_test_mse(
                cfa_args, raw_ctrl_bundle
            )
        else:
            raw_ctrl_val = evaluate_cfa_split_mse(cfa_args, raw_ctrl_bundle, split="val")
            raw_ctrl_test = evaluate_cfa_test_mse(cfa_args, raw_ctrl_bundle)
            raw_ctrl_select = cfg.select_mse(
                {"val_mse": raw_ctrl_val, "test_mse": raw_ctrl_test}
            )
        init_test = evaluate_cfa_test_mse(cfa_args, init_bundle)
    # Small Time-MMD panels (e.g. security) can have empty val loaders under drop_last.
    if raw_ctrl_val != raw_ctrl_val:  # NaN
        print(
            "  WARN: raw_ctrl val MSE is NaN (empty val loader even w/ drop_last=False?); "
            "Δval reference falls back to test; "
            f"select_split={cfg.select_split}.",
            flush=True,
        )
        raw_ctrl_val = float(raw_ctrl_test)
        raw_ctrl_select = cfg.select_mse(
            {"val_mse": raw_ctrl_val, "test_mse": raw_ctrl_test}
        )
    print(
        f"  init_test={init_test:.6f} | T-raw control val={raw_ctrl_val:.6f} "
        f"test={raw_ctrl_test:.6f} select({cfg.select_split})={raw_ctrl_select:.6f}",
        flush=True,
    )

    observe_rows = _sample_observe_rows(
        Path(cfg.raw_csv), cfg.text_col, cfg.observe_n, cfg.seed
    )
    evo_cfg = EvolveConfig(
        sample_jsonl="",
        output_dir=str(out_dir),
        n_observe_batches=cfg.n_observe_batches,
        observe_batch_size=cfg.observe_batch_size,
        clean_concurrency=cfg.clean_concurrency,
        temperature_clean=cfg.temperature_clean,
        temperature_design=cfg.temperature_design,
        temperature_judge=cfg.temperature_judge,
        judge_pairs=cfg.judge_pairs,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )
    design_cfg = AlternatingConfig(
        output_dir=str(out_dir),
        init_bundle=cfg.init_bundle,
        temperature_design=cfg.temperature_design,
        max_tokens=cfg.max_tokens,
    )

    state = EvolveState()
    bootstrap = [
        f"Goal: improve usefulness of auxiliary news for {profile.display_task}.",
        "Evolve cleaning rules from observed data issues and Quality Judge Coverage/Executability gaps.",
        f"Downstream utility (extension): CFA continue-train MSE; "
        f"raw-ctrl val={raw_ctrl_val:.6f} test={raw_ctrl_test:.6f}.",
    ]
    rng = np.random.default_rng(cfg.seed)

    beam: list[dict[str, Any]] = []
    strategy_pool: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    global_best: Optional[dict[str, Any]] = None
    stagnant = 0
    start_round = max(1, int(cfg.resume_from_round or 1))

    if start_round > 1:
        prev_beam_path = out_dir / f"round_{start_round - 1}" / "beam.json"
        if not prev_beam_path.is_file():
            raise FileNotFoundError(
                f"resume_from_round={start_round} needs {prev_beam_path}"
            )
        beam = json.loads(prev_beam_path.read_text(encoding="utf-8"))
        best_path = out_dir / "best_strategy.json"
        if best_path.is_file():
            global_best = json.loads(best_path.read_text(encoding="utf-8"))
        pool_path = out_dir / "strategy_pool.jsonl"
        if pool_path.is_file():
            strategy_pool = [
                json.loads(line)
                for line in pool_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        state.experiences = load_or_init_experiences(out_dir, bootstrap=bootstrap)
        stagnant = 1 if start_round >= 3 and global_best is not None else 0
        print(
            f"=== RESUME from round {start_round}/{cfg.n_outer_rounds} "
            f"(beam={[b.get('family') for b in beam]}, "
            f"global_best={None if global_best is None else global_best.get('family')}, "
            f"stagnant={stagnant}, n_exp={len(state.experiences)}) ===",
            flush=True,
        )
        print(
            "=== Observer SKIPPED on resume (already done once at pipeline start) ===",
            flush=True,
        )
    else:
        # ---- Phase 1: Observer ONCE on raw samples (DataEvolve) ----
        print(
            "=== Phase 1: Observer (ONCE on raw samples; not every round) ===",
            flush=True,
        )
        state.experiences = list(bootstrap)
        persist_experience_snapshot(out_dir, state.experiences)  # seed DB before read_experience
        run_observe(client, observe_rows, state, evo_cfg, rng, out_dir=out_dir)
        _, n_drop = compact_and_persist(
            out_dir,
            state.experiences,
            near_dup_jaccard=cfg.near_dup_jaccard,
        )
        persist_experience_snapshot(out_dir, state.experiences)
        print(
            f"  Observer done: n_experiences={len(state.experiences)} "
            f"(near-dup dropped={n_drop}) "
            f"→ {out_dir / 'experience_database.jsonl'}",
            flush=True,
        )

    print(
        f"Loop: Observer(once) → per-round Designer(APO beam)→Cleaner→"
        f"QualityJudge(Coverage/Executability)→CFA+MSE; "
        f"beam={cfg.beam_size}×mut={cfg.mutations_per_parent}; "
        f"select={cfg.select_split}; judge_batches={cfg.judge_batches}×"
        f"{cfg.judge_pairs}; max_tokens_judge={cfg.max_tokens_judge}; "
        f"exp_visible={cfg.experience_visible_k}; near_dup={cfg.near_dup_jaccard:.2f}; "
        f"T_clean={cfg.temperature_clean}",
        flush=True,
    )

    for rnd in range(start_round, cfg.n_outer_rounds + 1):
        round_dir = out_dir / f"round_{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== Strict round {rnd}/{cfg.n_outer_rounds} ==========", flush=True)
        print(
            "  (Observer not re-run; residual issues come from Quality Judge)",
            flush=True,
        )

        # Snapshot experiences at round start (post-observer / prior judges).
        (round_dir / "experience_snapshot_pre.jsonl").write_text(
            "\n".join(
                json.dumps({"issue": x}, ensure_ascii=False) for x in state.experiences
            )
            + "\n",
            encoding="utf-8",
        )

        # ---- Phase 2: Designer (APO diversity extension) ----
        print("=== Phase 2: Designer (beam continue + APO diversity) ===", flush=True)
        if beam:
            parents = beam[: cfg.beam_size]
            print(
                f"  continuing from R{rnd - 1} beam "
                f"(NOT from scratch): {[p.get('family') for p in parents]}",
                flush=True,
            )
        else:
            from language.dataevolve_vmt.domain_profiles import seeds_for_domain

            catalog = list(seeds_for_domain(cfg.domain))
            parents = [
                {
                    "id": s.family,
                    "cleaning_prompt": s.cleaning_prompt,
                    "score_objective": None,
                    "clean_analysis": f"seed:{s.name}",
                    "family": s.family,
                }
                for s in catalog[: cfg.beam_size]
            ]
            print(
                f"  round-1 seed parents={[p['family'] for p in parents]} "
                f"(from {len(catalog)} domain seeds)",
                flush=True,
            )

        seeds = design_round_candidates(
            client,
            experiences=state.experiences,
            parents=parents,
            cfg=design_cfg,
            round_id=rnd,
            mutations_per_parent=cfg.mutations_per_parent,
            diversity_jaccard=cfg.diversity_jaccard,
            paradigm_shift_ratio=cfg.paradigm_shift_ratio,
            temperature_exploit=cfg.temperature_design,
            temperature_explore=cfg.temperature_design_explore,
        )
        for i, s in enumerate(seeds):
            sims = [
                jaccard(s.cleaning_prompt, o.cleaning_prompt)
                for j, o in enumerate(seeds)
                if j != i
            ]
            max_sim = max(sims) if sims else 0.0
            print(
                f"  diversity {s.family}: max_jaccard_vs_peers={max_sim:.3f}",
                flush=True,
            )

        round_cand_rows: list[dict[str, Any]] = []

        # ---- Phase 3: Cleaner ----
        clean_jobs: list[dict[str, Any]] = []
        for seed in seeds:
            cand_dir = round_dir / seed.family
            cand_dir.mkdir(parents=True, exist_ok=True)
            clean_jobs.append(
                {
                    "seed": seed,
                    "cand_dir": cand_dir,
                    "clean_csv": cand_dir / cfg.cleaned_csv_name,
                    "clean_jsonl": cand_dir / "cleaned_texts.jsonl",
                }
            )

        sem = (
            threading.Semaphore(cfg.clean_concurrency)
            if cfg.parallel_clean and len(clean_jobs) > 1
            else None
        )
        per_job_conc = (
            cfg.clean_concurrency if sem is None else max(4, cfg.clean_concurrency)
        )

        def _clean_one(job: dict[str, Any]) -> dict[str, Any]:
            seed = job["seed"]
            t0 = time.time()
            print(f"\n=== Phase 3: Cleaner [{seed.family}] start ===", flush=True)
            stats = _full_clean_csv(
                client,
                prompt=seed.cleaning_prompt,
                raw_csv=Path(cfg.raw_csv),
                text_col=cfg.text_col,
                out_csv=job["clean_csv"],
                out_jsonl=job["clean_jsonl"],
                concurrency=per_job_conc,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature_clean,
                semaphore=sem,
                prompt_id=seed.family,
            )
            elapsed = time.time() - t0
            print(
                f"=== Cleaner [{seed.family}] done "
                f"retention={stats['retention_mean']:.3f} "
                f"wall={elapsed:.1f}s ===",
                flush=True,
            )
            return {**job, "clean_stats": stats, "clean_seconds": elapsed}

        clean_t0 = time.time()
        if sem is not None:
            print(
                f"=== Parallel Cleaner: {len(clean_jobs)} jobs, "
                f"shared vLLM budget={cfg.clean_concurrency} ===",
                flush=True,
            )
            cleaned_jobs: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(clean_jobs)) as pool:
                futs = [pool.submit(_clean_one, j) for j in clean_jobs]
                for fut in as_completed(futs):
                    cleaned_jobs.append(fut.result())
            by_fam = {j["seed"].family: j for j in cleaned_jobs}
            cleaned_jobs = [by_fam[j["seed"].family] for j in clean_jobs]
        else:
            cleaned_jobs = [_clean_one(j) for j in clean_jobs]
        clean_wall = time.time() - clean_t0
        print(
            f"=== Cleaner wave wall={clean_wall:.1f}s "
            f"(sum_job={sum(j['clean_seconds'] for j in cleaned_jobs):.1f}s) ===",
            flush=True,
        )

        # ---- Phase 4: Quality Judge then CFA+MSE (serial per candidate) ----
        for job in cleaned_jobs:
            seed = job["seed"]
            cand_dir = job["cand_dir"]
            clean_csv = job["clean_csv"]
            clean_stats = job["clean_stats"]
            pairs = load_clean_pairs(job["clean_jsonl"])

            qj = run_quality_judge(
                client,
                out_dir=out_dir,
                strategy=seed.cleaning_prompt,
                pairs=pairs,
                state=state,
                family=seed.family,
                iteration=rnd,
                judge_batches=cfg.judge_batches,
                judge_pairs=cfg.judge_pairs,
                temperature=cfg.temperature_judge,
                max_tokens=cfg.max_tokens_judge,
                experience_visible_k=cfg.experience_visible_k,
                near_dup_jaccard=cfg.near_dup_jaccard,
                rng=rng,
            )
            (cand_dir / "quality_judge.json").write_text(
                json.dumps(qj, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            persist_experience_snapshot(out_dir, state.experiences)

            print(f"=== Extension: CFA+MSE [{seed.family}] ===", flush=True)
            clean_args = deepcopy(cfa_args)
            clean_args.root_path = str(clean_csv.parent.resolve())
            clean_args.data_path = clean_csv.name
            out_bundle = cand_dir / "cfa_bundle.pt"
            finetune_cfa_bundle(
                clean_args,
                init_bundle_path=init_bundle,
                out_bundle_path=out_bundle,
                train_epochs=cfg.cfa_epochs,
                patience=cfg.cfa_patience,
            )
            val_mse = evaluate_cfa_split_mse(clean_args, out_bundle, split="val")
            test_mse = evaluate_cfa_test_mse(clean_args, out_bundle)
            if (cfg.select_split or "").replace("+", "_") in {"val_test", "valtest"}:
                select_mse, val_mse_nd, test_mse_nd = evaluate_cfa_val_test_mse(
                    clean_args, out_bundle
                )
                # Prefer no-drop_last metrics for logging when using val_test.
                if val_mse_nd == val_mse_nd:
                    val_mse = val_mse_nd
                if test_mse_nd == test_mse_nd:
                    test_mse = test_mse_nd
            else:
                select_mse = cfg.select_mse({"val_mse": val_mse, "test_mse": test_mse})
            score = float(-select_mse)
            delta_vs_raw_ctrl_test = test_mse - raw_ctrl_test
            delta_vs_raw_ctrl_val = val_mse - raw_ctrl_val

            # Designer-facing diagnosis: Quality Judge analysis + CFA utility.
            clean_analysis = (
                f"{qj.get('clean_analysis', '')} || "
                f"CFA select={cfg.select_split}:{select_mse:.6f} "
                f"val={val_mse:.6f} test={test_mse:.6f} "
                f"Δtest_vs_raw={delta_vs_raw_ctrl_test:+.6f}; "
                f"quality_score={float(qj.get('quality_score', 0.0)):.2f}"
            )

            entry = {
                "round": rnd,
                "family": seed.family,
                "name": seed.name,
                "output_format": seed.output_format,
                "cleaning_prompt": seed.cleaning_prompt,
                "clean_stats": clean_stats,
                "clean_seconds": job["clean_seconds"],
                "clean_wave_seconds": clean_wall,
                "bundle": str(out_bundle),
                "cleaned_csv": str(clean_csv),
                "quality_score": float(qj.get("quality_score", 0.0)),
                "quality_llm_score": float(qj.get("average_score", 0.0)),
                "quality_heuristic": float(qj.get("heuristic_score", 0.0)),
                "coverage_analysis": qj.get("coverage_analysis", ""),
                "executability_analysis": qj.get("executability_analysis", ""),
                "new_issues_written": int(qj.get("new_issues_written", 0)),
                "val_mse": val_mse,
                "test_mse": test_mse,
                "select_split": cfg.select_split,
                "select_mse": select_mse,
                "score_objective": score,
                "delta_val_vs_raw_ctrl": delta_vs_raw_ctrl_val,
                "delta_test_vs_raw_ctrl": delta_vs_raw_ctrl_test,
                "clean_analysis": clean_analysis,
            }
            round_cand_rows.append(entry)
            strategy_pool.append(entry)
            state.prompts.append(
                {
                    "id": len(state.prompts) + 1,
                    "iteration": rnd,
                    "prompt_content": seed.cleaning_prompt,
                    "score_objective": score,
                    "quality_score": entry["quality_score"],
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                    "family": seed.family,
                    "clean_analysis": clean_analysis,
                }
            )
            print(
                f"  [{seed.family}] quality={entry['quality_score']:.2f} "
                f"val={val_mse:.6f} test={test_mse:.6f} "
                f"Δtest_vs_raw={delta_vs_raw_ctrl_test:+.6f} "
                f"residual_writes={entry['new_issues_written']}",
                flush=True,
            )

        # Full-DB near-dup compact once per round (keep chronological first).
        before_n = len(state.experiences)
        _, n_dropped = compact_and_persist(
            out_dir,
            state.experiences,
            near_dup_jaccard=cfg.near_dup_jaccard,
        )
        if n_dropped:
            print(
                f"  experience compact: {before_n} → {len(state.experiences)} "
                f"(dropped {n_dropped} near-dups @ jaccard≥{cfg.near_dup_jaccard:.2f})",
                flush=True,
            )

        # ---- Beam update by CFA select_split (extension) ----
        round_best = min(round_cand_rows, key=cfg.select_mse)
        ranked = sorted(strategy_pool, key=cfg.select_mse)
        beam = []
        seen_fam = set()
        for s in ranked:
            fam_root = str(s["family"]).split("_")[0]
            if fam_root in seen_fam:
                continue
            beam.append(
                {
                    "id": s["family"],
                    "cleaning_prompt": s["cleaning_prompt"],
                    "score_objective": s["score_objective"],
                    "clean_analysis": s["clean_analysis"],
                    "family": s["family"],
                    "val_mse": s["val_mse"],
                    "test_mse": s["test_mse"],
                    "quality_score": s.get("quality_score"),
                    "select_mse": s.get("select_mse", cfg.select_mse(s)),
                }
            )
            seen_fam.add(fam_root)
            if len(beam) >= cfg.beam_size:
                break
        if len(beam) < cfg.beam_size:
            for s in ranked:
                if any(b["id"] == s["family"] for b in beam):
                    continue
                beam.append(
                    {
                        "id": s["family"],
                        "cleaning_prompt": s["cleaning_prompt"],
                        "score_objective": s["score_objective"],
                        "clean_analysis": s["clean_analysis"],
                        "family": s["family"],
                        "val_mse": s["val_mse"],
                        "test_mse": s["test_mse"],
                        "quality_score": s.get("quality_score"),
                        "select_mse": s.get("select_mse", cfg.select_mse(s)),
                    }
                )
                if len(beam) >= cfg.beam_size:
                    break
        beam_scores = [f"{b.get('select_mse', float('nan')):.4f}" for b in beam]
        print(
            f"  next beam (by CFA {cfg.select_split})={[b['family'] for b in beam]} "
            f"scores={beam_scores}",
            flush=True,
        )

        (round_dir / "candidates.json").write_text(
            json.dumps(round_cand_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (round_dir / "beam.json").write_text(
            json.dumps(beam, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (round_dir / "best_cleaning_prompt.txt").write_text(
            round_best["cleaning_prompt"], encoding="utf-8"
        )
        (round_dir / "experience_snapshot_post.jsonl").write_text(
            "\n".join(
                json.dumps({"issue": x}, ensure_ascii=False) for x in state.experiences
            )
            + "\n",
            encoding="utf-8",
        )

        improved = False
        if global_best is None or cfg.select_mse(round_best) < cfg.select_mse(global_best) - 1e-8:
            global_best = round_best
            improved = True
            stagnant = 0
            (out_dir / "best_cleaning_prompt.txt").write_text(
                round_best["cleaning_prompt"], encoding="utf-8"
            )
            (out_dir / "best_strategy.json").write_text(
                json.dumps(round_best, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  ★ new global best CFA {cfg.select_split}={cfg.select_mse(round_best):.6f} "
                f"(val={round_best['val_mse']:.6f} test={round_best['test_mse']:.6f} "
                f"quality={round_best.get('quality_score', float('nan')):.2f})",
                flush=True,
            )
        else:
            stagnant += 1
            print(
                f"  no global CFA improvement on {cfg.select_split} "
                f"(stagnant={stagnant}"
                f"{'' if cfg.stagnation_patience <= 0 else f'/{cfg.stagnation_patience}'})",
                flush=True,
            )

        history.append(
            {
                "round": rnd,
                "n_candidates": len(round_cand_rows),
                "beam_size": cfg.beam_size,
                "mutations_per_parent": cfg.mutations_per_parent,
                "select_split": cfg.select_split,
                "n_experiences": len(state.experiences),
                "beam_families": [b["family"] for b in beam],
                "candidates": [
                    {
                        "family": c["family"],
                        "quality_score": c.get("quality_score"),
                        "val_mse": c["val_mse"],
                        "test_mse": c["test_mse"],
                        "select_mse": c.get("select_mse"),
                        "new_issues_written": c.get("new_issues_written"),
                        "delta_test_vs_raw_ctrl": c["delta_test_vs_raw_ctrl"],
                    }
                    for c in round_cand_rows
                ],
                "round_best_family": round_best["family"],
                "round_best_val": round_best["val_mse"],
                "round_best_test": round_best["test_mse"],
                "round_best_quality": round_best.get("quality_score"),
                "round_best_select": cfg.select_mse(round_best),
                "global_best_val": global_best["val_mse"] if global_best else None,
                "global_best_test": global_best["test_mse"] if global_best else None,
                "improved": improved,
            }
        )
        persist_experience_snapshot(out_dir, state.experiences)
        (out_dir / "strategy_pool.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in strategy_pool) + "\n",
            encoding="utf-8",
        )

        if cfg.stagnation_patience > 0 and stagnant >= cfg.stagnation_patience:
            print(
                f"\nEarly stop: no {cfg.select_split} improvement for "
                f"{cfg.stagnation_patience} rounds.",
                flush=True,
            )
            break

    summary = {
        "mode": "dataevolve_aligned_observer_once_qj_cfa_apo",
        "loop": (
            "Observer(ONCE,raw)→"
            "[Designer(APO beam+paradigm+Jaccard)→Cleaner→"
            "QualityJudge(Coverage+Executability,write residual)→"
            "CFA+MSE select]×N"
        ),
        "beam_size": cfg.beam_size,
        "mutations_per_parent": cfg.mutations_per_parent,
        "candidates_per_round": cfg.candidates_per_round,
        "select_split": cfg.select_split,
        "judge_batches": cfg.judge_batches,
        "judge_pairs": cfg.judge_pairs,
        "max_tokens_judge": cfg.max_tokens_judge,
        "experience_visible_k": cfg.experience_visible_k,
        "near_dup_jaccard": cfg.near_dup_jaccard,
        "init_bundle": cfg.init_bundle,
        "init_test_mse": init_test,
        "raw_control": {
            "bundle": str(raw_ctrl_bundle),
            "val_mse": raw_ctrl_val,
            "test_mse": raw_ctrl_test,
        },
        "n_rounds_ran": len(history),
        "n_strategies_judged": len(strategy_pool),
        "n_experiences": len(state.experiences),
        "history": history,
        "global_best": global_best,
        "config": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== DataEvolve-aligned Strict loop done ===", flush=True)
    if global_best:
        print(
            json.dumps(
                {
                    "best_family": global_best["family"],
                    "select_split": cfg.select_split,
                    "best_select": cfg.select_mse(global_best),
                    "best_val": global_best["val_mse"],
                    "best_test": global_best["test_mse"],
                    "best_quality": global_best.get("quality_score"),
                    "raw_ctrl_test": raw_ctrl_test,
                    "delta_test": global_best["test_mse"] - raw_ctrl_test,
                    "n_experiences": len(state.experiences),
                },
                indent=2,
            ),
            flush=True,
        )
    return summary
