from __future__ import annotations

import json
import random
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from language import VLLMClient
from language.apo.baseline import BaselineSample, UnimodalBaselinePredictor
from language.apo.reward import REWARD_MODE_LABELS
from language.apo.runner import (
    APORunner,
    SEED_PROMPTS,
    apo_task_description,
    infer_apo_domain,
    seed_prompts_for_domain,
)
from language.apo.paradigm_prompts import PARADIGM_NAMES


@dataclass
class PromptCandidate:
    candidate_id: int
    meta_prompt: str
    mean_reward: float
    per_sample_rewards: List[float]
    round_index: int
    parent_id: Optional[int] = None
    source: str = "seed"
    mutation_strategy: Optional[str] = None
    selection_reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def rank_reward(self) -> float:
        """Reward used for beam ranking / final pick (val when set, else train)."""
        if self.selection_reward is not None:
            return float(self.selection_reward)
        return float(self.mean_reward)


@dataclass
class BeamRoundSnapshot:
    round_index: int
    beam: List[PromptCandidate]
    best_mean_reward: float
    evaluated_candidates: int


@dataclass
class BeamTrainResult:
    train_samples: List[BaselineSample]
    test_samples: List[BaselineSample] = field(default_factory=list)
    selection_samples: List[BaselineSample] = field(default_factory=list)
    history: List[BeamRoundSnapshot] = field(default_factory=list)
    best_candidate: Optional[PromptCandidate] = None
    initial_mean_reward: float = 0.0
    final_mean_reward: float = 0.0
    final_selection_reward: Optional[float] = None
    test_seed_reward: Optional[float] = None
    test_best_reward: Optional[float] = None
    summary: str = ""
    selection_mode: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_mode": self.selection_mode,
            "initial_mean_reward": self.initial_mean_reward,
            "final_mean_reward": self.final_mean_reward,
            "final_selection_reward": self.final_selection_reward,
            "test_seed_reward": self.test_seed_reward,
            "test_best_reward": self.test_best_reward,
            "summary": self.summary,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "train_windows": [s.window_index for s in self.train_samples],
            "selection_windows": [s.window_index for s in self.selection_samples],
            "test_windows": [s.window_index for s in self.test_samples],
            "history": [
                {
                    "round_index": snap.round_index,
                    "best_mean_reward": snap.best_mean_reward,
                    "evaluated_candidates": snap.evaluated_candidates,
                    "beam": [c.to_dict() for c in snap.beam],
                }
                for snap in self.history
            ],
        }


class MultiSampleBeamTrainer(APORunner):
    """Multi-sample beam search with diverse mutation strategies."""

    def __init__(
        self,
        baseline: UnimodalBaselinePredictor,
        scorer,
        client: VLLMClient,
        output_dir: str | Path,
        *,
        rewrite_temperature: float = 0.4,
        mutation_temperature: float = 0.9,
        diversity_jaccard: float = 0.78,
        paradigm_shift_ratio: float = 0.5,
    ):
        super().__init__(
            baseline,
            scorer,
            client,
            output_dir,
            rewrite_temperature=rewrite_temperature,
        )
        self.mutation_temperature = mutation_temperature
        self.diversity_jaccard = diversity_jaccard
        self.paradigm_shift_ratio = paradigm_shift_ratio
        self.rewrite_concurrency = 48
        self.cfa_batch_size = 16  # unified with rewrite / ProTeGi minibatch by default
        # ProTeGi-style knobs (minibatch MSE → textual gradients → prompt edits)
        self.minibatch_size = 48  # aligned with rewrite_concurrency for one parallel wave
        self.errors_per_gradient = 4
        self.gradients_per_error = 3
        self.steps_per_gradient = 1
        # How many fresh train minibatches to draw per parent when expanding.
        self.n_gradient_samples = 2
        self._next_candidate_id = 0
        self._rewrite_cache: Dict[Tuple[str, str, int], str] = {}
        self._reward_cache: Dict[Tuple[str, str, int], float] = {}
        self.apo_domain = infer_apo_domain(baseline.args)
        self.apo_task = apo_task_description(self.apo_domain)

    def _optimization_task_line(self) -> str:
        return (
            f"I'm optimizing a residual-extraction meta-prompt for {self.apo_task}."
        )

    def _reward_goal_line(self) -> str:
        mode = getattr(self.scorer, "reward_mode", "cfa_text_delta")
        label = REWARD_MODE_LABELS.get(mode, mode)
        return f"Goal: maximize reward = {label}."

    def _new_candidate_id(self) -> int:
        self._next_candidate_id += 1
        return self._next_candidate_id

    @staticmethod
    def _norm_prompt(prompt: str) -> str:
        return " ".join(prompt.split())

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def load_samples(self, split: str, window_indices: Sequence[int]) -> List[BaselineSample]:
        return [self.baseline.get_sample(split, idx) for idx in window_indices]

    def rewrite_with_prompt(self, sample: BaselineSample, meta_prompt: str) -> str:
        key = (self._norm_prompt(meta_prompt), sample.split, sample.window_index)
        if key not in self._rewrite_cache:
            self._rewrite_cache[key] = super().rewrite_with_prompt(sample, meta_prompt)
        return self._rewrite_cache[key]

    def rewrite_batch(
        self,
        samples: Sequence[BaselineSample],
        meta_prompt: str,
    ) -> List[str]:
        """Rewrite many windows concurrently; fill cache and preserve order."""
        prompt_key = self._norm_prompt(meta_prompt)
        out: List[Optional[str]] = [None] * len(samples)
        pending_idx: List[int] = []
        pending_prompts: List[str] = []

        for i, sample in enumerate(samples):
            cache_key = (prompt_key, sample.split, sample.window_index)
            cached = self._rewrite_cache.get(cache_key)
            if cached is not None:
                out[i] = cached
            else:
                pending_idx.append(i)
                pending_prompts.append(self._fill_template(meta_prompt, sample))

        if pending_prompts:
            print(
                f"  rewrite batch: {len(pending_prompts)} prompts "
                f"(concurrency={self.rewrite_concurrency})",
                flush=True,
            )
            texts = self.client.ask_many(
                pending_prompts,
                temperature=self.rewrite_temperature,
                enable_thinking=False,
                max_tokens=1024,
                concurrency=self.rewrite_concurrency,
            )
            for i, text in zip(pending_idx, texts):
                sample = samples[i]
                cleaned = text.strip()
                self._rewrite_cache[
                    (prompt_key, sample.split, sample.window_index)
                ] = cleaned
                out[i] = cleaned

        return [t if t is not None else "" for t in out]

    def evaluate_prompt(
        self,
        meta_prompt: str,
        samples: Sequence[BaselineSample],
        round_index: int,
        parent_id: Optional[int] = None,
        source: str = "evaluated",
        mutation_strategy: Optional[str] = None,
    ) -> PromptCandidate:
        rewards: List[float] = [0.0] * len(samples)
        prompt_key = self._norm_prompt(meta_prompt)
        total = len(samples)

        need_score: List[int] = []
        for i, sample in enumerate(samples):
            cache_key = (prompt_key, sample.split, sample.window_index)
            if cache_key in self._reward_cache:
                rewards[i] = self._reward_cache[cache_key]
            else:
                need_score.append(i)

        if need_score:
            todo_samples = [samples[i] for i in need_score]
            print(
                f"  score {source}/{mutation_strategy or '-'} "
                f"{len(need_score)}/{total} windows (batch rewrite)",
                flush=True,
            )
            rewrites = self.rewrite_batch(todo_samples, meta_prompt)
            print(
                f"  CFA batch score: {len(todo_samples)} windows "
                f"(cfa_batch_size={self.cfa_batch_size})",
                flush=True,
            )
            breakdowns = self.scorer.score_batch(
                list(todo_samples),
                list(rewrites),
                batch_size=self.cfa_batch_size,
            )
            for local_i, sample_i in enumerate(need_score):
                sample = samples[sample_i]
                reward = float(breakdowns[local_i].reward)
                self._reward_cache[
                    (prompt_key, sample.split, sample.window_index)
                ] = reward
                rewards[sample_i] = reward
                if (local_i + 1) % 50 == 0 or local_i + 1 == len(need_score):
                    print(
                        f"  scored [{local_i + 1}/{len(need_score)}] "
                        f"{sample.split}#{sample.window_index} reward={reward:.6f}",
                        flush=True,
                    )

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        return PromptCandidate(
            candidate_id=self._new_candidate_id(),
            meta_prompt=meta_prompt,
            mean_reward=mean_reward,
            per_sample_rewards=rewards,
            round_index=round_index,
            parent_id=parent_id,
            source=source,
            mutation_strategy=mutation_strategy,
        )

    def _rewards_on_samples(
        self,
        meta_prompt: str,
        samples: Sequence[BaselineSample],
    ) -> List[float]:
        """Score a minibatch under meta_prompt (cache + one concurrent rewrite wave)."""
        prompt_key = self._norm_prompt(meta_prompt)
        rewards: List[Optional[float]] = [None] * len(samples)
        need_idx: List[int] = []
        for i, sample in enumerate(samples):
            cache_key = (prompt_key, sample.split, sample.window_index)
            if cache_key in self._reward_cache:
                rewards[i] = self._reward_cache[cache_key]
            else:
                need_idx.append(i)
        if need_idx:
            todo = [samples[i] for i in need_idx]
            print(
                f"  minibatch score: {len(todo)} windows "
                f"(concurrency={self.rewrite_concurrency})",
                flush=True,
            )
            rewrites = self.rewrite_batch(todo, meta_prompt)
            breakdowns = self.scorer.score_batch(
                list(todo),
                list(rewrites),
                batch_size=self.cfa_batch_size,
            )
            for local_i, sample_i in enumerate(need_idx):
                sample = samples[sample_i]
                reward = float(breakdowns[local_i].reward)
                self._reward_cache[
                    (prompt_key, sample.split, sample.window_index)
                ] = reward
                rewards[sample_i] = reward
        return [float(r) if r is not None else 0.0 for r in rewards]

    def _sample_minibatch(
        self,
        samples: Sequence[BaselineSample],
        batch_size: Optional[int] = None,
    ) -> List[BaselineSample]:
        pool = list(samples)
        k = batch_size if batch_size is not None else self.minibatch_size
        k = max(1, min(k, len(pool)))
        return random.sample(pool, k=k)

    def _mse_error_string_from_minibatch(
        self,
        meta_prompt: str,
        samples: Sequence[BaselineSample],
        rewards: Sequence[float],
        n_errors: Optional[int] = None,
    ) -> str:
        """Build MSE/residual error string from worst windows inside one minibatch."""
        n = n_errors if n_errors is not None else self.errors_per_gradient
        indexed = list(zip(samples, rewards))
        indexed.sort(key=lambda item: item[1])
        worst = indexed[: max(1, n)]
        if len(indexed) > n:
            pool = indexed[n : min(len(indexed), n + 8)]
            if pool:
                worst = worst[:-1] + [random.choice(pool)]

        blocks: List[str] = []
        worst_samples = [sample for sample, _ in worst]
        worst_rewards = [reward for _, reward in worst]
        rewrites = self.rewrite_batch(worst_samples, meta_prompt)
        breakdowns = self.scorer.score_batch(
            list(worst_samples),
            list(rewrites),
            batch_size=self.cfa_batch_size,
        )
        reward_label = REWARD_MODE_LABELS.get(
            getattr(self.scorer, "reward_mode", "cfa_text_delta"),
            "reward",
        )
        for idx, (sample, sample_reward, rewrite, breakdown) in enumerate(
            zip(worst_samples, worst_rewards, rewrites, breakdowns),
            start=1,
        ):
            blocks.append(
                textwrap.dedent(
                    f"""
                    ## Example {idx}
                    Window={sample.window_index} split={sample.split}
                    sample_reward={sample_reward:.6f} (want maximize: {reward_label})
                    {breakdown.critique()}
                    rewrite_excerpt={rewrite[:450]}
                    """
                ).strip()
            )
        return "\n\n".join(blocks)

    # Keep old name as alias for any callers / logs.
    def _failure_report(
        self,
        parent: PromptCandidate,
        samples: Sequence[BaselineSample],
    ) -> str:
        rewards = self._rewards_on_samples(parent.meta_prompt, samples)
        return self._mse_error_string_from_minibatch(
            parent.meta_prompt, samples, rewards
        )

    @staticmethod
    def parse_tagged_text(text: str, start_tag: str = "<START>", end_tag: str = "<END>") -> List[str]:
        """Parse ProTeGi-style <START>...</END> segments."""
        texts: List[str] = []
        cursor = text
        while True:
            start_index = cursor.find(start_tag)
            if start_index == -1:
                break
            end_index = cursor.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(cursor[start_index:end_index].strip())
            cursor = cursor[end_index + len(end_tag) :]
        if texts:
            return texts
        # Fallback: numbered lines / bullets if the model ignored tags.
        bullets = re.findall(r"(?:^|\n)\s*(?:\d+[\).\]]|-)\s+(.+)", text)
        return [b.strip() for b in bullets if len(b.strip()) > 8]

    def _get_textual_gradients(
        self,
        meta_prompt: str,
        error_string: str,
        num_feedbacks: Optional[int] = None,
    ) -> List[str]:
        """ProTeGi `_get_gradients`: LLM criticizes the prompt from MSE errors."""
        k = num_feedbacks if num_feedbacks is not None else self.gradients_per_error
        gradient_prompt = f"""{self._optimization_task_line()}

{self._reward_goal_line()}
The prompt must extract residual drivers that move fusion toward y_true, not summarize news.

My current meta-prompt is:
\"\"\"{meta_prompt}\"\"\"

But this prompt gets the following examples wrong or under-rewarded (MSE / residual diagnostics):
{error_string}

Give {k} concrete reasons why this prompt could have produced rewrites that hurt or under-help fusion MSE.
Focus on missing residual directions, weak constraints, bad schema, or leaking generic encyclopedia text.
Wrap each reason with <START> and <END>.
"""
        raw = self.client.ask(
            gradient_prompt,
            temperature=self.mutation_temperature,
            enable_thinking=False,
            max_tokens=768,
        )
        gradients = self.parse_tagged_text(raw)
        return gradients[:k] if gradients else [raw.strip()[:400]] if raw.strip() else []

    def _build_apply_gradient_prompt(
        self,
        meta_prompt: str,
        error_string: str,
        feedback: str,
        steps: Optional[int] = None,
    ) -> str:
        steps = steps if steps is not None else self.steps_per_gradient
        return f"""{self._optimization_task_line()}

{self._reward_goal_line()}

My current meta-prompt is:
\"\"\"{meta_prompt}\"\"\"

It gets these examples wrong or under-rewarded:
{error_string}

Based on these examples, the problem with this prompt is that {feedback}

Based on the above information, write {steps} improved meta-prompt(s).
Hard requirements for every new prompt:
- Keep placeholders exactly as curly braces: {{start_date}} {{end_date}} {{history_stats}} {{baseline_forecast}} {{raw_fact}}
- Include a Window / Historical stats / Unimodal baseline forecast / Raw fact block that uses those five placeholders.
- Do not add other curly-brace fields, code fences, or y_true / true residual.
- Change instructions/schema/constraints enough to address the stated problem.
- Do NOT make adjective-only edits (e.g. specific→high-impact→actionable). Prefer structural changes: output format, reasoning steps, examples, JSON schema, multi-draft merge, or counterfactual framing.
- Return only the new prompt text inside <START> ... <END> tags (one block per prompt).

The {steps} new prompt(s) are:
"""

    def _parse_apply_gradient_candidates(self, raw: str) -> List[str]:
        candidates = self.parse_tagged_text(raw)
        if not candidates and raw.strip():
            candidates = [raw.strip()]
        cleaned: List[str] = []
        for cand in candidates:
            try:
                cleaned.append(self._ensure_prompt_placeholders(self._strip_fences(cand)))
            except ValueError as exc:
                repaired = self._repair_prompt_placeholders(self._strip_fences(cand))
                try:
                    cleaned.append(self._ensure_prompt_placeholders(repaired))
                    print(f"  repaired placeholders after: {exc}", flush=True)
                except ValueError as exc2:
                    print(f"  drop edited prompt: {exc2}", flush=True)
                    continue
        return cleaned

    def _apply_textual_gradient(
        self,
        meta_prompt: str,
        error_string: str,
        feedback: str,
        steps: Optional[int] = None,
    ) -> List[str]:
        """ProTeGi `apply_gradient`: edit prompt in the opposite direction of the critique."""
        raw = self.client.ask(
            self._build_apply_gradient_prompt(
                meta_prompt, error_string, feedback, steps=steps
            ),
            temperature=self.mutation_temperature,
            enable_thinking=False,
            max_tokens=1400,
        )
        return self._parse_apply_gradient_candidates(raw)

    def _apply_textual_gradients_parallel(
        self,
        meta_prompt: str,
        error_string: str,
        feedbacks: Sequence[str],
    ) -> List[List[str]]:
        """Apply multiple textual gradients concurrently (independent LLM calls)."""
        if not feedbacks:
            return []
        prompts = [
            self._build_apply_gradient_prompt(meta_prompt, error_string, feedback)
            for feedback in feedbacks
        ]
        if len(prompts) == 1:
            return [self._apply_textual_gradient(meta_prompt, error_string, feedbacks[0])]

        print(
            f"  apply gradients parallel: {len(prompts)} prompts "
            f"(concurrency={min(len(prompts), self.rewrite_concurrency)})",
            flush=True,
        )
        raws = self.client.ask_many(
            prompts,
            temperature=self.mutation_temperature,
            enable_thinking=False,
            max_tokens=1400,
            concurrency=min(len(prompts), self.rewrite_concurrency),
        )
        return [self._parse_apply_gradient_candidates(raw) for raw in raws]

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _repair_prompt_placeholders(prompt: str) -> str:
        """Append the standard context block if the editor dropped placeholders."""
        required = [
            "{start_date}",
            "{end_date}",
            "{history_stats}",
            "{baseline_forecast}",
            "{raw_fact}",
        ]
        if all(key in prompt for key in required):
            return prompt
        footer = """

Window:
- start_date={start_date}
- end_date={end_date}

Historical stats:
{history_stats}

Unimodal baseline forecast (no text):
{baseline_forecast}

Raw fact:
{raw_fact}
"""
        return prompt.rstrip() + footer

    def _is_diverse(self, prompt: str, existing: Sequence[str]) -> bool:
        norm = self._norm_prompt(prompt)
        for other in existing:
            if self._jaccard(norm, self._norm_prompt(other)) >= self.diversity_jaccard:
                return False
        return True

    def _save_mutation_artifacts(
        self,
        mut_dir: Path,
        stamp: str,
        *,
        parent: PromptCandidate,
        candidate: str,
        error_string: str = "",
        feedback: str = "",
        minibatch_info: str = "",
    ) -> None:
        (mut_dir / f"{stamp}_parent_prompt.txt").write_text(
            parent.meta_prompt, encoding="utf-8"
        )
        (mut_dir / f"{stamp}_child_prompt.txt").write_text(candidate, encoding="utf-8")
        if error_string:
            (mut_dir / f"{stamp}_error_string.txt").write_text(
                error_string, encoding="utf-8"
            )
        if feedback:
            (mut_dir / f"{stamp}_textual_gradient.txt").write_text(
                feedback, encoding="utf-8"
            )
        if minibatch_info:
            (mut_dir / f"{stamp}_minibatch.txt").write_text(
                minibatch_info, encoding="utf-8"
            )

    def _try_add_mutation(
        self,
        outputs: List[tuple[str, str]],
        existing: List[str],
        candidate: str,
        label: str,
        parent: PromptCandidate,
        mut_dir: Path,
        round_index: int,
        *,
        error_string: str = "",
        feedback: str = "",
        minibatch_info: str = "",
    ) -> bool:
        if self._norm_prompt(candidate) == self._norm_prompt(parent.meta_prompt):
            print("  drop: identical to parent", flush=True)
            return False
        if not self._is_diverse(candidate, existing + [p for p, _ in outputs]):
            print("  drop: not diverse enough", flush=True)
            return False
        outputs.append((candidate, label))
        existing.append(candidate)
        stamp = f"r{round_index}_p{parent.candidate_id}_{label}"
        self._save_mutation_artifacts(
            mut_dir,
            stamp,
            parent=parent,
            candidate=candidate,
            error_string=error_string,
            feedback=feedback,
            minibatch_info=minibatch_info,
        )
        return True

    def _propose_paradigm_catalog_mutations(
        self,
        parent: PromptCandidate,
        *,
        budget: int,
        round_index: int,
        existing: List[str],
        outputs: List[tuple[str, str]],
        mut_dir: Path,
    ) -> None:
        catalog = seed_prompts_for_domain(self.apo_domain)
        parent_strategy = parent.mutation_strategy or ""
        parent_norm = self._norm_prompt(parent.meta_prompt)

        for pname, ptext in catalog.items():
            if len(outputs) >= budget:
                break
            if pname == parent_strategy or self._norm_prompt(ptext) == parent_norm:
                continue
            try:
                candidate = self._ensure_prompt_placeholders(ptext)
            except ValueError:
                continue
            self._try_add_mutation(
                outputs,
                existing,
                candidate,
                f"paradigm_seed_{pname}",
                parent,
                mut_dir,
                round_index,
            )

    def _mutate_paradigm_transfer(
        self,
        parent: PromptCandidate,
        target_paradigm: str,
        target_template: str,
    ) -> List[str]:
        """LLM rewrite of parent prompt into a different output paradigm."""
        transfer_prompt = f"""{self._optimization_task_line()}

Convert the SOURCE meta-prompt into the TARGET PARADIGM structure below.
This is a large structural change, not an adjective tweak.

TARGET PARADIGM ({target_paradigm}):
\"\"\"{target_template[:1200]}\"\"\"

SOURCE meta-prompt:
\"\"\"{parent.meta_prompt}\"\"\"

Requirements:
- Preserve the five placeholders exactly: {{start_date}} {{end_date}} {{history_stats}} {{baseline_forecast}} {{raw_fact}}
- Include the standard Window / Historical stats / Unimodal baseline / Raw fact block with those placeholders.
- Adopt the TARGET paradigm's output format (few-shot examples, step-by-step reasoning, JSON schema, multi-draft merge, or counterfactual framing).
- Do not invent y_true or true residual fields.

Return ONE new meta-prompt inside <START> ... <END> tags.
"""
        raw = self.client.ask(
            transfer_prompt,
            temperature=self.mutation_temperature,
            enable_thinking=False,
            max_tokens=1600,
        )
        candidates = self.parse_tagged_text(raw)
        if not candidates and raw.strip():
            candidates = [raw.strip()]
        cleaned: List[str] = []
        for cand in candidates:
            try:
                cleaned.append(
                    self._ensure_prompt_placeholders(self._strip_fences(cand))
                )
            except ValueError:
                repaired = self._repair_prompt_placeholders(self._strip_fences(cand))
                try:
                    cleaned.append(self._ensure_prompt_placeholders(repaired))
                except ValueError:
                    continue
        return cleaned

    def _propose_paradigm_transfer_mutations(
        self,
        parent: PromptCandidate,
        *,
        budget: int,
        round_index: int,
        existing: List[str],
        outputs: List[tuple[str, str]],
        mut_dir: Path,
    ) -> None:
        catalog = seed_prompts_for_domain(self.apo_domain)
        parent_strategy = parent.mutation_strategy or ""
        targets = [
            name
            for name in PARADIGM_NAMES
            if name in catalog and name != parent_strategy
        ]
        random.shuffle(targets)

        for target in targets:
            if len(outputs) >= budget:
                break
            transferred = self._mutate_paradigm_transfer(
                parent, target, catalog[target]
            )
            for k, candidate in enumerate(transferred):
                if len(outputs) >= budget:
                    break
                self._try_add_mutation(
                    outputs,
                    existing,
                    candidate,
                    f"paradigm_transfer_{target}_s{k}",
                    parent,
                    mut_dir,
                    round_index,
                )

    def propose_mutations(
        self,
        parent: PromptCandidate,
        samples: Sequence[BaselineSample],
        num_mutations: int,
        round_index: int,
        existing_prompts: Sequence[str],
    ) -> List[tuple[str, str]]:
        """ProTeGi expand: sample train minibatches → MSE gradients → prompt edits.

        Each minibatch (default 16) is scored in one concurrent rewrite wave, then
        the worst windows drive textual gradients. Multiple minibatches are drawn
        per parent (`n_gradient_samples`), and training runs many such batches
        across beam parents × outer rounds.
        """
        outputs: List[tuple[str, str]] = []
        existing = list(existing_prompts)
        mut_dir = self.output_dir / "mutations"
        mut_dir.mkdir(parents=True, exist_ok=True)

        paradigm_budget = max(1, int(round(num_mutations * self.paradigm_shift_ratio)))
        protegi_budget = max(0, num_mutations - paradigm_budget)

        print(
            f"  mutation budget: paradigm={paradigm_budget} protegi={protegi_budget} "
            f"(total={num_mutations})",
            flush=True,
        )

        self._propose_paradigm_catalog_mutations(
            parent,
            budget=paradigm_budget,
            round_index=round_index,
            existing=existing,
            outputs=outputs,
            mut_dir=mut_dir,
        )
        if len(outputs) < paradigm_budget:
            self._propose_paradigm_transfer_mutations(
                parent,
                budget=paradigm_budget,
                round_index=round_index,
                existing=existing,
                outputs=outputs,
                mut_dir=mut_dir,
            )

        if protegi_budget <= 0:
            return outputs

        n_batches = max(1, self.n_gradient_samples)

        for batch_i in range(n_batches):
            if len(outputs) >= num_mutations:
                break
            minibatch = self._sample_minibatch(samples, self.minibatch_size)
            mb_rewards = self._rewards_on_samples(parent.meta_prompt, minibatch)
            mb_mean = float(np.mean(mb_rewards)) if mb_rewards else 0.0
            print(
                f"  protegi minibatch {batch_i + 1}/{n_batches} "
                f"size={len(minibatch)} mean_reward={mb_mean:.6f} "
                f"windows={[s.window_index for s in minibatch]}",
                flush=True,
            )
            error_string = self._mse_error_string_from_minibatch(
                parent.meta_prompt, minibatch, mb_rewards
            )
            gradients = self._get_textual_gradients(parent.meta_prompt, error_string)
            print(
                f"  protegi gradients for parent#{parent.candidate_id}: "
                f"{len(gradients)} (minibatch {batch_i + 1}/{n_batches})",
                flush=True,
            )
            pending_feedbacks: List[tuple[int, str]] = []
            for j, feedback in enumerate(gradients):
                if len(outputs) >= num_mutations:
                    break
                pending_feedbacks.append((j, feedback))

            edited_by_feedback = self._apply_textual_gradients_parallel(
                parent.meta_prompt,
                error_string,
                [feedback for _, feedback in pending_feedbacks],
            )
            for (j, feedback), edited in zip(pending_feedbacks, edited_by_feedback):
                if len(outputs) >= num_mutations:
                    break
                print(
                    f"  apply gradient f{j}: {feedback[:160].replace(chr(10), ' ')}",
                    flush=True,
                )
                print(f"  edited candidates from f{j}: {len(edited)}", flush=True)
                for k, candidate in enumerate(edited):
                    if len(outputs) >= num_mutations:
                        break
                    label = f"protegi_b{batch_i}_f{j}_s{k}"
                    minibatch_info = (
                        f"windows={[s.window_index for s in minibatch]}\n"
                        f"rewards={mb_rewards}\nmean={mb_mean:.6f}\n"
                    )
                    self._try_add_mutation(
                        outputs,
                        existing,
                        candidate,
                        label,
                        parent,
                        mut_dir,
                        round_index,
                        error_string=error_string,
                        feedback=feedback,
                        minibatch_info=minibatch_info,
                    )

        return outputs

    def _attach_selection_reward(
        self,
        candidate: PromptCandidate,
        selection_samples: Sequence[BaselineSample],
    ) -> PromptCandidate:
        sel = self.evaluate_prompt(
            candidate.meta_prompt,
            selection_samples,
            round_index=candidate.round_index,
            parent_id=candidate.parent_id,
            source=f"select_{candidate.source}",
            mutation_strategy=candidate.mutation_strategy,
        )
        candidate.selection_reward = sel.mean_reward
        print(
            f"  val-select id={candidate.candidate_id} "
            f"train_r={candidate.mean_reward:.6f} val_r={sel.mean_reward:.6f} "
            f"({candidate.mutation_strategy})",
            flush=True,
        )
        return candidate

    def train(
        self,
        train_samples: Sequence[BaselineSample],
        *,
        test_samples: Optional[Sequence[BaselineSample]] = None,
        selection_samples: Optional[Sequence[BaselineSample]] = None,
        beam_size: int = 4,
        outer_rounds: int = 3,
        mutations_per_parent: int = 3,
        seed_prompts: Optional[Dict[str, str]] = None,
    ) -> BeamTrainResult:
        seeds = seed_prompts or seed_prompts_for_domain(self.apo_domain)
        test_samples = list(test_samples or [])
        selection_samples = list(selection_samples or [])
        use_val_select = len(selection_samples) > 0
        result = BeamTrainResult(
            train_samples=list(train_samples),
            test_samples=test_samples,
            selection_samples=selection_samples,
            selection_mode="val" if use_val_select else "train",
        )

        evaluated: List[PromptCandidate] = []
        for name, prompt in seeds.items():
            cand = self.evaluate_prompt(
                prompt,
                train_samples,
                round_index=0,
                parent_id=None,
                source="seed",
                mutation_strategy=name,
            )
            if use_val_select:
                self._attach_selection_reward(cand, selection_samples)
            evaluated.append(cand)
        evaluated.sort(key=lambda item: item.rank_reward(), reverse=True)
        beam = evaluated[:beam_size]
        seed_best = evaluated[0]
        result.initial_mean_reward = seed_best.mean_reward
        result.history.append(
            BeamRoundSnapshot(
                round_index=0,
                beam=list(beam),
                best_mean_reward=beam[0].rank_reward(),
                evaluated_candidates=len(evaluated),
            )
        )

        for round_index in range(1, outer_rounds + 1):
            existing_prompts = [c.meta_prompt for c in evaluated]
            mutation_entries: List[tuple[str, int, str]] = []
            for parent in beam:
                mutations = self.propose_mutations(
                    parent,
                    train_samples,
                    num_mutations=mutations_per_parent,
                    round_index=round_index,
                    existing_prompts=existing_prompts,
                )
                for prompt, strategy in mutations:
                    mutation_entries.append((prompt, parent.candidate_id, strategy))
                    existing_prompts.append(prompt)

            for prompt, parent_id, strategy in mutation_entries:
                cand = self.evaluate_prompt(
                    prompt,
                    train_samples,
                    round_index=round_index,
                    parent_id=parent_id,
                    source="mutation",
                    mutation_strategy=strategy,
                )
                if use_val_select:
                    self._attach_selection_reward(cand, selection_samples)
                evaluated.append(cand)

            evaluated.sort(key=lambda item: item.rank_reward(), reverse=True)
            beam = evaluated[:beam_size]
            result.history.append(
                BeamRoundSnapshot(
                    round_index=round_index,
                    beam=list(beam),
                    best_mean_reward=beam[0].rank_reward(),
                    evaluated_candidates=len(evaluated),
                )
            )

        result.best_candidate = beam[0]
        result.final_mean_reward = beam[0].mean_reward
        result.final_selection_reward = beam[0].selection_reward

        if test_samples:
            seed_test = self.evaluate_prompt(
                seed_best.meta_prompt,
                test_samples,
                round_index=-1,
                source="test_seed",
                mutation_strategy=seed_best.mutation_strategy,
            )
            best_test = self.evaluate_prompt(
                beam[0].meta_prompt,
                test_samples,
                round_index=-1,
                source="test_best",
                mutation_strategy=beam[0].mutation_strategy,
            )
            result.test_seed_reward = seed_test.mean_reward
            result.test_best_reward = best_test.mean_reward

        delta = result.final_mean_reward - result.initial_mean_reward
        extra = ""
        if use_val_select and result.final_selection_reward is not None:
            extra += f" | val_select={result.final_selection_reward:.6f}"
        if result.test_seed_reward is not None and result.test_best_reward is not None:
            extra += (
                f" | held-out: {result.test_seed_reward:.6f} -> "
                f"{result.test_best_reward:.6f}"
            )
        mode = "val" if use_val_select else "train"
        if delta > 0:
            result.summary = (
                f"[{mode}-select] train reward {result.initial_mean_reward:.6f} -> "
                f"{result.final_mean_reward:.6f} (delta={delta:+.6f}){extra}"
            )
        elif delta < 0:
            result.summary = (
                f"[{mode}-select] train reward {result.initial_mean_reward:.6f} -> "
                f"{result.final_mean_reward:.6f} (delta={delta:+.6f}){extra}"
            )
        else:
            result.summary = (
                f"[{mode}-select] train reward {result.final_mean_reward:.6f}{extra}"
            )
        print(result.summary, flush=True)
        self._save_train_result(result)
        return result

    def _save_train_result(self, result: BeamTrainResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["timestamp"] = datetime.now().isoformat()
        if result.best_candidate:
            best_prompt_path = self.output_dir / "best_meta_prompt.txt"
            best_prompt_path.write_text(
                result.best_candidate.meta_prompt,
                encoding="utf-8",
            )
            payload["best_meta_prompt_path"] = str(best_prompt_path)
        out_file = self.output_dir / "beam_train_result.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
