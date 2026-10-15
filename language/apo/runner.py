from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol

from language import VLLMClient
from language.apo.baseline import BaselineSample, UnimodalBaselinePredictor
from language.apo.paradigm_prompts import AGRI_PARADIGM_PROMPTS, TRAFFIC_PARADIGM_PROMPTS
from language.apo.reward import RewardBreakdown


_PLACEHOLDERS = """
Window:
- start_date={start_date}
- end_date={end_date}

Historical stats:
{history_stats}

Unimodal baseline forecast (no text):
{baseline_forecast}

Raw fact:
{raw_fact}
""".strip()

AGRI_SEED_PROMPTS = {
    "residual_driver": f"""You are a residual evidence extractor for monthly U.S. retail broiler composite prices.

A unimodal time-series model already produced a baseline forecast of the OT price series. Your job is NOT to summarize the news and NOT to predict the full future. Extract only text clues that could explain why the true future may deviate from that baseline.

Do:
- Keep facts about supply (placements, slaughter, flock size), feed costs, disease outbreaks, demand/retail promotions, exports/imports, wholesale–retail spreads, and seasonal holiday effects that conflict with a smooth baseline.
- Tie each kept fact to a plausible residual effect: higher-than-baseline, lower-than-baseline, more volatile, or delayed effect.
- Prefer concrete numbers, regions, and dated observations over generic agriculture encyclopedia text.
- If evidence conflicts, keep both sides and say which is more local to the target window.

Do not:
- Repeat historical averages already implied by the baseline.
- Invent events, dates, or magnitudes not in the raw fact.
- Use the true future residual; you do not have it.
- Write a long essay.

Output format:
1) Drivers: 3-6 bullets, each <= 28 words, each starting with [higher|lower|volatile|persist|reverse]
2) Conflict: 1 bullet on the main contradictory signal, or "none"
3) Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>

{_PLACEHOLDERS}
""",
    "contrastive": f"""Compare the raw fact against the unimodal baseline forecast and keep only contrastive evidence.

Write as:
- KEEP: facts that imply retail broiler prices should be higher/lower/more volatile than the baseline path
- DROP: restatements of industry definitions, long-run averages, or facts already captured by history stats
- SCORE: for each KEEP bullet, add (direction, strength=weak|medium|strong)

Rules: no invented events; no true residual; <= 6 KEEP bullets; each <= 28 words.
End with: Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>.

{_PLACEHOLDERS}
""",
    "mechanism": f"""Extract causal mechanisms, not a news digest.

From the raw fact, recover at most 4 mechanisms that could move broiler prices off the unimodal baseline, e.g. feed-cost shock, disease cull, cold-storage drawdown, export ban/surge, holiday demand, or wholesale–retail spread squeeze.

For each mechanism write:
- name
- region / market
- expected residual direction (increase/decrease/volatile)
- one supporting phrase copied or lightly paraphrased from the fact

If a mechanism is only weakly supported, mark it weak. Do not invent mechanisms absent from the fact.
End with: Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>.

{_PLACEHOLDERS}
""",
}

TRAFFIC_SEED_PROMPTS = {
    "residual_driver": f"""You are a residual evidence extractor for monthly U.S. Vehicle Miles Traveled (VMT / OT).

A unimodal time-series model already produced a baseline forecast of the OT VMT series. Your job is NOT to summarize the news and NOT to predict the full future. Extract only text clues that could explain why the true future may deviate from that baseline.

Do:
- Keep facts about fuel prices, recessions/employment, holidays/travel seasons, road construction/closures, weather extremes, transit strikes, COVID-like mobility shocks, freight demand, and policy (tolls, WFH, stimulus) that conflict with a smooth baseline.
- Tie each kept fact to a plausible residual effect: higher-than-baseline, lower-than-baseline, more volatile, or delayed effect.
- Prefer concrete numbers, regions, and dated observations over generic transportation encyclopedia text.
- If evidence conflicts, keep both sides and say which is more local to the target window.

Do not:
- Repeat historical averages already implied by the baseline.
- Invent events, dates, or magnitudes not in the raw fact.
- Use the true future residual; you do not have it.
- Write a long essay.

Output format:
1) Drivers: 3-6 bullets, each <= 28 words, each starting with [higher|lower|volatile|persist|reverse]
2) Conflict: 1 bullet on the main contradictory signal, or "none"
3) Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>

{_PLACEHOLDERS}
""",
    "contrastive": f"""Compare the raw fact against the unimodal baseline forecast and keep only contrastive evidence for monthly U.S. VMT.

Write as:
- KEEP: facts that imply VMT should be higher/lower/more volatile than the baseline path
- DROP: restatements of highway definitions, long-run averages, or facts already captured by history stats
- SCORE: for each KEEP bullet, add (direction, strength=weak|medium|strong)

Rules: no invented events; no true residual; <= 6 KEEP bullets; each <= 28 words.
End with: Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>.

{_PLACEHOLDERS}
""",
    "mechanism": f"""Extract causal mechanisms for VMT residuals, not a news digest.

From the raw fact, recover at most 4 mechanisms that could move Vehicle Miles Traveled off the unimodal baseline, e.g. fuel-price shock, holiday travel surge, severe weather, road network disruption, labor/freight slowdown, or mobility-policy change (WFH, lockdown, tolls).

For each mechanism write:
- name
- region / corridor / market
- expected residual direction (increase/decrease/volatile)
- one supporting phrase copied or lightly paraphrased from the fact

If a mechanism is only weakly supported, mark it weak. Do not invent mechanisms absent from the fact.
End with: Residual hypothesis: <increase|decrease|flat|mixed> because <short reason>.

{_PLACEHOLDERS}
""",
}

TRAFFIC_SEED_PROMPTS.update(TRAFFIC_PARADIGM_PROMPTS)
AGRI_SEED_PROMPTS.update(AGRI_PARADIGM_PROMPTS)

DOMAIN_SEED_PROMPTS: Dict[str, Dict[str, str]] = {
    "agriculture": AGRI_SEED_PROMPTS,
    "traffic": TRAFFIC_SEED_PROMPTS,
}

# Backward-compatible aliases (agriculture default).
DEFAULT_META_PROMPT = AGRI_SEED_PROMPTS["residual_driver"]
SEED_PROMPTS = AGRI_SEED_PROMPTS


def seed_prompts_for_domain(domain: str) -> Dict[str, str]:
    domain = domain.lower().strip()
    if domain not in DOMAIN_SEED_PROMPTS:
        raise ValueError(f"No seed prompts for domain={domain!r}")
    return DOMAIN_SEED_PROMPTS[domain]


def infer_apo_domain(args) -> str:
    """Infer APO task domain from run.py args (root_path / data_path)."""
    root = str(getattr(args, "root_path", "")).lower()
    data = str(getattr(args, "data_path", "")).lower()
    if "traffic" in root or "vmt" in data:
        return "traffic"
    if "algriculture" in root or "agriculture" in root or "broiler" in data:
        return "agriculture"
    return "agriculture"


def apo_task_description(domain: str) -> str:
    domain = domain.lower().strip()
    if domain == "traffic":
        return "multimodal monthly U.S. Vehicle Miles Traveled (VMT) forecasting"
    return "multimodal monthly U.S. retail broiler composite price forecasting"


class RewardScorer(Protocol):
    def score(self, sample: BaselineSample, rewrite_text: str) -> RewardBreakdown: ...


class APORunner:
    """Shared rewrite/scoring helpers for beam search (Phase A)."""

    REWRITE_MAX_TOKENS = 1024
    INPUT_TOKEN_MARGIN = 256
    CHARS_PER_TOKEN_EST = 3.5

    def __init__(
        self,
        baseline: UnimodalBaselinePredictor,
        scorer: RewardScorer,
        client: VLLMClient,
        output_dir: str | Path,
        *,
        rewrite_temperature: float = 0.3,
        max_model_len: int | None = None,
    ):
        self.baseline = baseline
        self.scorer = scorer
        self.client = client
        self.output_dir = Path(output_dir)
        self.device = baseline.device
        self.rewrite_temperature = rewrite_temperature
        cfg_len = getattr(client.config, "max_model_len", None)
        self.max_model_len = int(max_model_len or cfg_len or 10240)

    def _max_input_tokens(self) -> int:
        return self.max_model_len - self.REWRITE_MAX_TOKENS - self.INPUT_TOKEN_MARGIN

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        return max(1, int(len(text) / cls.CHARS_PER_TOKEN_EST))

    @staticmethod
    def _truncate_field(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        suffix = "\n...[truncated for context limit]..."
        keep = max(0, max_chars - len(suffix))
        return text[:keep] + suffix

    def _build_prompt_context(self, sample: BaselineSample) -> Dict[str, str]:
        return {
            "start_date": sample.start_date,
            "end_date": sample.end_date,
            "history_stats": sample.history_stats,
            "baseline_forecast": sample.baseline_forecast,
            "raw_fact": sample.raw_text,
        }

    def _fill_template(self, meta_prompt: str, sample: BaselineSample) -> str:
        ctx = self._build_prompt_context(sample)
        filled = meta_prompt
        for key in ("start_date", "end_date", "history_stats", "baseline_forecast"):
            filled = filled.replace("{" + key + "}", str(ctx[key]))

        raw_slots = filled.count("{raw_fact}")
        if raw_slots == 0:
            return filled

        skeleton = filled.replace("{raw_fact}", "")
        overhead_tokens = self._estimate_tokens(skeleton)
        budget_tokens = self._max_input_tokens() - overhead_tokens
        if budget_tokens <= 0:
            raw_val = self._truncate_field(ctx["raw_fact"], 400)
        else:
            # Some mutated prompts repeat {raw_fact} many times; budget per slot.
            budget_chars = int(budget_tokens * self.CHARS_PER_TOKEN_EST)
            per_slot = max(400, budget_chars // raw_slots)
            raw_val = self._truncate_field(ctx["raw_fact"], per_slot)

        return filled.replace("{raw_fact}", raw_val)

    def rewrite_with_prompt(self, sample: BaselineSample, meta_prompt: str) -> str:
        prompt = self._fill_template(meta_prompt, sample)
        est_in = self._estimate_tokens(prompt)
        if est_in + self.REWRITE_MAX_TOKENS > self.max_model_len:
            raise ValueError(
                f"Filled prompt still too long: est_input_tokens={est_in} "
                f"max_model_len={self.max_model_len}"
            )
        return self.client.ask(
            prompt,
            temperature=self.rewrite_temperature,
            enable_thinking=False,
            max_tokens=self.REWRITE_MAX_TOKENS,
        ).strip()

    def score_rewrite(self, sample: BaselineSample, rewrite: str) -> RewardBreakdown:
        return self.scorer.score(sample, rewrite)

    @staticmethod
    def _ensure_prompt_placeholders(prompt: str) -> str:
        required = [
            "{start_date}",
            "{end_date}",
            "{history_stats}",
            "{baseline_forecast}",
            "{raw_fact}",
        ]
        missing = [key for key in required if key not in prompt]
        if missing:
            raise ValueError(f"Improved prompt lost placeholders: {missing}")
        return prompt
