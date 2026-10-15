"""LLM stages for economy event extraction."""
from language.lamp_timemmd import llm_stages_core
from language.lamp_timemmd.economy import prompts, schema

llm_stages_core.install_llm_stages(globals(), prompts=prompts, schema=schema)
