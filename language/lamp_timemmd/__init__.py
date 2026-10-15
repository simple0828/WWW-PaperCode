"""Generic Time-MMD Stage2 event packages (economy / energy / …).

Each domain is a thin subpackage exposing ``schema`` / ``prompts`` / ``llm_stages``
so ``load_lamp_sub(profile, …)`` works. Shared logic lives in ``*_core`` modules.
"""
