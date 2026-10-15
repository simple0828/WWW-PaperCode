"""Import domain lamp packages (schema / prompts / llm_stages)."""

from __future__ import annotations

import importlib
from types import ModuleType

from language.stage2.profiles import Stage2DomainProfile, get_stage2_profile


def load_lamp(domain_or_profile: str | Stage2DomainProfile) -> ModuleType:
    profile = (
        domain_or_profile
        if isinstance(domain_or_profile, Stage2DomainProfile)
        else get_stage2_profile(domain_or_profile)
    )
    return importlib.import_module(profile.lamp_package)


def load_lamp_sub(domain_or_profile: str | Stage2DomainProfile, sub: str) -> ModuleType:
    profile = (
        domain_or_profile
        if isinstance(domain_or_profile, Stage2DomainProfile)
        else get_stage2_profile(domain_or_profile)
    )
    return importlib.import_module(f"{profile.lamp_package}.{sub}")
