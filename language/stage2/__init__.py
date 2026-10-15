"""Stage2 package: multi-domain CFA residual (SoftLSTM / Fact / Explicit)."""

from language.stage2.profiles import (
    Stage2DomainProfile,
    get_stage2_profile,
    list_stage2_domains,
    register_stage2_profile,
)

__all__ = [
    "Stage2DomainProfile",
    "get_stage2_profile",
    "list_stage2_domains",
    "register_stage2_profile",
]
