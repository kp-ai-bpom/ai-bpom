from .config import AmbiguityConfig, get_ambiguity_config
from .service import AmbiguityService
from .types import (
    AmbiguityDetectionResult,
    AmbiguityResolutionResult,
    InterpretationOption,
)

__all__ = [
    "AmbiguityConfig",
    "AmbiguityService",
    "AmbiguityDetectionResult",
    "AmbiguityResolutionResult",
    "InterpretationOption",
    "get_ambiguity_config",
]
