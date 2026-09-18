"""PRD-248 — the decision seam.

Typed, calibrated decisions (Choice / Score / Noul) answered by a System One
model (TypeSafe Jev, direct or through OpenRouter) or by the platform's own
system LLM through an adapter that speaks the same shapes. Callers see one
``decide()`` that returns a result or ``None`` and never raises. It lives
beside the LLM manager because it is an inference seam, not a feature module.
"""
from .engine import (
    DEFAULT_DIALS,
    MODE_LIVE,
    MODE_OFF,
    MODE_SHADOW,
    MODES,
    SETTINGS_CATEGORY,
    DecisionEngine,
    Dials,
    get_decision_engine,
)
from .llm_adapter import PROVIDER_LLM, LLMDecisionAdapter
from .questions import (
    Choice,
    DecisionAnswer,
    DecisionResult,
    Noul,
    Question,
    Score,
    parse_answers,
    to_wire,
)
from .typesafe_client import (
    DEFAULT_MODELS,
    PROVIDER_OPENROUTER,
    PROVIDER_TYPESAFE,
    TypeSafeDecisionClient,
)

__all__ = [
    "Choice",
    "Score",
    "Noul",
    "Question",
    "DecisionAnswer",
    "DecisionResult",
    "DecisionEngine",
    "Dials",
    "DEFAULT_DIALS",
    "DEFAULT_MODELS",
    "LLMDecisionAdapter",
    "TypeSafeDecisionClient",
    "MODES",
    "MODE_OFF",
    "MODE_SHADOW",
    "MODE_LIVE",
    "PROVIDER_LLM",
    "PROVIDER_OPENROUTER",
    "PROVIDER_TYPESAFE",
    "SETTINGS_CATEGORY",
    "get_decision_engine",
    "parse_answers",
    "to_wire",
]
