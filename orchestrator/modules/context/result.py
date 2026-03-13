"""
ContextResult — Immutable result from ContextService.build_context().
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ContextResult:
    """Immutable result from ContextService.build_context()."""

    # Ready for LLM call
    system_prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"

    # Metadata for logging/debugging
    mode: str = ""
    sections_included: list[str] = field(default_factory=list)
    sections_trimmed: list[str] = field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = 0
    memory_context: Optional[str] = None
    user_name: Optional[str] = None
    preparation_time_ms: float = 0.0
