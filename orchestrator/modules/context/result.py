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

    # PRD-201 S1: the persisted assembly trace — "what did Auto know?".
    # ``sections`` is the per-section record the assembler already computes and
    # used to throw away: {name, priority, token_estimate, rendered_nonempty,
    # trimmed}. ``model`` + ``budget_total`` are the resolved driving model and
    # its model-aware budget ceiling (PRD-201 S3). ``injected_memory_ids`` is
    # best-effort memory-item provenance for the turn (empty when unknown).
    model: Optional[str] = None
    budget_total: int = 0
    sections: list[dict[str, Any]] = field(default_factory=list)
    injected_memory_ids: list[Any] = field(default_factory=list)

    # PRD-201 S4: the cache-stable prefix — the leading bytes of the system
    # prompt that are safe to mark with an Anthropic ``cache_control``
    # breakpoint (static identity/skills/catalog), assembled before the
    # volatile tail (memory/graph/datetime). ``None`` when no stable prefix was
    # identified. Consumed only by the Anthropic client seam; every other
    # consumer reads ``system_prompt`` unchanged.
    cacheable_prefix: Optional[str] = None

    def to_assembly_trace(self) -> dict[str, Any]:
        """The narrow, JSONB-serialisable assembly trace persisted per turn/run.

        This is the record the answerability win rides on (PRD-201 S1): mode,
        the driving model, the resolved budget ceiling, per-section token/trim
        detail, injected memory ids, prep time, and the assembled token
        estimate. Deliberately small — no rendered content, only shape.
        """
        return {
            "mode": self.mode,
            "model": self.model,
            "budget_total": self.budget_total,
            "token_estimate": self.token_estimate,
            "token_budget": self.token_budget,
            "prep_ms": round(self.preparation_time_ms, 1),
            "sections": self.sections,
            "sections_included": self.sections_included,
            "sections_trimmed": self.sections_trimmed,
            "injected_memory_ids": self.injected_memory_ids,
        }
