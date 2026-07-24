"""Policy plane — typed verdict + event vocabulary (PRD-174 W4).

The single typed gate the whole plane speaks in. Keeps Claude Code's separation
of model *intent* from *authorization* and its verdict semantics, re-keyed for
tenancy: every handler returns a :class:`Verdict`, and when handlers disagree
**deny outranks ask outranks allow** (:func:`merge_verdicts`).

Stdlib-only at import time on purpose — ``tool_loop`` (stdlib-only) and the unit
tests import this without dragging the heavy ``modules.tools`` package init
(asyncpg/pgvector). No SQLAlchemy, no config, no network here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Decision(str, Enum):
    """A policy verdict's disposition. ``str`` mixin so it serialises as its value.

    Ordering (deny > ask > allow) is enforced by :data:`_DECISION_RANK` /
    :func:`merge_verdicts`, not by enum member order.
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    DEFER = "defer"


# deny outranks ask outranks allow (PRD §4.2). ``defer`` sits between allow and
# ask: it means "no opinion / let a later stage decide", so it must never
# override an explicit allow but must yield to ask/deny.
_DECISION_RANK: Dict[Decision, int] = {
    Decision.ALLOW: 0,
    Decision.DEFER: 1,
    Decision.ASK: 2,
    Decision.DENY: 3,
}


class Event(str, Enum):
    """The event taxonomy (PRD §4.2) — Claude Code's hook points, tenant-scoped.

    The plane fires these in-process (NOT as shell hooks): handlers are typed
    Python callables with tenant scope only.
    """

    RUN_START = "RunStart"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_BATCH = "PostToolBatch"
    ROUND_END = "RoundEnd"
    RUN_END = "RunEnd"
    PRE_COMPACT = "PreCompact"


@dataclass(frozen=True)
class PolicyError:
    """Errors-as-data (PRD §4.2). A denial the *model* can read and adapt to,
    not an opaque failure.

    - ``code``: machine-stable reason slug (e.g. ``"permission_denied"``,
      ``"budget_exceeded"``, ``"rate_limited"``).
    - ``message_for_model``: one line the LLM sees as tool content so it can
      relay/adjust instead of erroring.
    - ``remediation``: what would make the call succeed (escalate, approve,
      wait, drop a field). May be ``None`` when nothing helps.
    - ``retryable``: whether an identical retry could ever succeed (rate limits:
      yes; a hard permission deny: no).
    """

    code: str
    message_for_model: str
    remediation: Optional[str] = None
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message_for_model": self.message_for_model,
            "remediation": self.remediation,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class Verdict:
    """The typed result of evaluating one tool call against the plane.

    Mirrors Claude Code's hook verdict, re-keyed for tenancy:

    - ``decision``: allow | deny | ask | defer.
    - ``updated_input``: a replacement tool-input dict the plane wants used
      instead of the model's (e.g. an injected ``_agent_id``/``field_id``).
      ``None`` = use the original input unchanged.
    - ``injected_context``: extra context string the plane wants folded into
      the model's view (prerequisite hints, policy notes). ``None`` = nothing.
    - ``reason``: human/audit-facing one-liner for why this disposition.
    - ``error``: on a deny/ask, the structured errors-as-data payload the model
      reads. Always present when ``decision`` is deny; present for ask when the
      ask carries a message; ``None`` for allow/defer.
    """

    decision: Decision
    reason: str = ""
    updated_input: Optional[Dict[str, Any]] = None
    injected_context: Optional[str] = None
    error: Optional[PolicyError] = None

    # -- constructors (readability at call sites) --------------------------

    @classmethod
    def allow(
        cls,
        reason: str = "",
        *,
        updated_input: Optional[Dict[str, Any]] = None,
        injected_context: Optional[str] = None,
    ) -> "Verdict":
        return cls(
            Decision.ALLOW,
            reason=reason,
            updated_input=updated_input,
            injected_context=injected_context,
        )

    @classmethod
    def defer(cls, reason: str = "") -> "Verdict":
        """No opinion — let a later stage decide. Never overrides an allow."""
        return cls(Decision.DEFER, reason=reason)

    @classmethod
    def deny(cls, error: PolicyError, *, reason: Optional[str] = None) -> "Verdict":
        return cls(
            Decision.DENY,
            reason=reason if reason is not None else error.message_for_model,
            error=error,
        )

    @classmethod
    def ask(
        cls,
        error: PolicyError,
        *,
        reason: Optional[str] = None,
        injected_context: Optional[str] = None,
    ) -> "Verdict":
        return cls(
            Decision.ASK,
            reason=reason if reason is not None else error.message_for_model,
            error=error,
            injected_context=injected_context,
        )

    # -- helpers -----------------------------------------------------------

    @property
    def blocks_execution(self) -> bool:
        """True when this verdict must stop the tool from executing.

        deny and ask both block: deny is final, ask suspends pending approval.
        allow and defer let the call proceed.
        """
        return self.decision in (Decision.DENY, Decision.ASK)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "updated_input": self.updated_input,
            "injected_context": self.injected_context,
            "error": self.error.to_dict() if self.error is not None else None,
        }


def merge_verdicts(*verdicts: Optional[Verdict]) -> Verdict:
    """Combine handler verdicts under **deny > ask > allow** (PRD §4.2).

    The highest-ranked disposition wins the ``decision`` and its ``error``.
    ``updated_input`` and ``injected_context`` from *allowing* handlers are still
    threaded through (an allow can rewrite input / inject context), but a
    blocking verdict (deny/ask) is authoritative for the decision. ``None`` and
    ``defer`` verdicts are treated as "no opinion".

    With no verdicts (or all ``None``/``defer``) the result is a plain allow —
    the plane never invents a block out of silence.
    """
    present = [v for v in verdicts if v is not None]
    if not present:
        return Verdict.allow("no policy opinion")

    winner = max(present, key=lambda v: _DECISION_RANK[v.decision])

    # Thread input-rewrites / injected-context from every non-blocking verdict
    # so an allowing handler's rewrite is not lost when another allows too.
    # A blocking winner owns the decision but we still surface its own rewrites.
    updated_input = winner.updated_input
    injected_parts = []
    for v in present:
        if v.injected_context:
            injected_parts.append(v.injected_context)
        if updated_input is None and v.updated_input is not None and not v.blocks_execution:
            updated_input = v.updated_input

    injected = "\n".join(injected_parts) if injected_parts else None

    if winner.decision in (Decision.ALLOW, Decision.DEFER):
        # Normalise a bare defer up to allow at the boundary so callers only
        # ever branch on allow/deny/ask.
        return Verdict.allow(
            winner.reason or "allowed",
            updated_input=updated_input,
            injected_context=injected,
        )
    return Verdict(
        decision=winner.decision,
        reason=winner.reason,
        updated_input=updated_input,
        injected_context=injected,
        error=winner.error,
    )
