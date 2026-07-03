"""Unified Policy Plane (PRD-174 Wave 4).

One typed gate, evaluated in one place, for every tool call — on every surface.
Auto's blueprint says what to *attempt*; this plane, and only this plane,
decides what *executes*. Guardrails become policy (DB-configured, one config,
one chokepoint), not code scattered across router dependencies.

Public surface:

- Flag: :func:`policy_plane_enabled` (``AUTOMATOS_POLICY_PLANE``, default OFF).
- Verdict vocabulary: :class:`Verdict`, :class:`Decision`, :class:`PolicyError`,
  :func:`merge_verdicts` (deny > ask > allow).
- Chokepoint: :class:`PolicyGate` + :class:`ToolCall` — invoked from
  ``UnifiedToolExecutor.execute_tool``.
- Event bus: :class:`PolicyBus`, :class:`Event`, :class:`EventContext`,
  :func:`get_policy_bus` — the ``on_pre_tool`` seam.
- Budget: :class:`BudgetExceeded`, :func:`check_budget`.
- Policy document: :func:`load_policy_document` (the Balanced defaults).
- Errors-as-data: :func:`verdict_to_result`, :func:`ensure_error_envelope`.

Everything is stdlib-only at import time (DB/registry imports are lazy) so the
tool loop and unit tests can import it without the heavy ``modules.tools`` init.
"""
from __future__ import annotations

from modules.policy.ai_act import (
    OversightMapping,
    OversightTier,
    classify_risk_tier,
    oversight_for_risk,
)
from modules.policy.audit_handler import (
    audit_policy_verdict,
    make_audit_handler,
    register_audit_handler,
)
from modules.policy.budget import BudgetDecision, BudgetExceeded, check_budget
from modules.policy.bus import (
    EventContext,
    Handler,
    PolicyBus,
    get_policy_bus,
    reset_policy_bus,
)
from modules.policy.errors import ensure_error_envelope, verdict_to_result
from modules.policy.flag import policy_plane_enabled
from modules.policy.gate import PolicyGate, ToolCall
from modules.policy.policy_document import (
    PolicyDocument,
    classify_action,
    load_policy_document,
    set_policy_document,
)
from modules.policy.types import (
    Decision,
    Event,
    PolicyError,
    Verdict,
    merge_verdicts,
)

__all__ = [
    # flag
    "policy_plane_enabled",
    # verdict vocabulary
    "Decision",
    "Event",
    "PolicyError",
    "Verdict",
    "merge_verdicts",
    # chokepoint
    "PolicyGate",
    "ToolCall",
    # bus
    "PolicyBus",
    "EventContext",
    "Handler",
    "get_policy_bus",
    "reset_policy_bus",
    # audit handler (S1 — the bus's Art.12 record-keeping seam)
    "audit_policy_verdict",
    "make_audit_handler",
    "register_audit_handler",
    # EU-AI-Act Art.14 oversight tiers (S6 scaffold — read by the S5 card)
    "OversightMapping",
    "OversightTier",
    "classify_risk_tier",
    "oversight_for_risk",
    # budget
    "BudgetDecision",
    "BudgetExceeded",
    "check_budget",
    # policy document
    "PolicyDocument",
    "classify_action",
    "load_policy_document",
    "set_policy_document",
    # errors-as-data
    "verdict_to_result",
    "ensure_error_envelope",
]
