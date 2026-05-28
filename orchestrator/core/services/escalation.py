"""Escalation levels — Wave 3.

Single 5-level ladder Auto uses to triage what flows where:

  L0  FYI       — informational, no action expected (digest material)
  L1  TASK      — needs work, no human decision (queue + execute)
  L2  APPROVAL  — needs Gerard's call before action
  L3  URGENT    — immediate attention (push channels)
  L4  SECURITY  — stop and escalate; no jokes, no assumptions

The ladder is **additive** — existing priority enums (CRITICAL/HIGH/MEDIUM/LOW)
on board_tasks and BudgetStatus on missions stay where they are. ``classify``
maps an event payload onto a level so dispatchers / queues can use a single
abstraction.

Routing convention: pair with ``auto_reporting.routes`` keys like
``"agent_error:urgent"`` or just ``"urgent"`` for severity-only fallback.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional


class EscalationLevel(IntEnum):
    FYI = 0
    TASK = 1
    APPROVAL = 2
    URGENT = 3
    SECURITY = 4

    @property
    def severity(self) -> str:
        return _LEVEL_TO_SEVERITY[self]

    @classmethod
    def from_severity(cls, severity: Optional[str]) -> "EscalationLevel":
        return _SEVERITY_TO_LEVEL.get((severity or "").lower(), cls.FYI)


_LEVEL_TO_SEVERITY: Dict[EscalationLevel, str] = {
    EscalationLevel.FYI: "info",
    EscalationLevel.TASK: "task",
    EscalationLevel.APPROVAL: "approval",
    EscalationLevel.URGENT: "urgent",
    EscalationLevel.SECURITY: "security",
}
_SEVERITY_TO_LEVEL: Dict[str, EscalationLevel] = {
    v: k for k, v in _LEVEL_TO_SEVERITY.items()
}


# Map existing priority strings → level
_PRIORITY_LEVEL: Dict[str, EscalationLevel] = {
    "urgent": EscalationLevel.URGENT,
    "critical": EscalationLevel.URGENT,
    "high": EscalationLevel.APPROVAL,
    "medium": EscalationLevel.TASK,
    "low": EscalationLevel.FYI,
}

# Map BudgetStatus → level
_BUDGET_LEVEL: Dict[str, EscalationLevel] = {
    "exceeded": EscalationLevel.URGENT,
    "critical": EscalationLevel.URGENT,
    "warning": EscalationLevel.APPROVAL,
    "healthy": EscalationLevel.FYI,
}


def classify(event: Dict[str, Any]) -> EscalationLevel:
    """Best-effort classification of an event payload onto the L0-L4 ladder.

    Recognised hints (first match wins):
      - ``event["security"]`` is truthy → SECURITY
      - ``event["requires_approval"]`` is truthy → APPROVAL
      - ``event["status"]`` in {"critical", "error", "failed"} → URGENT
      - ``event["budget_status"]`` mapped via _BUDGET_LEVEL
      - ``event["priority"]`` mapped via _PRIORITY_LEVEL
      - ``event["severity"]`` mapped via EscalationLevel.from_severity
      - default FYI

    Caller can override with an explicit ``escalation_level`` key.
    """
    if event.get("escalation_level") is not None:
        try:
            return EscalationLevel(int(event["escalation_level"]))
        except (ValueError, TypeError):
            pass

    if event.get("security"):
        return EscalationLevel.SECURITY

    if event.get("requires_approval"):
        return EscalationLevel.APPROVAL

    status = (event.get("status") or "").lower()
    if status in {"critical", "error", "failed"}:
        return EscalationLevel.URGENT

    budget_status = (event.get("budget_status") or "").lower()
    if budget_status in _BUDGET_LEVEL:
        return _BUDGET_LEVEL[budget_status]

    priority = (event.get("priority") or "").lower()
    if priority in _PRIORITY_LEVEL:
        return _PRIORITY_LEVEL[priority]

    if event.get("severity"):
        return EscalationLevel.from_severity(event["severity"])

    return EscalationLevel.FYI
