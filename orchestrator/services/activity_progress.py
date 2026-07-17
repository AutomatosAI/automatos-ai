"""PRD-221 S12 — plain-English progress labels for the activity feed.

Maps the real orchestration event vocabulary (the underscore-style EventType
enum) to short, human phrases shown as a feed item's ``last_progress`` line.
Keyed off the imported enum so a renamed/added event is caught by the test,
and any unmapped type falls back to a safe generic label (never a KeyError).
"""
from __future__ import annotations

from core.models.orchestration_enums import EventType

_GENERIC_LABEL = "Working…"

_LABELS = {
    # Run lifecycle
    EventType.RUN_CREATED.value: "Getting started",
    EventType.RUN_PLANNING_STARTED.value: "Planning the work",
    EventType.RUN_PLAN_READY.value: "Plan ready",
    EventType.RUN_APPROVED.value: "Approved — starting",
    EventType.RUN_AUTO_APPROVED.value: "Approved — starting",
    EventType.RUN_STARTED.value: "Working on it",
    EventType.RUN_PAUSED.value: "Paused",
    EventType.RUN_RESUMED.value: "Resumed",
    EventType.RUN_REPLANNING.value: "Re-planning after a failed step",
    EventType.RUN_REPLANNED.value: "New plan ready — continuing",
    EventType.RUN_BUDGET_WARNING.value: "Approaching the budget limit",
    EventType.RUN_BUDGET_EXCEEDED.value: "Budget exceeded — paused",
    EventType.RUN_VERIFYING.value: "Checking the results",
    EventType.RUN_AWAITING_HUMAN.value: "Waiting for your input",
    EventType.RUN_COMPLETED.value: "Completed",
    EventType.RUN_FAILED.value: "Failed — needs attention",
    EventType.RUN_CANCELLED.value: "Cancelled",
    # Task lifecycle
    EventType.TASK_CREATED.value: "New step queued",
    EventType.TASK_QUEUED.value: "Step queued",
    EventType.TASK_ASSIGNED.value: "Step assigned to an agent",
    EventType.TASK_STARTED.value: "Working on a step",
    EventType.TASK_CONTINUING.value: "Continuing a step",
    EventType.TASK_RESUMED.value: "Resumed a step",
    EventType.TASK_OUTPUT_SUBMITTED.value: "Step output ready",
    EventType.TASK_VERIFICATION_STARTED.value: "Checking a step's output",
    EventType.TASK_VERIFICATION_PASSED.value: "A step passed review",
    EventType.TASK_VERIFICATION_FAILED.value: "Output failed verification — retrying",
    EventType.TASK_VERIFICATION_COMPLETED.value: "Step review finished",
    EventType.TASK_HUMAN_REVIEW_REQUESTED.value: "Waiting for your review",
    EventType.TASK_HUMAN_APPROVED.value: "You approved a step",
    EventType.TASK_HUMAN_REJECTED.value: "You sent a step back",
    EventType.TASK_RETRYING.value: "Retrying a step",
    EventType.TASK_CRASHED.value: "A step crashed — recovering",
    EventType.TASK_FAILED.value: "A step failed — needs attention",
    EventType.TASK_SKIPPED.value: "A step was skipped",
    EventType.TASK_CANCELLED.value: "A step was cancelled",
    EventType.TASK_STALLED.value: "A step stalled",
    EventType.STALL_DETECTED.value: "Progress stalled",
}

# Events that should visibly flag the item as needing a human look.
_ATTENTION = {
    EventType.RUN_FAILED.value,
    EventType.RUN_AWAITING_HUMAN.value,
    EventType.RUN_BUDGET_EXCEEDED.value,
    EventType.TASK_VERIFICATION_FAILED.value,
    EventType.TASK_HUMAN_REVIEW_REQUESTED.value,
    EventType.TASK_CRASHED.value,
    EventType.TASK_FAILED.value,
    EventType.TASK_STALLED.value,
    EventType.STALL_DETECTED.value,
}


def progress_label(event_type: str) -> str:
    """Plain-English label for an event type; safe generic on unknown."""
    return _LABELS.get(event_type, _GENERIC_LABEL)


def progress_requires_attention(event_type: str) -> bool:
    """True if this event should flag the feed item for a human look."""
    return event_type in _ATTENTION
