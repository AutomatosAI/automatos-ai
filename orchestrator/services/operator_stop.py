"""A human's stop is a fact the machine may not undo (F036).

Night 1 (2026-09-18): the persona blocked 13 tickets. Every stop reached the
host — its log records all 13 "no longer ours — stopping its session", and each
session was killed within a quarter of a second. And then the tickets came
back. Ticket 136 was blocked at 18:18:06 UTC and re-claimed with a fresh
session at 18:19:17, ONE SECOND after someone answered a question its dead
session had asked. Ticket 231 came back at 23:56 in a batch of seven, off an
approval grant decided before the stop. The 74 KB written "after the stop" was
written by those new sessions.

The cause is that ``blocked`` meant two different things with one spelling:

* a MACHINE park — the ticket waits for an answer, an approval or the spend
  window, and the thing it waits for resumes it; and
* a HUMAN stop — someone looked at the work and said no.

Every auto-resume path checked ``status == 'blocked'`` and nothing else, so an
answer or a grant meant for the park silently reversed the stop.

The rule this module enforces: a status someone EXPLICITLY set to ``blocked``
or ``cancelled`` — through the board API or a platform tool acting for a
person — is recorded here, and only an explicit status change undoes it.
Machine parks write ``blocked`` internally and never set it, so they resume
exactly as before.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

OPERATOR_STOP_KEY = "operator_stop"
# The statuses a person uses to say "stop". Moving to any other status is how a
# person lifts their own stop.
STOPPING_STATUSES = frozenset({"blocked", "cancelled"})


def mark_operator_stop(task: Any, status: str, reason: Optional[str], by: str) -> None:
    """Record that a person stopped this ticket. Rebuilds runtime_ref (JSONB)."""
    ref: Dict[str, Any] = dict(getattr(task, "runtime_ref", None) or {})
    ref[OPERATOR_STOP_KEY] = {
        "status": status,
        "reason": (reason or "").strip()[:500] or None,
        "by": by,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    task.runtime_ref = ref


def clear_operator_stop(task: Any) -> None:
    """A person moved the ticket on — their stop is lifted."""
    ref: Dict[str, Any] = dict(getattr(task, "runtime_ref", None) or {})
    if OPERATOR_STOP_KEY in ref:
        ref.pop(OPERATOR_STOP_KEY, None)
        task.runtime_ref = ref


def operator_stop(task: Any) -> Optional[Dict[str, Any]]:
    """The recorded stop, or ``None`` when no person has stopped this ticket."""
    ref = getattr(task, "runtime_ref", None)
    if not isinstance(ref, dict):
        return None
    stop = ref.get(OPERATOR_STOP_KEY)
    return stop if isinstance(stop, dict) else None


def apply_explicit_status(task: Any, old_status: Optional[str], new_status: str,
                          reason: Optional[str], by: str) -> None:
    """Keep the stop marker in step with an EXPLICIT status change.

    Call from every path where a person — directly, or through a tool acting
    on their instruction — sets a ticket's status. Stopping records the stop;
    any other status lifts it.
    """
    if new_status in STOPPING_STATUSES:
        mark_operator_stop(task, new_status, reason, by)
    elif old_status in STOPPING_STATUSES or operator_stop(task):
        clear_operator_stop(task)
