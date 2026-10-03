"""PRD-252 R3 — why a ticket waits in Review or Blocked, as a code the board shows.

Review had six causes and Blocked seven, and the card said neither: "Review"
over a ticket the owner had asked to see looked the same as one whose named
file was missing, and "Blocked" over a question looked the same as a spend
hold. The codes come from what the ticket already records:

* a code a path wrote when it moved the ticket (``runtime_ref.review_reason``,
  stamped with the ``completed_at`` it set, so an older entry never speaks for
  a newer one);
* the notes a run's checks append to its result (the two ``*_NOTE_PREFIX``
  constants, which their writers build their notes from);
* the session's refused calls, the ticket's approval action, its review mode,
  its source, and the reason line a park wrote (``blocked_reason``).

Pure functions over the ticket's columns. The board's API serves both codes
(``api/board_tasks._board_dict``), and the Needs-you count reads
``MISSION_CHECKING`` to leave a mission's own checks out.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

REVIEW_REASON_KEY = "review_reason"
SPEND_HOLD_KEY = "spend_hold"

# What the run's checks append to a result that goes to review instead of done
# (services/result_files.py and services/result_substance.py write them).
FILE_MISSING_NOTE_PREFIX = "Not found when this ticket closed:"
NOTHING_DONE_NOTE_PREFIX = "Nothing was produced:"
# The lines a ticket out of attempts carries: the dispatcher's, for a run that
# never reported but left its files; the CLI host's park_exhausted. Both are
# written in place (test_prd252_ticket_reasons guards that they still match).
NEVER_REPORTED_PREFIX = "Finished, worker never reported"
ATTEMPTS_STOPPED_PREFIX = "Stopped after"

# Review
MISSION_CHECKING = "mission_checking"   # a mission step its mission is checking
MISSION_PLAN = "mission_plan"           # a mission's own card: its plan waits for the owner's OK
FILE_MISSING = "file_missing"
NOTHING_DONE = "nothing_done"
HELD_COMMAND = "held_command"           # a session's held tool call was refused
RETRIES_USED_UP = "retries_used_up"
APPROVAL_ACTION = "approval_action"     # filed for the owner's OK to run its action
STOPPED_WITH_WORK = "stopped_with_work"  # a playbook run stopped after finished work
MOVED_BY_YOU = "moved_by_you"
ASKED = "asked"                         # review_mode human (or llm, which has no reviewer)
ENDS_ON_A_QUESTION = "ends_on_a_question"  # F242: a playbook run's answer asks the owner something
UNEXPLAINED = "unexplained"

# Blocked
QUESTION = "question"
APPROVAL = "approval"
SPEND_CEILING = "spend_ceiling"
MISSION_PAUSED = "mission_paused"
STEP_FAILED = "step_failed"
STOPPED_BY_YOU = "stopped_by_you"
WAITING = "waiting"
OWNER_CHECK = "owner_check"             # F242: a mission paused until the owner checks a step

# F242: how a mission paused for the owner's check of a step says so (its stop
# detail, so its card's blocked line), followed by the step's number.
WAITING_FOR_YOUR_CHECK = "Waiting for your check of "


def with_review_reason(runtime_ref: Any, code: str, completed_at: Optional[datetime]) -> Dict[str, Any]:
    """``runtime_ref`` with ``code`` as the reason for this review entry (rebuilt)."""
    ref = dict(runtime_ref) if isinstance(runtime_ref, dict) else {}
    ref[REVIEW_REASON_KEY] = {"code": code, "at": completed_at.isoformat() if completed_at else None}
    return ref


def review_reason(task: Any) -> Optional[str]:
    """Why a ticket in review is there; None for a ticket that is not."""
    if getattr(task, "status", None) != "review":
        return None
    source = getattr(task, "source_type", None)
    if source == "orchestration_task":  # F242: a step the owner asked to check waits for them
        return ASKED if getattr(task, "review_mode", None) == "human" else MISSION_CHECKING
    if source == "orchestration":  # the card mirrors its run: in review = awaiting the plan's approval
        return MISSION_PLAN
    ref = _ref(task)
    return _recorded(task, ref) or _from_result(task, ref) or _from_ticket(task)


def blocked_code(task: Any) -> Optional[str]:
    """What a blocked ticket waits for; None for a ticket that is not blocked."""
    if getattr(task, "status", None) != "blocked":
        return None
    ref = _ref(task)
    reason = getattr(task, "blocked_reason", None) or ""
    stop = ref.get("operator_stop")
    if isinstance(stop, dict) and stop.get("status") == "blocked":
        return STOPPED_BY_YOU
    if SPEND_HOLD_KEY in ref:
        return SPEND_CEILING
    if "(ask #" in reason:
        return QUESTION
    if "grant #" in reason:
        return APPROVAL
    source = getattr(task, "source_type", None)
    if source == "orchestration":
        return OWNER_CHECK if reason.startswith(WAITING_FOR_YOUR_CHECK) else MISSION_PAUSED
    if source == "orchestration_task":
        return STEP_FAILED
    return WAITING


def _ref(task: Any) -> Dict[str, Any]:
    ref = getattr(task, "runtime_ref", None)
    return ref if isinstance(ref, dict) else {}


def _recorded(task: Any, ref: Dict[str, Any]) -> Optional[str]:
    """The code a path wrote for THIS review entry (its completed_at), if any."""
    entry = ref.get(REVIEW_REASON_KEY)
    completed_at = getattr(task, "completed_at", None)
    if not isinstance(entry, dict) or completed_at is None:
        return None
    return entry.get("code") if entry.get("at") == completed_at.isoformat() else None


def _from_result(task: Any, ref: Dict[str, Any]) -> Optional[str]:
    result = str(getattr(task, "result", None) or "")
    if FILE_MISSING_NOTE_PREFIX in result:
        return FILE_MISSING
    if NOTHING_DONE_NOTE_PREFIX in result:
        return NOTHING_DONE
    denials = ref.get("denials")
    if (isinstance(denials, int) and denials > 0) or ref.get("permission_denials"):
        return HELD_COMMAND
    if str(getattr(task, "review_feedback", None) or "").startswith((NEVER_REPORTED_PREFIX, ATTEMPTS_STOPPED_PREFIX)):
        return RETRIES_USED_UP
    return None


def _from_ticket(task: Any) -> str:
    planning = getattr(task, "planning_data", None)
    if isinstance(planning, dict) and planning.get("approval_action"):
        return APPROVAL_ACTION
    if getattr(task, "source_type", None) == "recipe":
        return STOPPED_WITH_WORK
    if getattr(task, "review_mode", None) in ("human", "llm"):
        return ASKED
    if not str(getattr(task, "result", None) or "").strip():
        return NOTHING_DONE
    return UNEXPLAINED
