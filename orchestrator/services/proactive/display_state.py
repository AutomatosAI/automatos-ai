"""PRD-268 §6.11.1, K13 — one status vocabulary, computed once, on the server.

Every row a person reads says where it stands (``display_state``), what happens next
(``next_step``, a sentence from :mod:`services.proactive.strings` in the person's language)
and, when it failed, what Try again does (``retry``). Clients map the state to a glyph, a
word and a colour role; they never work it out.

The states, by §6.11.1's table:

* ``needs_you``: a ``needs_you()`` row of kind review, question, approval or stuck. Only
  ``needs_you()`` says what waits for a person (PRD-252 R5), so only a needs-you kind gives
  this state. A ticket in Review read without one is its Mission's check, so it is working;
  an Assigned ticket read without kind stuck is working too.
* ``working``: a ticket Assigned or In progress; a Mission planning, running, replanning or
  checking its work.
* ``done``: a ticket Done or Closed; a completed Mission; a granted approval.
* ``failed``: kind failed; a failed ticket; a failed Mission.
* ``ready`` needs to know that a Deliverable is unopened (the seen flag, §12-Q18) or that a
  brief was made since (W2), so nothing gives it yet.
* Anything else (Inbox, Blocked, cancelled, paused, a pending grant read without its kind, a
  lapsed one, an unknown status) is ``None``: the client draws nothing, never a guess.

Try again, for a failed row only: a ticket runs again (``retry_ticket``, the board's Run
now), a Mission is replanned (``replan_mission``, which keeps its finished steps).
``rerun_mission`` (a new Mission from the same goal) completes the vocabulary.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from services.needs_you import KINDS, OPENS_MISSION
from services.proactive.strings import APPROVAL_PLAN, FAILED_MISSION, STUCK_REASONS, say

__all__ = [
    "BOARD_TASK", "DISPLAY_STATES", "DONE", "FAILED", "GRANT", "MISSION", "NEEDS_YOU", "READY",
    "REPLAN_MISSION", "RERUN_MISSION", "RETRIES", "RETRY_TICKET", "WORKING", "display_state",
    "needs_you_with_status", "next_step", "retry", "row_status",
]

NEEDS_YOU = "needs_you"
WORKING = "working"
READY = "ready"
DONE = "done"
FAILED = "failed"
DISPLAY_STATES = (NEEDS_YOU, WORKING, READY, DONE, FAILED)

# What a row is: a ticket, a Mission (its run), or an approval grant. A needs-you row
# says ``source`` for a grant and a Mission waiting for its plan's approval.
BOARD_TASK = "board_task"
MISSION = "mission"
GRANT = "grant"

RERUN_MISSION = "rerun_mission"
REPLAN_MISSION = "replan_mission"
RETRY_TICKET = "retry_ticket"
RETRIES = (RERUN_MISSION, REPLAN_MISSION, RETRY_TICKET)

KIND_STUCK = "stuck"
KIND_FAILED = "failed"
# The needs-you kinds that wait for a person; the other one, failed, is failed.
_ASKS_A_PERSON = frozenset(KINDS) - {KIND_FAILED}

_NOTHING: Mapping[str, str] = MappingProxyType({})
_BY_STATUS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    BOARD_TASK: MappingProxyType({
        "assigned": WORKING, "in_progress": WORKING, "review": WORKING,
        "done": DONE, "closed": DONE, "failed": FAILED,
    }),
    MISSION: MappingProxyType({
        "planning": WORKING, "replanning": WORKING, "running": WORKING, "verifying": WORKING,
        "completed": DONE, "failed": FAILED,
    }),
    GRANT: MappingProxyType({"granted": DONE}),
})
_RETRY: Mapping[str, str] = MappingProxyType({BOARD_TASK: RETRY_TICKET, MISSION: REPLAN_MISSION})
# A Mission's own approval and failure say what is done to the Mission.
_MISSION_SENTENCE: Mapping[str, str] = MappingProxyType({"approval": APPROVAL_PLAN, KIND_FAILED: FAILED_MISSION})


def display_state(source: str, status: Any, kind: Optional[str] = None) -> Optional[str]:
    """Where a row stands: a needs-you ``kind`` decides, else its ``source``'s ``status``
    (a string, or the enum member holding one). None when the table gives nothing."""
    if kind in _ASKS_A_PERSON:
        return NEEDS_YOU
    if kind == KIND_FAILED:
        return FAILED
    value = getattr(status, "value", status)
    return _BY_STATUS.get(source, _NOTHING).get(value) if isinstance(value, str) else None


def retry(source: str, status: Any, kind: Optional[str] = None) -> Optional[str]:
    """What Try again does for a failed row, None for a row that hasn't failed."""
    if display_state(source, status, kind) != FAILED:
        return None
    return _RETRY.get(source)


def next_step(locale: str, kind: str, *, source: str = BOARD_TASK, why: Optional[str] = None) -> Optional[str]:
    """A needs-you row's next step in ``locale``: a stuck ticket's reason, a Mission's own
    approval or failure, else its kind's sentence. None for a kind the table doesn't know."""
    if kind == KIND_STUCK and why in STUCK_REASONS:
        return say(locale, why)
    if source == MISSION and kind in _MISSION_SENTENCE:
        return say(locale, _MISSION_SENTENCE[kind])
    return say(locale, kind) if kind in KINDS else None


def _row_source(row: Mapping[str, Any]) -> str:
    """A needs-you row's source: what it names itself, else the Mission it opens (its
    decision is the Mission's), else its ticket."""
    named = row.get("source")
    if named in (GRANT, MISSION):
        return named
    return MISSION if row.get("opens") == OPENS_MISSION else BOARD_TASK


def row_status(kind: str, row: Mapping[str, Any], locale: str) -> Dict[str, Optional[str]]:
    """The three fields a needs-you row of ``kind`` gains."""
    source = _row_source(row)
    return {
        "display_state": display_state(source, None, kind),
        "next_step": next_step(locale, kind, source=source, why=row.get("why")),
        "retry": retry(source, None, kind),
    }


def needs_you_with_status(needs: Mapping[str, Any], locale: str) -> Dict[str, Any]:
    """``needs_you()``'s answer with each row's status, next step and retry, in ``locale``;
    its number and counts as they were."""
    rows = {kind: [{**row, **row_status(kind, row, locale)} for row in listed]
            for kind, listed in needs["rows"].items()}
    return {**needs, "rows": rows}
