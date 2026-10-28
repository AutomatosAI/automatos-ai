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

US-C17b: the Missions list and the board's ticket reads say the same, from their own status
(``mission_status``, ``ticket_status``). They read no needs-you kind, with one exception
each way: a Mission waiting for its plan's approval is a needs-you approval row (needs_you()
lists every one), and a ticket in Review is working only while its Mission checks it; one
the owner judges is needs_you()'s to say, so a board read draws nothing for it.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from core.models.orchestration_enums import RunState
from core.services.ticket_reasons import MISSION_CHECKING
from modules.proactive.locale import DEFAULT_LOCALE
from services.needs_you import KINDS, MISSION_CARD, OPENS_MISSION
from services.proactive.strings import APPROVAL_PLAN, FAILED_MISSION, STUCK_REASONS, say

__all__ = [
    "BOARD_TASK", "DISPLAY_STATES", "DONE", "FAILED", "GRANT", "MISSION", "NEEDS_YOU", "READY",
    "REPLAN_MISSION", "RERUN_MISSION", "RETRIES", "RETRY_TICKET", "WORKING", "display_state",
    "mission_status", "needs_you_with_status", "next_step", "retry", "row_status", "ticket_status",
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
KIND_APPROVAL = "approval"
REVIEW = "review"
# A Mission's step on the board; a Mission's own card is services.needs_you's MISSION_CARD.
MISSION_STEP = "orchestration_task"
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
_MISSION_SENTENCE: Mapping[str, str] = MappingProxyType({KIND_APPROVAL: APPROVAL_PLAN, KIND_FAILED: FAILED_MISSION})
# The needs-you kind a Mission's state reads as. needs_you() lists every Mission waiting for
# its plan's approval as an approval row (services/needs_you.py _MISSION_ROWS); a failed
# Mission is its own card's failed row.
_MISSION_STATE_KIND: Mapping[str, str] = MappingProxyType({
    RunState.AWAITING_APPROVAL.value: KIND_APPROVAL, RunState.FAILED.value: KIND_FAILED,
})
_NO_STATUS: Mapping[str, Optional[str]] = MappingProxyType(
    {"display_state": None, "next_step": None, "retry": None})


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


def mission_status(state: Any, locale: str = DEFAULT_LOCALE) -> Dict[str, Optional[str]]:
    """The three fields a Mission's row gains from its ``state`` (a string or the enum
    member): waiting for its plan's approval it needs you, failed it is replanned."""
    value = getattr(state, "value", state)
    kind = _MISSION_STATE_KIND.get(value) if isinstance(value, str) else None
    return {
        "display_state": display_state(MISSION, value, kind),
        "next_step": next_step(locale, kind, source=MISSION) if kind else None,
        "retry": retry(MISSION, value, kind),
    }


def ticket_status(task: Any, review_code: Optional[str], locale: str = DEFAULT_LOCALE) -> Dict[str, Optional[str]]:
    """The three fields a board ticket gains from its status. ``review_code`` is its
    ``review_reason``: in Review it is working only while its Mission checks it. A failed
    Mission card is its Mission's failure (replanned); a failed step is its Mission's to
    handle (services/needs_you.py), so nothing on the board tries it again."""
    status = getattr(task, "status", None)
    if status == REVIEW and review_code != MISSION_CHECKING:
        return dict(_NO_STATUS)
    state = display_state(BOARD_TASK, status)
    if state != FAILED:
        return {**_NO_STATUS, "display_state": state}
    source = getattr(task, "source_type", None)
    if source == MISSION_STEP:
        return {**_NO_STATUS, "display_state": FAILED}
    if source == MISSION_CARD and getattr(task, "orchestration_run_id", None):
        return mission_status(RunState.FAILED, locale)
    return {"display_state": FAILED, "next_step": next_step(locale, KIND_FAILED), "retry": RETRY_TICKET}


def needs_you_with_status(needs: Mapping[str, Any], locale: str) -> Dict[str, Any]:
    """``needs_you()``'s answer with each row's status, next step and retry, in ``locale``;
    its number and counts as they were."""
    rows = {kind: [{**row, **row_status(kind, row, locale)} for row in listed]
            for kind, listed in needs["rows"].items()}
    return {**needs, "rows": rows}
