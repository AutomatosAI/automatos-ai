"""PRD-252 R6 (F259): a drag on the board does what the matching button does.

A drag that needs a decision is refused, and the refusal names the button that
makes it. Review → Done ("Use Approve") and Review → Assigned ("Use Reject") were
the first two. Night 7's "Moving cards by hand" (3 Oct 2026) found the board
still accepting three more:

- Inbox → Done on a ticket nobody had worked on (#0044): Done, with no result.
- Inbox → Review on an empty ticket (#0045): a review of nothing, which Needs-you
  counted.
- In progress → Review or Done while the run was going (#0094 at 8 s, #0111 at
  5 s): the agent kept working, billed, and its answer never reached the card.

Where a ticket starts (``api/board_tasks._start_now``, shared by Run now and a
drag to In progress), a ticket waiting for the owner's answer is refused with a
pointer to the answer (``question_refusal``) instead of being started over.

A mission's tickets keep only the decision drags: the mission engine runs them,
and its own refusals (``mission_runs_it``) say where to act.

The same rules hold wherever a status changes: the general ``PATCH /{task_id}``
judges the ticket as that PATCH leaves it (``as_patched``), and Auto's status
tool keeps ``move_refusal`` (``modules/tools/discovery/ticket_moves.py``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.models.approval_grants import KIND_QUESTION, SUBJECT_BOARD_TASK
from services.run_cancel import is_playbook_card
from services.ticket_numbers import ticket_label

# #1094: a ticket with no agent is never in progress (a playbook's card has none
# by design: a move to In progress runs its playbook again, F259).
NO_AGENT_NO_PROGRESS = "Assign an agent first: a ticket with no agent cannot be in progress."

# A drag whose decision has its own button: the refusal names it.
DECISION_DRAGS = {
    ("review", "done"): "Use Approve on the ticket: a drag to Done would skip its approval step.",
    ("review", "assigned"): "Use Reject on the ticket: it sends the agent what to fix with the redo.",
}

# Where a ticket whose run is still going can be dragged: Cancelled stops the run
# (F245), and In progress changes nothing.
WHILE_RUNNING = frozenset({"in_progress", "cancelled"})

# Columns that hold finished work, so the card must carry some.
NEEDS_A_RESULT = frozenset({"done", "review"})

# Columns a ticket sits in before anyone has worked on it.
NOT_STARTED = frozenset({"inbox", "assigned"})

# Where a person can move a ticket that is not finished: it has no completion time.
UNFINISHED_BY_HAND = frozenset({"inbox", "assigned", "blocked"})


def drag_refusal(task: Any, new_status: str, *, running: bool, mission_ticket: bool) -> Optional[str]:
    """Why the board refuses to drag ``task`` to ``new_status``, naming the button
    that does it instead; None when the drag goes ahead."""
    decision = DECISION_DRAGS.get((task.status, new_status))
    if decision:
        return decision
    if new_status == "in_progress" and not task.assigned_agent_id and not is_playbook_card(task):
        return NO_AGENT_NO_PROGRESS
    return move_refusal(task, new_status, running=running, mission_ticket=mission_ticket)


def move_refusal(task: Any, new_status: str, *, running: bool, mission_ticket: bool) -> Optional[str]:
    """The rules every status change keeps, Auto's as well as a drag: a running
    ticket waits for its result or is cancelled first, and a finished column
    needs work on the card. None when the move goes ahead."""
    if mission_ticket:
        return None
    label = ticket_label(task, capital=True)
    if running and new_status not in WHILE_RUNNING:
        return (f"{label} is running: wait for its result, which lands on the card, "
                "or use Cancel to stop it first.")
    if new_status in NEEDS_A_RESULT and not (task.result or "").strip():
        return _nothing_to_show(task, label, "approve" if new_status == "done" else "review")
    return None


class _AsPatched:
    """A ticket read through the fields a PATCH is about to write."""

    def __init__(self, task: Any, after: Dict[str, Any]) -> None:
        self._task = task
        self._after = after

    def __getattr__(self, name: str) -> Any:
        after = self.__dict__["_after"]
        return after[name] if name in after else getattr(self.__dict__["_task"], name)


def as_patched(task: Any, **after: Any) -> Any:
    """``task`` as a PATCH would leave the fields in ``after``: its status change is
    judged on that ticket before anything is written (an agent or a result the same
    body sets counts, #1094)."""
    return _AsPatched(task, after)


def _nothing_to_show(task: Any, label: str, what: str) -> str:
    """``label`` opens a sentence ("Ticket #0052"); mid-sentence it is lower case."""
    if task.status == "failed":
        return f"{label}'s run failed and left nothing to {what}: use Run now to try again, or Cancel it."
    if task.status in NOT_STARTED:
        start = "use Run now" if task.assigned_agent_id else "assign an agent and use Run now"
        return (f"No one has worked on {ticket_label(task)} yet, so there is nothing to {what}: {start}, "
                "or Cancel it if it isn't needed.")
    return f"{label} has nothing to {what} yet: use Run now, or Cancel it."


def open_question_for(db: Any, task: Any) -> Optional[Any]:
    """The owner's question this ticket is parked on (an agent's ask, a session's
    hold or plan), or None. Only a blocked ticket waits on one."""
    if getattr(task, "status", None) != "blocked":
        return None
    from core.services.approval_grants import find_pending_grant

    grant = find_pending_grant(db, task.workspace_id, subject_type=SUBJECT_BOARD_TASK,
                               subject_id=str(task.id), kind=KIND_QUESTION)
    return grant if getattr(grant, "kind", None) == KIND_QUESTION else None


def question_refusal(db: Any, task: Any) -> Optional[str]:
    """Why ``task`` is not started over while it waits for the owner's answer: the
    answer is the decision, and it starts the ticket again with the answer in."""
    asked = open_question_for(db, task)
    if asked is None:
        return None
    # No '#': since PRD-252 R4 a '#' only ever names a ticket.
    return (f"{ticket_label(task, capital=True)} is waiting for your answer to its question: "
            "answer it in the Questions tab or on the ticket, and the answer starts it again.")
