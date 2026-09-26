"""F183 (night 6): a board ticket whose result only asks the owner waits for the owner.

Ticket #1097 (agent #324, 7 s) closed done on a result that was four questions to
the owner and no newsletter. Questions stayed empty all night: nothing but an
explicit platform_ask_human ever put a ticket's question there. F140's test for an
answer that only asks (services.playbook_owner_ask.owner_question, which F163
already runs on a mission's steps) now runs on every ticket result as well. Such a
ticket is parked the way platform_ask_human parks one (blocked, "Awaiting human
answer (ask #N)"). Its question goes to Questions and Telegram, and the answer
re-queues it. Answers are recorded in planning_data.human_qa, and the next run's
prompt carries them (ticket_answers_block), whether an API agent or a Claude Code
session picks it up. Nothing read them before, so a re-run would only have asked
again.

A ticket whose result another loop takes up keeps its path:
- a mission's step card or lane ticket, where the mission asks at the step (F163);
- a playbook's card, where the run stops to ask (F140);
- a chat's session ticket, where the chat turn waits on it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Source types whose result another loop consumes (see the module note).
OWNED_ELSEWHERE = frozenset({"mission", "orchestration_task", "recipe", "chat"})
# How many of a ticket's answered questions its next prompt carries (the latest).
ANSWERS_SHOWN = 5


def result_only_asks(task: Any, output: str, exec_result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """``{question, options}`` when this ticket's result only asks the owner and
    the ticket answers for itself (not a mission's, a playbook's or a chat's), else None."""
    from services.cli_ticket_lane import is_lane_owned
    from services.playbook_owner_ask import owner_question

    if getattr(task, "source_type", None) in OWNED_ELSEWHERE or is_lane_owned(task):
        return None
    brief = f"{getattr(task, 'title', None) or ''}\n{getattr(task, 'description', None) or ''}"
    return owner_question(output, exec_result or {}, brief)


async def park_if_the_result_asks(
    db: Session,
    *,
    task: Any,
    workspace_id: Any,
    agent_id: Optional[int],
    output: str,
    exec_result: Optional[Dict[str, Any]] = None,
) -> bool:
    """Park ``task`` behind its own question, as platform_ask_human parks a ticket,
    when its result only asks the owner. True iff parked. Never raises: a question
    that cannot be placed leaves the result to close as before."""
    ask = result_only_asks(task, output, exec_result)
    if ask is None:
        return False
    try:
        from core.models import Agent
        from modules.tools.discovery.handlers_asks import stage_question
        from services.board_events import notify_board_event

        agent = db.query(Agent).filter(Agent.id == agent_id).first() if agent_id else None
        notify_board_event(  # F118: rides the commit that parks the ticket
            db, workspace_id=str(workspace_id), task_id=task.id, status="blocked", event="task_updated",
        )
        await stage_question(
            db, UUID(str(workspace_id)),
            subject_type="board_task", subject_id=str(task.id),
            question=ask["question"], options=ask.get("options"),
            asked_by_agent_id=int(agent_id) if agent_id else None,
            agent_name=getattr(agent, "name", None), park=task,
        )
    except Exception:  # noqa: BLE001 — no question was placed: the result closes as it is
        logger.warning("[F183] could not ask the owner for ticket %s", getattr(task, "id", None), exc_info=True)
        return False
    logger.info("[F183] ticket %s only asked the owner: parked behind its question", task.id)
    return True


def ticket_answers_block(planning_data: Any) -> str:
    """The owner's answers to this ticket's questions (planning_data.human_qa, which
    answering a question writes), for the prompt of the run the answer re-queued."""
    qa = planning_data.get("human_qa") if isinstance(planning_data, dict) else None
    answered = [entry for entry in (qa or []) if isinstance(entry, dict) and str(entry.get("a") or "").strip()]
    if not answered:
        return ""
    lines = ["## The owner's answers",
             "This ticket asked the owner, and they answered. Use the answers; do not ask them again."]
    for entry in answered[-ANSWERS_SHOWN:]:
        lines += ["", f"**You asked:** {entry.get('q') or '(the question is on the ticket)'}",
                  f"**The owner answered:** {entry['a']}"]
    return "\n".join(lines)
