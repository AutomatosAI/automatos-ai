"""PRD-256 FX-011 (Decision D7): an agent's send on a ticket Auto wrote waits for the owner's click.

Night 12: ticket 2318 ("Confirm the order with the supplier") was written by Auto from the
owner's chat; 57 s after it started, its CLI agent ran GMAIL_SEND_EMAIL, before the owner's
"nothing goes to Kerbside" landed. ``owner_only.asks_before_a_send`` asked only in a person's
chat, and a session speaks as its agent (services/session_tools ``call_tool``).

Now a Composio send or publish on a ticket Auto wrote raises the same grant card
(``owner_only.send_ask``, FX-008's card text) whatever lane runs it: a ticket session
(``session_task_id``), an API agent's board run (``board_task_id``) or a playbook step
(``playbook_execution_id``, its run's card). Nothing is sent. The grant carries
:data:`AGENT_SEND` (the ticket, its lane, the agent), and a session's ticket parks on the
card at its turn's end, the way it parks on its own question (cli_host_service
``_park_for_answer``). The owner's click runs the send once (:func:`sends_on_the_click`,
from api/approval_grants ``_resume_tool_call``), writes it on the ticket and sends the
ticket back to work; a declined card fails the ticket with the reason
(:func:`declined_send`). A ticket a person wrote runs as today.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

AGENT_SEND = "agent_send"      # the grant's details marker
SESSION_LANE, BOARD_LANE, PLAYBOOK_LANE = "session", "board", "playbook"
# The run's server-built context key that names its ticket, per lane.
LANE_KEYS = ((SESSION_LANE, "session_task_id"), (BOARD_LANE, "board_task_id"), (PLAYBOOK_LANE, "playbook_execution_id"))
AUTO_SLUG = "auto-{workspace_id}"   # the workspace's orchestrator agent (core/seeds/seed_auto_agent)
WRITTEN_BY_AN_AGENT = "agent"
# PRD-204 S9: a run a platform call started is an Auto-launched playbook run.
LAUNCHED_BY_AUTO = "platform_action"
PLAYBOOK_CARD_SOURCE = "recipe"
STEP_SEPARATOR = ":"
FAILED = "failed"
NO_RECIPIENT = "(no recipient named)"
NO_SUBJECT = "(no subject)"
AT = "%Y-%m-%d %H:%M UTC"
CLICKER = "the owner's click"

CARD_RAISED = ("Card raised: send {subject} to {recipient}. Nothing goes out until the owner clicks; "
               "finish your report and end your turn.")
NO_CARD = ("This send waits for the owner's click, and its card could not be raised: nothing was sent. "
           "Say so in your report and end your turn.")
ON_ANOTHER_TICKET = ("This exact send already waits for the owner's click on ticket {task_id}: nothing was sent "
                     "from here. Say so in your report and end your turn.")
SENT = "Sent on the owner's click at {at}: {recipient}, {subject}"
NOT_SENT = "The owner clicked, but the send failed at {at}: {error}"
DECLINED = "The owner declined the send: {subject} to {recipient}"


def autos_ticket(db: Any, workspace_id: Any, caller_context: Any) -> Optional[Dict[str, Any]]:
    """``{lane, task_id, context}`` when the run works on Auto's brief, else None: a ticket
    Auto wrote, or a step of a playbook run Auto started (its card, or a CLI agent's step
    ticket; a run whose card is missing still asks, its ``task_id`` None).

    Only server-built context names the ticket (the session's token, the dispatcher, the
    playbook runner), never the call."""
    if db is None or not isinstance(caller_context, dict):
        return None
    for lane, key in LANE_KEYS:
        ref = caller_context.get(key)
        if ref is None or str(ref).strip() == "":
            continue
        task = _run_card(db, workspace_id, lane, ref)
        if not _on_autos_brief(db, workspace_id, task, _playbook_run(lane, ref, task)):
            return None
        return {"lane": lane, "task_id": int(task.id) if task is not None else None, "context": {key: ref}}
    return None


def waits_on_the_card(db: Any, workspace_id: Any, ticket: Dict[str, Any], ask: Dict[str, Any], *,
                      sent: Any, agent_id: Any) -> Dict[str, Any]:
    """The agent's result for a send on Auto's ticket: the card is raised, its grant marked
    with the ticket, a session's ticket parks on it at its turn's end."""
    recipient, subject = addressed(sent)
    grant_id = ask.get("grant_id")
    if not isinstance(grant_id, int):
        return {**ask, "message": NO_CARD}
    marker = {**ticket, "agent_id": _agent(agent_id), "recipient": recipient, "subject": subject}
    try:
        with db.begin_nested():  # a failure here undoes only the marker and the park, never the turn's work
            held_by = _mark(db, grant_id, marker)
            if held_by is None and ticket["lane"] == SESSION_LANE:
                _park_on(db, workspace_id, ticket["task_id"], grant_id, ask)
        db.commit()  # the tool's own commit (api/session_tools): the card must outlive the turn
    except Exception:  # noqa: BLE001 — logged; the agent is told nothing went out
        logger.exception("[agent_sends] the send card for ticket %s could not be kept", ticket.get("task_id"))
        return {**ask, "message": NO_CARD}
    if held_by is not None:
        return {**ask, "message": ON_ANOTHER_TICKET.format(task_id=held_by)}
    logger.info("[agent_sends] ticket %s: %s waits for the owner's click (grant %s)",
                ticket["task_id"], subject, grant_id)
    return {**ask, "message": CARD_RAISED.format(subject=subject, recipient=recipient)}


def addressed(sent: Any) -> Tuple[str, str]:
    """(recipients, subject) of a send, as its card shows them: every address, cc and bcc
    named (P256-FIX-RVW-26), so 'Card raised' and the click's note name who it goes to."""
    from modules.tools.discovery.card_question import SUBJECT_KEYS, first_said
    from modules.tools.discovery.card_question_sends import recipients_said
    from modules.tools.discovery.card_question_text import shown

    params = sent if isinstance(sent, dict) else {}
    subject = first_said(params, SUBJECT_KEYS)
    return recipients_said(params) or NO_RECIPIENT, shown(subject) if subject is not None else NO_SUBJECT


def send_marker(grant: Any) -> Optional[Dict[str, Any]]:
    """The :data:`AGENT_SEND` marker of a grant, or None."""
    details = getattr(grant, "details", None)
    marker = details.get(AGENT_SEND) if isinstance(details, dict) else None
    return marker if isinstance(marker, dict) and marker.get("lane") else None


def ticket_row(db: Any, workspace_id: Any, task_id: Any) -> Any:
    """This workspace's ticket, or None."""
    from core.models.core import BoardTask

    if task_id is None:
        return None
    return db.query(BoardTask).filter(BoardTask.id == int(task_id), BoardTask.workspace_id == workspace_id).first()


def _run_card(db: Any, workspace_id: Any, lane: str, ref: Any) -> Any:
    """The ticket the run works: the session's or the board run's own, a playbook run's card."""
    from core.models.core import BoardTask

    if lane != PLAYBOOK_LANE:
        try:
            return ticket_row(db, workspace_id, ref)
        except (TypeError, ValueError):
            return None
    return (db.query(BoardTask)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == PLAYBOOK_CARD_SOURCE,
                    BoardTask.source_id == str(ref))
            .first())


def _playbook_run(lane: str, ref: Any, task: Any) -> Optional[str]:
    """The playbook run a call works for: the playbook lane's own, or the run a CLI agent's
    step ticket names (``recipe:<run>:<step>``, api/recipe_executor); else None."""
    if lane == PLAYBOOK_LANE:
        return str(ref)
    if task is None or str(task.source_type or "") != PLAYBOOK_CARD_SOURCE or not task.source_id:
        return None
    parts = str(task.source_id).split(STEP_SEPARATOR)
    return parts[1] if len(parts) > 2 and parts[0] == PLAYBOOK_CARD_SOURCE else parts[0]


def _on_autos_brief(db: Any, workspace_id: Any, task: Any, run_id: Optional[str]) -> bool:
    """Auto wrote the ticket (an agent's ticket, and that agent is the workspace's Auto), or
    the playbook run it belongs to was started by Auto's platform call."""
    if task is not None and str(task.created_by_type or "") == WRITTEN_BY_AN_AGENT:
        auto = _auto_id(db, workspace_id)
        if auto is not None and str(task.created_by_id or "") == str(auto):
            return True
    if run_id is None:
        return False
    from core.models.core import RecipeExecution

    run = (db.query(RecipeExecution)
           .filter(RecipeExecution.execution_id == run_id, RecipeExecution.workspace_id == workspace_id)
           .first())
    return run is not None and run.triggered_by == LAUNCHED_BY_AUTO


def _auto_id(db: Any, workspace_id: Any) -> Optional[int]:
    from core.models.core import Agent

    row = (db.query(Agent.id)
           .filter(Agent.workspace_id == workspace_id, Agent.slug == AUTO_SLUG.format(workspace_id=workspace_id),
                   Agent.is_system_agent.is_(True))
           .first())
    return row[0] if row else None


def _agent(agent_id: Any) -> Optional[int]:
    try:
        return int(agent_id) if agent_id else None
    except (TypeError, ValueError):
        return None


def _mark(db: Any, grant_id: int, marker: Dict[str, Any]) -> Optional[str]:
    """The grant remembers the ticket and the agent the send runs as on the click. The same
    send asked on another ticket reuses its pending grant (``issue_tool_grant``): that
    ticket keeps it, and its id is returned (None when the grant is this ticket's)."""
    from core.models.approval_grants import ApprovalGrant

    grant = db.get(ApprovalGrant, grant_id)
    if grant is None:
        raise LookupError(f"grant {grant_id} is not on record")
    held = send_marker(grant)
    if held is not None and held.get("context") != marker.get("context"):
        return str(held.get("task_id") or held.get("context"))
    grant.details = {**(grant.details if isinstance(grant.details, dict) else {}), AGENT_SEND: marker}
    if grant.agent_id is None and marker.get("agent_id"):
        grant.agent_id = marker["agent_id"]
    return None


def _park_on(db: Any, workspace_id: Any, task_id: int, grant_id: int, ask: Dict[str, Any]) -> None:
    """The session's ticket parks on the card when its turn ends (``_park_for_answer``), as on
    its own question: an open entry in its ask ledger, once per card."""
    from services.cli_host_service import SESSION_ASKS_KEY, record_session_ask, session_asks
    from services.session_plans import SEND_KIND

    task = ticket_row(db, workspace_id, task_id)
    if task is None:
        return
    ref = dict(task.runtime_ref or {})
    if any(int(entry.get("grant_id") or 0) == grant_id for entry in session_asks(ref)):
        return
    ref = record_session_ask(ref, grant_id=grant_id, question=str(ask.get("question_md") or ask.get("message") or ""))
    asks = [{**entry, "kind": SEND_KIND} if int(entry.get("grant_id") or 0) == grant_id else entry
            for entry in session_asks(ref)]
    task.runtime_ref = {**ref, SESSION_ASKS_KEY: asks}


def now_said() -> str:
    return datetime.now(timezone.utc).strftime(AT)


async def sends_on_the_click(db: Any, grant: Any) -> bool:
    """The owner clicked a send card an agent raised on Auto's ticket: run it once, write it
    on the ticket, send the ticket back to work. False for any other grant."""
    from modules.tools.discovery.agent_sends_click import send_once, sent_on_the_ticket
    from services.click_results import executed_summary

    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY

    marker = send_marker(grant)
    if marker is None:
        return False
    if getattr(grant, "revoked_by", None) == GRANT_CONSUMED_BY:  # this click already sent it: never twice
        return True
    summary = executed_summary(await send_once(db, grant, marker))
    grant.details = {**(grant.details if isinstance(grant.details, dict) else {}), "executed_result": summary}
    db.commit()  # the claim and the outcome first: nothing after a send may undo them
    try:
        sent_on_the_ticket(db, grant, marker, summary)
    except Exception:  # noqa: BLE001 — logged; the send and its outcome are already kept on the grant
        logger.exception("[agent_sends] grant %s ran; its ticket %s could not be told", grant.id, marker.get("task_id"))
        db.rollback()
    return True


def declined_send(db: Any, grant: Any) -> bool:
    """The owner declined a send card an agent raised on Auto's ticket: the ticket fails with
    the reason. False for any other grant."""
    from modules.tools.discovery.agent_sends_click import fail_the_ticket

    marker = send_marker(grant)
    if marker is None:
        return False
    fail_the_ticket(db, grant, marker, DECLINED.format(subject=marker.get("subject"), recipient=marker.get("recipient")))
    return True


__all__ = ["AGENT_SEND", "CARD_RAISED", "DECLINED", "SENT", "addressed", "autos_ticket", "declined_send",
           "send_marker", "sends_on_the_click", "waits_on_the_card"]
