"""PRD-239 S2 — chat with a session agent (``runtime: cli``).

A session agent never runs in the LLM runtime (PRD-234 S1a). Talking to one in
the chat therefore means: the message becomes a board ticket its Claude Code
session works; the turn replies at once with what happened (ticket filed, host
online or not, waiting for approval or not) and a live card; the session's final
text lands in the conversation when the session ends (``deliver_session_reply``,
called from the host result path).

Continuity (owner decision D1): the ticket asks the host that ran this
conversation's previous session to ``--resume`` it, so the agent keeps the
whole exchange in its own context. The first turn of a conversation — or one
after the host changed — carries the recent conversation as written context.

Local edition only by construction: cli agents exist only under
``CLI_RUNTIME_ENABLED``; nothing here runs for an API agent.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from core.models.core import Agent
from services.cli_ticket_lane import (
    CHAT_SOURCE_TYPE,
    chat_origin_of,
    chat_source_id,
    file_cli_ticket,
    previous_session_of,
)

logger = logging.getLogger(__name__)

CONTEXT_TURNS = 6
CONTEXT_CHARS = 600
TITLE_CHARS = 80
REPLY_SOURCE_ORIGIN = "session_agent"


def _text_of(message: Dict[str, Any]) -> str:
    parts = message.get("parts") if isinstance(message, dict) else None
    if isinstance(parts, list):
        texts = [str(p.get("text")) for p in parts if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    content = message.get("content") if isinstance(message, dict) else None
    return str(content).strip() if content else ""


def conversation_context(history: Sequence[Dict[str, Any]], agent_name: str) -> str:
    """The recent exchange BEFORE the message being sent, as written context for
    a session that has no memory of this conversation yet. Empty when there is
    nothing before it."""
    turns = [m for m in (history or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    if turns and turns[-1].get("role") == "user":
        turns = turns[:-1]  # the message itself rides in the ticket body
    turns = [t for t in turns if _text_of(t)][-CONTEXT_TURNS:]
    if not turns:
        return ""
    lines = []
    for turn in turns:
        # Earlier replies in this chat may have come from Auto or another agent
        # (the operator can switch agents mid-conversation) — never attribute
        # them to this agent by name.
        who = "Operator" if turn.get("role") == "user" else "Assistant"
        text = _text_of(turn)
        if len(text) > CONTEXT_CHARS:
            text = text[:CONTEXT_CHARS].rstrip() + "…"
        lines.append(f"- **{who}:** {text}")
    return (
        "## Conversation so far\n"
        f"(earlier assistant replies may have come from Auto or another agent, not from {agent_name})\n"
        + "\n".join(lines)
    )


def ticket_title(agent_name: str, user_text: str) -> str:
    first = next((line.strip() for line in (user_text or "").splitlines() if line.strip()), "") or "(no text)"
    if len(first) > TITLE_CHARS:
        first = first[:TITLE_CHARS].rstrip() + "…"
    return f"Chat with {agent_name}: {first}"


def ticket_prompt(agent_name: str, user_text: str, history: Sequence[Dict[str, Any]], continuing: bool) -> str:
    head = (
        f"# Chat message for {agent_name}\n\n"
        "The operator sent you this message in the Automatos chat. Reply as you would in a "
        "conversation: answer directly, do the work the message asks for, and end with the "
        "reply the operator should read. Keep the reply itself concise.\n"
    )
    context = "" if continuing else conversation_context(history, agent_name)
    body = f"## Message\n{(user_text or '').strip()}"
    return head + ("\n" + context + "\n\n" if context else "\n") + body


def file_chat_ticket(
    db: Session,
    *,
    workspace_id: Any,
    chat_id: str,
    agent: Any,
    user_text: str,
    history: Sequence[Dict[str, Any]],
    user_id: Any,
) -> Tuple[Any, bool]:
    """File the ticket for one chat message. Returns ``(ticket, continuing)``
    where ``continuing`` says a previous session of this conversation exists and
    the ticket asks its host to resume it."""
    from services.board_consent import actor_from_user_id

    previous = previous_session_of(db, workspace_id, chat_id, agent.id)
    task = file_cli_ticket(
        db,
        workspace_id=workspace_id,
        agent_id=agent.id,
        title=ticket_title(agent.name, user_text),
        prompt=ticket_prompt(agent.name, user_text, history, continuing=previous is not None),
        source_type=CHAT_SOURCE_TYPE,
        source_id=chat_source_id(chat_id, uuid.uuid4().hex[:12]),
        tags=["chat"],
        actor=actor_from_user_id(user_id),
        created_by_type="user",
        created_by_id=str(user_id) if user_id is not None else None,
        resume_session_id=previous[0] if previous else None,
        resume_host_id=previous[1] if previous else None,
    )
    return task, previous is not None


def turn_line(agent_name: str, task: Any, continuing: bool) -> str:
    """What the turn says, honestly: filed, continuing or fresh, host or no host,
    waiting for approval or not, and where the reply will land."""
    line = f"{agent_name} runs as a Claude Code session on your CLI host, so I've filed ticket #{task.id} for this message"
    line += " — the session continues where your last exchange left off." if continuing else "."
    if getattr(task, "blocked_reason", None):
        line += " No CLI host is online right now; it starts the moment one connects."
    elif getattr(task, "status", None) == "blocked":
        line += " It is waiting for your approval on the board."
    line += f" {agent_name}'s reply lands here when the session ends."
    return line


async def produce_session_agent_turn(
    *,
    db: Session,
    workspace_id: Any,
    chat_id: str,
    agent_id: int,
    message_history: Sequence[Dict[str, Any]],
    user_text: str,
    user_id: Any,
) -> AsyncIterator[str]:
    """The chat turn for a session agent: chat id, one honest line, the ticket
    card, finish — and the assistant message persisted with the card."""
    from consumers.chatbot import ChatService
    from consumers.chatbot.streaming import get_streaming_handler
    from modules.tools.discovery.handlers_board_tasks import task_card

    handler = get_streaming_handler()
    yield handler.format_aisdk_chat_id(chat_id)

    agent = db.query(Agent).filter(Agent.id == int(agent_id)).first()
    if agent is None:
        line = f"Agent {agent_id} is not in this workspace."
        yield handler.format_aisdk_text(line)
        yield handler.format_aisdk_finish()
        return

    task, continuing = file_chat_ticket(
        db, workspace_id=workspace_id, chat_id=chat_id, agent=agent,
        user_text=user_text, history=message_history, user_id=user_id,
    )
    line = turn_line(agent.name, task, continuing)
    card = task_card(task, agent.name)
    yield handler.format_aisdk_text(line)
    yield handler.format_aisdk_tool_data({"task_card": card})
    yield handler.format_aisdk_finish()

    ChatService(db).save_message(
        chat_id=chat_id,
        role="assistant",
        parts=[{"type": "text", "text": line}, {"type": "task_card", "card": card}],
        workspace_id=str(workspace_id),
    )
    logger.info("[SessionAgentChat] chat %s → ticket #%s for agent %s (continuing=%s)", chat_id, task.id, agent.id, continuing)


def reply_text(agent_name: str, task: Any, exec_result: Dict[str, Any], status: str) -> str:
    """The session's ending as the conversation should read it."""
    ref = getattr(task, "runtime_ref", None)
    ref = ref if isinstance(ref, dict) else {}
    result = str(exec_result.get("result") or "").strip()
    if status in ("done", "review"):
        text = result or (
            f"{agent_name}'s session ended without a written reply (exit: {ref.get('exit_reason') or status})."
        )
        if status == "review":
            denials = ref.get("denials") if isinstance(ref.get("denials"), int) else 0
            note = f"_Ticket #{task.id} is held for review on the board"
            note += f": {denials} tool call{'s' if denials != 1 else ''} refused._" if denials else "._"
            text = f"{text}\n\n{note}"
        return text
    if status == "cancelled":
        return f"{agent_name}'s session for ticket #{task.id} was cancelled before it answered."
    error = str(exec_result.get("error") or "the session failed").strip()
    return f"{agent_name} could not finish ticket #{task.id}: {error}"


def deliver_session_reply(
    db: Session, task: Any, exec_result: Dict[str, Any], status: str, agent_name: Optional[str] = None
) -> Optional[object]:
    """Post the session's ending into the conversation that asked (fail-soft).
    No-op for tickets that are not chat tickets."""
    chat_id = chat_origin_of(task)
    if not chat_id:
        return None
    name = agent_name or f"Agent {getattr(task, 'assigned_agent_id', '?')}"
    try:
        from services.chat_messenger import deliver_background_message

        return deliver_background_message(
            db,
            workspace_id=task.workspace_id,
            text=reply_text(name, task, exec_result or {}, status),
            source={"origin": REPLY_SOURCE_ORIGIN, "label": f"{name} · Claude Code session", "task_id": task.id},
            chat_id=chat_id,
            clerk_user_id=None,
            link_type="task",
            link_id=str(task.id),
        )
    except Exception:  # noqa: BLE001 — the reply is a courtesy on top of the board's record
        logger.warning("[SessionAgentChat] reply delivery failed for ticket #%s", getattr(task, "id", "?"), exc_info=True)
        return None
