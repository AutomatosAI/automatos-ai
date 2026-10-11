"""PRD-256 P256-FIX-RVW-23: what an approval card says for a timer set for an agent.

``platform_schedule_task`` is owner-only now (Decision D1: an agent's timer is an
agent-setting change, as a playbook's timer is). The card names what the click files:
the agent that runs it (name and #id; the caller itself when a chat delivery names
none, the Inbox for an unassigned ticket), when it runs in words (a cron as
``cron_when.plain_cron`` says it, a one-shot's date and time), how it is delivered (a
ticket's review mode too), and the brief's first line, saying how many more it holds. The agent is read in the caller's workspace only; by the card,
its name has been bound to that one agent's id (``agent_binding``).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_text import said_line, value_line

AGENT_ID, CALLER = "agent_id", "_agent_id"
AGENT_LINE = "agent: '{name}' (agent #{id})"
NO_AGENT = "agent: none named; the ticket goes to the board's Inbox"
UNKNOWN_AGENT = "agent: #{id}"
ONE_SHOT = "one_shot"
CHAT, BOARD_TASK = "chat", "board_task"
DELIVERED = {CHAT: "a chat with the agent, opened when it fires",
             BOARD_TASK: "a ticket on the board, filed when it fires"}
ONCE = "once, on {at}"
AT = "%a %d %b %Y at %H:%M"
UTC = " (UTC)"
TIMES = "{count} times"
MAX_RUNS = "max_runs"
DEFAULT_REVIEW = "auto"
MORE_LINES = " (+{count} more line{s})"
# The card is raised only in a person's chat, where a brief that sends or orders is
# filed for their review unless the call named a review_mode (brief_sends, FX-010).
REVIEWS = {"human": "human (waits in Review for you before it closes)", "auto": "auto (closes by itself when done)"}


def schedule_task_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The agent, when it runs, how it is delivered (and how often at most), then the brief's first line."""
    deliver_as = params.get("deliver_as") or CHAT
    lines = [value_line(_agent(db, workspace_id, params, deliver_as)),
             said_line("runs", _when(params.get("task_type"), params.get("schedule"))),
             said_line("delivered as", DELIVERED.get(deliver_as, deliver_as))]
    if params.get(MAX_RUNS) is not None:
        lines.append(said_line("at most", TIMES.format(count=params[MAX_RUNS])))
    if deliver_as == BOARD_TASK:
        if params.get("title"):
            lines.append(said_line("ticket", params["title"]))
        lines.append(said_line("review", _review(params)))
    lines.append(said_line("brief", _first_line(params.get("description"))))
    return lines


def _review(params: Dict[str, Any]) -> str:
    """The review mode the ticket is filed with, as ``reviewed_by_a_person`` sets it on the click."""
    from modules.tools.discovery.brief_sends import HUMAN, sends_or_orders

    said = params.get("review_mode")
    mode = said or (HUMAN if sends_or_orders(params.get("title"), params.get("description")) else DEFAULT_REVIEW)
    return REVIEWS.get(mode, mode)


def _agent(db: Any, workspace_id: Any, params: Dict[str, Any], deliver_as: str) -> str:
    """The agent the timer runs as: the bound ``agent_id``, else the caller itself for a
    chat delivery, else none (an unassigned ticket)."""
    ident = params.get(AGENT_ID)
    if ident in (None, "") and deliver_as == CHAT:
        ident = params.get(CALLER)
    if ident in (None, ""):
        return NO_AGENT
    agent = _agent_row(db, workspace_id, ident)
    return AGENT_LINE.format(name=agent.name, id=agent.id) if agent is not None else UNKNOWN_AGENT.format(id=ident)


def _agent_row(db: Any, workspace_id: Any, ident: Any) -> Optional[Any]:
    from core.models import Agent

    try:
        agent_id = int(str(ident).lstrip("#"))
    except (TypeError, ValueError):
        return None
    return db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()


def _when(task_type: Any, schedule: Any) -> str:
    """"Mondays at 09:00 (UTC)" for a cron; "once, on Mon 12 Oct 2026 at 09:00 (UTC)" for a one-shot."""
    from modules.tools.discovery.cron_when import plain_cron

    said = str(schedule or "").strip()
    if task_type != ONE_SHOT:
        return plain_cron(said, None) if said else said
    try:
        at = datetime.fromisoformat(said.replace("Z", "+00:00"))
    except ValueError:
        return ONCE.format(at=said)
    if at.tzinfo is None:
        return ONCE.format(at=at.strftime(AT))
    return ONCE.format(at=at.astimezone(timezone.utc).strftime(AT)) + UTC


def _first_line(value: Any) -> str:
    """The brief's first line, saying how many more it holds."""
    lines = [line for line in str(value or "").strip().splitlines() if line.strip()]
    if not lines:
        return ""
    more = len(lines) - 1
    return lines[0] + (MORE_LINES.format(count=more, s="" if more == 1 else "s") if more else "")


__all__ = ["schedule_task_lines"]
