"""The chat's ticket card and the wait tool's words (PRD-238 S4/S6), out of
handlers_board_tasks.py. PRD-252 R4: they name a ticket by its number (#0042),
never by its id behind a '#'.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from services.ticket_numbers import format_number, ticket_label

#: PRD-238 S4: statuses at which a ticket has nothing more to wait for.
WAIT_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled", "review", "blocked"})


def task_card(task: Any, agent_name: Optional[str] = None) -> Dict[str, Any]:
    """PRD-238 S6: the compact, live-updatable card the chat renders for a ticket.

    Only ids, status, names, timestamps and the session's own counters from
    ``runtime_ref`` — never descriptions, transcripts or file contents.
    """
    ref = getattr(task, "runtime_ref", None) or {}
    ref = ref if isinstance(ref, dict) else {}
    tools = ref.get("recent_tools") or []
    last_tool = None
    if isinstance(tools, (list, tuple)) and tools:
        last = tools[-1]
        # F168: the host's entries carry ``tool``; ``name`` is the older shape.
        last_tool = (last.get("tool") or last.get("name")) if isinstance(last, dict) else str(last)
    files = ref.get("files_touched") or []
    return {
        "id": task.id,
        "number": format_number(getattr(task, "workspace_seq", None)),  # PRD-252 R4
        "title": task.title,
        "status": task.status,
        "assigned_agent": agent_name or "unassigned",
        "runtime": ref.get("runtime"),
        "last_tool": last_tool,
        "files_touched": len(files) if isinstance(files, (list, tuple)) else 0,
        "exit_reason": ref.get("exit_reason"),
        "denials": int(ref.get("denials") or 0) if isinstance(ref.get("denials"), int) else 0,
        "started_at": str(task.started_at) if getattr(task, "started_at", None) else None,
        "completed_at": str(task.completed_at) if getattr(task, "completed_at", None) else None,
    }


def _progress_line(card: Dict[str, Any], waited_s: int) -> str:
    who = card.get("assigned_agent") or "the agent"
    # PRD-252 R4: the ticket's number, never its id behind a '#'
    bits = [f"{who} is working on ticket {card.get('number') or card['id']} · {waited_s} s"]
    if card.get("last_tool"):
        bits.append(f"last tool: {card['last_tool']}")
    if card.get("files_touched"):
        n = card["files_touched"]
        bits.append(f"{n} file{'s' if n != 1 else ''} touched")
    return " · ".join(bits)


def _wait_budget(params: Dict[str, Any]) -> Tuple[int, int]:
    """How long this wait may last (``max_wait_seconds``, capped by the budget) and how often it looks."""
    from config import config

    budget = max(1, int(config.CHATBOT_WAIT_BUDGET_S))
    requested = params.get("max_wait_seconds")
    try:
        limit = min(budget, int(requested)) if requested else budget
    except (TypeError, ValueError):
        limit = budget
    return limit, max(1, int(config.CHATBOT_WAIT_POLL_S))


def _wait_result(task: Any, card: Dict[str, Any], waited: int, limit: int) -> Dict[str, Any]:
    """The wait's answer. PRD-252 R4: it names the ticket by its number."""
    terminal = task.status in WAIT_TERMINAL_STATUSES
    label = ticket_label(task, capital=True)
    return {
        "success": True,
        "terminal": terminal,
        "status": task.status if terminal else "still running",
        "waited_seconds": waited,
        "budget_seconds": limit,
        "task": card,
        "message": (
            f"{label} ended: {task.status}."
            if terminal
            else f"{label} is still running after {waited} s — the watcher will report back when it ends."
        ),
        "frontend_data": {"task_card": card},
    }
