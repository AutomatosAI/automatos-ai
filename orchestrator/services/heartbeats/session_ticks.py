"""A Claude Code agent's heartbeat reports when its session ends, not when it is queued.

A ``runtime: cli`` agent's heartbeat files a board ticket for the paired host (PRD-234 S3)
and returns at once with one finding, "queued for your Claude Code session as ticket #0330",
which was handled like a result. On the Automatos HQ install (13 Nov 2026) the News & Research
agent's hourly heartbeat filed, every hour, a "standup" report holding only that line and a
"Review News & Research heartbeat — queued for …" ticket that ran Auto on nothing: 24 cards a
day, and the empty report hid the agent's real digest from ``get_latest_report``. An agent
reporting to Telegram, a webhook or the workspace got the same line as a notification.

Now a tick that only queued a session (``only_queued_a_session``) files no report and sends
nothing. When the session's ticket finishes, ``services.task_report`` files its report and
calls ``report_finished_session``, which hands what the session found to the heartbeat's own
dispatcher, so every ``report_to`` route gets it, as an API agent's tick does.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

HEARTBEAT_SOURCE = "heartbeat"
# What a Claude Code agent's tick records when it files its session ticket
# (``HeartbeatService._agent_tick``: ``{"action": "file_cli_ticket", "task_id": …}``).
FILE_CLI_TICKET = "file_cli_ticket"
FAILED_TICKET_STATUSES = ("failed", "cancelled")
SESSION_FINDING = "llm_analysis"          # the check the dispatcher reads its message from
SESSION_ERROR_FINDING = "exec_error"


def only_queued_a_session(actions: Any) -> bool:
    """True when a heartbeat tick's actions are the session ticket(s) it filed and nothing else:
    the tick has no result yet, the session will."""
    try:
        items = json.loads(actions) if isinstance(actions, str) else actions
    except ValueError:
        return False
    if not isinstance(items, list) or not items:
        return False
    return all(isinstance(a, dict) and a.get("action") == FILE_CLI_TICKET for a in items)


def is_session_heartbeat_ticket(task: Any) -> bool:
    """A Claude Code agent's heartbeat ticket (``cli_ticket_lane``: source ``heartbeat``,
    ``agent:<id>``), never Auto's review ticket (whose source is the tick's own id)."""
    from services.cli_ticket_lane import source_id_for

    source_id = str(getattr(task, "source_id", None) or "")
    return getattr(task, "source_type", None) == HEARTBEAT_SOURCE and source_id.startswith(source_id_for("agent", ""))


def finished_session_result(workspace_id: str, task: Any, answer: str) -> Dict[str, Any]:
    """The heartbeat result a finished session's ticket stands for, in the tick's own shape."""
    failed = getattr(task, "status", None) in FAILED_TICKET_STATUSES or bool(getattr(task, "error_message", None))
    if answer:
        findings: List[Dict[str, Any]] = [{"check": SESSION_FINDING, "detail": answer}]
    else:
        findings = [{"check": SESSION_ERROR_FINDING, "detail": str(getattr(task, "error_message", None) or "")}]
    return {"workspace_id": str(workspace_id), "source_type": "agent", "source_id": str(task.assigned_agent_id),
            "status": "error" if failed else "success", "findings": findings, "actions_taken": []}


async def report_finished_session(workspace_id: str, task: Any, *, answer: str) -> bool:
    """When a Claude Code agent's heartbeat ticket finishes, what its session found goes where
    the agent's heartbeat reports (Auto's review ticket, a webhook or the workspace's
    channels). True when it was handed on; never raises (the dispatcher logs its own failures)."""
    if not is_session_heartbeat_ticket(task) or task.assigned_agent_id is None:
        return False
    from services.heartbeat_service import get_heartbeat_service

    await get_heartbeat_service()._dispatch_heartbeat_notification(finished_session_result(workspace_id, task, answer))
    return True


__all__ = ["finished_session_result", "is_session_heartbeat_ticket", "only_queued_a_session",
           "report_finished_session"]
