"""Diagnostics handlers for PlatformActionExecutor: what went wrong, grouped by cause.

``platform_get_workspace_errors`` reads this workspace's failed cards, failed LLM calls
and failed tool runs over a window and returns them grouped by cause
(``failure_causes``). Every query is scoped to the caller's workspace.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery.board_waiting import FAILED, MISSION_STEP
from modules.tools.discovery.failure_causes import Failure, as_payload, group_by_cause

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 1
MAX_DAYS = 14
# The most recent rows read per source; a window with more is reported as truncated.
ROWS_PER_SOURCE = 500
LLM_FAILED = ("error", "timeout", "rate_limited")
TOOL_FAILED = ("error", "timeout")


def _naive_utc(at: Optional[datetime]) -> Optional[datetime]:
    """One clock for every source: cards are timezone-aware, usage and tool logs are naive UTC."""
    if at is None or at.tzinfo is None:
        return at
    return at.astimezone(timezone.utc).replace(tzinfo=None)


def _window_days(params: Dict[str, Any]) -> int:
    try:
        days = int(params.get("days") or DEFAULT_DAYS)
    except (TypeError, ValueError):
        days = DEFAULT_DAYS
    return max(1, min(days, MAX_DAYS))


def _failed_cards(db: Session, workspace_id: UUID, since: datetime) -> List[Failure]:
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    rows = (db.query(BoardTask)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.status == FAILED,
                    BoardTask.source_type != MISSION_STEP, BoardTask.updated_at >= since)
            .order_by(BoardTask.updated_at.desc()).limit(ROWS_PER_SOURCE).all())
    numbers = ticket_numbers(db, workspace_id, rows) if rows else {}
    return [Failure(source="card", ref=str(numbers.get(t.id) or t.id),
                    message=t.error_message or t.blocked_reason or "",
                    at=_naive_utc(t.updated_at), agent_id=t.assigned_agent_id) for t in rows]


def _failed_llm_calls(db: Session, workspace_id: UUID, since: datetime) -> List[Failure]:
    from core.models.core import LLMUsage

    rows = (db.query(LLMUsage.id, LLMUsage.status, LLMUsage.error_message, LLMUsage.created_at,
                     LLMUsage.agent_id, LLMUsage.model_id)
            .filter(LLMUsage.workspace_id == workspace_id, LLMUsage.status.in_(LLM_FAILED),
                    LLMUsage.created_at >= _naive_utc(since))
            .order_by(LLMUsage.created_at.desc()).limit(ROWS_PER_SOURCE).all())
    # A call booked as timeout / rate_limited with no message still carries its cause in the status.
    return [Failure(source="llm_call", ref=f"{r.id} ({r.model_id})", message=r.error_message or r.status,
                    at=_naive_utc(r.created_at), agent_id=r.agent_id) for r in rows]


def _failed_tool_runs(db: Session, workspace_id: UUID, since: datetime) -> List[Failure]:
    from core.models.composio_cache import ToolExecutionLog

    rows = (db.query(ToolExecutionLog.id, ToolExecutionLog.status, ToolExecutionLog.error_message,
                     ToolExecutionLog.executed_at, ToolExecutionLog.agent_id, ToolExecutionLog.action_name)
            .filter(ToolExecutionLog.workspace_id == workspace_id, ToolExecutionLog.status.in_(TOOL_FAILED),
                    ToolExecutionLog.executed_at >= _naive_utc(since))
            .order_by(ToolExecutionLog.executed_at.desc()).limit(ROWS_PER_SOURCE).all())
    return [Failure(source="tool_run", ref=f"{r.id} ({r.action_name})", message=r.error_message or r.status,
                    at=_naive_utc(r.executed_at), agent_id=r.agent_id) for r in rows]


async def get_workspace_errors(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """This workspace's failures over the last ``days`` (1–14, default 1), grouped by cause."""
    days = _window_days(params)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    sources = {"card": _failed_cards, "llm_call": _failed_llm_calls, "tool_run": _failed_tool_runs}
    failures: List[Failure] = []
    truncated: List[str] = []
    for name, read in sources.items():
        rows = read(db, workspace_id, since)
        failures.extend(rows)
        if len(rows) >= ROWS_PER_SOURCE:
            truncated.append(name)
    causes = as_payload(group_by_cause(failures))
    return {
        "success": True,
        "period_days": days,
        "total_failures": len(failures),
        "causes": causes,
        "truncated_sources": truncated,
        "note": ("Each failure is under the first cause its error text matches. 'other' holds the rest; "
                 "open an example by its card number or id."),
    }


__all__ = ["get_workspace_errors"]
