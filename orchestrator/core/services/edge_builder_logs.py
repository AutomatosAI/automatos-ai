"""
PRD-139 US-003 / P256-FIX-N14-1 (F416): the nightly EdgeBuilder's log read.

Night 14 (12–13 Oct, F416): build_edges loaded every tool_execution_logs row of its
30-day window in one ``.all()`` (whole ORM rows, the input and output JSON payloads
included) and the backend was OOMKilled at 03:50. The read now pages through the
window by id, loads only the columns the edges and affinities use, and keeps one
slim dict per row; a page's ORM rows are dropped before the next is read.

PRD-263 US-018: the read keeps production rows, the synthetic signal rows (PRD-232
gap and shown markers) and untagged rows; rows tagged ``eval`` or ``replay`` stay out
of the Knowledge Graph builder. Replay traffic is still logged as production until it
is tagged, so today this removes no replay row. A wait for the owner's click
(``asked``) is never read: build_edges counts every status but ``success`` as a
failure, so an ask would become a ``fails_for_intent`` for its tool.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session, load_only

from core.models.composio_cache import ToolExecutionLog

_DEFAULT_LOG_PAGE_SIZE = 1000  # config.EDGE_BUILDER_LOG_PAGE_SIZE overrides

# The telemetry_source values the builder reads, besides untagged (NULL) rows: the
# literals modules/tools/execution/telemetry.py writes (its default and
# SYNTHETIC_SIGNAL_SOURCE). Kept literal so this read never loads the tool package.
READ_SOURCES = ("production", "synthetic_signal")
# telemetry.ASKED_STATUS: a call that waited for the owner's click, never read.
ASKED_STATUS = "asked"

# The columns _log_dict reads; the JSON payloads (input_parameters, output_result,
# token_usage) are never loaded.
_LOG_COLUMNS = (
    ToolExecutionLog.id,
    ToolExecutionLog.agent_id,
    ToolExecutionLog.workspace_id,
    ToolExecutionLog.action_name,
    ToolExecutionLog.app_name,
    ToolExecutionLog.status,
    ToolExecutionLog.user_query,
    ToolExecutionLog.executed_at,
    ToolExecutionLog.router_decision,
)


def log_page_size() -> int:
    """Rows per page of the window read, from config (at least one)."""
    from config import config

    return max(1, int(getattr(config, "EDGE_BUILDER_LOG_PAGE_SIZE", _DEFAULT_LOG_PAGE_SIZE)))


def load_logs(
    db: Session,
    cutoff: datetime,
    workspace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load tool_execution_logs rows from cutoff onwards, in pages, as dicts.

    Pages are keyed on id (a page boundary never skips or repeats a row); the
    result is ordered by executed_at, ties by id, as the sequence edges expect.
    """
    page_size = log_page_size()
    results: List[Dict[str, Any]] = []
    last_id = 0
    while True:
        page = _read_page(db, cutoff, workspace_id, last_id, page_size)
        results.extend(_log_dict(row) for row in page)
        if len(page) < page_size:
            break
        last_id = page[-1].id
    results.sort(key=lambda log: (log["executed_at"], log["id"]))
    return results


def _read_page(
    db: Session,
    cutoff: datetime,
    workspace_id: Optional[str],
    after_id: int,
    page_size: int,
) -> List[Any]:
    """One page of the window: rows with id > after_id, the slim columns only, from
    the sources the builder reads and never an ask."""
    query = (
        db.query(ToolExecutionLog)
        .options(load_only(*_LOG_COLUMNS))
        .filter(ToolExecutionLog.executed_at >= cutoff)
        .filter(ToolExecutionLog.id > after_id)
        .filter(or_(ToolExecutionLog.telemetry_source.in_(READ_SOURCES),
                    ToolExecutionLog.telemetry_source.is_(None)))
        .filter(ToolExecutionLog.status != ASKED_STATUS)
    )
    if workspace_id:
        query = query.filter(ToolExecutionLog.workspace_id == workspace_id)
    return query.order_by(ToolExecutionLog.id.asc()).limit(page_size).all()


def _log_dict(row: Any) -> Dict[str, Any]:
    """The slim dict build_edges works on, from one log row."""
    router = row.router_decision or {}
    # PRD-232 US-011: the surfaced action set (router_decision.candidates) is
    # the 'shown' half of shown-vs-used decay. Present on execution rows and
    # on the synthetic __tool_shown__ rows record_selection persists.
    shown = router.get("candidates")
    return {
        "id": row.id,
        "agent_id": row.agent_id,
        "workspace_id": str(row.workspace_id) if row.workspace_id else None,
        "action_name": row.action_name,
        "app_name": row.app_name,
        "status": row.status,
        "user_query": row.user_query,
        "executed_at": row.executed_at,
        "turn_id": router.get("turn_id"),
        "conversation_id": router.get("conversation_id"),
        "shown_actions": list(shown) if isinstance(shown, (list, tuple)) else [],
    }
