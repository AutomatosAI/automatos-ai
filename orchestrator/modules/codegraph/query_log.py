"""The code graph's query log: one ``codegraph_query_logs`` row per search.

Moved out of ``CodeGraphService._log_query`` (codegraph_service.py is over 800 lines).

The table stores ``duration_ms`` and a NOT NULL ``workspace_id``. The row is telemetry,
so a failed write never fails the search: it is logged and rolled back.

PRD-268 P268W1-RVW-1 (K17): ``platform_codegraph_search`` and ``platform_codegraph_get_symbol``
are labelled read, so a check-in's search logs no row (``core.read_only_call``).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.read_only_call import only_reading

logger = logging.getLogger(__name__)

_INSERT = text("""
    INSERT INTO codegraph_query_logs
    (project_id, query_type, query_text, results_count, duration_ms, workspace_id)
    VALUES (:project_id, :query_type, :query_text, :results_count, :duration_ms, :workspace_id)
""")


def log_query(db: Session, project_id: Any, query_type: str, query_text: str, results_count: int,
              duration_ms: float, workspace_id: Optional[str]) -> None:
    """Write the search's log row on ``db`` and commit; nothing in a check-in. Best-effort."""
    if only_reading():
        return
    try:
        db.execute(_INSERT, {
            "project_id": project_id,
            "query_type": query_type,
            "query_text": query_text,
            "results_count": results_count,
            "duration_ms": duration_ms,
            "workspace_id": workspace_id,
        })
        db.commit()
    except Exception:  # noqa: BLE001 — logged; a telemetry write never fails the search
        logger.exception("CodeGraph query-log write failed (non-fatal)")
        _roll_back(db)


def _roll_back(db: Session) -> None:
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 — logged; the search goes on
        logger.exception("CodeGraph query-log rollback failed")


__all__ = ["log_query"]
