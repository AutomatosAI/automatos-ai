"""``audit_logs`` on the in-memory SQLite the theme and Sync route tests use (#1657).

``AuditLog`` declares Postgres's UUID and JSONB, which SQLite can't create, so the table is made here
with the same columns; ``AuditService.log`` writes it through the real model. ``rows`` reads it back
as plain dicts, and ``written`` is one row as text, for checking that nothing secret was stored.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import sqlalchemy as sa

from core.workspaces.audit import AuditLog

AUDIT_LOGS_DDL = sa.text(
    "CREATE TABLE audit_logs (id INTEGER PRIMARY KEY, workspace_id CHAR(32) NOT NULL, user_id INTEGER, "
    "actor_type VARCHAR(20) NOT NULL DEFAULT 'user', action VARCHAR(100) NOT NULL, resource_type VARCHAR(50), "
    "resource_id VARCHAR(255), resource_name VARCHAR(255), details JSON, ip_address VARCHAR(45), "
    "user_agent VARCHAR(500), created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
)
COLUMNS = tuple(column.name for column in AuditLog.__table__.columns)


def create_audit_logs(conn: Any) -> None:
    """Make ``audit_logs`` on a SQLite connection."""
    conn.execute(AUDIT_LOGS_DDL)


def rows(db: Any, action: str) -> List[Dict[str, Any]]:
    """Every ``audit_logs`` row with ``action``, oldest first, as dicts."""
    db.expire_all()
    found = db.query(AuditLog).filter(AuditLog.action == action).order_by(AuditLog.id).all()
    return [{name: getattr(row, name) for name in COLUMNS} for row in found]


def written(row: Dict[str, Any]) -> str:
    """The whole row as text: whatever a reader of the table could see."""
    return json.dumps(row, default=str)


__all__ = ["AUDIT_LOGS_DDL", "create_audit_logs", "rows", "written"]
