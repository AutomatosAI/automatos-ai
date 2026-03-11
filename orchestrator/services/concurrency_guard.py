"""
ConcurrencyGuard (Symphony-inspired bounded concurrency)
=========================================================
Enforces per-workspace execution limits with state-specific caps.

Config lives in workspace.plan_limits:
{
    "max_concurrent_total": 5,
    "max_concurrent_running": 3,
    "max_concurrent_pending": 10
}

Defaults (from config.py) if not set in workspace:
- max_concurrent_total: 3 (matches WORKER_CONCURRENCY)
- max_concurrent_running: 3
- max_concurrent_pending: 10
"""

import logging
from dataclasses import dataclass, field
from typing import Dict
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcurrencyResult:
    allowed: bool
    reason: str = ""
    current_running: int = 0
    current_pending: int = 0
    limits: Dict[str, int] = field(default_factory=dict)


def _get_limits(workspace_plan_limits: dict) -> Dict[str, int]:
    """Extract concurrency limits from workspace plan_limits, falling back to config defaults."""
    from config import config as app_config

    defaults = {
        "max_concurrent_total": app_config.DEFAULT_MAX_CONCURRENT_TOTAL,
        "max_concurrent_running": app_config.DEFAULT_MAX_CONCURRENT_RUNNING,
        "max_concurrent_pending": app_config.DEFAULT_MAX_CONCURRENT_PENDING,
    }
    if not workspace_plan_limits:
        return defaults

    return {
        key: int(workspace_plan_limits.get(key, defaults[key]))
        for key in defaults
    }


async def check_concurrency(workspace_id: UUID, db: Session) -> ConcurrencyResult:
    """Check if workspace can start a new execution.

    Queries recipe_executions for current running/pending counts and compares
    against workspace plan_limits (or config defaults).

    Returns ConcurrencyResult with allowed=True if a new execution can start.
    """
    from core.models.workspaces import Workspace

    # Load workspace limits
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    plan_limits_raw = (workspace.plan_limits or {}) if workspace else {}
    limits = _get_limits(plan_limits_raw)

    # Count current executions by status
    counts = db.execute(
        text(
            "SELECT status, COUNT(*) AS cnt "
            "FROM recipe_executions "
            "WHERE workspace_id = :ws_id AND status IN ('running', 'pending') "
            "GROUP BY status"
        ),
        {"ws_id": str(workspace_id)},
    ).fetchall()

    current_running = 0
    current_pending = 0
    for row in counts:
        if row[0] == "running":
            current_running = int(row[1])
        elif row[0] == "pending":
            current_pending = int(row[1])

    current_total = current_running + current_pending

    # Check limits
    if current_total >= limits["max_concurrent_total"]:
        return ConcurrencyResult(
            allowed=False,
            reason=f"Total concurrent executions ({current_total}) >= limit ({limits['max_concurrent_total']})",
            current_running=current_running,
            current_pending=current_pending,
            limits=limits,
        )

    if current_running >= limits["max_concurrent_running"]:
        return ConcurrencyResult(
            allowed=False,
            reason=f"Running executions ({current_running}) >= limit ({limits['max_concurrent_running']})",
            current_running=current_running,
            current_pending=current_pending,
            limits=limits,
        )

    if current_pending >= limits["max_concurrent_pending"]:
        return ConcurrencyResult(
            allowed=False,
            reason=f"Pending executions ({current_pending}) >= limit ({limits['max_concurrent_pending']})",
            current_running=current_running,
            current_pending=current_pending,
            limits=limits,
        )

    return ConcurrencyResult(
        allowed=True,
        current_running=current_running,
        current_pending=current_pending,
        limits=limits,
    )
