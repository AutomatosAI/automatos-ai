"""PRD-140 — bypass audit writer.

Records every system-agent / workspace-owner permission bypass into a
queryable table so the workspace owner can answer "did anyone bypass
anything in the last 7 days?" without grepping logs.

The writer is fail-soft: if the audit insert fails, the caller still
proceeds (permission is already granted) but a warning is logged. We
never block a legitimate bypass on audit-storage failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import text

from core.security.hierarchy_permissions import PermissionDecision

logger = logging.getLogger(__name__)


def record_bypass(
    db,
    *,
    decision: PermissionDecision,
    workspace_id: UUID | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert one row into ``permission_bypass_log`` for a granted bypass.

    No-op when ``decision.bypass`` is False. Caller owns the transaction —
    we do not commit here, matching the pattern used by NotificationDispatcher
    and ReportService.
    """
    if not decision.bypass:
        return

    try:
        db.execute(
            text(
                """
                INSERT INTO permission_bypass_log
                    (workspace_id, actor_agent_id, actor_name, actor_kind,
                     target_type, target_id, change_type, reason, source, metadata)
                VALUES
                    (:workspace_id, :actor_agent_id, :actor_name, :actor_kind,
                     :target_type, :target_id, :change_type, :reason, :source, :metadata)
                """
            ),
            {
                "workspace_id": str(workspace_id),
                "actor_agent_id": decision.actor_agent_id,
                "actor_name": decision.actor_name or "unknown",
                "actor_kind": decision.bypass_kind or "unknown",
                "target_type": decision.target_type or "unknown",
                "target_id": decision.target_id,
                "change_type": decision.change_type or "unknown",
                "reason": decision.reason,
                "source": decision.source,
                "metadata": json.dumps(metadata or {}),
            },
        )
    except Exception as exc:
        # Never block a legitimate bypass on audit failure — but make the
        # gap visible so the operator knows audit coverage degraded.
        logger.error(
            "[bypass_audit] failed to record bypass actor=%s target=%s/%s change=%s: %s",
            decision.actor_name,
            decision.target_type,
            decision.target_id,
            decision.change_type,
            exc,
            exc_info=True,
        )
