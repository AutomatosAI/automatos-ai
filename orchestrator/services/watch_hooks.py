"""
Watch terminal hooks -- PRD-204 S3
==================================

THE documented fail-soft seam between producers and the watch registry.
Producers (mission transition choke point, playbook executor terminal
blocks, board-task terminal dispatch) call :func:`watch_ingest_terminal`
at their terminal boundaries; a raising watch service must NEVER break the
producer, so this wrapper catches EVERYTHING and logs (the
knowledge_flywheel / _dispatch_mission_event pattern).

This is the one place a blanket ``except Exception`` around watch ingest is
sanctioned -- everywhere else watch errors propagate normally.

Sync + DB-only by design: the mission choke point (``transition_run``) is
synchronous, so the hook cannot await. The ingest joins the CALLER's
transaction (no commit here); notifications stay with the producers (S4)
and the watcher tick (S5).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def watch_ingest_terminal(
    db: Session,
    workspace_id: UUID | str,
    target_type: str,
    target_id: str,
    terminal_state: str,
    summary: Optional[str] = None,
    cost_snapshot: Optional[Dict[str, Any]] = None,
    output_pointer: Optional[str] = None,
) -> None:
    """Report a target's terminal state to its live watch -- fail-soft.

    No-op when the target has no live watch. Idempotent per
    (watch, target): duplicate deliveries are swallowed by the event_key
    unique constraint. Never raises into the producer.
    """
    try:
        from services.watch_service import WatchService

        WatchService.ingest_terminal(
            db,
            workspace_id=workspace_id,
            target_type=target_type,
            target_id=str(target_id),
            terminal_state=terminal_state,
            summary=summary,
            cost_snapshot=cost_snapshot,
            output_pointer=output_pointer,
        )
    except Exception:  # noqa: BLE001 -- the sanctioned fail-soft seam (PRD-204 S3)
        logger.warning(
            "[WatchHooks] terminal ingest failed for %s:%s (state=%s) -- "
            "producer unaffected",
            target_type,
            target_id,
            terminal_state,
            exc_info=True,
        )
