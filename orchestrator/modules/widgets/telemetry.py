"""
Widget telemetry helper (PRD-008-A Phase 4).

Fire-and-forget writer for ``widget_event_log``. Modeled on the
PRD-139 ``modules/tools/execution/telemetry.py`` pattern: never raises,
never blocks the caller, never propagates DB or validation errors.

Usage::

    from modules.widgets.telemetry import log_widget_event

    await log_widget_event(
        db,
        site_id=site_id,
        event_type="callback_requested",
        session_id="sess_abc",
        event_data={"phone_hash": "..."},
    )

    # Or fire-and-forget, no await:
    asyncio.create_task(log_widget_event(db, site_id=site_id, event_type="..."))
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def log_widget_event(
    db: Session,
    *,
    site_id: UUID,
    event_type: str,
    session_id: Optional[str] = None,
    event_data: Optional[dict] = None,
) -> None:
    """Append a single row to ``widget_event_log``.

    Fire-and-forget. ANY failure (DB error, unknown event_type, JSONB
    serialisation problem) is logged at WARNING and swallowed — telemetry
    must NEVER fail the calling business path.

    Unknown event types are still written (so we don't lose data), but
    logged at WARNING so the omission from ``WIDGET_EVENT_TYPES`` shows
    up in observability.
    """
    try:
        from core.models.widget_event_log import WIDGET_EVENT_TYPES, WidgetEventLog

        if event_type not in WIDGET_EVENT_TYPES:
            logger.warning(
                "widget_event_log: unknown event_type=%r (writing anyway). "
                "Add it to WIDGET_EVENT_TYPES if intentional.",
                event_type,
            )

        row = WidgetEventLog(
            site_id=site_id,
            session_id=session_id[:64] if session_id else None,
            event_type=event_type[:64],
            event_data=event_data or {},
        )
        db.add(row)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — fire-and-forget by contract
        logger.warning(
            "widget_event_log write failed (event_type=%r site_id=%s): %s",
            event_type,
            site_id,
            exc,
        )
        # Best-effort rollback. If even this fails, give up — the caller
        # must not see the exception.
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
