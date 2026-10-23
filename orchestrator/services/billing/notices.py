"""Telling the workspace's owner what billing did (PRD-265 W2).

A failed payment, a plan that ended, or a Business offer reaches the owner's
notification destinations (the bell, and wherever they route these events). A
notice is sent after the change it reports has committed; one that can't be sent is
logged and never undoes the change.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

NOTICE_PAYMENT_FAILED = "billing_payment_failed"
NOTICE_SUBSCRIPTION_ENDED = "billing_subscription_ended"
NOTICE_BUSINESS_OFFERED = "billing_business_offered"


@dataclass(frozen=True)
class Notice:
    event_type: str
    title: str
    message: str
    status: str = "warning"


def send_notice(db: Any, workspace_id: Any, notice: Notice) -> None:
    from core.services.notification_dispatcher import NotificationDispatcher

    try:
        asyncio.run(NotificationDispatcher(db, str(workspace_id)).dispatch(
            event_type=notice.event_type, title=notice.title, message=notice.message, status=notice.status,
        ))
    except Exception:  # noqa: BLE001 — the change is committed; a notice that can't be sent never undoes it
        db.rollback()
        logger.exception("[billing] %s notice for workspace %s not sent", notice.event_type, workspace_id)
