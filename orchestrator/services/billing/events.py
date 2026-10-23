"""Receive a payment provider's webhook exactly once (PRD-265 US-010, FR-6).

1. The provider verifies the signature (a bad one never reaches the database).
2. The event is stored in ``billing_events`` and committed **before** it is
   processed; a redelivery of the same ``(provider, event_id)`` inserts nothing.
3. Processing locks the event's row (two deliveries at once queue), skips an
   event already settled, and lands the plan, the account, the ledger and the
   event's outcome in one transaction.
4. When processing fails the transaction rolls back, the row is marked
   ``failed`` with the error's type, and the error goes back to the route as a
   500, so the provider redelivers and the next delivery tries again.

After the commit: the owner's notice (a failed payment, a plan that ended) and a
fresh balance-alert check, so a top-up clears a "running low" state at once.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from services.billing.apply import IGNORED, NEEDS_REVIEW, PROCESSED, Outcome, apply_fact
from services.billing.facts import WebhookEvent

logger = logging.getLogger(__name__)

RECEIVED = "received"
FAILED = "failed"
SETTLED = frozenset({PROCESSED, IGNORED, NEEDS_REVIEW})


def store(db: Any, event: WebhookEvent) -> None:
    """Keep the event before acting on it (a redelivery inserts nothing). Commits."""
    db.execute(text(
        "INSERT INTO billing_events (provider, event_id, event_type, livemode, status, payload) "
        "VALUES (:provider, :event_id, :event_type, :livemode, 'received', CAST(:payload AS jsonb)) "
        "ON CONFLICT ON CONSTRAINT uq_billing_events_provider_event DO NOTHING"
    ), {"provider": event.provider, "event_id": event.event_id, "event_type": event.event_type,
        "livemode": event.livemode, "payload": json.dumps(event.data, default=str)})
    db.commit()


def _lock(db: Any, event: WebhookEvent) -> tuple[int, str]:
    row = db.execute(text(
        "SELECT id, status FROM billing_events WHERE provider = :provider AND event_id = :event_id FOR UPDATE"
    ), {"provider": event.provider, "event_id": event.event_id}).fetchone()
    return int(row[0]), str(row[1])


def _settle(db: Any, row_id: int, outcome: Outcome) -> None:
    db.execute(text(
        "UPDATE billing_events SET status = :status, workspace_id = CAST(:ws AS uuid), payment_ref = :ref, "
        "grant_source = :grant, amount_cents = :amount, note = :note, attempts = attempts + 1, "
        "processed_at = now() WHERE id = :id"
    ), {"status": outcome.status, "ws": str(outcome.workspace_id) if outcome.workspace_id else None,
        "ref": outcome.payment_ref, "grant": outcome.grant_source, "amount": outcome.amount_cents,
        "note": outcome.note, "id": row_id})


def _mark_failed(db: Any, event: WebhookEvent, error: BaseException) -> None:
    db.execute(text(
        "UPDATE billing_events SET status = 'failed', attempts = attempts + 1, note = :note "
        "WHERE provider = :provider AND event_id = :event_id"
    ), {"note": f"{type(error).__name__} (see the server log)", "provider": event.provider,
        "event_id": event.event_id})
    db.commit()


def _after_commit(db: Any, outcome: Outcome) -> None:
    from services.billing.notices import send_notice
    from services.credits.alerts import check_alerts

    if not outcome.workspace_id:
        return
    try:
        check_alerts(db, str(outcome.workspace_id))
    except Exception:  # noqa: BLE001 — the payment is settled; an alert that can't be sent never undoes it
        db.rollback()
        logger.exception("[billing] balance alert for workspace %s not checked", outcome.workspace_id)
    if outcome.notice is not None:
        send_notice(db, outcome.workspace_id, outcome.notice)


def process(db: Any, event: WebhookEvent, provider: Any) -> str:
    """Act on a stored event once. Returns its status. Raises when it failed."""
    row_id, status = _lock(db, event)
    if status in SETTLED:
        db.commit()
        return status
    outcome = apply_fact(db, provider.translate(event), provider=event.provider)
    _settle(db, row_id, outcome)
    db.commit()
    logger.info("[billing] %s %s → %s (%s)", event.event_type, event.event_id, outcome.status, outcome.note or "")
    _after_commit(db, outcome)
    return outcome.status


def receive(db: Any, provider: Any, payload: bytes, signature: str) -> str:
    """Verify, store and process one webhook. ``SignatureError`` when it isn't
    the provider's; any processing error is re-raised after the row is marked failed."""
    event = provider.parse_webhook(payload, signature)
    store(db, event)
    try:
        return process(db, event, provider)
    except Exception as exc:
        db.rollback()
        logger.exception("[billing] %s %s failed; the provider will redeliver it", event.event_type, event.event_id)
        _mark_failed(db, event, exc)
        raise
