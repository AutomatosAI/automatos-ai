"""PRD-266 US-010e (§4.4, §4.8, K15) — the receipts job: Expo's later word on every kept ticket.

The fan-out keeps each ticket Expo accepted in ``mobile_push_tickets`` (US-011b). Only the
ticket's receipt, read by its id some minutes later, says whether Apple or Google took the
message, and a ``DeviceNotRegistered`` receipt is how Expo says the app left the phone.
``run_receipts`` runs on the leader's scheduler every ``config.PUSH_RECEIPTS_INTERVAL_SECONDS``
(``register``, called from ``services.schedule_reconcile.start_schedule_reconcile``):

1. Tickets ``PUSH_RECEIPT_MAX_AGE_SECONDS`` old are deleted unread: Expo keeps receipts a day.
2. The rest, once ``PUSH_RECEIPT_MIN_AGE_SECONDS`` old (Expo's advice: wait before asking),
   oldest first, ``RECEIPTS_BATCH`` ids at a time. Each batch's reads end before the sender
   is asked, so no transaction is held while Expo answers.
3. Per receipt: ok → the ticket goes; ``DeviceNotRegistered`` → that phone's push is off with
   the reason and a ``push_failed`` event, its token kept (K15), and the ticket goes; any other
   error → Expo's code is logged and the ticket goes. A ticket with no receipt yet stays for
   the next run.

A local install asks Expo nothing while its operator has not agreed to Expo's push service
(D21, K13): its tickets only expire. Synchronous: APScheduler runs it in its thread pool, off
the event loop. It commits once per batch, never raises (a failure is logged and rolled back;
the tickets it had not settled stay) and logs counts and Expo's error codes only, never a
push token.
"""
from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from config import config
from core.models.mobile_devices import MobileDevice
from core.models.mobile_push import MobilePushTicket
from services.push import consent
from services.push.fanout import turn_push_off
from services.push.sender import DEVICE_NOT_REGISTERED, TICKET_OK, PushReceipt, PushSender

logger = logging.getLogger(__name__)

# Expo's advice: ask for a ticket's receipt no sooner than 15 minutes after the send.
PUSH_RECEIPT_MIN_AGE_SECONDS = 900
# Expo keeps a receipt for a day: a ticket that old is deleted unread.
PUSH_RECEIPT_MAX_AGE_SECONDS = 86_400
# Expo's getReceipts limit: the ids one request asks about, and one commit settles.
RECEIPTS_BATCH = 1000
RECEIPTS_JOB_ID = "push_receipts_tick"
# A receipt error with no code (never Expo's: the sender always names one).
UNNAMED_ERROR = "unnamed"

COUNT_KEYS = ("expired", "asked", "delivered", "failed", "disabled", "waiting")

SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class _Due:
    """One ticket waiting for its receipt, as plain values: the batch's reads end before Expo is asked."""

    ticket_id: str
    device_id: uuid.UUID
    workspace_id: uuid.UUID
    created_at: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expire(db: Session, now: datetime) -> int:
    """Delete the tickets Expo no longer keeps receipts for; how many went."""
    cutoff = now - timedelta(seconds=PUSH_RECEIPT_MAX_AGE_SECONDS)
    return db.query(MobilePushTicket).filter(MobilePushTicket.created_at <= cutoff).delete(synchronize_session=False)


def _due_batch(db: Session, now: datetime, after: Optional[_Due]) -> List[_Due]:
    """The next ``RECEIPTS_BATCH`` tickets old enough to ask about, oldest first, after ``after``.

    One fan-out's tickets share a ``created_at`` (its transaction's), so the ticket id breaks
    the tie and a ticket that stays (no receipt yet) is never read twice in one run.
    """
    ticket = MobilePushTicket
    window = (
        ticket.created_at <= now - timedelta(seconds=PUSH_RECEIPT_MIN_AGE_SECONDS),
        ticket.created_at > now - timedelta(seconds=PUSH_RECEIPT_MAX_AGE_SECONDS),
    )
    keyset = () if after is None else (or_(
        ticket.created_at > after.created_at,
        and_(ticket.created_at == after.created_at, ticket.ticket_id > after.ticket_id),
    ),)
    rows = (
        db.query(ticket.ticket_id, ticket.device_id, ticket.workspace_id, ticket.created_at)
        .filter(*window, *keyset)
        .order_by(ticket.created_at, ticket.ticket_id)
        .limit(RECEIPTS_BATCH)
        .all()
    )
    return [_Due(*row) for row in rows]


def _turn_off(db: Session, device_id: uuid.UUID, workspace_id: uuid.UUID, now: datetime) -> bool:
    """K15 for one phone, under a row lock in the ticket's workspace.

    Only while its push is on and it holds a token: a phone revoked since (token cleared,
    ``device_revoked``) or already off keeps its reason.
    """
    device = (
        db.query(MobileDevice)
        .filter(MobileDevice.id == device_id, MobileDevice.workspace_id == workspace_id,
                MobileDevice.push_enabled.is_(True), MobileDevice.expo_push_token.isnot(None))
        .with_for_update()
        .first()
    )
    if device is None:
        return False
    turn_push_off(device, now)
    return True


def _log_errors(errors: Mapping[str, int]) -> None:
    """One line with Expo's error codes and their counts, when any receipt was an error."""
    if errors:
        summary = ", ".join(f"{code}={count}" for code, count in sorted(errors.items()))
        logger.info("Push receipts: %d errors (%s)", sum(errors.values()), summary)


def _settle(db: Session, batch: Sequence[_Due], receipts: Mapping[str, PushReceipt], now: datetime) -> Dict[str, int]:
    """Apply one batch's receipts on ``db``: every answered ticket goes, phones that left go off.

    Several tickets of one phone in a batch turn it off once (one ``push_failed`` event).
    """
    answered = [due for due in batch if due.ticket_id in receipts]
    failed = [due for due in answered if receipts[due.ticket_id].status != TICKET_OK]
    errors = Counter(receipts[due.ticket_id].error or UNNAMED_ERROR for due in failed)
    gone: Dict[Tuple[uuid.UUID, uuid.UUID], None] = dict.fromkeys(
        (due.device_id, due.workspace_id) for due in failed if receipts[due.ticket_id].error == DEVICE_NOT_REGISTERED
    )
    disabled = sum(1 for device_id, workspace_id in gone if _turn_off(db, device_id, workspace_id, now))
    if answered:
        (db.query(MobilePushTicket)
         .filter(MobilePushTicket.ticket_id.in_([due.ticket_id for due in answered]))
         .delete(synchronize_session=False))
    _log_errors(errors)
    return {"asked": len(batch), "delivered": len(answered) - len(failed), "failed": len(failed),
            "disabled": disabled, "waiting": len(batch) - len(answered)}


def _batches(db: Session, sender: PushSender, now: datetime) -> Iterator[Dict[str, int]]:
    """Expire, then ask and settle batch by batch; yields each commit's counts once it commits."""
    expired = _expire(db, now)
    db.commit()
    yield {"expired": expired}
    if not consent.expo_push_allowed():
        logger.info("Push receipts: Expo's push service is not agreed on this install; nothing asked")
        return
    after: Optional[_Due] = None
    while batch := _due_batch(db, now, after):
        db.rollback()  # the reads end: nothing was written since the last commit
        receipts = sender.receipts([due.ticket_id for due in batch])
        settled = _settle(db, batch, receipts, now)
        db.commit()
        yield settled
        after = batch[-1]


def _added(counts: Mapping[str, int], more: Mapping[str, int]) -> Dict[str, int]:
    return {key: counts.get(key, 0) + more.get(key, 0) for key in COUNT_KEYS}


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def _default_sender() -> PushSender:
    from services.push.expo import ExpoPushSender

    return ExpoPushSender()


def _close(db: Optional[Session], *, rollback: bool) -> None:
    """Roll back (after a failure) and close; a session that will not close is logged, never raised."""
    if db is None:
        return
    try:
        if rollback:
            db.rollback()
        db.close()
    except Exception:
        logger.exception("Push receipts: the session did not close cleanly")


def run_receipts(
    *,
    sender: Optional[PushSender] = None,
    session_factory: Optional[SessionFactory] = None,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """Read Expo's receipts for every kept ticket old enough, on a session of its own.

    Never raises: a failure is logged and rolled back, and the counts say what was committed
    before it. The counts: expired (deleted unread), asked, delivered, failed (an error
    receipt), disabled (phones turned off: DeviceNotRegistered) and waiting (no receipt yet).
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    db: Optional[Session] = None
    try:
        db = (session_factory or _open_session)()
        for committed in _batches(db, sender or _default_sender(), now or _now()):
            counts = _added(counts, committed)
    except Exception:
        logger.exception("Push receipts: the run failed; the tickets it had not settled stay (%s)", counts)
        _close(db, rollback=True)
        return counts
    _close(db, rollback=False)
    logger.info("Push receipts: %s", counts)
    return counts


def register(scheduler: Any) -> bool:
    """The receipts job on the leader's scheduler; ``False`` when this worker hosts none.

    No job args: a Redis job store pickles them (``schedule_reconcile._tick``), and the job
    needs none.
    """
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    interval = int(config.PUSH_RECEIPTS_INTERVAL_SECONDS)
    scheduler.add_job(
        run_receipts, IntervalTrigger(seconds=interval), id=RECEIPTS_JOB_ID,
        replace_existing=True, max_instances=1, coalesce=True,
    )
    logger.info("Push receipts: read every %ds", interval)
    return True


__all__ = [
    "PUSH_RECEIPT_MAX_AGE_SECONDS",
    "PUSH_RECEIPT_MIN_AGE_SECONDS",
    "RECEIPTS_BATCH",
    "RECEIPTS_JOB_ID",
    "register",
    "run_receipts",
]
