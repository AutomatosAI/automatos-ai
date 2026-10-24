"""PRD-266 US-011b (§4.4, K6, K7, K13, K15) — an event's push, to the right phones, off the event loop.

The notification dispatcher calls ``hand_over(db, event)`` for an event in
``payload.PUSH_EVENT_TYPES`` (US-010d). ``hand_over`` asks one cheap question on the caller's
session — does this workspace have any phone push could reach? — and only then hands
``deliver`` to ``core.best_effort.off_loop``: inline where no event loop runs, on the
best-effort threads where one does, so a request never waits on Expo (K6).

``deliver`` does the real work on a session of its own, when the push is sent:

1. ``recipients.recipients`` — who may be pushed this type, membership and role read now.
2. ``eligible_devices`` — their active ios/android phones in this workspace with push on, an
   Expo token and a push key: ``clerk`` rows on SaaS, ``paired`` rows on a local install (§8),
   none at all from a local install until the operator agreed (``consent``, K13), minus the
   people whose push switch for the event is off (K7: a ``notification_preferences`` row with
   destination ``push``; the person's row wins over the workspace's, no row means on).
3. One signed message per phone (``payload.build_payload`` / ``to_message``): a question's
   grant loaded in this workspace only (while it still waits), a review's ticket number,
   redaction from ``workspace.settings``. The reads then end, so no transaction or
   connection is held while Expo answers.
4. The sender (``ExpoPushSender`` unless one is given): every accepted ticket is kept in
   ``mobile_push_tickets`` for the receipts job; a DeviceNotRegistered ticket turns that
   phone's push off with the reason and a ``push_failed`` event, and keeps its token (K15),
   unless the phone registered a new token meanwhile.

It commits once (the writes after the send), never raises, and logs ids and counts only:
never a push token, a push key, a question, an option or a body.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from config import config
from core.best_effort import off_loop
from core.models.approval_grants import ApprovalGrant, GrantStatus
from core.models.core import BoardTask
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from core.models.mobile_push import MobilePushTicket
from core.models.workspaces import Workspace
from services.mobile.events import PUSH_FAILED, with_event
from services.push import consent
from services.push.payload import (
    build_payload,
    push_type_for,
    redact_questions_enabled,
    to_message,
)
from services.push.recipients import recipients
from services.push.sender import DEVICE_NOT_REGISTERED, TICKET_OK, PushMessage, PushSender, PushTicket
from services.ticket_numbers import ticket_numbers

logger = logging.getLogger(__name__)

# A paired browser ('web') never receives a push: only phones do.
PUSH_PLATFORMS = ("ios", "android")
PUSH_DESTINATION = "push"
# The links whose id is an approval grant's (a question's or a tool approval's).
GRANT_LINK_TYPES = ("question", "approval_grant")
# A review's link is the ticket's id (US-012a).
TICKET_LINK_TYPE = "task"

# A person's push switch for one event (K7): theirs, or the workspace's (user_id NULL).
_PUSH_SWITCHES_SQL = text(
    "SELECT user_id, enabled FROM notification_preferences "
    "WHERE workspace_id = CAST(:ws AS uuid) AND event_type = :event_type "
    "  AND destination = :destination AND (user_id IS NULL OR user_id IN :uids)"
).bindparams(bindparam("uids", expanding=True))

COUNT_KEYS = ("recipients", "devices", "sent", "refused", "disabled")

SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class PushEvent:
    """What the dispatcher knows about one notification that may push (plain values only).

    ``notification_id`` is the in-app row's id; ``user_id`` (``users.id``) names the one person
    the event is for, or None for everyone the push type reaches.
    """

    workspace_id: Any
    event_type: str
    notification_id: Optional[str] = None
    link_type: Optional[str] = None
    link_id: Optional[str] = None
    user_id: Optional[int] = None
    agent_name: Optional[str] = None
    severity: Optional[str] = None


@dataclass(frozen=True)
class _Context:
    """What every phone's payload shares: the grant, the ticket's number, redaction."""

    grant: Optional[ApprovalGrant]
    ticket_number: Optional[str]
    redact: bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _workspace_uuid(workspace_id: Any) -> uuid.UUID:
    return workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))


def _edition_kind() -> str:
    """§8: SaaS phones sign in with Clerk; a local install's phones are paired."""
    return (MobileDeviceKind.PAIRED if config.IS_LOCAL_EDITION else MobileDeviceKind.CLERK).value


def _device_filters(workspace_id: Any) -> Tuple[Any, ...]:
    """A phone push can reach in this workspace, whoever holds it."""
    return (
        MobileDevice.workspace_id == _workspace_uuid(workspace_id),
        MobileDevice.kind == _edition_kind(),
        MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
        MobileDevice.push_enabled.is_(True),
        MobileDevice.expo_push_token.isnot(None),
        MobileDevice.push_key.isnot(None),
        MobileDevice.platform.in_(PUSH_PLATFORMS),
    )


def _switched_off(db: Session, event: PushEvent, user_ids: Sequence[int]) -> Set[int]:
    """The people whose push switch for this event is off: their own row, else the workspace's."""
    rows = db.execute(_PUSH_SWITCHES_SQL, {
        "ws": str(event.workspace_id), "event_type": event.event_type,
        "destination": PUSH_DESTINATION, "uids": list(user_ids),
    }).fetchall()
    default_on = all(bool(enabled) for person, enabled in rows if person is None)
    own: Dict[int, bool] = {}
    for person, enabled in rows:
        if person is not None:
            own[int(person)] = own.get(int(person), True) and bool(enabled)
    return {person for person in user_ids if not own.get(person, default_on)}


def eligible_devices(db: Session, event: PushEvent, user_ids: Iterable[int]) -> List[MobileDevice]:
    """The phones of these people in the event's workspace that this event may be pushed to."""
    people = sorted(set(user_ids))
    if push_type_for(event.event_type) is None or not people or not consent.expo_push_allowed():
        return []
    off = _switched_off(db, event, people)
    reachable = [person for person in people if person not in off]
    if not reachable:
        return []
    return (
        db.query(MobileDevice)
        .filter(*_device_filters(event.workspace_id), MobileDevice.user_id.in_(reachable))
        .order_by(MobileDevice.created_at, MobileDevice.id)
        .all()
    )


def _grant(db: Session, event: PushEvent) -> Optional[ApprovalGrant]:
    """The question's (or approval's) grant, in this workspace only, while it still waits.

    One answered, denied, revoked or expired by now is not read: the push goes generic, so a
    phone never offers buttons ``/answer`` would refuse.
    """
    link_id = str(event.link_id or "")
    if event.link_type not in GRANT_LINK_TYPES or not link_id.isdigit():
        return None
    return db.query(ApprovalGrant).filter(
        ApprovalGrant.id == int(link_id),
        ApprovalGrant.workspace_id == _workspace_uuid(event.workspace_id),
        ApprovalGrant.status == GrantStatus.PENDING.value,
    ).first()


def _ticket_number(db: Session, event: PushEvent) -> Optional[str]:
    """The reviewed ticket's number (#0419), in this workspace only."""
    link_id = str(event.link_id or "")
    if event.link_type != TICKET_LINK_TYPE or not link_id.isdigit():
        return None
    workspace_id = _workspace_uuid(event.workspace_id)
    task = db.query(BoardTask).filter(BoardTask.id == int(link_id), BoardTask.workspace_id == workspace_id).first()
    return ticket_numbers(db, workspace_id, [task]).get(task.id) if task is not None else None


def _context(db: Session, event: PushEvent) -> _Context:
    settings = db.query(Workspace.settings).filter(Workspace.id == _workspace_uuid(event.workspace_id)).scalar()
    return _Context(
        grant=_grant(db, event),
        ticket_number=_ticket_number(db, event),
        redact=redact_questions_enabled(settings),
    )


def _message(device: MobileDevice, event: PushEvent, context: _Context, now: datetime) -> PushMessage:
    """One phone's signed message (its own push key)."""
    payload = build_payload(
        device.push_key,
        event_type=event.event_type,
        workspace_id=device.workspace_id,
        notification_id=event.notification_id,
        link_type=event.link_type,
        link_id=event.link_id,
        grant=context.grant,
        ticket_number=context.ticket_number,
        redact=context.redact,
        now=now,
    )
    return to_message(device, payload, event_type=event.event_type, agent_name=event.agent_name)


@dataclass(frozen=True)
class _Sent:
    """One phone a message went to, as plain values: the write after the send reads only these."""

    device_id: uuid.UUID
    workspace_id: uuid.UUID
    token: str


def _turn_off(db: Session, sent: _Sent, now: datetime) -> bool:
    """K15: the phone left; push is off with Expo's reason, the token is kept.

    Read again under a row lock, and only while the row still holds the token that was sent
    to: a phone that registered a new token (or was revoked) during the send is left alone.
    """
    device = (
        db.query(MobileDevice)
        .filter(MobileDevice.id == sent.device_id, MobileDevice.workspace_id == sent.workspace_id,
                MobileDevice.expo_push_token == sent.token)
        .with_for_update()
        .first()
    )
    if device is None:
        return False
    device.push_enabled = False
    device.push_disabled_reason = DEVICE_NOT_REGISTERED
    device.events = with_event(device.events, PUSH_FAILED, now)
    return True


def _record(db: Session, push_type: str, sent: Sequence[_Sent],
            tickets: Sequence[PushTicket], now: datetime) -> Dict[str, int]:
    """Keep every accepted ticket for the receipts job; turn off the phones Expo says left.

    A message the sender gave no ticket for counts as refused.
    """
    if len(tickets) != len(sent):
        logger.error("Push fan-out: %d tickets for %d messages", len(tickets), len(sent))
    answered = list(zip(sent, tickets))
    accepted = [(to, ticket) for to, ticket in answered if ticket.status == TICKET_OK and ticket.ticket_id]
    db.add_all([
        MobilePushTicket(ticket_id=ticket.ticket_id, device_id=to.device_id,
                         workspace_id=to.workspace_id, push_type=push_type)
        for to, ticket in accepted
    ])
    gone = [to for to, ticket in answered if ticket.status != TICKET_OK and ticket.error == DEVICE_NOT_REGISTERED]
    disabled = sum(1 for to in gone if _turn_off(db, to, now))
    return {"sent": len(accepted), "refused": len(sent) - len(accepted), "disabled": disabled}


def _push(db: Session, event: PushEvent, sender: PushSender) -> Dict[str, int]:
    """Resolve, build, send and record one event's push on ``db``; the caller commits.

    The reads end (``rollback``: nothing is written before the send) before the sender is
    called, so no connection or transaction is held while Expo answers.
    """
    push_type = push_type_for(event.event_type)
    if push_type is None:
        return dict.fromkeys(COUNT_KEYS, 0)
    people = recipients(db, event.workspace_id, push_type, event.user_id)
    devices = eligible_devices(db, event, people)
    counts = {**dict.fromkeys(COUNT_KEYS, 0), "recipients": len(people), "devices": len(devices)}
    if not devices:
        return counts
    now = _now()
    context = _context(db, event)
    messages = [_message(device, event, context, now) for device in devices]
    sent = [_Sent(device.id, device.workspace_id, device.expo_push_token) for device in devices]
    db.rollback()
    tickets = sender.send(messages)
    return {**counts, **_record(db, push_type, sent, tickets, now)}


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def _default_sender() -> PushSender:
    from services.push.expo import ExpoPushSender

    return ExpoPushSender()


def deliver(
    event: PushEvent,
    *,
    sender: Optional[PushSender] = None,
    session_factory: Optional[SessionFactory] = None,
) -> Dict[str, int]:
    """Push one event to every phone it may reach, on a session of its own; commits once.

    Never raises: a failure is logged (ids only) and rolled back, and the counts are all 0.
    Otherwise the counts: recipients, devices, sent (tickets kept), refused, disabled
    (DeviceNotRegistered).
    """
    db: Optional[Session] = None
    try:
        db = (session_factory or _open_session)()
        counts = _push(db, event, sender or _default_sender())
        db.commit()
    except Exception:
        logger.exception("Push fan-out failed for workspace %s (%s)", event.workspace_id, event.event_type)
        _end(db, rollback=True)
        return dict.fromkeys(COUNT_KEYS, 0)
    _end(db, rollback=False)
    logger.info("Push fan-out: workspace %s %s notification %s: %s", event.workspace_id,
                event.event_type, event.notification_id, counts)
    return counts


def _end(db: Optional[Session], *, rollback: bool) -> None:
    """Roll back (after a failure) and close; a session that will not close is logged, never raised."""
    if db is None:
        return
    try:
        if rollback:
            db.rollback()
        db.close()
    except Exception:
        logger.exception("Push fan-out: the session did not close cleanly")


@off_loop
def _deliver_off_loop(event: PushEvent) -> None:
    """``deliver`` on the best-effort threads when an event loop runs; inline when none does.

    The dispatcher's ``dispatch`` is a coroutine, so its hand-over always finds a loop.
    """
    deliver(event)


def has_eligible_device(db: Session, workspace_id: Any) -> bool:
    """One EXISTS on the caller's session: may any phone in this workspace be pushed at all?"""
    reachable = db.query(MobileDevice.id).filter(*_device_filters(workspace_id))
    return bool(db.query(reachable.exists()).scalar())


def _worth_handing_over(db: Session, event: PushEvent) -> bool:
    """A push event, the operator's answer on a local install, and one reachable phone.

    The EXISTS runs in a savepoint, so a failed read never aborts the caller's transaction.
    """
    if push_type_for(event.event_type) is None or not consent.expo_push_allowed():
        return False
    with db.begin_nested():
        return has_eligible_device(db, event.workspace_id)


def hand_over(db: Session, event: PushEvent) -> bool:
    """Hand the event's push to ``deliver`` off the event loop, when there is anyone to push to.

    No recipient work here (``deliver`` reads who, at send time). False, with nothing handed
    over, when no phone in the workspace can be pushed; False and logged when the check
    itself fails. Never raises.
    """
    try:
        if not _worth_handing_over(db, event):
            return False
        _deliver_off_loop(event)
    except Exception:
        logger.exception("Push hand-over failed for workspace %s (%s)", event.workspace_id, event.event_type)
        return False
    return True


__all__ = ["PushEvent", "deliver", "eligible_devices", "hand_over", "has_eligible_device"]
