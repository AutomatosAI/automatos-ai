"""PRD-268 §6.3, K7-K9 — a follow-up: Auto's word to check again, kept as one scheduled row.

A follow-up is one ``agent_scheduled_tasks`` row (K7): ``deliver_as`` 'follow_up', a
one-shot whose creator and target are the workspace's Auto, ``for_user_id`` the person
whose chat turn scheduled it, ``origin_chat_id`` that chat, ``description`` the reason, and
``payload`` ``{about}`` when the call named one. ``created_by_user_id`` is NULL: it is no
operator's row. It never goes through ``ScheduledTaskService.create_task``, so
``DELIVERY_MODES`` stays the operator's and an agent's, and none of their caps count it
(K8: every generic reader leaves it out). The person and the chat come from the server
only (``modules.tools.discovery.follow_up_context``).

* ``schedule_follow_up`` takes the workspace's follow-up lock
  (``pg_advisory_xact_lock(namespace, hashtext(workspace_id))``, K9), counts what the caps
  count and inserts the row in that one transaction, so six calls at once cannot pass a
  cap of five. It commits, registers the one-shot as the service registers one (the
  reconcile tick registers it within ``RECONCILE_INTERVAL_SECONDS`` on any other worker),
  then writes the audit row (W1's K15: after the commit). The lock is per workspace, not
  per person: a per-person key could not bound the workspace's 100 a day.
* ``list_follow_ups``: the person's own, pending by default, optionally in one chat.
* ``cancel_follow_up``: the person's own pending one only; another's, or another
  workspace's, is not found. It removes the job and writes the audit row.

Every statement keeps ``deliver_as = 'follow_up'`` a SQL literal, so the partial index
``ix_scheduled_tasks_follow_ups`` serves it. The audit rows never carry the reason, which
comes from a private chat.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.llm.auto_model_sync import AUTO_SLUG_PATTERN
from modules.proactive.follow_up_caps import CapBroken, FollowUpCaps, FollowUpCounts, count_broken

logger = logging.getLogger(__name__)

__all__ = [
    "CANCELLED", "FIRED", "FollowUp", "FollowUpRequest", "NOT_FOUND", "PENDING", "STATUSES",
    "auto_agent_id", "cancel_follow_up", "list_follow_ups", "platform_caps", "schedule_follow_up",
]

# The workspace's follow-up lock, in a key space of its own ('fwup').
FOLLOW_UP_LOCK_NAMESPACE = 0x66777570
ONE_SHOT = "one_shot"
# A follow-up as the person sees it, from its row's status.
PENDING, FIRED, CANCELLED = "pending", "fired", "cancelled"
STATUSES = (PENDING, FIRED, CANCELLED)
# A one-shot that fired is 'completed' (or 'failed', should a run ever mark one so).
_ROW_STATUS = {"active": PENDING, "completed": FIRED, "failed": FIRED, "cancelled": CANCELLED}
NOT_FOUND = "not_found"
DAY = timedelta(hours=24)
LIST_LIMIT = 100
SCHEDULED_ACTION = "proactive:follow_up_scheduled"
CANCELLED_ACTION = "proactive:follow_up_cancelled"
AUDIT_RESOURCE = "follow_up"

_LOCK = text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:workspace_id))")
_AUTO = text("SELECT id FROM agents WHERE workspace_id = CAST(:ws AS uuid) AND is_system_agent IS TRUE "
             "AND slug LIKE :pattern ORDER BY id LIMIT 1")
_COUNTS = text("""
    SELECT
        count(*) FILTER (WHERE for_user_id = :user AND created_by_agent_id = :agent AND status = 'active'),
        count(*) FILTER (WHERE for_user_id = :user AND created_by_agent_id = :agent AND created_at > :since),
        count(*) FILTER (WHERE created_at > :since)
    FROM agent_scheduled_tasks
    WHERE workspace_id = CAST(:ws AS uuid) AND deliver_as = 'follow_up'
""")
_INSERT = text("""
    INSERT INTO agent_scheduled_tasks
        (workspace_id, created_by_agent_id, target_agent_id, task_type, description, schedule,
         next_run_at, origin_chat_id, deliver_as, payload, created_by_user_id, for_user_id)
    VALUES
        (CAST(:ws AS uuid), :agent, :agent, 'one_shot', :reason, :schedule,
         :at, CAST(:chat AS uuid), 'follow_up', CAST(:payload AS jsonb), NULL, :user)
    RETURNING id
""")
_LIST = text("""
    SELECT id, next_run_at, description, status, origin_chat_id
    FROM agent_scheduled_tasks
    WHERE workspace_id = CAST(:ws AS uuid) AND deliver_as = 'follow_up' AND for_user_id = :user
      AND status IN ('active', 'completed', 'failed', 'cancelled')
      AND (CAST(:only_pending AS boolean) IS FALSE OR status = 'active')
      AND (CAST(:chat AS uuid) IS NULL OR origin_chat_id = CAST(:chat AS uuid))
    ORDER BY status <> 'active', CASE WHEN status = 'active' THEN next_run_at END, next_run_at DESC, id
    LIMIT :limit
""")
_CANCEL = text("""
    UPDATE agent_scheduled_tasks SET status = 'cancelled', updated_at = NOW()
    WHERE id = :id AND workspace_id = CAST(:ws AS uuid) AND deliver_as = 'follow_up'
      AND for_user_id = :user AND status = 'active'
    RETURNING id, next_run_at, origin_chat_id
""")
_STATUS = text("""
    SELECT status FROM agent_scheduled_tasks
    WHERE id = :id AND workspace_id = CAST(:ws AS uuid) AND deliver_as = 'follow_up' AND for_user_id = :user
""")


@dataclass(frozen=True)
class FollowUpRequest:
    """One follow-up to schedule, every fact the server's: the workspace, the person, their
    chat, Auto's agent id, the time (UTC), the reason and what it is about."""

    workspace_id: UUID
    user_id: int
    chat_id: UUID
    agent_id: int
    at: datetime
    reason: str
    about: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class FollowUp:
    """A follow-up as its person sees it."""

    id: int
    at: Optional[datetime]
    reason: str
    status: str
    chat_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """``{id, at, reason, status, chat_id}``, the time as UTC ISO."""
        return {"id": self.id, "at": self.at.isoformat() if self.at else None, "reason": self.reason,
                "status": self.status, "chat_id": self.chat_id}


def platform_caps() -> FollowUpCaps:
    """The follow-up caps as ``config`` sets them (K9)."""
    return FollowUpCaps(
        min_minutes=config.PROACTIVE_FOLLOW_UP_MIN_MINUTES,
        max_days=config.PROACTIVE_FOLLOW_UP_MAX_DAYS,
        max_pending=config.PROACTIVE_FOLLOW_UP_MAX_PENDING,
        max_per_day=config.PROACTIVE_FOLLOW_UP_MAX_PER_DAY,
        max_per_workspace_per_day=config.PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY,
        reason_chars=config.PROACTIVE_FOLLOW_UP_REASON_CHARS,
    )


def auto_agent_id(db: Session, workspace_id: UUID) -> Optional[int]:
    """The workspace's Auto: its system agent whose slug is ``auto-…`` (K7), or None."""
    row = db.execute(_AUTO, {"ws": str(workspace_id), "pattern": AUTO_SLUG_PATTERN}).first()
    return int(row[0]) if row is not None else None


def _counts(db: Session, request: FollowUpRequest, now: datetime) -> FollowUpCounts:
    row = db.execute(_COUNTS, {"ws": str(request.workspace_id), "user": request.user_id,
                               "agent": request.agent_id, "since": now - DAY}).one()
    return FollowUpCounts(pending=int(row[0]), today=int(row[1]), workspace_today=int(row[2]))


def _insert(db: Session, request: FollowUpRequest) -> int:
    payload = {"about": dict(request.about)} if request.about else None
    return int(db.execute(_INSERT, {
        "ws": str(request.workspace_id), "agent": request.agent_id, "reason": request.reason,
        "schedule": request.at.isoformat(), "at": request.at, "chat": str(request.chat_id),
        "payload": json.dumps(payload) if payload else None, "user": request.user_id,
    }).scalar_one())


def _service(db: Session, workspace_id: UUID) -> Any:
    from services.scheduled_task_service import ScheduledTaskService

    return ScheduledTaskService(db, workspace_id)


def _audit(db: Session, workspace_id: UUID, user_id: int, action: str, follow_up_id: int,
           details: Dict[str, Any]) -> None:
    """The follow-up's audit row, after its commit (W1's K15); never its reason."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(workspace_id=str(workspace_id), user_id=user_id, action=action, actor_type="agent",
                         resource_type=AUDIT_RESOURCE, resource_id=str(follow_up_id), details=details)


def schedule_follow_up(db: Session, request: FollowUpRequest, caps: FollowUpCaps,
                       now: datetime) -> Union[int, CapBroken]:
    """Schedule ``request`` under the workspace's lock: its id, or the count cap it breaks
    (nothing written, the lock released). The time caps are the caller's to check first."""
    db.execute(_LOCK, {"namespace": FOLLOW_UP_LOCK_NAMESPACE, "workspace_id": str(request.workspace_id)})
    broken = count_broken(caps, _counts(db, request, now))
    if broken is not None:
        db.rollback()
        return broken
    follow_up_id = _insert(db, request)
    db.commit()
    _service(db, request.workspace_id)._register_with_scheduler(
        follow_up_id, ONE_SHOT, request.at.isoformat(), request.agent_id)
    logger.info("[follow-up] %d scheduled for user %s in %s at %s", follow_up_id, request.user_id,
                request.workspace_id, request.at.isoformat())
    _audit(db, request.workspace_id, request.user_id, SCHEDULED_ACTION, follow_up_id,
           {"at": request.at.isoformat(), "agent_id": request.agent_id, "chat_id": str(request.chat_id)})
    return follow_up_id


def _follow_up(row: Any) -> FollowUp:
    return FollowUp(id=int(row.id), at=row.next_run_at, reason=str(row.description or ""),
                    status=_ROW_STATUS[str(row.status)],
                    chat_id=str(row.origin_chat_id) if row.origin_chat_id else None)


def list_follow_ups(db: Session, workspace_id: UUID, user_id: int, *, chat_id: Optional[UUID] = None,
                    only_pending: bool = True) -> List[FollowUp]:
    """The person's own follow-ups in the workspace: pending only by default, soonest first;
    else the pending ones, then those that fired or were cancelled, newest first; in one chat
    when ``chat_id`` names it."""
    rows = db.execute(_LIST, {"ws": str(workspace_id), "user": user_id, "only_pending": only_pending,
                              "chat": str(chat_id) if chat_id else None, "limit": LIST_LIMIT}).fetchall()
    return [_follow_up(row) for row in rows]


def _not_cancelled(db: Session, workspace_id: UUID, user_id: int, follow_up_id: int) -> str:
    """Why the person's follow-up was not cancelled: it already fired or was cancelled, or it
    is not theirs (or not there) at all."""
    row = db.execute(_STATUS, {"id": follow_up_id, "ws": str(workspace_id), "user": user_id}).first()
    status = _ROW_STATUS.get(str(row[0])) if row is not None else None
    return status if status in (FIRED, CANCELLED) else NOT_FOUND


def cancel_follow_up(db: Session, workspace_id: UUID, user_id: int,
                     follow_up_id: int) -> Tuple[str, Optional[FollowUp]]:
    """Cancel the person's own pending follow-up: ``(CANCELLED, it)``. Otherwise nothing
    changes and the first item says why: ``NOT_FOUND`` (another person's, another
    workspace's, or none), ``FIRED`` or ``CANCELLED`` (already)."""
    row = db.execute(_CANCEL, {"id": follow_up_id, "ws": str(workspace_id), "user": user_id}).first()
    if row is None:
        db.rollback()
        return _not_cancelled(db, workspace_id, user_id, follow_up_id), None
    db.commit()
    _service(db, workspace_id)._reschedule(follow_up_id, "cancelled")
    logger.info("[follow-up] %d cancelled by user %s in %s", follow_up_id, user_id, workspace_id)
    _audit(db, workspace_id, user_id, CANCELLED_ACTION, follow_up_id,
           {"at": row.next_run_at.isoformat() if row.next_run_at else None})
    return CANCELLED, FollowUp(id=follow_up_id, at=row.next_run_at, reason="", status=CANCELLED,
                               chat_id=str(row.origin_chat_id) if row.origin_chat_id else None)
