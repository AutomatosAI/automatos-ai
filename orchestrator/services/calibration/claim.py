"""PRD-268 §6.11.5, K15, K16 — the person's calibration record: offered once, claimed first.

``proactive_calibrations`` holds one row per (workspace, person), and its unique
``uq_proactive_calibrations_person`` is the offered-once claim: ``INSERT … ON CONFLICT DO
NOTHING``, committed before anything is read (the brief's claim, ``services.mobile.digest.lease``).
Whoever inserts the row holds the offer; every other claimant, at once or later, gets nothing
and answers the record already there.

The claim of an offer the person agreed to is a lease on its ``consented_at`` (the brief's
``posted_at``, P268W2-RVW-1): a claimant that died between its claim and its post (a restart
during the model call) leaves an ``open`` row with no message, which ``take_over`` hands to the
next consent once it is ``CLAIM_STALE`` old, so the person is not refused for good. ``hold``
keeps the built calibration only while the caller still holds the lease, and stamps it anew, so
a slow claimant whose claim was taken over posts nothing (``LeaseLost``); ``release`` gives back
only a lease the caller still holds.

* ``claim``: ``open`` with ``consented_at`` (consent: the build follows), or ``declined`` with
  ``decided_at`` (no build and no model call); None when the person was offered already.
* ``abandoned``, ``take_over``, ``hold``, ``release``: the lease, as above. An abandoned claim is
  offered again (GET's ``offer``); a released record goes while no message carries it, so the
  person is offered again.
* ``record_of`` and ``own_record``: the person's own record in the workspace, by person or by id.
  Anyone else's, and another workspace's, is None (the routes answer 404).
* ``posted``: the chat and the message the calibration was posted as.
* ``mark_rejected`` / ``unmark_rejected`` and ``mark_answered`` / ``unmark_answered``: an
  answer's state change on the person's own row, under its lock, committed before its memory
  write (so no transaction stays open across the write), and given back when the write fails
  (K16: the line's state does not change). A line is rejected, and the calibration answered,
  only while it is ``open``; a line already rejected, or a calibration already answered, is
  left as it is.
* ``view``: the record as the routes answer it.

Every function that writes commits. Every read loads the row as it is now, never as the session
last saw it. ``ProactiveCalibration`` is the row (``core.models.proactive``).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.models.proactive import CALIBRATION_ANSWERED, CALIBRATION_DECLINED, CALIBRATION_OPEN, ProactiveCalibration

__all__ = [
    "CLAIM_STALE", "Lease", "LeaseLost", "abandoned", "claim", "hold", "mark_answered", "mark_rejected", "own_record",
    "posted", "record_of", "release", "take_over", "unmark_answered", "unmark_rejected", "view",
]

# The unique the claim conflicts on: one record per person per workspace.
CLAIM_KEY = ["workspace_id", "user_id"]
# An offer that has not posted in this long died with its claimant: its build is one model call,
# which answers or fails well within it (the brief's BRIEF_CLAIM_STALE_SECONDS default).
CLAIM_STALE = timedelta(minutes=10)


class LeaseLost(Exception):
    """The claim was taken over by a later consent: this claimant posts nothing."""


@dataclass(frozen=True)
class Lease:
    """A claim this caller holds: the record and the ``consented_at`` it was stamped with."""

    id: uuid.UUID
    at: datetime


def _uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def claim(db: Session, workspace_id: Any, user_id: int, *, consent: bool, now: datetime) -> Optional[Lease]:
    """Claim the person's calibration in the workspace, committed: the lease on the new record,
    or None when they were offered it already (consented, declined or answered)."""
    values = {
        "id": uuid.uuid4(), "workspace_id": _uuid(workspace_id), "user_id": int(user_id),
        "status": CALIBRATION_OPEN if consent else CALIBRATION_DECLINED,
        "consented_at": now if consent else None, "decided_at": None if consent else now,
    }
    statement = (
        pg_insert(ProactiveCalibration).values(**values)
        .on_conflict_do_nothing(index_elements=CLAIM_KEY)
        .returning(ProactiveCalibration.id)
    )
    claimed = db.execute(statement).scalar()
    db.commit()
    return Lease(claimed, now) if claimed is not None else None


def _unposted(calibration_id: Any, at: datetime) -> tuple:
    return (ProactiveCalibration.id == _uuid(calibration_id), ProactiveCalibration.message_id.is_(None),
            ProactiveCalibration.consented_at == at)


def abandoned(record: ProactiveCalibration, now: datetime) -> bool:
    """The record is a claim whose claimant died before its post: ``open``, with no message, and
    ``CLAIM_STALE`` old at ``now`` (``take_over`` takes exactly these)."""
    return (record.status == CALIBRATION_OPEN and record.message_id is None and record.consented_at is not None
            and record.consented_at < now - CLAIM_STALE)


def take_over(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, now: datetime) -> Optional[Lease]:
    """Take over the person's ``open`` claim that never posted once it is ``CLAIM_STALE`` old,
    committed: one compare-and-set, so one caller wins; None for every other, and while fresh."""
    taken = (
        db.query(ProactiveCalibration)
        .filter(
            ProactiveCalibration.id == _uuid(calibration_id),
            ProactiveCalibration.workspace_id == _uuid(workspace_id),
            ProactiveCalibration.user_id == int(user_id),
            ProactiveCalibration.status == CALIBRATION_OPEN,
            ProactiveCalibration.message_id.is_(None),
            ProactiveCalibration.consented_at < now - CLAIM_STALE,
        )
        .update({ProactiveCalibration.consented_at: now}, synchronize_session=False)
    )
    db.commit()
    return Lease(_uuid(calibration_id), now) if taken == 1 else None


def hold(db: Session, lease: Lease, calibration: Mapping[str, Any], now: datetime) -> Lease:
    """Keep the built ``Calibration@1`` on its record while ``lease`` is still held, stamped
    ``now`` and committed: the lease now held. Raises ``LeaseLost`` when it was taken over."""
    kept = db.query(ProactiveCalibration).filter(*_unposted(lease.id, lease.at)).update(
        {ProactiveCalibration.calibration: dict(calibration), ProactiveCalibration.consented_at: now},
        synchronize_session=False)
    db.commit()
    if kept != 1:
        raise LeaseLost("the calibration's claim was taken over")
    return Lease(lease.id, now)


def release(db: Session, lease: Lease) -> None:
    """Give the claim back (its build or post failed), committed: the record goes while the
    caller still holds it and no message carries it, so the person is offered again."""
    db.query(ProactiveCalibration).filter(*_unposted(lease.id, lease.at)).delete(synchronize_session=False)
    db.commit()


def record_of(db: Session, workspace_id: Any, user_id: int) -> Optional[ProactiveCalibration]:
    """The person's record in the workspace; None before they were offered calibration."""
    return (
        db.query(ProactiveCalibration)
        .filter(ProactiveCalibration.workspace_id == _uuid(workspace_id), ProactiveCalibration.user_id == int(user_id))
        .populate_existing()
        .first()
    )


def _own(workspace_id: Any, user_id: int, calibration_id: Any) -> tuple:
    return (ProactiveCalibration.id == _uuid(calibration_id), ProactiveCalibration.workspace_id == _uuid(workspace_id),
            ProactiveCalibration.user_id == int(user_id))


def own_record(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> Optional[ProactiveCalibration]:
    """The record ``calibration_id`` while it is the person's own in the workspace; None otherwise."""
    return db.query(ProactiveCalibration).filter(*_own(workspace_id, user_id, calibration_id)).populate_existing().first()


def _locked(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> Optional[ProactiveCalibration]:
    return (
        db.query(ProactiveCalibration)
        .filter(*_own(workspace_id, user_id, calibration_id))
        .with_for_update()
        .populate_existing()
        .one_or_none()
    )


def posted(db: Session, calibration_id: Any, chat_id: Any, message_id: Any) -> None:
    """Record the chat and the message the calibration was posted as, committed."""
    db.query(ProactiveCalibration).filter(ProactiveCalibration.id == _uuid(calibration_id)).update(
        {ProactiveCalibration.chat_id: _uuid(chat_id), ProactiveCalibration.message_id: _uuid(message_id)},
        synchronize_session=False)
    db.commit()


def mark_rejected(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, line_id: str) -> bool:
    """Reject ``line_id`` under the row's lock, committed: True when this call rejected it; False
    (nothing changed) when it was rejected already or the calibration is no longer open."""
    row = _locked(db, workspace_id, user_id, calibration_id)
    if row is None or row.status != CALIBRATION_OPEN or line_id in (row.rejected_lines or []):
        db.rollback()
        return False
    row.rejected_lines = [*(row.rejected_lines or []), line_id]
    db.commit()
    return True


def unmark_rejected(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, line_id: str) -> None:
    """Give a rejection back (its correction was not written), committed."""
    row = _locked(db, workspace_id, user_id, calibration_id)
    if row is not None:
        row.rejected_lines = [kept for kept in (row.rejected_lines or []) if kept != line_id]
    db.commit()


def mark_answered(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, line_ids: Sequence[str],
                  now: datetime) -> Optional[List[str]]:
    """Answer the calibration under the row's lock, committed: every line of ``line_ids`` not
    rejected is accepted, and those ids are returned; None (nothing changed) when it is no
    longer open."""
    row = _locked(db, workspace_id, user_id, calibration_id)
    if row is None or row.status != CALIBRATION_OPEN:
        db.rollback()
        return None
    rejected = set(row.rejected_lines or [])
    accepted = [line_id for line_id in line_ids if line_id not in rejected]
    row.status, row.accepted_lines, row.decided_at = CALIBRATION_ANSWERED, accepted, now
    db.commit()
    return accepted


def unmark_answered(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> None:
    """Give an answer back (a memory was not written): open again, nothing accepted, committed."""
    row = _locked(db, workspace_id, user_id, calibration_id)
    if row is not None:
        row.status, row.accepted_lines, row.decided_at = CALIBRATION_OPEN, [], None
    db.commit()


def view(record: ProactiveCalibration) -> Dict[str, Any]:
    """The record as the routes answer it: its status, its ``Calibration@1`` (None until it is
    built, and for a declined one), the lines rejected and accepted, and where it was posted."""
    saved = record.to_dict()
    return {
        "status": saved["status"],
        "calibration": saved["calibration"] or None,
        "rejected_lines": saved["rejected_lines"],
        "accepted_lines": saved["accepted_lines"],
        "chat_id": saved["chat_id"],
        "message_id": saved["message_id"],
    }
