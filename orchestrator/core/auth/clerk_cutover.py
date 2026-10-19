"""The Clerk production cutover's one-time relink of returning users (19 Oct 2026).

Moving Clerk from its development instance to production gives every returning
person a new Clerk user ID. ``users`` rows are found by ``clerk_user_id``, and
sign-in never moves a row (``core.auth.user_link``), so each returning row is moved
to its production ID here, once, by the operator step
``scripts/clerk_cutover_relink.py``. The rules:

- **Old-instance IDs only.** A row moves only from the Clerk ID it holds now to
  the production user whose ``external_id`` is that ID. clerk/migration-tool stores
  each development user's ``id`` as the imported user's ``external_id``, so the
  match is the import's own link, not an email.
- **Verified email only.** The row's email must also be one of that production
  user's verified addresses (case-insensitive).
- **At most once.** ``clerk_user_relinks.user_id`` is unique: a row is relinked
  once, to one production identity. A rollback moves it back and stamps
  ``reverted_at``; a retried cutover re-applies only that same pair.
- **Audited.** Every move writes or updates its ``clerk_user_relinks`` row in the
  same transaction, and logs both Clerk IDs at WARNING.

Each move is a compare-and-set on the Clerk ID the plan saw. Nothing here commits:
the script commits a whole run as one transaction, or nothing.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Clerk's verification status for a verified email address.
VERIFIED = "verified"

# Forward outcomes. Only RELINK and REAPPLY write.
RELINK = "relink"                    # move the row and record the move
REAPPLY = "reapply"                  # after a rollback: re-apply the recorded pair
ALREADY_LINKED = "already_linked"    # a row already holds the production ID; nothing to move
NOT_IMPORTED = "not_imported"        # no external_id: the person is new on production
NO_ROW = "no_row"                    # no users row holds the old ID
EMAIL_NOT_VERIFIED = "email_not_verified"  # the row's email is not a verified address of the user
TARGET_TAKEN = "target_taken"        # another users row already holds the production ID
RELINKED_BEFORE = "relinked_before"  # the row's one relink is recorded, to another pair
DUPLICATE_EXTERNAL_ID = "duplicate_external_id"  # two production users name the same old ID

# Rollback outcomes. Only REVERT writes.
REVERT = "revert"                    # move the row back to its old ID
ALREADY_REVERTED = "already_reverted"  # the row already holds its old ID
CHANGED_SINCE = "changed_since"      # the row holds neither ID: left for a person to look at

WRITING_OUTCOMES = frozenset({RELINK, REAPPLY, REVERT})


class RelinkConflict(RuntimeError):
    """A row changed between the plan and the write: the run must roll back."""


@dataclass(frozen=True)
class ProductionUser:
    """The fields of a production Clerk user that the relink reads."""

    clerk_user_id: str
    external_id: Optional[str]
    verified_emails: frozenset


@dataclass(frozen=True)
class AppUser:
    """A ``users`` row: its id, Clerk ID and email."""

    id: int
    clerk_user_id: Optional[str]
    email: str


@dataclass(frozen=True)
class RelinkRecord:
    """A row's ``clerk_user_relinks`` record."""

    user_id: int
    old_clerk_user_id: str
    new_clerk_user_id: str
    reverted: bool


@dataclass(frozen=True)
class Decision:
    """What the run does, or why it does nothing, for one production user or record."""

    outcome: str
    new_clerk_user_id: str
    old_clerk_user_id: Optional[str] = None
    user_id: Optional[int] = None
    verified_email: Optional[str] = None


def _verified_emails(payload: dict) -> frozenset:
    """The user's verified email addresses, lowercased."""
    verified = set()
    for entry in payload.get("email_addresses") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("email_address"), str):
            continue
        if (entry.get("verification") or {}).get("status") == VERIFIED:
            verified.add(entry["email_address"].strip().lower())
    return frozenset(verified)


def production_user(payload: dict) -> ProductionUser:
    """Read one Clerk Backend API user object; ValueError when it has no id."""
    clerk_user_id = payload.get("id")
    if not isinstance(clerk_user_id, str) or not clerk_user_id:
        raise ValueError("A Clerk user in the list has no id")
    external_id = payload.get("external_id")
    if not isinstance(external_id, str) or not external_id:
        external_id = None
    return ProductionUser(clerk_user_id, external_id, _verified_emails(payload))


def _decide_for_row(prod: ProductionUser, old_row: AppUser, record: Optional[RelinkRecord], pair: dict) -> Decision:
    """The forward outcome once a row holds the old ID and no row holds the new one."""
    email = old_row.email.strip().lower()
    if email not in prod.verified_emails:
        return Decision(EMAIL_NOT_VERIFIED, user_id=old_row.id, **pair)
    if record is None:
        return Decision(RELINK, user_id=old_row.id, verified_email=email, **pair)
    same_pair = (record.old_clerk_user_id, record.new_clerk_user_id) == (prod.external_id, prod.clerk_user_id)
    if same_pair and record.reverted:
        return Decision(REAPPLY, user_id=old_row.id, verified_email=email, **pair)
    return Decision(RELINKED_BEFORE, user_id=old_row.id, **pair)


def decide(
    prod: ProductionUser,
    old_row: Optional[AppUser],
    new_row: Optional[AppUser],
    record: Optional[RelinkRecord],
) -> Decision:
    """The forward outcome for one production user.

    ``old_row`` holds the user's ``external_id`` now, ``new_row`` holds its
    production ID, and ``record`` is ``old_row``'s relink record.
    """
    pair = {"new_clerk_user_id": prod.clerk_user_id, "old_clerk_user_id": prod.external_id}
    if prod.external_id is None:
        return Decision(NOT_IMPORTED, **pair)
    if old_row is None and new_row is None:
        return Decision(NO_ROW, **pair)
    if old_row is None:
        return Decision(ALREADY_LINKED, user_id=new_row.id, **pair)
    if new_row is not None:
        return Decision(TARGET_TAKEN, user_id=old_row.id, **pair)
    return _decide_for_row(prod, old_row, record, pair)


def _app_user(db, clerk_user_id: str) -> Optional[AppUser]:
    row = db.execute(
        text("SELECT id, clerk_user_id, email FROM users WHERE clerk_user_id = :cid"),
        {"cid": clerk_user_id},
    ).fetchone()
    return AppUser(id=row[0], clerk_user_id=row[1], email=row[2]) if row else None


def _record(db, user_id: int) -> Optional[RelinkRecord]:
    row = db.execute(
        text(
            "SELECT user_id, old_clerk_user_id, new_clerk_user_id, reverted_at IS NOT NULL "
            "FROM clerk_user_relinks WHERE user_id = :uid"
        ),
        {"uid": user_id},
    ).fetchone()
    return RelinkRecord(row[0], row[1], row[2], bool(row[3])) if row else None


def plan_relinks(db, prod_users: Sequence[ProductionUser]) -> List[Decision]:
    """One forward decision per production user. Reads only."""
    claims = Counter(p.external_id for p in prod_users if p.external_id)
    decisions = []
    for prod in prod_users:
        if prod.external_id and claims[prod.external_id] > 1:
            decisions.append(Decision(
                DUPLICATE_EXTERNAL_ID, new_clerk_user_id=prod.clerk_user_id, old_clerk_user_id=prod.external_id,
            ))
            continue
        old_row = _app_user(db, prod.external_id) if prod.external_id else None
        new_row = _app_user(db, prod.clerk_user_id)
        record = _record(db, old_row.id) if old_row else None
        decisions.append(decide(prod, old_row, new_row, record))
    return decisions


def _move(db, user_id: int, from_id: str, to_id: str) -> None:
    """Compare-and-set ``users.clerk_user_id``; RelinkConflict when the row changed."""
    moved = db.execute(
        text("UPDATE users SET clerk_user_id = :to_id WHERE id = :uid AND clerk_user_id = :from_id"),
        {"to_id": to_id, "uid": user_id, "from_id": from_id},
    )
    if moved.rowcount != 1:
        raise RelinkConflict(f"users.id={user_id} no longer holds clerk_user_id={from_id}")


def _record_relink(db, decision: Decision) -> None:
    params = {
        "uid": decision.user_id,
        "old": decision.old_clerk_user_id,
        "new": decision.new_clerk_user_id,
        "email": decision.verified_email,
    }
    if decision.outcome == RELINK:
        db.execute(
            text(
                "INSERT INTO clerk_user_relinks (user_id, old_clerk_user_id, new_clerk_user_id, verified_email) "
                "VALUES (:uid, :old, :new, :email)"
            ),
            params,
        )
        return
    updated = db.execute(
        text(
            "UPDATE clerk_user_relinks SET reverted_at = NULL, relinked_at = now(), verified_email = :email "
            "WHERE user_id = :uid AND old_clerk_user_id = :old AND new_clerk_user_id = :new "
            "AND reverted_at IS NOT NULL"
        ),
        params,
    )
    if updated.rowcount != 1:
        raise RelinkConflict(f"users.id={decision.user_id}: its reverted relink record changed")


def apply_relinks(db, decisions: Iterable[Decision]) -> int:
    """Move every RELINK and REAPPLY row and record it. Not committed. Returns the count."""
    moved = 0
    for decision in decisions:
        if decision.outcome not in (RELINK, REAPPLY):
            continue
        _move(db, decision.user_id, decision.old_clerk_user_id, decision.new_clerk_user_id)
        _record_relink(db, decision)
        logger.warning(
            "Clerk cutover (%s): users.id=%s from clerk_user_id=%s to %s (external_id match, verified email)",
            decision.outcome, decision.user_id, decision.old_clerk_user_id, decision.new_clerk_user_id,
        )
        moved += 1
    return moved


def plan_rollback(db) -> List[Decision]:
    """One rollback decision per relink not yet reverted. Reads only; needs no Clerk call."""
    rows = db.execute(
        text(
            "SELECT r.user_id, r.old_clerk_user_id, r.new_clerk_user_id, u.clerk_user_id "
            "FROM clerk_user_relinks r JOIN users u ON u.id = r.user_id "
            "WHERE r.reverted_at IS NULL ORDER BY r.user_id"
        )
    ).fetchall()
    decisions = []
    for user_id, old_id, new_id, current in rows:
        outcome = REVERT if current == new_id else ALREADY_REVERTED if current == old_id else CHANGED_SINCE
        decisions.append(Decision(outcome, new_clerk_user_id=new_id, old_clerk_user_id=old_id, user_id=user_id))
    return decisions


def apply_rollback(db, decisions: Iterable[Decision]) -> int:
    """Move every REVERT row back to its old Clerk ID and stamp its record. Not committed."""
    moved = 0
    for decision in decisions:
        if decision.outcome != REVERT:
            continue
        _move(db, decision.user_id, decision.new_clerk_user_id, decision.old_clerk_user_id)
        db.execute(
            text("UPDATE clerk_user_relinks SET reverted_at = now() WHERE user_id = :uid AND reverted_at IS NULL"),
            {"uid": decision.user_id},
        )
        logger.warning(
            "Clerk cutover rollback: users.id=%s from clerk_user_id=%s back to %s",
            decision.user_id, decision.new_clerk_user_id, decision.old_clerk_user_id,
        )
        moved += 1
    return moved


def summarize(decisions: Iterable[Decision]) -> Counter:
    """How many decisions have each outcome."""
    return Counter(d.outcome for d in decisions)
