"""Find or create the ``users`` row behind a Clerk user, never by moving someone else's.

``users`` rows are found by ``clerk_user_id``. ``users.email`` is unique, and so is
``users.username`` (provisioning sets it to the email). Provisioning used to insert
with ``ON CONFLICT (clerk_user_id) DO NOTHING``, swallow the email
``IntegrityError``, read the row back by the new Clerk ID, find nothing and crash
on ``uid_row[0]``: a 500 for anyone whose email already had a row under another
Clerk ID (every returning user after the Clerk production cutover).

Sign-in never moves a row from one Clerk ID to another. A row is found by Clerk ID
or a new one is inserted; when the email or username is already held by another
row, the caller gets a clear 409, never a 500. Returning users are moved to their
production Clerk IDs once, by the cutover's operator step
(``scripts/clerk_cutover_relink.py``, ``core.auth.clerk_cutover``), which matches on
the import's own ID link and a verified email and records every move.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text

logger = logging.getLogger(__name__)

# The email a users row gets when the Clerk token carried none.
PENDING_EMAIL_TEMPLATE = "{clerk_user_id}@pending"

LINK_CONFLICT_MESSAGE = (
    "This sign-in matches an existing account that is not linked to it. "
    "Contact Automatos support."
)


def user_id_for_clerk_id(db, clerk_user_id: str) -> Optional[int]:
    """The ``users.id`` linked to this Clerk user ID, or None."""
    row = db.execute(
        text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
        {"cid": clerk_user_id},
    ).fetchone()
    return row[0] if row else None


def _row_with_email(db, email: str):
    """``(id, clerk_user_id)`` of the oldest row with this email, case-insensitive, or None."""
    return db.execute(
        text(
            "SELECT id, clerk_user_id FROM users "
            "WHERE LOWER(email) = LOWER(:email) ORDER BY id LIMIT 1"
        ),
        {"email": email},
    ).fetchone()


def _link_conflict(clerk_user_id: str, reason: str) -> HTTPException:
    logger.error("No users row for clerk_user_id=%s: %s", clerk_user_id, reason)
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=LINK_CONFLICT_MESSAGE)


def ensure_user_row(
    db, clerk_user_id: str, email: Optional[str] = None, name: Optional[str] = None,
) -> int:
    """The ``users.id`` for this Clerk user: found by Clerk ID, or inserted.

    A row that already has the email (case-insensitive: the unique index is not,
    so ``Jane@Example.com`` would otherwise get a second row) belongs to someone
    else, or to a returning user the cutover step has not relinked yet: the caller
    gets a 409 with a clear message, never a 500, and no row changes. Idempotent,
    and safe when two first requests race: the insert skips on ANY unique conflict
    (Clerk ID, email or username) and the row is read back by Clerk ID; nothing
    found then (another row holds the username) is the same 409.
    """
    existing = user_id_for_clerk_id(db, clerk_user_id)
    if existing is not None:
        return existing
    if email:
        holder = _row_with_email(db, email)
        if holder:
            raise _link_conflict(clerk_user_id, f"users.id={holder[0]} already has its email")
    db.execute(
        text(
            "INSERT INTO users (username, email, clerk_user_id, name, is_active) "
            "VALUES (:username, :email, :cid, :name, true) "
            "ON CONFLICT DO NOTHING"
        ),
        {
            "username": email or clerk_user_id,
            "email": email or PENDING_EMAIL_TEMPLATE.format(clerk_user_id=clerk_user_id),
            "cid": clerk_user_id,
            "name": name,
        },
    )
    created = user_id_for_clerk_id(db, clerk_user_id)
    if created is not None:
        return created
    raise _link_conflict(clerk_user_id, "another row holds its email or username")
