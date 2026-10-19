"""Find, relink or create the ``users`` row behind a Clerk user.

``users`` rows are found by ``clerk_user_id``. Moving Clerk from its development
instance to production gives every returning person a NEW Clerk user ID with the
SAME email. ``users.email`` is unique, and so is ``users.username`` (provisioning
sets it to the email), so inserting a second row for that person cannot succeed.
Provisioning used to insert with ``ON CONFLICT (clerk_user_id) DO NOTHING``,
swallow the email ``IntegrityError``, read the row back by the new Clerk ID, find
nothing and crash on ``uid_row[0]``: every returning user got a 500.

Here the row is found by Clerk ID first, then by email (case-insensitive, the way
pending invitations are matched) and moved to the new Clerk ID. Only a person with
no row at all gets a new one. The email is the one in the signature-verified Clerk
session token (``ClerkAuth.extract_user_info``), the same address invitation
acceptance (``api/team.py``) already matches users on.
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
    "This sign-in matches an existing account that could not be linked automatically. "
    "Contact Automatos support."
)


def user_id_for_clerk_id(db, clerk_user_id: str) -> Optional[int]:
    """The ``users.id`` linked to this Clerk user ID, or None."""
    row = db.execute(
        text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
        {"cid": clerk_user_id},
    ).fetchone()
    return row[0] if row else None


def relink_user_by_email(db, clerk_user_id: str, email: Optional[str]) -> Optional[int]:
    """Move the ``users`` row with this email to ``clerk_user_id`` and return its id.

    None when there is no email or no row has it. Not committed: the caller commits
    with the rest of its work. The previous Clerk ID is logged, so every relink can
    be traced.
    """
    if not email:
        return None
    row = db.execute(
        text(
            "SELECT id, clerk_user_id FROM users "
            "WHERE LOWER(email) = LOWER(:email) ORDER BY id LIMIT 1"
        ),
        {"email": email},
    ).fetchone()
    if not row:
        return None
    user_id, previous_clerk_id = row[0], row[1]
    if previous_clerk_id == clerk_user_id:
        return user_id
    db.execute(
        text("UPDATE users SET clerk_user_id = :cid WHERE id = :id"),
        {"cid": clerk_user_id, "id": user_id},
    )
    logger.warning(
        "Relinked users.id=%s from clerk_user_id=%s to %s (same email, new Clerk user ID)",
        user_id, previous_clerk_id, clerk_user_id,
    )
    return user_id


def ensure_user_row(
    db, clerk_user_id: str, email: Optional[str] = None, name: Optional[str] = None,
) -> int:
    """The ``users.id`` for this Clerk user: found by Clerk ID, relinked by email, or inserted.

    Idempotent, and safe when two first requests race: the insert skips on ANY unique
    conflict (Clerk ID, email or username) and the row is read back by Clerk ID. When
    that still finds nothing (another row holds the username), the caller gets a 409
    with a clear message, never a 500.
    """
    existing = user_id_for_clerk_id(db, clerk_user_id)
    if existing is not None:
        return existing
    relinked = relink_user_by_email(db, clerk_user_id, email)
    if relinked is not None:
        return relinked
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
    logger.error(
        "No users row for clerk_user_id=%s after the insert: another row holds its username",
        clerk_user_id,
    )
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=LINK_CONFLICT_MESSAGE)
