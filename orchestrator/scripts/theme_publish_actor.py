"""PRD-261 (D5) — create or find the theme publish key's service user, and print its ``users.id``.

``POST /api/marketplace/themes/publish`` acts as one ``users`` row, ``THEME_PUBLISH_ACTOR``: the
Design Kit's publish job is not a person, so it gets a row of its own, never a person's. Run it
once on the server, then set the printed id as ``THEME_PUBLISH_ACTOR``::

    docker compose exec backend python -m scripts.theme_publish_actor   # local edition
    python -m scripts.theme_publish_actor                               # hosted: in the backend service's shell

Idempotent: the row is found by its email (``SERVICE_EMAIL``, a reserved ``.invalid`` address no
mailbox can verify), else inserted, and the same id is printed every time. The row has no Clerk id,
so no sign-in resolves to it (``core.auth.user_link.ensure_user_row`` never attaches a row with this
email to a Clerk user either), no workspace membership, and no system role: ``users`` stores
none, so it is a platform admin nowhere. What it may do is the publish endpoint's alone: list,
approve and sign versions under ``THEME_PUBLISH_PUBLISHERS``.

Exit 0 with the id; 2 when a row with the email exists but is linked to a Clerk user (someone
signed in with it: nothing is changed, and the endpoint must not act as that person); 1 when
the database fails, printed by the error's type only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Optional, Sequence, TextIO

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
SERVICE_USERNAME = "theme-publisher"
SERVICE_EMAIL = "theme-publisher@automatos.invalid"
SERVICE_NAME = "Theme publisher (Design Kit publish job)"

_FIND = text("SELECT id, clerk_user_id FROM users WHERE LOWER(email) = LOWER(:email) ORDER BY id LIMIT 1")
_INSERT = text(
    "INSERT INTO users (username, email, name, is_active) "
    "VALUES (:username, :email, :name, true) ON CONFLICT DO NOTHING"
)

REPORT = (
    "Theme publish service user: users.id {user_pk} ({state}).\n"
    "Set THEME_PUBLISH_ACTOR={user_pk} on the backend, next to THEME_PUBLISH_KEY_SHA256 and "
    "THEME_PUBLISH_PUBLISHERS."
)
LINKED = (
    "A users row with {email} exists and is linked to a Clerk user (users.id {user_pk}), so it is a "
    "person's sign-in, not a service user. Nothing was changed; do not use it as THEME_PUBLISH_ACTOR."
)
NOT_CREATED = "The service user could not be created: another users row holds its username. Nothing was written."
DATABASE_FAILED = "The database refused the service user ({kind}). Nothing was written."


class ActorLinked(RuntimeError):
    """The email's row belongs to a Clerk sign-in, so it is not the service user."""


def _found(db: Session) -> Optional[tuple]:
    return db.execute(_FIND, {"email": SERVICE_EMAIL}).fetchone()


def ensure_publish_actor(db: Session) -> tuple:
    """``(users.id, created)`` of the service user: found by its email, else inserted and committed.

    Raises :class:`ActorLinked` when the email's row has a Clerk id, and ``LookupError`` when the
    insert was skipped because another row holds the username.
    """
    row = _found(db)
    if row is None:
        db.execute(_INSERT, {"username": SERVICE_USERNAME, "email": SERVICE_EMAIL, "name": SERVICE_NAME})
        db.commit()
        created = _found(db)
        if created is None:
            raise LookupError(NOT_CREATED)
        return int(created[0]), True
    if row[1]:
        raise ActorLinked(LINKED.format(email=SERVICE_EMAIL, user_pk=row[0]))
    return int(row[0]), False


def _default_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def main(argv: Optional[Sequence[str]] = None, session_factory: Callable[[], Session] = _default_session,
         out: TextIO = sys.stdout) -> int:
    """Create or find the service user and print the id to set as ``THEME_PUBLISH_ACTOR``."""
    argparse.ArgumentParser(description="Create or find the theme publish key's service user").parse_args(argv)
    db = session_factory()
    try:
        user_pk, created = ensure_publish_actor(db)
    except ActorLinked as linked:
        print(str(linked), file=out)
        return EXIT_REFUSED
    except LookupError as missing:
        print(str(missing), file=out)
        return EXIT_FAILED
    except SQLAlchemyError as failed:
        db.rollback()
        print(DATABASE_FAILED.format(kind=type(failed).__name__), file=out)
        return EXIT_FAILED
    finally:
        db.close()
    print(REPORT.format(user_pk=user_pk, state="created" if created else "already there"), file=out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
