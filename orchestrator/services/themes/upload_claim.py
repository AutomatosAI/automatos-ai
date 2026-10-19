"""PRD-261 C-02a — one upload (or Marketplace submission) of the same bytes at a time.

The review stage's LLM calls run before the package row exists, so the row's unique index can't
stop two overlapping uploads of identical bytes (a timed-out request sent again) from both paying
for the stage. The first upload claims (scope, digest), where the scope is the workspace for a
private upload and ``marketplace`` for a submission, until its block exits:

* a session-level Postgres advisory lock (``pg_try_advisory_lock``), across every worker process,
  on one dedicated connection kept for the block (the ``harness_runs.workspace_lock`` pattern).
  The statement that takes it commits at once, so that connection waits idle, never idle in
  transaction, while the stage runs; the route ends its own read transaction before the stage, so
  the request's session holds no connection meanwhile. The lock is released on the way out, or
  its connection dropped when the release fails;
* a claim in this process, for a database without advisory locks (SQLite, in the unit tests).

A second upload of the same bytes is refused at once (409, try again) rather than holding a
threadpool thread for minutes; once the first commits, the retry finds its row and pays nothing.

Synchronous (a plain SQLAlchemy session): called from plain-``def`` routes only.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Set

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The upload claim's key space in the database ('thmu'), apart from every other advisory lock.
UPLOAD_LOCK_NAMESPACE = 0x74686D75
_TRY_LOCK = text("SELECT pg_try_advisory_lock(:namespace, hashtext(:key))")
_UNLOCK = text("SELECT pg_advisory_unlock(:namespace, hashtext(:key))")
_PROCESS_GUARD = threading.Lock()
_PROCESS_CLAIMS: Set[str] = set()


def claim_key(scope: Any, digest: str) -> str:
    """The claim's key: one scope's (a workspace, or the Marketplace) upload of one set of bytes."""
    return f"{scope}:{digest}"


def _claim_in_process(key: str) -> bool:
    with _PROCESS_GUARD:
        if key in _PROCESS_CLAIMS:
            return False
        _PROCESS_CLAIMS.add(key)
        return True


def _release_in_process(key: str) -> None:
    with _PROCESS_GUARD:
        _PROCESS_CLAIMS.discard(key)


def _unlock(conn: Any, params: Dict[str, Any]) -> None:
    """Release the lock; when that fails, drop the connection (which releases it) so the pool
    never hands out a connection still holding it."""
    try:
        conn.execute(_UNLOCK, params)
        conn.commit()
    except Exception:
        logger.exception("[Themes] Releasing the upload claim %s failed; dropping its connection", params["key"])
        conn.invalidate()


@contextmanager
def _claim_in_database(db: Session, key: str) -> Iterator[bool]:
    """The session-level advisory lock on its own connection on Postgres; any other database has one process."""
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        yield True
        return
    params = {"namespace": UPLOAD_LOCK_NAMESPACE, "key": key}
    with bind.engine.connect() as conn:
        held = bool(conn.execute(_TRY_LOCK, params).scalar())
        conn.commit()
        try:
            yield held
        finally:
            if held:
                _unlock(conn, params)


@contextmanager
def upload_claim(db: Session, scope: Any, digest: str) -> Iterator[bool]:
    """Yields True when this upload holds (scope, digest), False when another upload does.

    Both claims, in the database and in this process, last until the block exits: past the
    upload's commit, or its refusal.
    """
    key = claim_key(scope, digest)
    if not _claim_in_process(key):
        yield False
        return
    try:
        with _claim_in_database(db, key) as held:
            yield held
    finally:
        _release_in_process(key)
