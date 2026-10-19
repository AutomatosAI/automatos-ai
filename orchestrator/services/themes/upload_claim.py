"""PRD-261 C-02a — one upload of the same bytes into a workspace at a time.

The review stage's LLM calls run before the package row exists, so the row's unique index can't
stop two overlapping uploads of identical bytes (a timed-out request sent again) from both paying
for the stage. The first upload claims (workspace, digest) for its transaction:

* a Postgres advisory lock (``pg_try_advisory_xact_lock``), across every worker process, held until
  the upload commits or rolls back;
* a claim in this process, for a database without advisory locks (SQLite, in the unit tests).

A second upload of the same bytes is refused at once (409, try again) rather than holding a
threadpool thread for minutes; once the first commits, the retry finds its row and pays nothing.

Synchronous (a plain SQLAlchemy session): called from plain-``def`` routes only.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Set

from sqlalchemy import text
from sqlalchemy.orm import Session

# The upload claim's key space in the database ('thmu'), apart from every other advisory lock.
UPLOAD_LOCK_NAMESPACE = 0x74686D75
_TRY_LOCK = text("SELECT pg_try_advisory_xact_lock(:namespace, hashtext(:key))")
_PROCESS_GUARD = threading.Lock()
_PROCESS_CLAIMS: Set[str] = set()


def claim_key(workspace_id: Any, digest: str) -> str:
    """The claim's key: one workspace's upload of one set of bytes."""
    return f"{workspace_id}:{digest}"


def _claim_in_process(key: str) -> bool:
    with _PROCESS_GUARD:
        if key in _PROCESS_CLAIMS:
            return False
        _PROCESS_CLAIMS.add(key)
        return True


def _release_in_process(key: str) -> None:
    with _PROCESS_GUARD:
        _PROCESS_CLAIMS.discard(key)


def _claim_in_database(db: Session, key: str) -> bool:
    """The transaction-scoped advisory lock on Postgres; any other database has one process."""
    if db.get_bind().dialect.name != "postgresql":
        return True
    return bool(db.execute(_TRY_LOCK, {"namespace": UPLOAD_LOCK_NAMESPACE, "key": key}).scalar())


@contextmanager
def upload_claim(db: Session, workspace_id: Any, digest: str) -> Iterator[bool]:
    """Yields True when this upload holds (workspace, digest), False when another upload does.

    The database lock lasts until ``db``'s transaction ends (the upload's commit, or the rollback
    of a refused one); the process claim until the block exits.
    """
    key = claim_key(workspace_id, digest)
    if not _claim_in_process(key):
        yield False
        return
    try:
        yield _claim_in_database(db, key)
    finally:
        _release_in_process(key)
