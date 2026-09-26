"""F105-B (night 3): a chat turn does not hold a pool connection while the model thinks.

A turn's session opened a transaction with its first read — the user's clerk
id, the agent, the workspace, the tools — and kept it, with its connection,
through every model call of the tool loop until the turn's final commit.
Night 3's pool (10 + 20) ran dry on turns doing exactly that: up to 9
connections sat "idle in transaction" for 16–38 s, their last statements the
turns' opening reads.

``release_if_read_only`` ends a transaction that has only READ — a rollback
that loses nothing — so its connection goes back to the pool; the next query
opens a new transaction. (``database.end_open_transaction`` instead COMMITS
whatever is pending: right for a tick that owns its writes, wrong mid-turn.)
Objects the session loaded are expired and re-read on their next use. A
transaction that wrote anything, or took a lock, is left exactly as it was. "Wrote" is judged per statement on the connection, so a raw
``text()`` INSERT counts the same as an ORM flush; only a plain SELECT or SHOW
is a read, and a SELECT that locks (``FOR UPDATE``/``SHARE``, advisory locks),
sets config or selects INTO counts as a write. A session whose transaction
began before these listeners were registered is never released.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_WROTE = "f105_wrote"                    # on a connection: this transaction wrote or locked
_CONNECTIONS = "f105_connections"        # on a session: the connections its transaction uses
_TRACKED = "f105_tracked"                # on a session: its transaction began under these listeners

_PLAIN_READ = re.compile(r"^\s*(?:/\*.*?\*/\s*|--[^\n]*\n\s*)*(?:SELECT|SHOW)\b", re.IGNORECASE | re.DOTALL)
_LOCKS_OR_WRITES = re.compile(
    # F119: pg_notify is delivered at COMMIT — a transaction holding one is not
    # read-only; releasing it (a rollback) drops the notice. F200: the try-lock
    # holds the agent count; releasing it would reopen the double create.
    r"\bFOR\s+(?:NO\s+KEY\s+)?(?:UPDATE|SHARE|KEY\s+SHARE)\b|pg_(?:try_)?advisory|set_config|\bINTO\b|nextval|setval|pg_notify",
    re.IGNORECASE,
)


def is_plain_read(statement: str) -> bool:
    """True only for a SELECT/SHOW that neither locks nor writes."""
    return bool(_PLAIN_READ.match(statement or "")) and not _LOCKS_OR_WRITES.search(statement)


@event.listens_for(Engine, "begin")
def _transaction_starts_clean(conn) -> None:
    conn.info.pop(_WROTE, None)


@event.listens_for(Engine, "before_cursor_execute")
def _mark_writes(conn, cursor, statement, parameters, context, executemany) -> None:
    if executemany or not is_plain_read(statement):
        conn.info[_WROTE] = True


@event.listens_for(Session, "after_begin")
def _track_connection(session, transaction, connection) -> None:
    session.info[_TRACKED] = True
    session.info.setdefault(_CONNECTIONS, []).append(connection)


@event.listens_for(Session, "after_transaction_end")
def _forget_connections(session, transaction) -> None:
    if transaction.parent is None:
        session.info.pop(_CONNECTIONS, None)
        session.info.pop(_TRACKED, None)


def _wrote(connection: Any) -> bool:
    try:
        return bool(connection.info.get(_WROTE))
    except Exception:  # noqa: BLE001 — cannot tell: keep the transaction
        return True


def release_if_read_only(session: Any) -> bool:
    """End ``session``'s transaction if it has only read, returning its
    connection to the pool. True when released; never raises."""
    try:
        if session is None or not session.in_transaction() or not session.info.get(_TRACKED):
            return False
        if session.new or session.dirty or session.deleted:
            return False
        if any(_wrote(c) for c in session.info.get(_CONNECTIONS, [])):
            return False
        session.rollback()
        return True
    except Exception:  # noqa: BLE001 — a failed release leaves the turn as it was
        logger.debug("read-only release skipped", exc_info=True)
        return False
