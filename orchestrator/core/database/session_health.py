"""A request session left in a failed transaction fails everything after it (F074).

Postgres aborts a transaction on its first failed statement; until it is rolled
back, every later statement on that connection fails "current transaction is
aborted". A chat turn runs every tool on one request session, so a tool that
catches its own SQL error and returns it as data — ``search_multimodal`` on
night 1 — silently takes down every tool after it, query_database included.
The unified executor calls :func:`rollback_if_aborted` after each tool.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# psycopg2.extensions.TRANSACTION_STATUS_INERROR — the connection's transaction
# has failed and accepts nothing but a rollback.
_TRANSACTION_IN_ERROR = 3


def transaction_aborted(session: Any) -> bool:
    """True when the session's open database transaction has failed."""
    try:
        if session is None or not session.in_transaction():
            return False
        if not session.is_active:  # a failed flush: SQLAlchemy already knows
            return True
        dbapi = session.connection().connection.dbapi_connection
        status = getattr(getattr(dbapi, "info", None), "transaction_status", None)
        return status == _TRANSACTION_IN_ERROR
    except Exception:  # noqa: BLE001 — a probe must never become the failure
        return False


def rollback_if_aborted(session: Any, culprit: str) -> bool:
    """Roll a failed transaction back so the rest of the turn can run, naming
    the culprit so the failed statement can be found. True when it rolled back.

    Nothing is lost that was not already lost: an aborted transaction can only
    end in a rollback.
    """
    if not transaction_aborted(session):
        return False
    logger.warning(
        "[session-health] %s left the request session in a failed transaction — "
        "rolled back so the rest of the turn can run (F074)",
        culprit,
    )
    try:
        session.rollback()
    except Exception:  # noqa: BLE001
        logger.warning("[session-health] the rollback after %s failed", culprit, exc_info=True)
        return False
    return True
