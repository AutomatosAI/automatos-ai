"""Playbook repeated-failure circuit breaker (PRD-185 S4).

A cron-scheduled playbook that fails on every run re-fires forever. That is what
happened in 2026-06: an OpenRouter 402 made one playbook fail on every tick, and
because nothing stopped it, it re-fired daily and spammed the same failure for
~17 days. The board even reported green over it (fixed separately in #504).

This breaker closes that loop. It reads the playbook's recent execution history
(``recipe_executions``) and, once the last N *terminal* runs are all ``failed``,
reports the breaker OPEN so the cron scheduler skips re-firing until a human
intervenes. The state is **derived, not stored** — no new column, no migration:

- A manual run (which bypasses the breaker) that *succeeds* writes a ``completed``
  row, which breaks the all-failed streak, so the breaker auto-closes on the next
  tick. There is no explicit "reset" to forget to call.
- While the breaker is open the scheduler creates no new execution rows, so the
  history stays stable and the breaker stays open — "fails, alerts, stops" — until
  that manual success (or a config change) clears it.

The decision (:func:`is_breaker_open`) is a pure function over a status list so it
is trivially unit-testable with no DB; :func:`breaker_is_open` is the thin DB-bound
wrapper the scheduler calls.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)

# Only terminal executions count toward the streak. A pending/running row is not
# yet an outcome; a cancelled run was deliberately stopped, so it is neither a
# success that should reset the breaker nor a failure that should trip it.
_TERMINAL_STATUSES = ("completed", "failed")


def is_breaker_open(recent_statuses: Sequence[str], threshold: int) -> bool:
    """Pure decision: is the breaker open?

    Args:
        recent_statuses: The playbook's most recent terminal execution statuses,
            newest first (only ``completed`` / ``failed`` — see ``_TERMINAL_STATUSES``).
        threshold: Number of consecutive failures that trips the breaker. ``<= 0``
            disables it.

    Returns:
        True when there are at least ``threshold`` terminal runs and the most
        recent ``threshold`` of them are all ``failed``.
    """
    if threshold <= 0:
        return False
    if len(recent_statuses) < threshold:
        return False
    return all(status == "failed" for status in recent_statuses[:threshold])


def _recent_terminal_statuses(db: Session, recipe_id: int, limit: int) -> List[str]:
    """Fetch the newest ``limit`` terminal execution statuses for a playbook.

    Uses the ``idx_recipe_executions_recipe_status`` index on
    ``(recipe_id, status)``; ordered by ``started_at`` desc so "newest first".
    """
    from core.models.core import RecipeExecution

    rows = (
        db.query(RecipeExecution.status)
        .filter(
            RecipeExecution.recipe_id == recipe_id,
            RecipeExecution.status.in_(_TERMINAL_STATUSES),
        )
        .order_by(RecipeExecution.started_at.desc())
        .limit(limit)
        .all()
    )
    return [row.status for row in rows]


def breaker_is_open(db: Session, recipe_id: int) -> bool:
    """True if playbook ``recipe_id`` should be paused from cron re-firing.

    Thin DB-bound wrapper over :func:`is_breaker_open`. Never raises — a breaker
    that cannot read history must not take the scheduler down, so on any error it
    fails *closed* (returns False = "do not block the run") and logs.
    """
    threshold = config.PLAYBOOK_BREAKER_THRESHOLD
    if threshold <= 0:
        return False
    try:
        statuses = _recent_terminal_statuses(db, recipe_id, threshold)
    except Exception as exc:  # pragma: no cover - defensive; never block on read error
        logger.warning(
            "[PlaybookBreaker] Could not read execution history for playbook %s: %s",
            recipe_id, exc,
        )
        return False
    return is_breaker_open(statuses, threshold)
