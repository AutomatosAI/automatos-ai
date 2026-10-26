"""The ``notifications`` table's bounded columns, and the one cut that makes a value fit.

``alembic/versions/prd128_notifications.py`` made the table (there is no ORM model): ``event_type``
VARCHAR(50), ``title`` VARCHAR(255), ``link_type`` VARCHAR(30), ``agent_name`` VARCHAR(100) and
``status`` VARCHAR(20); ``message`` and ``link_id`` are TEXT. One value longer than its column
fails the whole INSERT (StringDataRightTruncation: on 26 Oct 2026 a Mission step's 300-character
title). So the dispatcher's in-app row passes through :func:`fit_columns` first: a cut value keeps
its first characters and ends in ``…``, so whoever reads it sees it was cut.

:func:`clamp` is also the cut the notice builders make to their own, shorter limits
(``services/notification_events``). Pure: the standard library only.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional

ELLIPSIS = "…"

# Each bounded column's length in characters, as prd128_notifications.py declares it
# (Postgres counts VARCHAR(n) in characters, as len() does).
NOTIFICATION_COLUMN_CHARS: Mapping[str, int] = MappingProxyType({
    "event_type": 50,
    "title": 255,
    "link_type": 30,
    "agent_name": 100,
    "status": 20,
})


def clamp(value: Optional[str], limit: int) -> Optional[str]:
    """``value`` at most ``limit`` characters, ending in ``…`` when it was cut; None stays None."""
    if value is None or len(value) <= limit:
        return value
    return value[: limit - len(ELLIPSIS)] + ELLIPSIS


def fit_columns(params: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the insert's ``params`` with each bounded column's text cut to fit its column."""
    return {key: _fit(key, value) for key, value in params.items()}


def _fit(column: str, value: Any) -> Any:
    """``value`` cut to ``column``'s length when the column is bounded and the value is text."""
    limit = NOTIFICATION_COLUMN_CHARS.get(column)
    return clamp(value, limit) if limit is not None and isinstance(value, str) else value
