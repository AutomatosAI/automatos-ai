"""Spoken turns skip the connected-apps search (voice latency C-2).

A turn sent with ``spokenMode`` on ``POST /api/chat`` runs the Composio search before its
first model call, like a typed turn: a lookup bounded at ``COMPOSIO_LOOKUP_TIMEOUT_SECONDS``.
While ``SPOKEN_MODE_SKIP_COMPOSIO`` is true a spoken turn does not run it, on Auto's lane and
on an agent chosen by ``agentId`` alike. This is the shape Auto Live has run since PRD-207.

What the skip removes: the typed per-action tools, the scope message and the hint lines.
What it leaves: the generic ``composio_execute`` tool wherever the registry grants it, and
every platform tool. No prompt text is added.

While the switch is false the turn passes the lane's own value, as before.
"""
from __future__ import annotations

import logging

from config import config

logger = logging.getLogger(__name__)


def voice_skips_apps(lane_skips: bool, spoken: bool) -> bool:
    """True only when the spoken rule is why this turn skips the apps search: a spoken turn,
    ``SPOKEN_MODE_SKIP_COMPOSIO`` on, and the lane itself would have searched."""
    return bool(spoken) and not lane_skips and bool(config.SPOKEN_MODE_SKIP_COMPOSIO)


def spoken_skips_apps(lane_skips: bool, spoken: bool) -> bool:
    """What the turn passes as ``skip_composio``: the lane's own skip, or the spoken rule's.
    Logs one line when the spoken rule is the reason."""
    if not voice_skips_apps(lane_skips, spoken):
        return lane_skips
    logger.info("[chat] spoken turn: connected-apps search skipped (SPOKEN_MODE_SKIP_COMPOSIO)")
    return True


__all__ = ["spoken_skips_apps", "voice_skips_apps"]
