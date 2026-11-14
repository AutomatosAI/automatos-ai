"""Spoken turns skip the connected-apps search (voice latency C-2).

A turn sent with ``spokenMode`` on ``POST /api/chat`` runs the Composio search before its
first model call, like a typed turn: a lookup bounded at ``COMPOSIO_LOOKUP_TIMEOUT_SECONDS``.
While ``SPOKEN_MODE_SKIP_COMPOSIO`` is true a spoken turn does not run it, on Auto's lane and
on an agent chosen by ``agentId`` alike. This is the shape Auto Live has run since PRD-207.

What the skip removes: the typed per-action tools, the scope message and the hint lines.
What it leaves: the generic ``composio_execute`` tool wherever the registry grants it, and
every platform tool. No prompt text is added.

While the switch is false the turn passes the lane's own value, as before.

Naming a connected app brings the typed tools back (voice latency C-3). While
``SPOKEN_MODE_APPS_WHEN_NAMED`` is true too, a turn the spoken rule itself skipped runs the
search after all when the person's own words name an app Auto may use ("check my Gmail").
Auto's own lane only: never with ``agentId``, never on the ASSIGN lane, never on a turn the
lane already skips. The list of apps is the search's own, read on a thread and session of its
own under ``APP_CHECK_TIMEOUT_SECONDS``; a slow or failed read is "not named". The check's
log lines carry a flag and a duration, never the person's words.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Iterable, List

from config import config

logger = logging.getLogger(__name__)

APP_CHECK_TIMEOUT_SECONDS = 0.5
MIN_APP_NAME_CHARS = 4
MAX_APP_NAME_WORDS = 3
MS_PER_SECOND = 1000
_WORD = re.compile(r"[^\W_]+")  # a run of letters and digits


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


def _words(text: str) -> List[str]:
    """``text`` lower-cased and cut into words of letters and digits."""
    return _WORD.findall((text or "").lower())


def names_an_app(text: str, apps: Iterable[str]) -> bool:
    """Pure. An app's name is said as whole consecutive words: the text is lower-cased and cut
    into words of letters and digits; every run of 1 to MAX_APP_NAME_WORDS adjacent words,
    joined, is compared for equality with each app's name (lower-cased, letters and digits
    only). Names under MIN_APP_NAME_CHARS never match."""
    names = {"".join(_words(app)) for app in apps or ()}
    names = {name for name in names if len(name) >= MIN_APP_NAME_CHARS}
    if not names:
        return False
    words = _words(text)
    return any(
        "".join(words[start:start + size]) in names
        for size in range(1, MAX_APP_NAME_WORDS + 1)
        for start in range(len(words) - size + 1)
    )


def allowed_apps(agent_id: int, workspace_id: Any) -> List[str]:
    """Sync, on a session of its own (SessionLocal, closed in finally): the list the search
    itself uses, ``ComposioToolService(db)._resolve_allowed_apps(agent_id, workspace_id)``.
    Local queries only, no SDK call."""
    from core.database.database import SessionLocal
    from modules.tools.services.composio_tool_service import ComposioToolService

    db = SessionLocal()
    try:
        return list(ComposioToolService(db)._resolve_allowed_apps(agent_id, workspace_id))
    finally:
        db.close()


def _may_name_an_app(lane: Any, agent_chosen: bool) -> bool:
    """Whether the name check applies to a turn the spoken rule skipped: the switch is on, no
    agent was chosen, and the lane is not ASSIGN (Auto answers on its own lane)."""
    from consumers.chatbot.auto import Action

    if agent_chosen or not config.SPOKEN_MODE_APPS_WHEN_NAMED:
        return False
    assessment = getattr(lane, "assessment", None)
    return getattr(assessment, "action", None) != Action.ASSIGN


async def _named(agent_id: int, workspace_id: Any, text: str) -> bool:
    """Whether ``text`` names an app the agent may use. The read is bounded; one that is
    slow or fails logs one warning and reads as not named. One info line per check."""
    started = time.monotonic()
    named = False
    try:
        apps = await asyncio.wait_for(
            asyncio.to_thread(allowed_apps, agent_id, workspace_id), APP_CHECK_TIMEOUT_SECONDS)
        named = names_an_app(text, apps)
    except Exception as exc:  # noqa: BLE001 — a timeout included: the turn goes on skipped, never fails
        logger.warning("[spoken] apps check failed (%s): read as not named", type(exc).__name__)
    check_ms = int((time.monotonic() - started) * MS_PER_SECOND)
    logger.info("[spoken] apps check: named=%s check_ms=%s", named, check_ms)
    return named


async def apps_skipped(lane: Any, *, spoken: bool, agent_chosen: bool, workspace_id: Any,
                       text: str) -> bool:
    """What the turn passes as skip_composio. The lane's own value unless the spoken rule is
    the reason; then True unless SPOKEN_MODE_APPS_WHEN_NAMED is on, Auto answers on its own
    lane, and the person named an app Auto may use."""
    lane_skips = lane.skip_composio
    if not voice_skips_apps(lane_skips, spoken):
        return lane_skips
    if _may_name_an_app(lane, agent_chosen) and await _named(lane.agent_id, workspace_id, text):
        return False
    return spoken_skips_apps(lane_skips, spoken)


__all__ = [
    "APP_CHECK_TIMEOUT_SECONDS", "allowed_apps", "apps_skipped", "names_an_app", "spoken_skips_apps",
    "voice_skips_apps",
]
