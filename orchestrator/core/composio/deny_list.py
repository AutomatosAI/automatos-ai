"""
PRD-251 S0.6 (D16) — the Composio deny list, platform-wide
==========================================================

Composio actions that spend real money or act outside Automatos (Higgsfield's
billing purchase and trial changes, website create/deploy/publish, contest
entries, raw app invocation) are refused before any network call: for every
caller — agents, Playbooks and the API — in every workspace, whatever the
policy plane mode and whatever the capability classifier says. Buying credits,
changing plans and deploying stay with a person, in the tool's own interface.

The list is DATA, never a code constant: the ``composio.denied_actions`` system
setting, a JSON list of action slugs, seeded by the ``prd251_socials`` migration
and edited by the super-admin in Settings → System Settings. Slugs match
case-insensitively.

The list is cached per process for ``config.COMPOSIO_DENY_LIST_CACHE_TTL_SECONDS``
(30 s by default), so removing a slug unblocks it within one TTL, with no restart
or deploy. A warm cache answers from memory; a stale one answers from memory too
and starts ONE background refresh. Once the list has been read, a Composio call
never waits on the database: the F105 lesson, where synchronous pool waits froze
the event loop, applies to every execution path.

``composio_action_denial(slug)`` is the ONE check. ``composio_action_denial_async``
gives the same decision to code on the event loop: a cold cache is read in a worker
thread, never on the loop. Every Composio execution entry point calls one of them
before any network call; ``tests/test_prd251_composio_deny.py`` finds them by grep
and holds each one to it.

* No row, or an empty value → nothing is denied (a stack without the seed).
* A value that is not a JSON list of strings → EVERY action is refused, with
  the reason, until it is fixed: a guard on real money fails closed.
* A read that cannot complete (an exhausted pool, a timeout, a dropped
  connection) → the last list read is kept and the failure is logged at WARNING;
  with nothing read yet in this process, EVERY action is refused and the failure
  is logged at ERROR. The read is strict: never ``get_system_setting``, whose catch-all returns the
  default on any failure and would turn "could not read" into "nothing denied".
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

KEY_DENIED_ACTIONS = "denied_actions"  # in SettingCategory.COMPOSIO

BLOCKED_PREFIX = "This action is blocked in Automatos: "
DENIED_REASON = (
    "{slug} is on the platform's Composio deny list. Buying credits, changing plans "
    "and deploying stay with a person, in the tool's own interface."
)
UNREADABLE_REASON = (
    "the Composio deny list (system setting composio.denied_actions) is not a JSON "
    "list of action slugs, so every Composio action is refused until a super-admin "
    "fixes it in Settings → System Settings."
)
READ_FAILED_REASON = (
    "the Composio deny list (system setting composio.denied_actions) could not be "
    "read, so no Composio action runs until it can be."
)
ERROR_TYPE_DENIED = "action_denied"


def parse_denied_actions(raw: Optional[str]) -> FrozenSet[str]:
    """The setting's value → the denied slugs, upper-cased.

    No value (``None``, blank, or not text — the column only ever holds text) →
    nothing denied. Raises ``ValueError`` unless the value is a JSON list of
    strings.
    """
    if not isinstance(raw, str) or not raw.strip():
        return frozenset()
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(slug, str) for slug in value):
        raise ValueError("composio.denied_actions must be a JSON list of action slugs")
    return frozenset(slug.strip().upper() for slug in value if slug.strip())


def _read_denied_actions() -> Optional[str]:
    """The setting's stored value, ``None`` when there is no row. Raises when
    the read cannot complete."""
    # Lazy: core.llm.manager imports every LLM provider client, and this module
    # is imported by the Composio client itself.
    from core.llm.manager import read_system_setting
    from core.models.system_settings import SettingCategory

    return read_system_setting(SettingCategory.COMPOSIO.value, KEY_DENIED_ACTIONS)


# The last COMPLETED read: (outcome, expires_at on the monotonic clock). The outcome
# is the denied slugs, or _UNREADABLE when the stored value is not a JSON list. A
# failed read is never cached: with nothing cached, the next call reads again.
_UNREADABLE = object()
_MISSING = object()
_cache: Optional[Tuple[object, float]] = None
_generation = 0  # bumped by reset_cache(), so a refresh that started before it never lands after it
_refresh_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _ttl_seconds() -> float:
    from config import config

    return float(config.COMPOSIO_DENY_LIST_CACHE_TTL_SECONDS)


def reset_cache() -> None:
    """Forget the cached list, so the next check reads it again (tests)."""
    global _cache, _generation
    _generation += 1
    _cache = None


def _outcome_of(raw: Optional[str]) -> object:
    try:
        return parse_denied_actions(raw)
    except ValueError:
        logger.error(
            "[ComposioDenyList] composio.denied_actions is unreadable (%r); every Composio action is refused until it is fixed",
            raw,
        )
        return _UNREADABLE


def _refresh() -> Optional[object]:
    """Read the list once (single-flight) and cache the outcome.

    Returns the outcome. When the read cannot complete, the cached outcome is
    kept (WARNING) and returned; with nothing cached, returns ``None`` (ERROR),
    which refuses every action. Blocks the calling thread, so it runs off the
    event loop: in a worker thread, a background thread, or synchronous code.
    """
    global _cache
    with _refresh_lock:
        generation = _generation
        cached = _cache
        now = _now()
        if cached is not None and now < cached[1]:
            return cached[0]  # another thread refreshed while this one waited
        try:
            raw = _read_denied_actions()
        except Exception:  # noqa: BLE001 — any read that did not complete
            if cached is not None:
                logger.warning(
                    "[ComposioDenyList] composio.denied_actions could not be refreshed; keeping the cached list",
                    exc_info=True,
                )
                if generation == _generation:
                    _cache = (cached[0], now + _ttl_seconds())  # retry after one TTL, not on every call
                return cached[0]
            logger.error(
                "[ComposioDenyList] composio.denied_actions could not be read and nothing is cached; "
                "refusing Composio actions until it can be",
                exc_info=True,
            )
            return None
        outcome = _outcome_of(raw)
        if generation == _generation:
            _cache = (outcome, now + _ttl_seconds())
        return outcome


def _start_background_refresh() -> None:
    """Refresh in a daemon thread, at most one at a time; never blocks the caller."""
    if _refresh_lock.locked():
        return
    threading.Thread(target=_refresh, name="composio-deny-list-refresh", daemon=True).start()


def _cached_outcome() -> object:
    """The cached outcome without a database read. Fresh → it. Stale → it, with a
    background refresh started. Cold → ``_MISSING``."""
    cached = _cache
    if cached is None:
        return _MISSING
    if _now() >= cached[1]:
        _start_background_refresh()
        cached = _cache or cached  # an inline refresh (tests) has already landed
    return cached[0]


def _decide(outcome: Optional[object], slug: Any) -> Optional[str]:
    if outcome is None:
        return BLOCKED_PREFIX + READ_FAILED_REASON
    if outcome is _UNREADABLE:
        return BLOCKED_PREFIX + UNREADABLE_REASON
    normalized = str(slug or "").strip().upper()
    if normalized and normalized in outcome:
        logger.warning("[ComposioDenyList] refused %s", normalized)
        return BLOCKED_PREFIX + DENIED_REASON.format(slug=normalized)
    return None


def composio_action_denial(slug: Any) -> Optional[str]:
    """``"This action is blocked in Automatos: <reason>"`` when ``slug`` may not
    run, else ``None``. Never raises. A warm cache answers from memory; only a
    cold cache reads, in the calling thread — so code on the event loop calls
    :func:`composio_action_denial_async` instead."""
    outcome = _cached_outcome()
    if outcome is _MISSING:
        outcome = _refresh()
    return _decide(outcome, slug)


async def composio_action_denial_async(slug: Any) -> Optional[str]:
    """The same decision as :func:`composio_action_denial`, for code on the event
    loop: a warm cache answers from memory, and a cold cache is read in a worker
    thread (``asyncio.to_thread``), never on the loop."""
    outcome = _cached_outcome()
    if outcome is _MISSING:
        outcome = await asyncio.to_thread(_refresh)
    return _decide(outcome, slug)


def denied_result(denial: str) -> Dict[str, Any]:
    """The standard failed-execution envelope for a refused action."""
    return {"success": False, "data": None, "error": denial, "error_type": ERROR_TYPE_DENIED}
