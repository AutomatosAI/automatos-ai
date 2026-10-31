"""PRD-268 §6.11.3 — run the readiness probes, hold the last check, and write ``Readiness@1``.

* **A check** runs every item's probe at once, each in its own daemon thread on its own
  session, and waits ``config.READINESS_PROBE_TIMEOUT_SECONDS`` for them. A probe that has
  not answered when the wait ends reads ``not_checked`` ("Didn't answer in time"); its thread
  is left to finish on its own (a daemon, so a hung probe never holds up shutdown) and its
  answer is not read. A probe that raises reads ``needs_setup`` with ``unreadable``, the
  error in the server log.
* **The last check** of each workspace is held for the check window (``readiness_check`` in
  ``core.security.rate_limiter.DEFAULT_LIMITS``, the window "Check again" is limited to),
  whatever it found: :func:`current` answers it while it is held, else checks;
  :func:`check_now` always checks. One check of a workspace runs at a time, so reads that
  find nothing held wait for one check rather than each starting their own. Each worker
  holds its own.
* **The detail** is the finding's sentence in the request's language. The local edition adds
  where the probe reached and how long it took (``localhost:5432 · 12 ms``); SaaS never
  shows a host or a latency. A fix is shown only on an item that needs setting up.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple
from uuid import UUID

from cachetools import TTLCache
from sqlalchemy.orm import Session

from config import config
from core.security.rate_limiter import DEFAULT_LIMITS
from services.readiness import probes
from services.readiness.contract import Readiness, ReadinessFix, ReadinessItem
from services.readiness.items import FIXES, NEEDS_SETUP, NOT_ANSWERED, NOT_CHECKED, UNREADABLE, Finding, Item
from services.readiness.strings import say

logger = logging.getLogger(__name__)

CHECK_OPERATION = "readiness_check"
VERSION = 1
THREAD_PREFIX = "readiness"
HELD_WORKSPACES = 1024
# Locks a workspace's check takes, shared by workspaces whose ids hash alike.
CHECK_STRIPES = 64
MS_PER_SECOND = 1000
DETAIL_SEPARATOR = " · "
LATENCY_UNIT = "ms"


@dataclass(frozen=True)
class Result:
    """An item's finding, and how long its probe took (None when it did not answer)."""

    item: Item
    finding: Finding
    ms: Optional[int] = None


@dataclass(frozen=True)
class Checked:
    """One check: when, and each item's result in the items' order."""

    checked_at: datetime
    results: Tuple[Result, ...]


def check_window() -> int:
    """The seconds a check is held, the window "Check again" is limited to."""
    return int(DEFAULT_LIMITS[CHECK_OPERATION][1])


class HeldChecks:
    """Each workspace's last check, held for the check window."""

    def __init__(self, ttl_seconds: int) -> None:
        self._held: TTLCache = TTLCache(maxsize=HELD_WORKSPACES, ttl=ttl_seconds)
        self._lock = threading.Lock()
        self._checking = tuple(threading.Lock() for _ in range(CHECK_STRIPES))

    def checking(self, workspace_id: UUID) -> threading.Lock:
        """The lock a check of this workspace runs under, one check at a time."""
        return self._checking[hash(str(workspace_id)) % len(self._checking)]

    def get(self, workspace_id: UUID) -> Optional[Checked]:
        """The workspace's last check while it is held, else None."""
        with self._lock:
            return self._held.get(str(workspace_id))

    def put(self, workspace_id: UUID, checked: Checked) -> None:
        """Hold the workspace's check, replacing the one before."""
        with self._lock:
            self._held[str(workspace_id)] = checked


# Module-level so a test can start from nothing.
held = HeldChecks(check_window())


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def _probe(item: Item, workspace_id: UUID) -> Tuple[Finding, int]:
    """One item's probe on its own session: what it found, and how long it took."""
    start = time.monotonic()
    db: Optional[Session] = None
    try:
        db = _open_session()
        finding = item.probe(db, workspace_id)
    except Exception:
        logger.exception("readiness: the %s probe failed", item.id)
        finding = Finding(NEEDS_SETUP, UNREADABLE)
    finally:
        if db is not None:
            db.close()
    return finding, round((time.monotonic() - start) * MS_PER_SECOND)


def _start(item: Item, workspace_id: UUID) -> Future:
    """Run one item's probe in a daemon thread; the future holds its answer once it has one."""
    future: Future = Future()
    thread = threading.Thread(target=lambda: future.set_result(_probe(item, workspace_id)),
                              name=f"{THREAD_PREFIX}-{item.id}", daemon=True)
    thread.start()
    return future


def _result(item: Item, future: Future) -> Result:
    if not future.done():
        return Result(item, Finding(NOT_CHECKED, NOT_ANSWERED))
    finding, ms = future.result()
    return Result(item, finding, ms)


def check(workspace_id: UUID, items: Optional[Sequence[Item]] = None) -> Checked:
    """Run every item's probe at once; one that does not answer in time is ``not_checked``."""
    items = tuple(probes.ITEMS if items is None else items)
    futures = [_start(item, workspace_id) for item in items]
    wait(futures, timeout=config.READINESS_PROBE_TIMEOUT_SECONDS)
    return Checked(datetime.now(timezone.utc), tuple(_result(item, f) for item, f in zip(items, futures)))


def _detail(result: Result, locale: str, local: bool) -> str:
    """The finding's sentence; on the local edition, where it reached and how long it took too."""
    sentence = say(locale, result.finding.detail)
    if not local:
        return sentence
    latency = f"{result.ms} {LATENCY_UNIT}" if result.ms is not None else None
    return DETAIL_SEPARATOR.join(part for part in (sentence, result.finding.where, latency) if part)


def _fix(finding: Finding, locale: str) -> Optional[ReadinessFix]:
    fix = FIXES.get(finding.fix) if finding.state == NEEDS_SETUP and finding.fix else None
    return ReadinessFix(label=say(locale, fix.label), href=fix.href) if fix else None


def _item(result: Result, locale: str, local: bool) -> ReadinessItem:
    return ReadinessItem(
        id=result.item.id, label=say(locale, result.item.id), detail=_detail(result, locale, local),
        state=result.finding.state, fix=_fix(result.finding, locale), needed_by=list(result.item.needed_by),
    )


def render(checked: Checked, locale: str) -> Dict[str, Any]:
    """``Readiness@1`` in ``locale``, for this edition."""
    local = config.IS_LOCAL_EDITION
    readiness = Readiness(v=VERSION, checked_at=checked.checked_at.isoformat(),
                          items=[_item(result, locale, local) for result in checked.results])
    return readiness.model_dump(exclude_none=True)


def _check_and_hold(workspace_id: UUID) -> Checked:
    checked = check(workspace_id)
    held.put(workspace_id, checked)
    return checked


def check_now(workspace_id: UUID, locale: str) -> Dict[str, Any]:
    """Check every item now, hold the check, and answer it."""
    with held.checking(workspace_id):
        checked = _check_and_hold(workspace_id)
    return render(checked, locale)


def current(workspace_id: UUID, locale: str) -> Dict[str, Any]:
    """The workspace's last check while it is held, else a check now (one at a time)."""
    checked = held.get(workspace_id)
    if checked is None:
        with held.checking(workspace_id):
            checked = held.get(workspace_id) or _check_and_hold(workspace_id)
    return render(checked, locale)
