"""F458 (night 18): how full the connection pool is, and who holds it when it is full.

The app has one pool (core/database/database.py: 10 + 20 connections, a 30 s
wait). Asking it for a connection when none is free blocks for that whole wait,
so two callers read its counters first instead:

* the best-effort writers (core/best_effort.py, F105), which drop a row rather
  than queue behind requests;
* GET /health (core/health_probe.py). Night 18: the pool ran dry at 08:01Z, the
  probe's SELECT 1 waited for a connection past curl's 10 s three times in a row,
  and the entrypoint's liveness watchdog stopped a backend that was still serving.

Night 18's log showed the TimeoutError and nothing about who held the 30
connections (F105 hit the same wall on night 3 and needed a pg_stat_activity
sampler). ``watch(engine)`` remembers each checked-out connection's holder: when,
which thread, which asyncio task, which request. When a checkout leaves no
connection free, one WARNING, at most every DB_POOL_FULL_LOG_SECONDS, lists the
holders oldest first with where each one is now: a task's await chain, or a
thread's stack, as code lines only (file, line, function; never locals or
values), like the loop watchdog. Bookkeeping per checkout is a few attribute
reads and a dict write; the frames are read only for the report. A listener never
raises into a checkout. DB_POOL_FULL_LOG_SECONDS=0 attaches nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import sysconfig
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import event

logger = logging.getLogger(__name__)

APP_FRAMES = 8           # the innermost app-code frames shown per holder
MAX_HOLDERS_LISTED = 40  # the pool's limit is 30
_ROOT = str(Path(__file__).resolve().parents[2])
_LIBRARY_DIRS = tuple(sorted({sysconfig.get_paths()[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}))


def _app_pool() -> Any:
    from core.database import database

    return database.engine.pool


def pool_has_room(pool: Optional[Any] = None) -> bool:
    """A connection is idle, or one more may be opened (the app's pool when none
    is given). Unreadable → True: a faked or foreign pool never stops a caller."""
    try:
        pool = _app_pool() if pool is None else pool
        return pool.checkedin() > 0 or pool.overflow() < pool._max_overflow
    except Exception:  # noqa: BLE001 — a faked or foreign pool: just go ahead
        return True


def pool_numbers(pool: Optional[Any] = None) -> Dict[str, int]:
    """Connections checked out, and the most the pool opens. Empty when unreadable."""
    try:
        pool = _app_pool() if pool is None else pool
        return {"checked_out": int(pool.checkedout()), "limit": int(pool.size() + pool._max_overflow)}
    except Exception:  # noqa: BLE001 — a faked or foreign pool has no numbers to show
        return {}


@dataclass(frozen=True)
class Holder:
    """Who checked a connection out, read at the checkout."""

    since: float
    thread_id: int
    thread_name: str
    task: Optional["weakref.ReferenceType[asyncio.Task]"]
    request: str


def _request_line() -> str:
    """The request being served ("GET /api/x req=ab12"), path hashes redacted (PRD-264 FR-17)."""
    from core.monitoring.automatos_logging import http_method_var, http_path_var, request_id_var
    from core.monitoring.path_digests import redact_digest_segments

    path = http_path_var.get()
    if not path:
        return ""
    return f" {http_method_var.get()} {redact_digest_segments(path)} req={request_id_var.get()}".rstrip()


def _current_task() -> Optional["weakref.ReferenceType[asyncio.Task]"]:
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no event loop in this thread
        return None
    return weakref.ref(task) if task is not None else None


def _is_app_frame(filename: str) -> bool:
    return not (filename.startswith("<") or filename.startswith(_LIBRARY_DIRS)
                or "site-packages" in filename or filename == __file__)


def _line(filename: str, lineno: int, name: str) -> str:
    shown = os.path.relpath(filename, _ROOT) if filename.startswith(_ROOT) else filename
    return f"{shown}:{lineno} {name}"


def _await_chain(coro: Any) -> List[str]:
    """A task's coroutine and everything it awaits, outermost first."""
    lines = []
    while coro is not None and len(lines) < 64:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            break
        if _is_app_frame(frame.f_code.co_filename):
            lines.append(_line(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name))
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return lines[-APP_FRAMES:]


def _thread_stack(frame: Any) -> List[str]:
    lines = []
    while frame is not None:
        if _is_app_frame(frame.f_code.co_filename):
            lines.append(_line(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name))
        frame = frame.f_back
    return list(reversed(lines))[-APP_FRAMES:]


def _where(holder: Holder, frames: Dict[int, Any]) -> Tuple[str, List[str]]:
    """Who the holder is, and the code lines it is at now."""
    task = holder.task() if holder.task is not None else None
    if task is not None and not task.done():
        return f"task {task.get_name()} ({getattr(task.get_coro(), '__qualname__', '?')})", _await_chain(task.get_coro())
    if holder.task is not None:
        return "a task that has finished without giving the connection back", []
    frame = frames.get(holder.thread_id)
    if frame is None:
        return f"thread {holder.thread_name}, which has ended without giving the connection back", []
    return f"thread {holder.thread_name}", _thread_stack(frame)


class PoolWatch:
    """The holders of one engine's checked-out connections, and the full-pool report."""

    def __init__(self, engine: Any, every_s: float) -> None:
        self._engine = engine
        self._every_s = every_s
        self._lock = threading.Lock()
        self._holders: Dict[int, Holder] = {}
        self._last_report: Optional[float] = None

    def holder_count(self) -> int:
        with self._lock:
            return len(self._holders)

    def on_checkout(self, _dbapi_connection: Any, record: Any, _proxy: Any) -> None:
        try:
            self._remember(record)
            if not pool_has_room(self._engine.pool) and self._report_due(time.monotonic()):
                self._report()
        except Exception:  # noqa: BLE001 — never fail a checkout over its bookkeeping
            logger.debug("[pool-watch] a checkout was not recorded", exc_info=True)

    def on_return(self, _dbapi_connection: Any, record: Any) -> None:
        with self._lock:
            self._holders.pop(id(record), None)

    def _remember(self, record: Any) -> None:
        thread = threading.current_thread()
        holder = Holder(time.monotonic(), thread.ident or 0, thread.name, _current_task(), _request_line())
        with self._lock:
            self._holders[id(record)] = holder

    def _report_due(self, now: float) -> bool:
        with self._lock:
            if self._last_report is not None and now - self._last_report < self._every_s:
                return False
            self._last_report = now
            return True

    def _report(self) -> None:
        now = time.monotonic()
        with self._lock:
            holders = sorted(self._holders.values(), key=lambda h: h.since)
        frames = sys._current_frames()
        lines = []
        for holder in holders[:MAX_HOLDERS_LISTED]:
            who, at = _where(holder, frames)
            lines.append(f"  {now - holder.since:7.1f}s  {who}{holder.request}" + (f"\n      at {' > '.join(at)}" if at else ""))
        numbers = pool_numbers(self._engine.pool)
        logger.warning(
            "[pool-watch] the connection pool is full (%s of %s checked out); who holds it, oldest first:\n%s",
            numbers.get("checked_out", "?"), numbers.get("limit", "?"), "\n".join(lines) or "  (none recorded)",
        )


def watch(engine: Any, every_s: Optional[float] = None) -> Optional[PoolWatch]:
    """Watch ``engine``'s pool (DB_POOL_FULL_LOG_SECONDS when ``every_s`` is None).
    Returns None, having attached nothing, when that is 0 or the listeners cannot attach."""
    try:
        if every_s is None:
            from config import config

            every_s = float(config.DB_POOL_FULL_LOG_SECONDS)
        if every_s <= 0:
            return None
        pool_watch = PoolWatch(engine, every_s)
        event.listen(engine, "checkout", pool_watch.on_checkout)
        event.listen(engine, "checkin", pool_watch.on_return)
        event.listen(engine, "detach", pool_watch.on_return)
        return pool_watch
    except Exception:  # noqa: BLE001 — a watch never takes the app down
        logger.warning("[pool-watch] not started", exc_info=True)
        return None
