"""PRD-180 S1 (F090) — real-time board-event fan-out via Postgres LISTEN/NOTIFY.

The board SSE was a timed ping with zero subscribers; the Command Centre polled.
This module makes push honest: every board-task mutation fires ``pg_notify`` on
the ``board_events`` channel, and the SSE ``LISTEN``s that channel and forwards
each event to the subscribed client — so the Command Centre sees a status change
sub-second instead of on a poll tick.

Reuses the exact primitive the dispatch spine proved (``services.board_dispatcher``):
a dedicated raw psycopg2 connection blocking on ``select`` + ``connection.poll()``.
That LISTEN is blocking, so it lives in a worker thread and bridges each
notification into a per-connection ``asyncio.Queue`` the async SSE generator
drains. NOTIFY buys latency, not correctness: a dropped notification only means
the next real event (or the heartbeat) refreshes the client.

The dispatch channel (``board_task_available``) is deliberately separate — it is
a claimant wakeup for the dispatch loop, not a UI refresh signal.
"""
from __future__ import annotations

import asyncio
import json
import logging
import select
import threading
from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Clients LISTEN on this channel; every board-task mutation NOTIFYs it. Distinct
# from the dispatcher's ``board_task_available`` (a claimant wakeup, not a UI ping).
NOTIFY_CHANNEL = "board_events"


def notify_board_event(
    db: Session,
    *,
    workspace_id,
    task_id: int,
    status: str,
    event: str = "task_changed",
) -> None:
    """Fire ``pg_notify`` so subscribed Command Centres refresh sub-second.

    Best-effort by design: a failed NOTIFY only costs the client its next
    heartbeat's latency, never correctness, so it never raises into the caller's
    request path. The payload is small and JSON so the SSE can forward it as-is.
    """
    try:
        payload = json.dumps(
            {
                "workspace_id": str(workspace_id),
                "task_id": task_id,
                "status": status,
                "event": event,
            }
        )
        db.execute(
            text("SELECT pg_notify(:chan, :payload)"),
            {"chan": NOTIFY_CHANNEL, "payload": payload},
        )
    except Exception:  # noqa: BLE001 — NOTIFY is an optimisation, not a guarantee
        logger.debug(
            "[board_events] pg_notify failed for task %s", task_id, exc_info=True
        )


def notify_chat_event(
    db: Session,
    *,
    workspace_id,
    chat_id,
    user_id,
) -> None:
    """PRD-205 S7 (backend half): fire ``chat_changed`` on the SAME channel.

    A sibling of :func:`notify_board_event` -- one new event value, zero new
    moving parts (Section 8 Q4). ChatMessenger calls this after its commit so
    an open chat learns a background message landed; the client filters by
    ``user_id`` (the target chat owner's integer ``users.id``) and refetches
    ``/messages`` for the matching ``chat_id``. Workspace isolation is the
    stream's existing ``frame_for_payload`` gate -- it keys ONLY on
    ``workspace_id``, so this payload (no task_id/status) passes untouched.

    Best-effort by design, exactly like the board emitter: a failed NOTIFY
    costs latency (the next open/fetch shows the message), never correctness,
    and never raises into the messenger.
    """
    try:
        payload = json.dumps(
            {
                "workspace_id": str(workspace_id),
                "chat_id": str(chat_id),
                "user_id": user_id,
                "event": "chat_changed",
            }
        )
        db.execute(
            text("SELECT pg_notify(:chan, :payload)"),
            {"chan": NOTIFY_CHANNEL, "payload": payload},
        )
    except Exception:  # noqa: BLE001 -- NOTIFY is an optimisation, not a guarantee
        logger.debug(
            "[board_events] pg_notify failed for chat %s", chat_id, exc_info=True
        )


class _SSEListener(threading.Thread):
    """Holds one raw LISTEN connection and pushes notifications onto a queue.

    psycopg2 ``LISTEN`` blocks, so it runs in its own daemon thread and hands
    each notification to the event loop via ``call_soon_threadsafe`` onto an
    ``asyncio.Queue``. Mirrors ``board_dispatcher._NotifyListener`` but targets a
    per-connection queue (fan-out to one SSE client) instead of a wake Event.
    """

    def __init__(
        self,
        queue: "asyncio.Queue[str]",
        loop: "asyncio.AbstractEventLoop",
    ):
        super().__init__(daemon=True, name="board-events-listen")
        self._queue = queue
        self._loop = loop
        self._stop = threading.Event()
        self._raw = None

    def run(self) -> None:
        try:
            from core.database.database import engine

            self._raw = engine.raw_connection()
            self._raw.connection.autocommit = True
            cur = self._raw.cursor()
            cur.execute(f"LISTEN {NOTIFY_CHANNEL}")
            logger.debug("[board_events] LISTEN %s active", NOTIFY_CHANNEL)
            while not self._stop.is_set():
                if select.select([self._raw.connection], [], [], 1.0)[0]:
                    self._raw.connection.poll()
                    while self._raw.connection.notifies:
                        note = self._raw.connection.notifies.pop(0)
                        self._loop.call_soon_threadsafe(
                            self._queue.put_nowait, note.payload
                        )
        except Exception:  # noqa: BLE001 — listener is best-effort; heartbeat covers
            logger.warning(
                "[board_events] LISTEN stopped — SSE falls back to heartbeat",
                exc_info=True,
            )
        finally:
            # Return the connection to the pool CLEAN. This connection was put in
            # ``autocommit=True`` for LISTEN; SQLAlchemy's reset-on-return rolls
            # back but does NOT clear psycopg2 ``autocommit``, so a leaked
            # autocommit connection would silently auto-commit the next
            # ``SessionLocal()`` caller's statements — a transactional-integrity
            # bug (and the cause of the W2-S5 idle-in-transaction regression).
            try:
                if self._raw is not None:
                    try:
                        self._raw.connection.autocommit = False
                    except Exception:  # noqa: BLE001
                        pass
                    self._raw.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stop.set()


async def board_event_stream(
    workspace_id: str,
    *,
    heartbeat_seconds: float,
) -> AsyncIterator[str]:
    """Yield SSE frames for one workspace, driven by real NOTIFY events.

    Emits a ``board_changed`` frame on connect (so the client renders current
    state immediately), then forwards every ``board_events`` NOTIFY whose payload
    is for *this* workspace as a ``board_changed`` frame carrying the changed
    task's id + status. A heartbeat comment (``:hb``) every ``heartbeat_seconds``
    keeps the connection alive and lets the client detect a dead stream — it does
    NOT drive refreshes (real events do).

    Workspace isolation is enforced here: notifications for other workspaces are
    dropped, so a tenant only ever sees its own board changes.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[str]" = asyncio.Queue()
    listener = _SSEListener(queue, loop)
    listener.start()

    try:
        # Initial sync frame so the client renders immediately, not on first event.
        yield _frame("board_changed", {"workspace_id": workspace_id})

        while True:
            payload: Optional[str] = None
            try:
                payload = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_seconds
                )
            except asyncio.TimeoutError:
                # No event this window → heartbeat comment (connection liveness).
                yield ": hb\n\n"
                continue

            frame = frame_for_payload(payload, workspace_id)
            if frame is None:
                # Malformed or another tenant's event — never leak cross-workspace.
                continue
            yield frame
    except asyncio.CancelledError:
        # Client disconnected — tear the listener down and exit cleanly.
        raise
    finally:
        listener.stop()


def frame_for_payload(payload: str, workspace_id: str) -> Optional[str]:
    """Pure routing decision: SSE frame for a NOTIFY payload, or ``None`` to drop.

    Drops anything not destined for ``workspace_id`` (the tenant-isolation gate)
    and anything malformed. Kept pure (no I/O) so the isolation + framing logic is
    unit-testable without a Postgres connection — the PG NOTIFY channel is the
    only thing that needs mocking, and it is mocked by feeding payloads here.
    """
    data = _parse_payload(payload)
    if data is None or data.get("workspace_id") != workspace_id:
        return None
    return _frame("board_changed", data)


def _frame(event: str, data: dict) -> str:
    """Format one SSE frame (``event:`` + ``data:``)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _parse_payload(payload: str) -> Optional[dict]:
    """Parse a NOTIFY payload as JSON; ``None`` on anything malformed."""
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None
