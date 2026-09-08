"""PRD-237 S7 — a chat turn finishes even when the browser goes away.

Before this module the assistant reply was generated inside the HTTP response
generator: a reload, a closed tab or a navigation that aborted the fetch closed
the generator, the LLM stream was cancelled and ``save_message`` never ran — the
reply was lost. Now the turn is a **producer task** that owns its own lifetime
and DB session; the HTTP response is only a **consumer** reading from a queue.

* The consumer going away (disconnect) does NOT cancel the producer. The turn
  completes, the reply is saved, and — because the client missed it — a
  ``chat_changed`` NOTIFY is fired by the caller's ``on_complete`` hook so the
  reloaded page merges the reply live (PRD-205 S7 lane, zero new transport).
* An explicit Stop must still stop the model: ``TurnRegistry.request_cancel``
  cancels a turn running in THIS process directly and, because production runs
  several uvicorn workers, also sets a short-lived Redis marker the producer
  polls between chunks. Redis is optional — without it, cancel is process-local
  (the same scope the existing per-process session queue already has).
* ``TurnRegistry.is_in_flight`` answers ``GET /api/chat/{id}``'s ``turnInFlight``
  so a reloaded page can show "Auto is still replying" honestly.

Nothing here knows about FastAPI or the streaming service — it is plain asyncio
so it can be tested without the app.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: A turn older than this is presumed dead if its marker is still around.
INFLIGHT_TTL_S = 15 * 60
#: A cancel request that nobody consumed within this window is forgotten.
CANCEL_TTL_S = 120
#: How often the producer looks for a cross-process cancel marker.
CANCEL_POLL_S = 0.5

_INFLIGHT_KEY = "chat:turn:inflight:{chat_id}"
_CANCEL_KEY = "chat:turn:cancel:{chat_id}"

OnComplete = Callable[..., Awaitable[None]]
Producer = Callable[[], AsyncIterator[str]]


class _Done:
    """Queue sentinel — the producer has nothing more to say."""


_DONE = _Done()


class TurnRegistry:
    """Which chats have a turn in flight, and how to stop one.

    Process-local truth (``_tasks``) plus best-effort Redis markers so the
    answer is right across workers. Every Redis failure degrades to local-only
    and is logged once at debug level — availability of the chat never depends
    on Redis.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._redis = None
        self._redis_resolved = False

    # -- redis (optional) ---------------------------------------------------

    def _client(self):
        if self._redis_resolved:
            return self._redis
        self._redis_resolved = True
        try:
            from core.redis.client import get_redis_client

            base = get_redis_client()
            if base is None:
                return None
            import redis.asyncio as aioredis

            self._redis = aioredis.Redis(
                host=base.host,
                port=base.port,
                password=base.password,
                db=base.db,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        except Exception:  # noqa: BLE001 — optional dependency, degrade to local
            logger.debug("[chat_turns] redis unavailable — process-local registry only", exc_info=True)
            self._redis = None
        return self._redis

    async def _redis_call(self, method: str, *args, **kwargs):
        client = self._client()
        if client is None:
            return None
        try:
            return await getattr(client, method)(*args, **kwargs)
        except Exception:  # noqa: BLE001 — marker ops are an optimisation
            logger.debug("[chat_turns] redis %s failed", method, exc_info=True)
            return None

    # -- lifecycle ----------------------------------------------------------

    def attach(self, chat_id: str, task: asyncio.Task) -> None:
        self._tasks[chat_id] = task

    def detach(self, chat_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(chat_id) is task:
            del self._tasks[chat_id]

    async def mark_inflight(self, chat_id: str) -> None:
        await self._redis_call("set", _INFLIGHT_KEY.format(chat_id=chat_id), "1", ex=INFLIGHT_TTL_S)

    async def clear_inflight(self, chat_id: str) -> None:
        await self._redis_call("delete", _INFLIGHT_KEY.format(chat_id=chat_id))

    async def is_in_flight(self, chat_id: str) -> bool:
        task = self._tasks.get(chat_id)
        if task is not None and not task.done():
            return True
        return bool(await self._redis_call("exists", _INFLIGHT_KEY.format(chat_id=chat_id)))

    # -- cancel -------------------------------------------------------------

    async def request_cancel(self, chat_id: str) -> bool:
        """Stop the turn for ``chat_id``. True when something was reachable."""
        task = self._tasks.get(chat_id)
        local = task is not None and not task.done()
        if local:
            task.cancel()
        marked = await self._redis_call("set", _CANCEL_KEY.format(chat_id=chat_id), "1", ex=CANCEL_TTL_S)
        return local or bool(marked)

    async def cancel_requested(self, chat_id: str) -> bool:
        """Consume a cross-process cancel marker (one-shot; GET+DEL works on any Redis)."""
        key = _CANCEL_KEY.format(chat_id=chat_id)
        if not await self._redis_call("get", key):
            return False
        await self._redis_call("delete", key)
        return True

    @property
    def local_turns(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())


_registry: Optional[TurnRegistry] = None


def get_turn_registry() -> TurnRegistry:
    global _registry
    if _registry is None:
        _registry = TurnRegistry()
    return _registry


def error_frame(message: str, code: Optional[str] = None) -> str:
    """The AI-SDK data-stream error line the frontend parses (``e:``) — the same
    ``{"message", "code"}`` shape the streaming handler emits (PRD-239 S4)."""
    payload = {"message": message}
    if code:
        payload["code"] = code
    return "e:" + json.dumps(payload) + "\n"


async def _produce_into(
    queue: "asyncio.Queue",
    *,
    chat_id: str,
    produce: Producer,
    registry: TurnRegistry,
) -> dict:
    """Drain the producer into the queue; report how it ended."""
    outcome = {"completed": False, "cancelled": False}
    last_poll = time.monotonic()
    source = produce()
    try:
        async for chunk in source:
            await queue.put(chunk)
            now = time.monotonic()
            if now - last_poll >= CANCEL_POLL_S:
                last_poll = now
                if await registry.cancel_requested(chat_id):
                    raise asyncio.CancelledError("cancel requested for chat %s" % chat_id)
        outcome["completed"] = True
    except asyncio.CancelledError:
        outcome["cancelled"] = True
        logger.info("[chat_turns] turn cancelled for chat %s", chat_id)
    except Exception as exc:  # noqa: BLE001 — surface to the client, never lose the loop
        logger.exception("[chat_turns] turn failed for chat %s", chat_id)
        await queue.put(error_frame(str(exc) or exc.__class__.__name__))
    finally:
        # Release the producer's resources (its DB session) now, not at GC time.
        try:
            await source.aclose()
        except Exception:  # noqa: BLE001 — already finished or failed; nothing to hold
            pass
    return outcome


async def _run_producer(
    queue: "asyncio.Queue",
    *,
    chat_id: str,
    produce: Producer,
    on_complete: Optional[OnComplete],
    client_gone: asyncio.Event,
    registry: TurnRegistry,
) -> None:
    task = asyncio.current_task()
    if task is not None:
        registry.attach(chat_id, task)
    await registry.mark_inflight(chat_id)
    outcome = {"completed": False, "cancelled": True}
    try:
        outcome = await _produce_into(queue, chat_id=chat_id, produce=produce, registry=registry)
    finally:
        await queue.put(_DONE)
        if task is not None:
            registry.detach(chat_id, task)
        await registry.clear_inflight(chat_id)
        if on_complete is not None:
            try:
                await on_complete(
                    completed=outcome["completed"],
                    cancelled=outcome["cancelled"],
                    client_gone=client_gone.is_set(),
                )
            except Exception:  # noqa: BLE001 — the hook is an optimisation
                logger.debug("[chat_turns] on_complete failed for chat %s", chat_id, exc_info=True)


async def run_detached_turn(
    *,
    chat_id: str,
    produce: Producer,
    on_complete: Optional[OnComplete] = None,
    registry: Optional[TurnRegistry] = None,
) -> AsyncIterator[str]:
    """Start the turn now; yield its chunks for as long as the client listens.

    ``produce`` is called exactly once, inside a task the response does not own.
    ``on_complete(completed=, cancelled=, client_gone=)`` runs when the producer
    ends, whatever happened to the consumer. Closing this generator early (the
    HTTP client went away) only records ``client_gone`` — the turn carries on.
    """
    reg = registry or get_turn_registry()
    queue: "asyncio.Queue" = asyncio.Queue()
    client_gone = asyncio.Event()
    asyncio.create_task(
        _run_producer(
            queue,
            chat_id=chat_id,
            produce=produce,
            on_complete=on_complete,
            client_gone=client_gone,
            registry=reg,
        ),
        name=f"chat-turn:{chat_id}",
    )
    finished = False
    try:
        while True:
            item = await queue.get()
            if isinstance(item, _Done):
                finished = True
                return
            yield item
    finally:
        if not finished:
            client_gone.set()
