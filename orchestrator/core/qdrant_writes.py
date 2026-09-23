"""F103 (night 3): a Qdrant write that times out is tried once more.

Night 3 lost memory writes to Qdrant 408s inside the F105 event-loop freezes
(19 of them in 63 ms at one point). Both memory stores write through here —
durable memory (L3) and the mission field: a write that times out (a Qdrant
408, or the client's own timeout) is tried once more after
MEMORY_WRITE_RETRY_PAUSE_S; anything else, or a second timeout, is raised for
the caller to report.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config import config

logger = logging.getLogger(__name__)


def is_timeout(exc: BaseException) -> bool:
    """The store did not answer in time: a 408 from Qdrant, or the client's own timeout."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 408
    if isinstance(exc, ResponseHandlingException):  # the client wraps its transport errors
        exc = exc.source
    return isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError))


async def upsert_retrying_a_timeout_once(client: Any, collection_name: str, points: List[Any]) -> None:
    """``client.upsert`` — tried once more after a short pause if it timed out."""
    for attempt in (1, 2):
        try:
            await client.upsert(collection_name=collection_name, points=points)
            return
        except Exception as exc:
            if attempt == 2 or not is_timeout(exc):
                raise
            logger.info("[Qdrant] write to %s timed out (%r) — retrying once", collection_name, exc)
            await asyncio.sleep(config.MEMORY_WRITE_RETRY_PAUSE_S)
