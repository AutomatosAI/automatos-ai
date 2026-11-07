"""PRD-273 §5.1 — ``memory/durable.jsonl`` and ``memory/field.jsonl``: Qdrant's memories, no vectors.

Durable memory (``DurableMemoryStore``) travels as its text, namespace, data-subject tag and
metadata, so an import writes each one back through ``DurableMemoryStore.add`` (which embeds
it with the target's own model and dedups by content). Field memory travels as its key,
value, strength and provenance. Both stores are scrolled a page at a time with no cap
(``scroll_workspace``), written as they arrive.

Both live in Qdrant, which only runs where ``QDRANT_URL`` is set: elsewhere the parts are
written empty and the manifest says the store was off. A store that is on but fails stops
the export: a file that silently lost Auto's memories would be worse than none.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

from config import config

DURABLE_PART = "memory/durable.jsonl"
FIELD_PART = "memory/field.jsonl"
FIELD_KEYS = (
    "key", "value", "strength", "agent_id", "mission_id", "task_id", "subject_id", "field_id",
    "created_at", "last_accessed", "access_count", "expired_at",
)


def durable_line(point_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """One durable memory as the file carries it."""
    return {
        "id": point_id,
        "text": payload.get("content"),
        "namespace": payload.get("namespace"),
        "subject_id": payload.get("subject_id"),
        "metadata": payload.get("metadata") or {},
        "created_at": payload.get("created_at"),
        "content_hash": payload.get("content_hash"),
    }


def field_line(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One field-memory pattern as the file carries it."""
    return {key: payload.get(key) for key in FIELD_KEYS}


async def durable_pages(workspace_id: str) -> AsyncIterator[List[Dict[str, Any]]]:
    """The workspace's durable memories, a page of file lines at a time."""
    from modules.memory.durable_store import DurableMemoryStore

    async for page in DurableMemoryStore().scroll_workspace(workspace_id):
        yield [durable_line(point_id, payload) for point_id, payload in page]


async def field_pages(workspace_id: str) -> AsyncIterator[List[Dict[str, Any]]]:
    """The workspace's field-memory patterns, a page of file lines at a time."""
    from modules.context.adapters.vector_field import VectorFieldSharedContext

    async for page in VectorFieldSharedContext().scroll_workspace(workspace_id):
        yield [field_line(payload) for payload in page]


async def _drain(pages: AsyncIterator[List[Dict[str, Any]]], sink: Any) -> None:
    async for page in pages:
        for line in page:
            sink.write_row(line)


def memory_available() -> bool:
    """Whether this edition runs Qdrant (both memory stores are on)."""
    return bool(config.QDRANT_URL)


def write_memory(writer: Any, workspace_id: str) -> Dict[str, bool]:
    """Write both memory parts (empty when Qdrant is off); which stores were on."""
    on = memory_available()
    for name, pages in ((DURABLE_PART, durable_pages), (FIELD_PART, field_pages)):
        with writer.part(name) as sink:
            if on:
                asyncio.run(_drain(pages(str(workspace_id)), sink))
    return {"durable": on, "field": on}


__all__ = ["DURABLE_PART", "FIELD_PART", "durable_line", "field_line", "memory_available", "write_memory"]
