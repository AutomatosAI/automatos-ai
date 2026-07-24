"""Anthropic memory-tool backend (PRD-201 S5).

The Anthropic memory tool (``memory_20250818``) is **client-executed**: the model
emits ``memory`` tool calls (``view`` / ``create`` / ``str_replace`` / ``insert``
/ ``delete`` / ``rename``) against a ``/memories`` directory, and the harness must
run them and return the result. This module is that harness for the headless
Anthropic loop.

Two concerns are separated:

- :func:`resolve_memory_path` — the **traversal guard**. Every model-supplied
  path is canonicalised and confined to ``/memories`` before any storage call;
  ``..``, absolute paths outside ``/memories`` and URL-encoded traversal are
  rejected. This is pure and the security-critical, always-tested part.
- :class:`MemoryToolBackend` — command dispatch against a swappable
  :class:`MemoryStore`. The default :class:`DurableMemoryStoreBackend` delegates
  to the platform's existing PRD-187 durable memory plane
  (``UnifiedMemoryService``) rather than standing up a parallel memory vendor —
  the §8-Q6 **recommendation**. Whether a semantic durable store is the right
  substrate for the memory tool's *file* model (vs a scoped per-workspace scratch
  surface, or a fresh file plane) is §8-Q6, Gerard's call; the store is behind a
  Protocol so swapping it is one class.
"""
from __future__ import annotations

import logging
import posixpath
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The one virtual root the memory tool operates in. Not a real filesystem dir —
# it is a namespace over the durable store — so the guard is purely lexical.
MEMORY_ROOT = "/memories"

# Anthropic memory-tool declaration (confirmed via the claude-api skill; flag for
# build-time re-confirmation per §8-Q2 — it is a versioned beta surface).
MEMORY_TOOL_TYPE = "memory_20250818"


class MemoryPathError(ValueError):
    """A model-supplied memory path escaped ``/memories`` (traversal guard)."""


def resolve_memory_path(path: str) -> str:
    """Canonicalise a model-supplied memory path and confine it to ``/memories``.

    Rejects ``..`` traversal, absolute paths outside ``/memories``, URL-encoded
    traversal (``%2e`` / ``%2f``) and NUL bytes. Returns the normalised POSIX
    path, always at or under ``/memories``.
    """
    if not path or not isinstance(path, str):
        raise MemoryPathError("empty memory path")
    lowered = path.lower()
    if "%2e" in lowered or "%2f" in lowered or "\x00" in path:
        raise MemoryPathError(f"illegal memory path: {path!r}")
    # An absolute path must already be under /memories; a relative path is joined
    # under it. normpath collapses any ``..`` — the result is then re-checked so a
    # collapse that escaped the root is rejected.
    candidate = path if path.startswith("/") else posixpath.join(MEMORY_ROOT, path)
    normalized = posixpath.normpath(candidate)
    if normalized != MEMORY_ROOT and not normalized.startswith(MEMORY_ROOT + "/"):
        raise MemoryPathError(f"memory path escapes {MEMORY_ROOT}: {path!r}")
    return normalized


def memory_tool_definition() -> Dict[str, str]:
    """The Anthropic memory-tool declaration block (client-executed)."""
    return {"type": MEMORY_TOOL_TYPE, "name": "memory"}


class MemoryStore:
    """Minimal path-keyed store the memory tool runs against (swappable substrate).

    Async so a durable/DB-backed implementation fits. Implementations do NOT need
    to guard paths — :class:`MemoryToolBackend` calls :func:`resolve_memory_path`
    before every store call.
    """

    async def get_text(self, workspace_id: Any, path: str) -> Optional[str]:
        raise NotImplementedError

    async def put_text(self, workspace_id: Any, path: str, text: str) -> None:
        raise NotImplementedError

    async def delete(self, workspace_id: Any, path: str) -> None:
        raise NotImplementedError

    async def list_paths(self, workspace_id: Any, prefix: str) -> List[str]:
        raise NotImplementedError


class MemoryToolBackend:
    """Executes Anthropic memory-tool commands against a :class:`MemoryStore`.

    Every model-supplied path passes :func:`resolve_memory_path` FIRST, so a
    ``../`` escape is rejected before any storage call. Command handlers never
    raise into the tool loop — a failure comes back as an ``Error: ...`` string
    the model can read and adapt to (errors-as-data).
    """

    def __init__(self, store: MemoryStore, workspace_id: Any = None) -> None:
        self._store = store
        self._workspace_id = workspace_id

    async def handle(self, command_input: Dict[str, Any]) -> str:
        cmd = (command_input or {}).get("command")
        try:
            if cmd == "view":
                return await self._view(command_input)
            if cmd == "create":
                return await self._create(command_input)
            if cmd == "str_replace":
                return await self._str_replace(command_input)
            if cmd == "insert":
                return await self._insert(command_input)
            if cmd == "delete":
                return await self._delete(command_input)
            if cmd == "rename":
                return await self._rename(command_input)
            return f"Error: unknown memory command {cmd!r}"
        except MemoryPathError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # never fail the tool loop
            logger.warning("[memory-tool] command %s failed: %s", cmd, exc)
            return f"Error: memory command failed: {exc}"

    async def _view(self, inp: Dict[str, Any]) -> str:
        path = resolve_memory_path(inp.get("path") or MEMORY_ROOT)
        text = await self._store.get_text(self._workspace_id, path)
        if text is not None:
            return text
        entries = await self._store.list_paths(self._workspace_id, path)
        return "\n".join(entries) if entries else f"(empty: {path})"

    async def _create(self, inp: Dict[str, Any]) -> str:
        path = resolve_memory_path(inp.get("path"))
        await self._store.put_text(self._workspace_id, path, inp.get("file_text", ""))
        return f"Created {path}"

    async def _str_replace(self, inp: Dict[str, Any]) -> str:
        path = resolve_memory_path(inp.get("path"))
        text = await self._store.get_text(self._workspace_id, path) or ""
        old = inp.get("old_str", "")
        new = inp.get("new_str", "")
        if old and old not in text:
            return f"Error: old_str not found in {path}"
        await self._store.put_text(self._workspace_id, path, text.replace(old, new, 1))
        return f"Edited {path}"

    async def _insert(self, inp: Dict[str, Any]) -> str:
        path = resolve_memory_path(inp.get("path"))
        text = await self._store.get_text(self._workspace_id, path) or ""
        lines = text.split("\n")
        idx = int(inp.get("insert_line", 0) or 0)
        idx = max(0, min(idx, len(lines)))
        lines.insert(idx, inp.get("insert_text", ""))
        await self._store.put_text(self._workspace_id, path, "\n".join(lines))
        return f"Inserted into {path}"

    async def _delete(self, inp: Dict[str, Any]) -> str:
        path = resolve_memory_path(inp.get("path"))
        await self._store.delete(self._workspace_id, path)
        return f"Deleted {path}"

    async def _rename(self, inp: Dict[str, Any]) -> str:
        old = resolve_memory_path(inp.get("old_path") or inp.get("path"))
        new = resolve_memory_path(inp.get("new_path"))
        text = await self._store.get_text(self._workspace_id, old) or ""
        await self._store.put_text(self._workspace_id, new, text)
        await self._store.delete(self._workspace_id, old)
        return f"Renamed {old} -> {new}"


class DurableMemoryStoreBackend(MemoryStore):
    """:class:`MemoryStore` over the PRD-187 durable memory plane (§8-Q6 default).

    Delegates to the existing ``UnifiedMemoryService`` so the memory tool reuses
    the platform's durable store rather than opening a parallel file plane. Each
    memory "file" is a durable long-term entry tagged with its ``memory_path`` in
    metadata; exact-path get/list matches on that tag.

    §8-Q6 caveat (surfaced, not hidden): a *semantic* durable store is an
    imperfect substrate for the memory tool's *file* model — exact-path retrieval
    over vector search is best-effort, and ``delete`` depends on the store
    exposing a by-id/by-metadata delete. Gerard chooses the final substrate
    (durable Qdrant vs a scoped per-workspace scratch surface vs a fresh file
    plane); this adapter is the recommended reuse, swappable behind
    :class:`MemoryStore`.
    """

    _CATEGORY = "memory_tool"

    def __init__(self, service: Any = None) -> None:
        self._service = service

    def _svc(self) -> Any:
        if self._service is None:
            from modules.memory.unified_memory_service import UnifiedMemoryService

            self._service = UnifiedMemoryService.get_instance()
        return self._service

    @staticmethod
    def _entry_path(entry: Any) -> Optional[str]:
        md = (entry.get("metadata") or {}) if isinstance(entry, dict) else {}
        return md.get("memory_path")

    @staticmethod
    def _entry_text(entry: Any) -> str:
        if not isinstance(entry, dict):
            return ""
        return entry.get("content") or entry.get("memory") or entry.get("text") or ""

    async def get_text(self, workspace_id: Any, path: str) -> Optional[str]:
        results = await self._svc().search_long_term(
            workspace_id=workspace_id, query=path, limit=25
        )
        for entry in results or []:
            if self._entry_path(entry) == path:
                return self._entry_text(entry)
        return None

    async def put_text(self, workspace_id: Any, path: str, text: str) -> None:
        # Overwrite semantics: drop any prior entry at this path first.
        await self.delete(workspace_id, path)
        await self._svc().store_long_term(
            workspace_id=workspace_id,
            content=text,
            category=self._CATEGORY,
            metadata={"memory_path": path},
        )

    async def delete(self, workspace_id: Any, path: str) -> None:
        svc = self._svc()
        deleter = getattr(svc, "delete_long_term", None) or getattr(svc, "delete", None)
        if deleter is None:
            logger.info(
                "[memory-tool] durable store has no delete — leaving stale entry "
                "for %s (§8-Q6: substrate is Gerard's call)", path,
            )
            return
        try:
            results = await svc.search_long_term(workspace_id=workspace_id, query=path, limit=25)
            for entry in results or []:
                if self._entry_path(entry) == path and isinstance(entry, dict) and entry.get("id"):
                    await deleter(workspace_id=workspace_id, memory_id=entry["id"])
        except Exception as exc:
            logger.debug("[memory-tool] delete best-effort failed for %s: %s", path, exc)

    async def list_paths(self, workspace_id: Any, prefix: str) -> List[str]:
        results = await self._svc().search_long_term(
            workspace_id=workspace_id, query=prefix or MEMORY_ROOT, limit=50
        )
        paths = {
            p
            for entry in (results or [])
            if (p := self._entry_path(entry)) and p.startswith(prefix)
        }
        return sorted(paths)
