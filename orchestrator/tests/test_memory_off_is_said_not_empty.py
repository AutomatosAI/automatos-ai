"""With no embedding provider, Auto says memory is off on this install, never that it is empty.

7 Nov 2026, a self-hosted install: the embedding provider was ``openrouter`` and no key was found,
so the embedding manager fell back to the degraded stand-in that refuses to embed. Every memory
write failed (346 errors in 72 h, only in the logs), every search found nothing, and Auto told its
owner "My memory is empty… the workspace holds no stored memories".

Now, when the memory store has no embedding provider:

* Auto's memory section (a chat turn that asked memory something) carries the Memory Is Off block;
* platform_search_memory, platform_browse_memories and platform_get_memory_stats answer that memory
  is off, with the reason code, when they find nothing; a read that found something is unchanged;
* platform_store_memory says the memory was NOT saved because memory is off.

A widget visitor's turn is never told, and with a working provider nothing changes. The memory
service is faked; the durable store's ``embeddings_ready`` is what is asked.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from core.llm.embedding_status import MEMORY_OFF, NO_EMBEDDING_PROVIDER
from modules.context.sections.base import SectionContext
from modules.context.sections.memory import MEMORY_OFF_HEADING, MemorySection

OFF_WORDS = "Memory is off on this install (no embedding provider)"
FOUND = {"id": "m1", "memory": "Wholesale minimum is 3 kg.", "score": 0.9, "created_at": "2026-11-01"}


class _Memory:
    """The memory service: what its durable store says of its embedder, and what it finds."""

    is_durable_configured = True

    def __init__(self, ready: bool, found=()):
        self._durable = NS(embeddings_ready=lambda: ready)
        self.found = list(found)

    async def search_long_term(self, workspace_id, query, agent_id=None, limit=10, **_):
        return list(self.found) if agent_id is None else []

    async def get_all_memories(self, workspace_id, agent_id=None, limit=20, **_):
        return list(self.found) if agent_id is None else []

    async def store_long_term(self, workspace_id, content, agent_id=None, metadata=None, **_):
        return {"success": False, "error": "store_long_term failed"}

    def namespace(self, workspace_id):
        return NS(resolve=lambda agent_id=None: f"ws:{workspace_id}")


@pytest.fixture
def memory(monkeypatch):
    """Install a memory service: ``memory(ready, found)``."""
    from modules.memory import unified_memory_service

    def install(ready: bool, found=()):
        service = _Memory(ready, found)
        monkeypatch.setattr(unified_memory_service, "get_unified_memory_service", lambda: service)
        return service

    return install


# ── Auto's context ──────────────────────────────────────────────────────────

def _render(widget=False, query="What do you remember about me?") -> str:
    ctx = SectionContext(agent=NS(id=322, name="Auto"), workspace_id="ws", widget_mode=widget,
                         messages=[{"role": "user", "content": query}], kwargs={})
    with patch.object(MemorySection, "_try_context_router", return_value=None), \
            patch.object(MemorySection, "_build_from_smart_memory", return_value=""):
        return asyncio.run(MemorySection().render(ctx))


def test_the_memory_section_says_memory_is_off_not_empty(memory):
    memory(ready=False)

    rendered = _render()

    assert rendered == f"{MEMORY_OFF_HEADING}\n{MEMORY_OFF}"
    assert OFF_WORDS in rendered and "never that it is empty" in rendered


def test_a_working_store_that_recalls_nothing_adds_no_block(memory):
    memory(ready=True)

    assert _render() == ""


def test_a_switched_off_store_is_not_called_off_for_want_of_a_provider(memory):
    """No QDRANT_URL: memory is off for another reason, which the tools already word."""
    memory(ready=False)._durable.enabled = False

    assert _render() == ""


def test_a_widget_visitor_is_never_told(memory):
    memory(ready=False)

    assert _render(widget=True) == ""


# ── Auto's memory tools ─────────────────────────────────────────────────────

def _assert_off(answer):
    assert answer["success"] is False and answer["memory_off"] is True
    assert answer["reason_code"] == NO_EMBEDDING_PROVIDER
    assert OFF_WORDS in answer["error"] and "empty" not in answer["error"].split("never")[0]


def test_search_with_no_provider_says_memory_is_off(memory, db_session, seed_workspace):
    from modules.tools.discovery.handlers_search import search_memory

    memory(ready=False)

    _assert_off(asyncio.run(search_memory(db_session, UUID(seed_workspace()), {"query": "wholesale"})))


def test_browse_and_stats_with_no_provider_say_memory_is_off(memory, db_session, seed_workspace):
    from modules.tools.discovery.handlers_search import browse_memories
    from modules.tools.discovery.handlers_workspace import get_memory_stats

    memory(ready=False)
    ws = UUID(seed_workspace())

    _assert_off(asyncio.run(browse_memories(db_session, ws, {})))
    _assert_off(asyncio.run(get_memory_stats(db_session, ws, {})))


def test_a_store_that_failed_with_no_provider_says_not_saved_because_memory_is_off(memory):
    from modules.tools.discovery.handlers_workspace import store_memory

    memory(ready=False)

    answer = asyncio.run(store_memory(None, uuid4(), {"content": "We post on Tuesdays."}))

    _assert_off(answer)
    assert answer["error"].startswith("Memory NOT saved.")


def test_with_a_provider_an_empty_search_is_still_an_empty_search(memory, db_session, seed_workspace):
    from modules.tools.discovery.handlers_search import search_memory

    memory(ready=True)

    answer = asyncio.run(search_memory(db_session, UUID(seed_workspace()), {"query": "wholesale"}))

    assert answer["success"] is True and answer["total"] == 0 and "memory_off" not in answer


def test_a_read_that_found_something_is_never_replaced(memory, db_session, seed_workspace):
    from modules.tools.discovery.handlers_search import browse_memories

    memory(ready=False, found=[FOUND])

    answer = asyncio.run(browse_memories(db_session, UUID(seed_workspace()), {}))

    assert answer["success"] is True and answer["total"] == 1


def test_a_refusal_of_the_calls_own_parameters_stands(memory, db_session, seed_workspace):
    from modules.tools.discovery.handlers_search import search_memory

    memory(ready=False)

    answer = asyncio.run(search_memory(db_session, UUID(seed_workspace()), {"query": " "}))

    assert answer == {"success": False, "error": "query parameter is required"}
