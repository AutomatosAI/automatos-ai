"""PRD-274 Wave 3 (§5.5): what an end user's turn recalls is theirs and the workspace's, never another's.

Wherever their turn reads the workspace tier it reads their namespace with it; a read of what they own
(their standing rules) reads their namespace; the turn's subject is theirs, not the key's shared service
user's; the caches key on them; and the memory section reads no other conversation and no daily log.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.security.end_user_turn import EndUser, end_user_turn

WS = "11111111-1111-1111-1111-111111111111"
WORKSPACE, AGENT = f"mem:{WS}", f"mem:{WS}:agent:42"
PLAYER, OTHER = EndUser(WS, "player-7"), EndUser(WS, "player-8")
THEIRS = f"mem:{WS}:end_user:player-7"


def _store():
    """The real durable store, its Qdrant calls replaced by recorders of the filter each one is given."""
    from modules.memory.durable_store import DurableMemoryStore

    store = object.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done = True, True
    filters = []

    async def query(query, flt, limit):
        filters.append(flt)
        return []

    async def scroll(flt, limit):
        filters.append(flt)
        return []

    store._query, store._record_search = query, (lambda *args, **kwargs: None)
    store._scroll = scroll
    return store, filters


def _namespaces(flt) -> set:
    condition = next(c for c in flt.must if c.key == "namespace")
    match = condition.match
    return set(getattr(match, "any", None) or [match.value])


def test_a_search_of_the_workspace_reads_their_namespace_with_it():
    store, filters = _store()
    with end_user_turn(PLAYER):
        asyncio.run(store.search("night shift", WORKSPACE, workspace_id=WS))
        asyncio.run(store.search("night shift", AGENT, workspace_id=WS, also_namespaces=(WORKSPACE,)))
        asyncio.run(store.search("night shift", AGENT, workspace_id=WS))
    assert [_namespaces(f) for f in filters] == [{WORKSPACE, THEIRS}, {AGENT, WORKSPACE, THEIRS}, {AGENT}]


def test_no_other_turn_reads_an_end_users_namespace():
    store, filters = _store()
    asyncio.run(store.search("night shift", WORKSPACE, workspace_id=WS))
    with end_user_turn(OTHER):
        asyncio.run(store.search("night shift", WORKSPACE, workspace_id=WS))
    assert THEIRS not in _namespaces(filters[0]) | _namespaces(filters[1])


def test_their_own_rows_are_read_from_their_namespace():
    from modules.memory.injection_filter import STANDING_RULE_FILTER, STANDING_RULE_OWNER_KEY

    store, filters = _store()
    with end_user_turn(PLAYER):
        asyncio.run(store.get_where_any(WORKSPACE, STANDING_RULE_FILTER, where={STANDING_RULE_OWNER_KEY: "end_user:player-7"}))
        asyncio.run(store.get_where_any(WORKSPACE, STANDING_RULE_FILTER, where={STANDING_RULE_OWNER_KEY: "user:9"}))
        with pytest.raises(ValueError):
            asyncio.run(store.get_where_any("mem:22222222-2222-2222-2222-222222222222", STANDING_RULE_FILTER,
                                            where={STANDING_RULE_OWNER_KEY: "end_user:player-7"}))
    assert [_namespaces(f) for f in filters] == [{THEIRS}, {WORKSPACE}]


def test_the_turns_subject_is_theirs_not_the_keys_service_user():
    from consumers.chatbot.service import StreamingChatService

    service = object.__new__(StreamingChatService)
    service.widget_mode = False
    service._bind_turn_person(9)
    assert service._viewer_subject_id == "user:9"
    with end_user_turn(PLAYER):
        service._bind_turn_person(9)
        assert service._viewer_subject_id == "end_user:player-7"

        stored = {}

        async def store(user_message, assistant_response, chat_id, subject_id=None, receipts=None):
            stored["subject_id"] = subject_id
            return False

        service._smart_chat = SimpleNamespace(store=store)

        async def drain():
            return [chunk async for chunk in service._remember_the_turn("I work nights now", "Noted.", "c1", 9)]

        asyncio.run(drain())
    assert stored["subject_id"] == "end_user:player-7"


def test_the_caches_key_on_the_end_user():
    from consumers.chatbot.smart_memory import SmartMemoryManager
    from modules.memory.unified_memory_service import MemoryNamespace

    manager = object.__new__(SmartMemoryManager)
    keys = []
    for turn in (None, PLAYER, OTHER):
        with end_user_turn(turn):
            keys.append((MemoryNamespace(WS).cache_key(None, "q"), manager._get_cache_key(WS, 42, "q", "s")))
    assert len({k[0] for k in keys}) == 3 and len({k[1] for k in keys}) == 3


def test_the_memory_section_reads_no_other_conversation_on_their_turn():
    from modules.context.sections.memory import MemorySection

    section = object.__new__(MemorySection)
    chat_turn = SimpleNamespace(widget_mode=False, context_mode="lean", kwargs={})
    assert section._reads_the_workspace(chat_turn) is True
    with end_user_turn(PLAYER):
        assert section._reads_the_workspace(chat_turn) is False
