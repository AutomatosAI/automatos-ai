"""PRD-274 Wave 3 (§5.5): what an end user's turn writes stays theirs (``core.security.end_user_turn``).

During an end user's turn every durable write lands in their namespace, tagged with their subject, and
nothing is written where the whole workspace reads: no L2 row (so no ``mem:{ws}:l2`` mirror) and no daily
log. On any other turn every write goes where it always has.
"""
from __future__ import annotations

import asyncio

import pytest

from core.security.end_user_turn import EndUser, current_end_user, end_user_key, end_user_turn, routed

WS = "11111111-1111-1111-1111-111111111111"
PLAYER = EndUser(WS, "player-7")


def _captured_add():
    """The real durable store's ``add``, stopped at its dedup lookup, which sees the final namespace."""
    from modules.memory.durable_store import DurableMemoryStore

    store = object.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done = True, True
    seen = {}

    async def deduped(user_id, content_hash, metadata):
        seen.update(user_id=user_id, metadata=metadata)
        return {"success": True, "id": "existing"}

    store._deduped = deduped
    return store, seen


@pytest.mark.parametrize("namespace", [f"mem:{WS}", f"mem:{WS}:agent:42"])
def test_an_end_users_turn_writes_to_their_namespace(namespace):
    store, seen = _captured_add()
    with end_user_turn(PLAYER):
        asyncio.run(store.add([{"role": "user", "content": "prefers the night shift"}], user_id=namespace,
                              metadata={"type": "preference"}, workspace_id=WS, subject_id="user:9"))
    assert seen["user_id"] == f"mem:{WS}:end_user:player-7"
    assert seen["metadata"] == {"type": "preference", "end_user_id": "player-7"}


def test_any_other_turn_writes_where_it_always_has():
    store, seen = _captured_add()
    asyncio.run(store.add([{"role": "user", "content": "x"}], user_id=f"mem:{WS}", workspace_id=WS))
    assert seen["user_id"] == f"mem:{WS}" and current_end_user() is None


def test_the_write_carries_their_subject_and_never_leaves_the_workspace():
    assert routed(PLAYER, f"mem:{WS}:agent:42", None) == {
        "user_id": f"mem:{WS}:end_user:player-7", "subject_id": "end_user:player-7",
        "metadata": {"end_user_id": "player-7"}}
    owned = routed(PLAYER, f"mem:{WS}", {"owner": "user:9", "scope": "private"})["metadata"]
    assert owned == {"owner": "end_user:player-7", "scope": "private", "end_user_id": "player-7"}
    for elsewhere in ("mem:22222222-2222-2222-2222-222222222222", f"mem:{WS}0", "agent:42"):
        with pytest.raises(ValueError):
            routed(PLAYER, elsewhere, {})


def test_nothing_goes_where_the_whole_workspace_reads(monkeypatch):
    from consumers.chatbot.smart_memory import SmartMemoryManager
    from modules.memory.unified_memory_service import UnifiedMemoryService

    service = object.__new__(UnifiedMemoryService)

    def never(*args, **kwargs):
        raise AssertionError("written during an end user's turn")

    service._store_short_term_sync = never
    service._durable = type("Durable", (), {"add": never})()
    manager = object.__new__(SmartMemoryManager)
    with end_user_turn(PLAYER):
        assert asyncio.run(service.store_short_term(WS, "a transcript", content_type="exchange")) is None
        assert asyncio.run(service.store_daily_log(WS, "today")) == {
            "success": False, "skipped": "an end user's turn: nothing is shared with the workspace"}
        assert asyncio.run(manager.store_daily_summary(WS, "hello there", "hi")) is False


def test_the_cache_key_names_the_end_user_only_on_their_turn():
    assert end_user_key() == ""
    with end_user_turn(PLAYER):
        assert end_user_key() == "eu=player-7" and current_end_user() == PLAYER
        with end_user_turn(None):
            assert end_user_key() == ""
    assert current_end_user() is None
