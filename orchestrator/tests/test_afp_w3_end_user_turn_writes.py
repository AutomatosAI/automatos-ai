"""PRD-274 Wave 3 (§5.5): what an end user's turn writes stays theirs (``core.security.end_user_turn``).

During an end user's turn every durable write lands in their namespace, tagged with their subject, and
nothing is written where the whole workspace reads: no L2 row (so no ``mem:{ws}:l2`` mirror) and no daily
log. On any other turn every write goes where it always has.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.security.end_user_turn import ERASED, EndUser, current_end_user, end_user_key, end_user_turn, routed

WS = "11111111-1111-1111-1111-111111111111"
PLAYER = EndUser(WS, "player-7")


@pytest.fixture(autouse=True)
def consent(monkeypatch):
    """The end user's consent, as the write checks it before and after: on record unless a test says."""
    answers = []

    async def consented(end_user):
        return answers.pop(0) if answers else True

    monkeypatch.setattr("core.security.end_user_turn._consented", consented)
    return answers


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


def test_no_write_is_made_once_their_erase_has_started(consent):
    store, seen = _captured_add()
    consent.append(False)
    with end_user_turn(PLAYER):
        answer = asyncio.run(store.add([{"role": "user", "content": "x"}], user_id=f"mem:{WS}", workspace_id=WS))
    assert answer == {"success": False, "skipped": ERASED} and seen == {}


def test_a_write_their_erase_overtook_is_taken_back(consent):
    store, seen = _captured_add()
    deleted = []

    async def delete(memory_ids, user_id, workspace_id=None):
        deleted.append((tuple(memory_ids), user_id))
        return True

    store.delete = delete
    consent.extend([True, False])  # on record before the write; erasing once it landed
    with end_user_turn(PLAYER):
        answer = asyncio.run(store.add([{"role": "user", "content": "x"}], user_id=f"mem:{WS}", workspace_id=WS))
    assert seen["user_id"] == f"mem:{WS}:end_user:player-7"
    assert deleted == [(("existing",), f"mem:{WS}:end_user:player-7")] and answer["skipped"] == ERASED



def _session_service():
    """The real ``update_session`` on a dict standing in for Redis."""
    from modules.memory.unified_memory_service import UnifiedMemoryService

    service = object.__new__(UnifiedMemoryService)
    saved = {}
    conn = SimpleNamespace(get=saved.get, setex=lambda key, ttl, value: saved.__setitem__(key, value),
                           delete=lambda *keys: [saved.pop(k, None) for k in keys])
    service._get_redis = lambda: SimpleNamespace(get_redis=lambda: conn)
    return service, saved


SESSION = f"mem:session:{WS}:chat-1"


def _update(service):
    asyncio.run(service.update_session(workspace_id=WS, conversation_id="chat-1", user_msg="hi", assistant_msg="hello"))


def test_their_session_is_written_while_their_consent_counts():
    service, saved = _session_service()
    with end_user_turn(PLAYER):
        _update(service)
    assert "User: hi" in saved[SESSION]


def test_no_session_is_written_once_their_erase_has_started(consent):
    service, saved = _session_service()
    consent.append(False)
    with end_user_turn(PLAYER):
        _update(service)
    assert saved == {}


def test_a_session_their_erase_overtook_is_cleared(consent):
    service, saved = _session_service()
    consent.extend([True, False])  # on record before the write; erasing once it landed
    with end_user_turn(PLAYER):
        _update(service)
    assert saved == {}


def test_any_other_turns_session_is_written_as_always(consent):
    service, saved = _session_service()
    consent.extend([False, False])  # never asked: no end user's turn
    _update(service)
    assert SESSION in saved and consent == [False, False]


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr("core.security.end_user_turn.CLEAR_RETRY_SECONDS", 0)
    recorded = []
    monkeypatch.setattr("core.security.end_user_turn._record_left_session", lambda *a: recorded.append(a))
    return recorded


def test_a_consent_check_that_fails_after_the_write_clears_the_session_and_raises(monkeypatch):
    service, saved = _session_service()
    answers = iter([True])

    async def consented(end_user):
        try:
            return next(answers)
        except StopIteration:
            raise ConnectionError("database unreachable") from None

    monkeypatch.setattr("core.security.end_user_turn._consented", consented)
    with end_user_turn(PLAYER), pytest.raises(ConnectionError):
        _update(service)
    assert saved == {}


def test_a_durable_write_whose_consent_check_fails_is_taken_back_and_raised(monkeypatch):
    store, seen = _captured_add()
    deleted = []

    async def delete(memory_ids, user_id, workspace_id=None):
        deleted.append(tuple(memory_ids))

    async def consented(end_user):
        if seen:
            raise ConnectionError("database unreachable")
        return True

    store.delete = delete
    monkeypatch.setattr("core.security.end_user_turn._consented", consented)
    with end_user_turn(PLAYER), pytest.raises(ConnectionError):
        asyncio.run(store.add([{"role": "user", "content": "x"}], user_id=f"mem:{WS}", workspace_id=WS))
    assert deleted == [("existing",)]


def _flaky(service, saved, failures):
    """Redis's delete fails ``failures`` times, then works."""
    conn = service._get_redis().get_redis()
    tries = []

    def delete(*keys):
        tries.append(keys)
        if len(tries) <= failures:
            raise ConnectionError("redis unreachable")
        return [saved.pop(k, None) for k in keys]

    service._get_redis = lambda: SimpleNamespace(get_redis=lambda: SimpleNamespace(get=conn.get, setex=conn.setex,
                                                                                    delete=delete))
    return tries


def test_a_session_clear_that_fails_is_tried_again(consent, no_wait):
    service, saved = _session_service()
    tries = _flaky(service, saved, failures=2)
    consent.extend([True, False])
    with end_user_turn(PLAYER):
        _update(service)
    assert saved == {} and len(tries) == 3 and no_wait == []


def test_a_session_that_cant_be_cleared_is_recorded_and_raised(consent, no_wait):
    service, saved = _session_service()
    tries = _flaky(service, saved, failures=99)
    consent.extend([True, False])
    with end_user_turn(PLAYER), pytest.raises(ConnectionError):
        _update(service)
    assert SESSION in saved and len(tries) == 3 and len(no_wait) == 1 and no_wait[0][1] == SESSION
