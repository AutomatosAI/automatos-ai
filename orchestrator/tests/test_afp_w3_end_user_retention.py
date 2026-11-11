"""PRD-274 Wave 3 (§5.5): an end user's retention deletes what is older than their ``retention_days``.

Their facts made before the cutoff (retired ones too), their messages sent before it, and their conversations
it leaves empty that were opened before it. An end user with no retention, or whose erase has started, is left
as it is; one that fails is counted and the rest are still swept. Real database; the durable store is a stand-in
for the sweep, and the store's own prune is checked against a stand-in Qdrant client.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from services import end_user_retention

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


def test_the_store_prunes_a_namespace_of_what_was_made_before_the_cutoff():
    from modules.memory.durable_store import DurableMemoryStore

    old, new = (NOW - timedelta(days=40)).isoformat(), (NOW - timedelta(days=2)).isoformat()
    pages = {None: ([SimpleNamespace(id="old-1", payload={"created_at": old}),
                     SimpleNamespace(id="new-1", payload={"created_at": new})], "page-2"),
             "page-2": ([SimpleNamespace(id="old-2", payload={"created_at": "2026-09-01T08:00:00Z"}),
                         SimpleNamespace(id="undated", payload={}),
                         SimpleNamespace(id="garbled", payload={"created_at": "last tuesday"})], None)}
    seen = {"filters": [], "deleted": []}

    async def scroll(collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        seen["filters"].append(scroll_filter)
        return pages[offset]

    async def delete(collection_name, points_selector):
        seen["deleted"].extend(points_selector)

    store = object.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "memories"
    store._client = SimpleNamespace(scroll=scroll, delete=delete)
    namespace = "mem:ws:end_user:shopper-7"
    assert asyncio.run(store.prune_older_than(namespace, NOW - timedelta(days=30))) == 2
    assert seen["deleted"] == ["old-1", "old-2"]
    assert all(f.must[0].match.value == namespace for f in seen["filters"])


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.end_users import EndUser

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    EndUser.__table__.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128)"))
        conn.execute(text("ALTER TABLE end_users ADD COLUMN IF NOT EXISTS erasing_at TIMESTAMPTZ"))
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w3-retention')"), {"id": ws})
    person = s.execute(text("INSERT INTO users (username, email) VALUES (:u, :e) RETURNING id"),
                       {"u": f"retention-{ws[:8]}", "e": f"retention-{ws[:8]}@example.com"}).scalar()
    s.commit()
    s.close()
    yield ws, person
    s = new_session.sweep()
    for table in ("chats", "end_users"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM users WHERE id = :u"), {"u": person})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _end_user(s, ws, end_user_id, days=None, erasing=False):
    s.execute(text("INSERT INTO end_users (id, workspace_id, end_user_id, retention_days, erasing_at) VALUES "
                   "(gen_random_uuid(), CAST(:w AS uuid), :e, :d, CASE WHEN :x THEN now() END)"),
              {"w": ws, "e": end_user_id, "d": days, "x": erasing})


def _chat(s, ws, user_id, end_user_id, opened_days_ago, message_days_ago=()):
    chat_id = s.execute(text(
        "INSERT INTO chats (id, user_id, workspace_id, title, visibility, end_user_id, created_at, updated_at) VALUES "
        "(gen_random_uuid(), :u, CAST(:w AS uuid), 'a conversation', 'private', :e, "
        "now() - make_interval(days => :d), now() - make_interval(days => :d)) RETURNING id::text"),
        {"u": user_id, "w": ws, "e": end_user_id, "d": opened_days_ago}).scalar()
    for days in message_days_ago:
        s.execute(text("INSERT INTO messages (id, chat_id, workspace_id, role, parts, created_at) VALUES "
                       "(gen_random_uuid(), CAST(:c AS uuid), CAST(:w AS uuid), 'user', CAST('[]' AS json), "
                       "now() - make_interval(days => :d))"), {"c": chat_id, "w": ws, "d": days})
    return chat_id


def _messages(new_session, chat_id):
    s = new_session()
    try:
        return s.execute(text("SELECT count(*) FROM messages WHERE chat_id = CAST(:c AS uuid)"), {"c": chat_id}).scalar()
    finally:
        s.close()


def _chats(new_session, ws):
    s = new_session()
    try:
        return set(s.execute(text("SELECT id::text FROM chats WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalars())
    finally:
        s.close()


def _durable(monkeypatch, fails_for=()):
    pruned = []

    async def prune_older_than(namespace, cutoff):
        if any(namespace.endswith(f":end_user:{end_user}") for end_user in fails_for):
            raise ConnectionError("qdrant unreachable")
        pruned.append((namespace, cutoff))
        return 3

    service = SimpleNamespace(_durable=SimpleNamespace(prune_older_than=prune_older_than))
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service", lambda: service)
    return pruned


def test_the_sweep_deletes_what_is_older_than_each_end_users_retention(new_session, workspace, monkeypatch):
    ws, person = workspace
    pruned = _durable(monkeypatch)
    s = new_session()
    _end_user(s, ws, "kept-30", days=30)
    _end_user(s, ws, "kept-forever")
    _end_user(s, ws, "being-erased", days=30, erasing=True)
    gone = _chat(s, ws, person, "kept-30", 60, (50, 45))
    mixed = _chat(s, ws, person, "kept-30", 60, (50, 2))
    just_opened = _chat(s, ws, person, "kept-30", 0)
    others = [_chat(s, ws, person, "kept-forever", 60, (50,)), _chat(s, ws, person, "being-erased", 60, (50,)),
              _chat(s, ws, person, None, 60, (50,))]
    s.commit()
    s.close()
    totals = asyncio.run(end_user_retention.sweep_end_user_retention(now=datetime.now(timezone.utc)))
    mine = [(ns, cutoff) for ns, cutoff in pruned if ns.startswith(f"mem:{ws}:")]
    assert [ns for ns, _ in mine] == [f"mem:{ws}:end_user:kept-30"]
    assert timedelta(days=29, hours=23) < datetime.now(timezone.utc) - mine[0][1] < timedelta(days=30, hours=1)
    assert _chats(new_session, ws) == {mixed, just_opened, *others} and gone not in _chats(new_session, ws)
    assert _messages(new_session, mixed) == 1 and all(_messages(new_session, c) == 1 for c in others)
    assert totals["messages"] >= 3 and totals["chats"] >= 1 and totals["memories"] >= 3


def test_one_end_user_that_fails_is_counted_and_the_rest_are_swept(new_session, workspace, monkeypatch):
    ws, person = workspace
    _durable(monkeypatch, fails_for=("broken",))
    s = new_session()
    _end_user(s, ws, "broken", days=30)
    _end_user(s, ws, "fine", days=30)
    broken = _chat(s, ws, person, "broken", 60, (50,))
    fine = _chat(s, ws, person, "fine", 60, (50,))
    s.commit()
    s.close()
    totals = asyncio.run(end_user_retention.sweep_end_user_retention(now=datetime.now(timezone.utc)))
    assert totals["failed"] >= 1
    assert broken in _chats(new_session, ws) and _messages(new_session, broken) == 1
    assert fine not in _chats(new_session, ws)
