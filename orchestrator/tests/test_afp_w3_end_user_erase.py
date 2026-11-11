"""PRD-274 Wave 3 (§5.5): erasing an end user deletes everything they left (``DELETE /api/memory/end-users/{id}``).

Their namespace, their conversations (``chats.end_user_id``) with their messages, those conversations'
sessions, then their consent record. Another end user's conversations, the key's own and a member's stay.
Real database, real keys; the durable store and Redis are stand-ins.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from api import end_user_memory as eum
from api.chat import get_user_id
from core.auth.scopes import MEMORY_READ, MEMORY_WRITE
from modules.memory.unified_memory_service import MemoryNamespace


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.end_users import EndUser

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    EndUser.__table__.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128)"))
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w3-end-user-erase')"), {"id": ws})
    person = s.execute(text("INSERT INTO users (username, email) VALUES (:u, :e) RETURNING id"),
                       {"u": f"erase-{ws[:8]}", "e": f"erase-{ws[:8]}@example.com"}).scalar()
    s.commit()
    s.close()
    yield ws, person
    s = new_session.sweep()
    for table in ("chats", "end_users", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM users WHERE id = :u"), {"u": person})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


class _Redis:
    def __init__(self, fails=False):
        self.deleted, self.fails = [], fails

    def get_redis(self):
        return self

    def delete(self, *keys):
        if self.fails:
            raise ConnectionError("redis unreachable")
        self.deleted.extend(keys)
        return len(keys)


def _service(monkeypatch, redis):
    erased = []

    async def erase_namespace(user_id):
        erased.append(user_id)
        return 2

    fake = SimpleNamespace(_durable=SimpleNamespace(erase_namespace=erase_namespace), _get_redis=lambda: redis,
                           namespace=MemoryNamespace)
    monkeypatch.setattr("api.widget_memory._get_memory_service", lambda: fake)
    return erased


def _key(new_session, ws):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme support", key_type="server",
                                            permissions=[MEMORY_READ, MEMORY_WRITE])
    finally:
        s.close()
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=uuid.UUID(made["id"]),
                           user=SimpleNamespace(id=f"sdk:{made['id']}", email=None, clerk_user_id=None))


def _chat(s, ws, user_id, end_user_id):
    chat_id = s.execute(text(
        "INSERT INTO chats (id, user_id, workspace_id, title, visibility, end_user_id) VALUES (gen_random_uuid(), :u, "
        "CAST(:w AS uuid), 'a conversation', 'private', :e) RETURNING id::text"), {"u": user_id, "w": ws, "e": end_user_id}).scalar()
    s.execute(text("INSERT INTO messages (id, chat_id, workspace_id, role, parts) VALUES (gen_random_uuid(), "
                   "CAST(:c AS uuid), CAST(:w AS uuid), 'user', CAST('[]' AS json))"), {"c": chat_id, "w": ws})
    return chat_id


def _left(new_session, ws):
    s = new_session()
    try:
        chats = dict(s.execute(text("SELECT id::text, end_user_id FROM chats WHERE workspace_id = CAST(:w AS uuid)"),
                               {"w": ws}).all())
        messages = s.execute(text("SELECT count(*) FROM messages m JOIN chats c ON c.id = m.chat_id "
                                  "WHERE c.workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalar()
        records = s.execute(text("SELECT end_user_id FROM end_users WHERE workspace_id = CAST(:w AS uuid)"),
                            {"w": ws}).scalars().all()
        return chats, messages, records
    finally:
        s.close()


def _erase(new_session, ctx, end_user_id):
    s = new_session()
    try:
        return asyncio.run(eum.erase_end_user(end_user_id=end_user_id, ctx=ctx, db=s))
    finally:
        s.close()


def _setup(new_session, ws, person):
    key = _key(new_session, ws)
    s = new_session()
    service_user = get_user_id(s, key)
    for end_user in ("shopper-7", "shopper-8"):
        s.execute(text("INSERT INTO end_users (id, workspace_id, end_user_id) VALUES (gen_random_uuid(), "
                       "CAST(:w AS uuid), :e)"), {"w": ws, "e": end_user})
    theirs = [_chat(s, ws, service_user, "shopper-7"), _chat(s, ws, service_user, "shopper-7")]
    kept = [_chat(s, ws, service_user, "shopper-8"), _chat(s, ws, service_user, None), _chat(s, ws, person, None)]
    s.commit()
    s.close()
    return key, theirs, kept


def test_erasing_an_end_user_deletes_their_conversations_and_sessions(new_session, workspace, monkeypatch):
    ws, person = workspace
    redis = _Redis()
    erased = _service(monkeypatch, redis)
    key, theirs, kept = _setup(new_session, ws, person)
    done = _erase(new_session, key, "shopper-7")
    assert done == {"end_user_id": "shopper-7", "erased": True, "memories_deleted": 2, "chats_deleted": 2,
                    "sessions_cleared": True}
    assert erased == [f"mem:{ws}:end_user:shopper-7"]
    assert sorted(redis.deleted) == sorted(f"mem:session:{ws}:{c}" for c in theirs)
    chats, messages, records = _left(new_session, ws)
    assert set(chats) == set(kept) and messages == len(kept) and records == ["shopper-8"]


def test_sessions_redis_cant_reach_are_reported_not_hidden(new_session, workspace, monkeypatch):
    ws, person = workspace
    _service(monkeypatch, _Redis(fails=True))
    key, theirs, _ = _setup(new_session, ws, person)
    done = _erase(new_session, key, "shopper-7")
    assert (done["chats_deleted"], done["sessions_cleared"]) == (2, False)
    assert not set(theirs) & set(_left(new_session, ws)[0])
