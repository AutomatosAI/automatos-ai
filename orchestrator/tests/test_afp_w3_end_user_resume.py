"""PRD-274 Wave 3 (§5.7): "where were we", for one of a product's end users (``GET /api/memory/end-users/{id}/resume``).

Their own recent conversations with each one's checkpoint, and the decisions and open loops in their facts and
the workspace's, read as them: a private fact of theirs is shown, a person's private fact and, for a key, an
audience-limited one are not, and no other end user's conversation or fact ever is. Real database, real keys;
the memory service is a stand-in.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from api import end_user_memory as eum
from api.chat import get_user_id
from core.auth.scopes import MEMORY_READ, MEMORY_WRITE


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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w3-resume')"), {"id": ws})
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    users = s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid) "
                           "AND service_user_id IS NOT NULL"), {"id": ws}).scalars().all()
    for table in ("chats", "end_users", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    if users:
        s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": users})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _fact(fid, text_, fact_type, **meta):
    return {"id": fid, "memory": text_, "created_at": f"2026-11-{fid[-2:]}T10:00:00+00:00",
            "metadata": {"type": fact_type, **meta}}


THEIRS = [_fact("their-10", "Switched to the yearly plan.", "decision"),
          _fact("their-11", "Wants the invoice re-sent.", "open_loop", scope="private", owner="end_user:shopper-7")]
WORKSPACE = [_fact("ws-09", "Refunds go out on Fridays.", "decision"),
             _fact("ws-08", "A member's own reminder.", "decision", scope="private", owner="user:1"),
             _fact("ws-07", "Margins for the finance team.", "decision", audience=["finance"])]


@pytest.fixture
def memory(monkeypatch):
    read = []

    async def get_all_memories_scoped(user_id, limit=100, workspace_id=None):
        read.append(user_id)
        return [dict(m) for m in THEIRS] if user_id.endswith(":end_user:shopper-7") else [_fact("other-12", "x", "decision")]

    async def get_all_memories(workspace_id, agent_id=None, limit=100):
        return [dict(m) for m in WORKSPACE]

    service = SimpleNamespace(is_durable_configured=True, get_all_memories_scoped=get_all_memories_scoped,
                              get_all_memories=get_all_memories)
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service", lambda: service)
    return read


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


def _setup(new_session, ws, key):
    s = new_session()
    service_user = get_user_id(s, key)
    for end_user in ("shopper-7", "shopper-8"):
        s.execute(text("INSERT INTO end_users (id, workspace_id, end_user_id) VALUES (gen_random_uuid(), "
                       "CAST(:w AS uuid), :e)"), {"w": ws, "e": end_user})
    chats = {}
    for name, end_user, step in (("billing", "shopper-7", "Re-send the invoice"), ("delivery", "shopper-7", None),
                                 ("theirs", "shopper-8", "Not for shopper-7"), ("keys-own", None, None)):
        summary = json.dumps({"topic": name, "next_step": step}) if step else None
        chats[name] = s.execute(text(
            "INSERT INTO chats (id, user_id, workspace_id, title, visibility, end_user_id, summary) VALUES "
            "(gen_random_uuid(), :u, CAST(:w AS uuid), :t, 'private', :e, CAST(:s AS jsonb)) RETURNING id::text"),
            {"u": service_user, "w": ws, "t": name, "e": end_user, "s": summary}).scalar()
    s.commit()
    s.close()
    return chats


def _resume(new_session, ctx, end_user_id):
    s = new_session()
    try:
        return asyncio.run(eum.resume_end_user(end_user_id=end_user_id, ctx=ctx, db=s))
    finally:
        s.close()


def test_where_they_left_off_is_theirs_and_the_workspaces_never_another_end_users(new_session, workspace, memory):
    key = _key(new_session, workspace)
    chats = _setup(new_session, workspace, key)
    resumed = _resume(new_session, key, "shopper-7")
    assert resumed["end_user_id"] == "shopper-7"
    assert {t["chat_id"] for t in resumed["threads"]} == {chats["billing"], chats["delivery"]}
    assert resumed["suggested_next_steps"] == ["Re-send the invoice"]
    assert [d["id"] for d in resumed["recent_decisions"]] == ["their-10", "ws-09"]
    assert [o["id"] for o in resumed["open_loops"]] == ["their-11"]
    assert memory == [f"mem:{workspace}:end_user:shopper-7"]


def test_an_unknown_end_user_is_not_found_and_one_being_erased_is_refused(new_session, workspace, memory):
    key = _key(new_session, workspace)
    _setup(new_session, workspace, key)
    with pytest.raises(HTTPException) as unknown:
        _resume(new_session, key, "shopper-9")
    assert unknown.value.status_code == 404
    s = new_session()
    s.execute(text("UPDATE end_users SET erasing_at = now() WHERE workspace_id = CAST(:w AS uuid) "
                   "AND end_user_id = 'shopper-7'"), {"w": workspace})
    s.commit()
    s.close()
    with pytest.raises(HTTPException) as erasing:
        _resume(new_session, key, "shopper-7")
    assert erasing.value.status_code == 409 and memory == []

