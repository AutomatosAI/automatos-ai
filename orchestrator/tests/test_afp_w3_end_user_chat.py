"""PRD-274 Wave 3 (§5.5): a server key's chat turn for one of its product's end users.

``POST /api/chat`` from a key may name ``endUserId``: the end user the turn is for. Their consent must be
on record (409 otherwise), and a person's turn names none (422). The turn runs as theirs
(``core.security.end_user_turn``). Their conversation is tagged with them, only turns for them continue
it, and a key's turn for no end user continues none of theirs. Real database, real keys.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import text

from api import chat, chat_server_key
from api.chat_models import ChatRequest, MessagePart
from core.auth.scopes import AGENTS_CHAT
from core.security.end_user_turn import EndUser, current_end_user

LEAN = {"runtime_profile": "lean"}


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.end_users import EndUser as EndUserRecord

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    EndUserRecord.__table__.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128)"))
        conn.execute(text("ALTER TABLE end_users ADD COLUMN IF NOT EXISTS erasing_at TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS model_allow_list JSONB"))
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS service_user_id INTEGER "
                          "REFERENCES users(id) ON DELETE SET NULL"))
    yield eng
    eng.dispose()


@pytest.fixture
def lean_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w3-end-user-chat')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Concierge', 'custom', CAST(:w AS uuid), 'active', CAST(:c AS jsonb)) RETURNING id"),
        {"w": ws, "c": json.dumps(LEAN)}).scalar()
    s.commit()
    s.close()
    yield ws, agent
    s = new_session.sweep()
    users = s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid) "
                           "AND service_user_id IS NOT NULL"), {"id": ws}).scalars().all()
    for table in ("chats", "end_users", "agents", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    if users:
        s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": users})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _key(new_session, ws):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme support", key_type="server", permissions=[AGENTS_CHAT])
    finally:
        s.close()
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=made["id"],
                           user=SimpleNamespace(id=f"sdk:{made['id']}", email=None, clerk_user_id=None))


def _consent(new_session, ws, end_user_id):
    s = new_session()
    s.execute(text("INSERT INTO end_users (id, workspace_id, end_user_id) VALUES (gen_random_uuid(), "
                   "CAST(:w AS uuid), :e)"), {"w": ws, "e": end_user_id})
    s.commit()
    s.close()


def _checked(new_session, ctx, **request):
    s = new_session()
    try:
        return chat._checked_lean_request(s, ctx, ChatRequest(message={"content": "Where is my order?"}, **request))
    finally:
        s.close()


def test_a_keys_turn_for_an_end_user_needs_their_consent(new_session, workspace, lean_on):
    ws, agent = workspace
    key = _key(new_session, ws)
    with pytest.raises(HTTPException) as refused:
        _checked(new_session, key, agentId=agent, endUserId="shopper-7")
    assert refused.value.status_code == 409 and "consent" in refused.value.detail
    _consent(new_session, ws, "shopper-7")
    end_user = _checked(new_session, key, agentId=agent, endUserId="shopper-7").end_user
    assert (end_user.workspace_id, end_user.end_user_id) == (ws, "shopper-7") and end_user.record_id  # their record
    assert _checked(new_session, key, agentId=agent).end_user is None


def test_a_persons_turn_names_no_end_user(new_session, workspace):
    ws, agent = workspace
    person = SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk", api_key_id=None,
                             user=SimpleNamespace(id="user_1", email="priya@example.com", clerk_user_id="user_1"))
    with pytest.raises(HTTPException) as refused:
        _checked(new_session, person, agentId=agent, endUserId="shopper-7")
    assert (refused.value.status_code, refused.value.detail) == (422, chat_server_key.NOT_A_PERSONS)


@pytest.mark.parametrize("bad", ["shopper 7", "shopper/7", "", "x" * 129])
def test_an_end_user_id_is_the_products_own_id(bad):
    with pytest.raises(ValidationError):
        ChatRequest(message={"content": "hi"}, endUserId=bad)


def _open(new_session, ctx, **request):
    from consumers.chatbot import ChatService

    s = new_session()
    try:
        user_id = chat.get_user_id(s, ctx)
        parts = [MessagePart(type="text", text="Where is my order?")]
        return chat._open_chat(s, ChatService(s), ChatRequest(message={"content": "x"}, **request), ctx, user_id, parts)
    finally:
        s.close()


def _tag_of(new_session, chat_id):
    s = new_session()
    try:
        return s.execute(text("SELECT end_user_id FROM chats WHERE id = CAST(:id AS uuid)"), {"id": chat_id}).scalar()
    finally:
        s.close()


def test_each_end_users_conversation_is_theirs_alone(new_session, workspace):
    ws, _ = workspace
    key = _key(new_session, ws)
    _consent(new_session, ws, "shopper-7")
    theirs = _open(new_session, key, endUserId="shopper-7")
    assert _tag_of(new_session, theirs) == "shopper-7"
    assert _open(new_session, key, id=theirs, endUserId="shopper-7") == theirs
    keys_own = _open(new_session, key)
    assert _tag_of(new_session, keys_own) is None
    for chat_id, end_user in ((theirs, "shopper-8"), (theirs, None), (keys_own, "shopper-7")):
        with pytest.raises(HTTPException) as refused:
            _open(new_session, key, id=chat_id, endUserId=end_user)
        assert (refused.value.status_code, refused.value.detail) == (403, chat_server_key.NOT_THIS_END_USERS_CHAT)


def test_the_turn_runs_as_its_end_user(monkeypatch):
    from consumers.chatbot.lean_request import LeanRequest
    from consumers.chatbot.service import StreamingChatService

    service = object.__new__(StreamingChatService)
    service.widget_mode, service.widget_scopes, service.widget_team, service.widget_agent_lock = False, None, None, None
    service.db, service.workspace_id = None, None
    seen = []

    async def scoped(*args, **kwargs):
        seen.append(current_end_user())
        yield "chunk"

    monkeypatch.setattr(service, "_stream_response_with_agent_scoped", scoped)
    monkeypatch.setattr(service, "_hidden_action_scope", lambda agent_id: nullcontext())

    async def drain(lean):
        return [c async for c in service.stream_response_with_agent("c1", [], 42, 9, lean_request=lean)]

    player = EndUser(str(uuid.uuid4()), "shopper-7")
    asyncio.run(drain(LeanRequest(end_user=player)))
    asyncio.run(drain(None))
    assert seen == [player, None] and current_end_user() is None


def test_a_new_conversation_is_saved_with_its_end_user_in_one_write(new_session, workspace, engine):
    from sqlalchemy import event

    ws, _ = workspace
    key = _key(new_session, ws)
    _consent(new_session, ws, "shopper-7")
    statements = []

    def seen(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()).upper())

    event.listen(engine, "before_cursor_execute", seen)
    try:
        theirs = _open(new_session, key, endUserId="shopper-7")
    finally:
        event.remove(engine, "before_cursor_execute", seen)
    inserts = [s for s in statements if s.startswith("INSERT INTO CHATS")]
    assert len(inserts) == 1 and "END_USER_ID" in inserts[0]
    assert not [s for s in statements if s.startswith("UPDATE CHATS")]
    assert _tag_of(new_session, theirs) == "shopper-7"
