"""PRD-274 Wave 1 (§5.2, §8-5, §8-10): a product's server key chats with a lean agent.

``POST /api/chat`` takes a member with ``agents:execute``, as before, or a server key listing
``agents:chat``. A key's turn runs as its own non-login service user, never user 1 (the owner), so
it owns its own chats and no admin-only tool is its. It chats only with a lean agent of its
workspace, in no person's mode, and a model it names must be on the key's allow-list and the
agent's. Its chat turns count towards its tokens a minute. Real database, real keys.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import text

from api import chat, chat_server_key
from api.chat_models import ChatRequest
from core.auth.scopes import AGENTS_CHAT

ALLOWED = [{"provider": "anthropic", "model_id": "claude-haiku-5-5"}]
LEAN = {"runtime_profile": "lean", "model_allow_list": ALLOWED + [{"provider": "anthropic", "model_id": "claude-sonnet-5-5"}]}


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:  # a test database built before the columns
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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-headless-chat')"), {"id": ws})
    agents = {}
    for name, configuration in (("GM", LEAN), ("Generalist", {})):
        agents[name] = s.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS jsonb)) RETURNING id"),
            {"n": name, "w": ws, "c": json.dumps(configuration)}).scalar()
    s.commit()
    s.close()
    yield ws, agents
    s = new_session.sweep()
    users = s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid) "
                           "AND service_user_id IS NOT NULL"), {"id": ws}).scalars().all()
    for table in ("llm_usage", "chats", "agents", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    if users:
        s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": users})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _key(new_session, ws, model_allow_list=ALLOWED):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme game", key_type="server",
                                            permissions=[AGENTS_CHAT], model_allow_list=model_allow_list)
    finally:
        s.close()
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=made["id"],
                           user=SimpleNamespace(id=f"sdk:{made['id']}", email=None, clerk_user_id=None,
                                                system_role="service"))


def _ask(new_session, ctx, **request):
    s = new_session()
    try:
        return chat._checked_lean_request(s, ctx, ChatRequest(message={"content": "Roll for the gate."}, **request))
    finally:
        s.close()


def _refused(call):
    with pytest.raises(HTTPException) as refused:
        call()
    return refused.value.status_code, refused.value.detail


def test_a_key_runs_as_its_own_service_user_never_the_owner(new_session, workspace):
    from core.models.core import User

    ws, _ = workspace
    key, other = _key(new_session, ws), _key(new_session, ws)
    s = new_session()
    try:
        first, again, theirs = chat.get_user_id(s, key), chat.get_user_id(s, key), chat.get_user_id(s, other)
        user = s.get(User, first)
        assert first == again and first != theirs and first != 1
        assert (user.clerk_user_id, user.username, user.email.endswith("@keys.automatos.invalid")) == (
            None, f"sdk-{key.api_key_id}", True)
        member = s.execute(text("SELECT count(*) FROM workspace_members WHERE user_id = :u"), {"u": first}).scalar()
        assert member == 0
    finally:
        s.close()


def test_a_key_chats_only_with_a_lean_agent_of_its_workspace_in_no_persons_mode(new_session, workspace, lean_on):
    ws, agents = workspace
    key = _key(new_session, ws)
    assert _refused(lambda: _ask(new_session, key))[0] == 422                                  # no agent
    assert _refused(lambda: _ask(new_session, key, agentId=agents["Generalist"]))[0] == 422    # not lean
    stranger = _key(new_session, ws)
    stranger.workspace_id = uuid.uuid4()
    assert _refused(lambda: _ask(new_session, stranger, agentId=agents["GM"]))[0] == 422       # another workspace's
    for mode in ("missionMode", "planMode", "spokenMode"):
        assert _refused(lambda mode=mode: _ask(new_session, key, agentId=agents["GM"], **{mode: True})) == (
            422, chat_server_key.NOT_A_KEYS_MODE)
    assert _ask(new_session, key, agentId=agents["GM"]).model is None                          # the agent's own model


def test_a_model_a_key_names_must_be_on_the_keys_list_and_the_agents(new_session, workspace, lean_on):
    ws, agents = workspace
    key, listless = _key(new_session, ws), _key(new_session, ws, model_allow_list=None)
    assert _ask(new_session, key, agentId=agents["GM"], model="claude-haiku-5-5").model.model == "claude-haiku-5-5"
    status, detail = _refused(lambda: _ask(new_session, key, agentId=agents["GM"], model="claude-sonnet-5-5"))
    assert status == 422 and "key's model allow-list" in detail                                # the agent's, not the key's
    assert _refused(lambda: _ask(new_session, listless, agentId=agents["GM"], model="claude-haiku-5-5"))[0] == 422


def test_a_member_is_checked_as_before(new_session, workspace, lean_on):
    ws, agents = workspace
    member = SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                             user=SimpleNamespace(id=1, email=None, clerk_user_id=None))
    assert _ask(new_session, member, agentId=agents["GM"], model="claude-sonnet-5-5").model.model == "claude-sonnet-5-5"
    assert _ask(new_session, member).model is None


def test_a_keys_chat_turns_count_towards_its_tokens_a_minute(new_session, workspace):
    from services import key_limits

    ws, _ = workspace
    key = _key(new_session, ws)
    s = new_session()
    owner = chat.get_user_id(s, key)
    chat_id = s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility) VALUES (gen_random_uuid(), "
                             ":u, CAST(:w AS uuid), 'game', 'private') RETURNING id"), {"u": owner, "w": ws}).scalar()
    s.execute(text("INSERT INTO llm_usage (workspace_id, model_id, provider, tier, execution_id, input_tokens, "
                   "output_tokens, total_tokens, input_cost, output_cost, total_cost, created_at) VALUES "
                   "(CAST(:w AS uuid), 'm', 'p', 'direct', :ex, 0, 700, 700, 0, 0, 0, now())"),
              {"w": ws, "ex": f"chat:{chat_id}"})
    s.commit()
    assert key_limits.tokens_last_minute(s, ws, key.user.id, key.api_key_id) == 700
    s.close()


def test_the_chat_route_takes_agents_chat_and_keys_carry_a_checked_allow_list():
    from fastapi.routing import APIRoute, iter_route_contexts

    from api.api_keys import VALID_PERMISSIONS, ApiKeyCreateRequest
    from core.auth.server_key_gate import KEY_SCOPE_MARKER_ATTR
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app, limiter

    route = next(r for r in iter_route_contexts(app.routes)
                 if isinstance(r.original_route, APIRoute) and r.path == "/api/chat" and "POST" in r.methods)
    marked = [(getattr(d.call, PERMISSION_MARKER_ATTR, None), getattr(d.call, KEY_SCOPE_MARKER_ATTR))
              for d in route.dependant.dependencies if getattr(d.call, KEY_SCOPE_MARKER_ATTR, None)]
    assert marked == [("agents:execute", AGENTS_CHAT)] and "api.chat.stream_chat" in limiter._exempt_routes
    assert AGENTS_CHAT in VALID_PERMISSIONS
    assert ApiKeyCreateRequest(name="g", model_allow_list=ALLOWED).model_allow_list == ALLOWED
    with pytest.raises(ValidationError):
        ApiKeyCreateRequest(name="g", model_allow_list=[{"model_id": "claude-haiku-5-5"}])
