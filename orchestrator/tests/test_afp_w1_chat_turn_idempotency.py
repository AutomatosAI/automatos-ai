"""PRD-274 Wave 1 (§5.2): ``Idempotency-Key`` on a product's server key's chat turn (``POST /api/chat``).

A key's turn sent with the header runs once in a window (``api.chat_turn_keys``). A repeat of a turn that
answered is answered with its saved answer, framed as the turn framed it, and runs nothing: no model call,
no memory write, no message saved. A repeat while it runs, and the same key on a different request, are
409s; a malformed key is a 422. A failed or dead turn may run again, a turn refused before it started
leaves its key free, and a key past the window runs as new (its expired rows are deleted on use). A
person's turn, and a key's turn without the header, run as before.

The route runs as in production with its model stood in for: a streaming service that answers, saves its
answer as the real one does, then counts what the real one would remember. The detached turn and its
end-of-turn hooks are the real ones. Real database, real keys.
"""
from __future__ import annotations

import asyncio
import functools
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from api import agent_execute, chat, chat_server_key, chat_turn_keys
from api.chat_models import ChatRequest
from consumers.chatbot import ChatService
from consumers.chatbot.model_history import NO_ANSWER_SENTENCES
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.turn_errors import TURN_ERROR_ORIGIN
from core.auth.scopes import AGENTS_CHAT
from core.models.chat_turn_keys import DONE, FAILED, RUNNING, UNIQUE_INDEX, ChatTurnKey
from services import chat_turns

LEAN = {"runtime_profile": "lean"}
ASKED = "Roll for the gate."
ANSWER = "The gate opens on a 14."
KEY = "turn-0042"
MIGRATION = "chat_turn_keys_20261116"
VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:  # a test database built before these
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS model_allow_list JSONB"))
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS service_user_id INTEGER "
                          "REFERENCES users(id) ON DELETE SET NULL"))
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128)"))
    ChatTurnKey.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def lean_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)


@pytest.fixture
def workspace(new_session, lean_on):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-chat-turn-keys')"), {"id": ws})
    agent = s.execute(text(  # a key names its agent by public id (api.chat_server_key.with_its_agents_id)
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, public_id) "
        "VALUES ('GM', 'custom', CAST(:w AS uuid), 'active', CAST(:c AS jsonb), gen_random_uuid()) RETURNING public_id"),
        {"w": ws, "c": json.dumps(LEAN)}).scalar()
    s.commit()
    s.close()
    yield ws, str(agent)
    s = new_session.sweep()
    users = s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid) "
                           "AND service_user_id IS NOT NULL"), {"id": ws}).scalars().all()
    for table in ("chat_turn_keys", "messages", "chats", "agents", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    if users:
        s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": users})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


@pytest.fixture
def route(monkeypatch):
    """The chat route with its model stood in for. Each turn the stand-in runs is a model call (``turns``):
    it answers ``ANSWER`` and saves it as the real service does, then counts the memory write the real
    service makes after its save (``remembered``). The detached turn and its hooks are the real ones."""
    ran = NS(turns=[], remembered=[])

    class _Answers:
        def __init__(self, db, workspace_id=None):
            self.db, self.workspace_id = db, workspace_id
            self.streaming_handler = get_streaming_handler()

        async def stream_response_with_agent(self, *, chat_id, **kwargs):
            ran.turns.append(chat_id)
            handler = self.streaming_handler
            yield handler.format_aisdk_chat_id(chat_id)
            yield handler.format_aisdk_text(ANSWER)
            yield handler.format_aisdk_finish()
            ChatService(self.db).save_message(chat_id=chat_id, role="assistant", workspace_id=self.workspace_id,
                                              parts=[{"type": "text", "text": ANSWER}])
            ran.remembered.append(chat_id)

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _lane(db, ctx, request, message_text, history_length, before=""):
        return NS(agent_id=request.agentId, assessment=None, session_agent=False, suggest_mission=False)

    async def _no_apps(*args, **kwargs):
        return True

    local = chat_turns.TurnRegistry()
    local._redis_resolved = True  # never dial Redis in tests
    monkeypatch.setattr(chat, "StreamingChatService", _Answers)
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Queue())
    monkeypatch.setattr(chat, "_turn_lane", _lane)
    monkeypatch.setattr(chat, "apps_skipped", _no_apps)
    monkeypatch.setattr(chat, "turn_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat, "run_detached_turn", functools.partial(chat_turns.run_detached_turn, registry=local))
    return ran


class _RouteRan(Exception):
    """The route's own body began: the request ran as it runs without the header."""


def _route_ran(*args, **kwargs):
    raise _RouteRan()


def _key(new_session, ws):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme game", key_type="server", permissions=[AGENTS_CHAT])
    finally:
        s.close()
    return NS(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=made["id"],
              user=NS(id=f"sdk:{made['id']}", email=None, clerk_user_id=None, system_role="service"))


def _ask(new_session, ctx, key=KEY, **body):
    """One ``POST /api/chat`` for ``ctx`` as the route runs it: every frame read, and the turn's end waited for."""
    request = ChatRequest(message={"content": ASKED}, **body)

    async def run():
        s = new_session()
        try:
            response = await chat.stream_chat(request=request, ctx=ctx, db=s, idempotency_key=key)
            frames = [chunk async for chunk in response.body_iterator]
        finally:
            s.close()
        others = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        await asyncio.gather(*others, return_exceptions=True)  # the detached turn and its end-of-turn hook
        return response, frames

    return asyncio.run(run())


def _call(new_session, ctx, header, **body):
    """The route called for ``ctx`` with ``header``, its response not read."""
    s = new_session()
    try:
        request = ChatRequest(message={"content": ASKED}, **body)
        return asyncio.run(chat.stream_chat(request=request, ctx=ctx, db=s, idempotency_key=header))
    finally:
        s.close()


def _claim(new_session, ctx, key, request):
    s = new_session()
    try:
        return chat_turn_keys.claim(s, ctx, key, chat_turn_keys.request_hash(request))
    finally:
        s.close()


def _refused(call):
    with pytest.raises(HTTPException) as refused:
        call()
    return refused.value.status_code, refused.value.detail


def _sql(new_session, statement, **params):
    s = new_session()
    try:
        s.execute(text(statement), params)
        s.commit()
    finally:
        s.close()


def _backdate(new_session, ws, hours=0, seconds=0):
    _sql(new_session, "UPDATE chat_turn_keys SET created_at = created_at - make_interval(hours => :h, secs => :s) "
         "WHERE workspace_id = CAST(:w AS uuid)", h=hours, s=seconds, w=ws)


def _rows(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT * FROM chat_turn_keys WHERE workspace_id = CAST(:w AS uuid) ORDER BY created_at"),
                         {"w": ws}).mappings().all()
    finally:
        s.close()


def _agent_id(new_session, public_id):
    """The agent's integer id: how a person names it (a key names it by ``public_id``)."""
    s = new_session()
    try:
        return s.execute(text("SELECT id FROM agents WHERE public_id = CAST(:p AS uuid)"), {"p": public_id}).scalar_one()
    finally:
        s.close()


def _messages(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT count(*) FROM messages WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalar()
    finally:
        s.close()


# --- a repeat ---------------------------------------------------------------------------------------------

def test_a_repeat_after_the_turn_answered_replays_its_answer_and_runs_nothing(new_session, workspace, route,
                                                                              monkeypatch):
    ws, agent = workspace
    key = _key(new_session, ws)
    first, said = _ask(new_session, key, agentId=agent)
    (row,) = _rows(new_session, ws)
    chat_id = str(row["chat_id"])
    assert row["state"] == DONE and row["answer_message_id"] is not None and route.turns == [chat_id]
    assert agent_execute.REPLAYED_HEADER not in first.headers
    saved = _messages(new_session, ws)

    monkeypatch.setattr(chat, "get_user_id", _route_ran)  # the route's body: the repeat never enters it
    monkeypatch.setattr(chat, "ChatService", _route_ran)
    again, replayed = _ask(new_session, key, agentId=agent)
    handler = get_streaming_handler()
    assert replayed == said == [handler.format_aisdk_chat_id(chat_id), handler.format_aisdk_text(ANSWER),
                                handler.format_aisdk_finish()]
    assert again.status_code == 200 and again.headers[agent_execute.REPLAYED_HEADER] == "true"
    assert again.media_type == first.media_type and again.headers["x-vercel-ai-data-stream"] == "v1"
    assert route.turns == [chat_id] and route.remembered == [chat_id]  # no second model call, no memory write
    assert _messages(new_session, ws) == saved and [r["state"] for r in _rows(new_session, ws)] == [DONE]


def test_the_row_keeps_no_words_of_the_turn(new_session, workspace, route):
    ws, agent = workspace
    _ask(new_session, _key(new_session, ws), agentId=agent)
    (row,) = _rows(new_session, ws)
    assert {c.name for c in ChatTurnKey.__table__.columns} == {
        "id", "workspace_id", "api_key_id", "idempotency_key", "request_hash", "state", "chat_id",
        "answer_message_id", "created_at"}
    assert not any(ASKED in str(value) or ANSWER in str(value) for value in row.values())


def test_a_repeat_while_the_turn_runs_or_on_a_different_request_is_a_conflict(new_session, workspace, route,
                                                                              monkeypatch):
    ws, agent = workspace
    key = _key(new_session, ws)
    _claim(new_session, key, KEY, ChatRequest(message={"content": ASKED}, agentId=agent))  # its turn runs
    monkeypatch.setattr(chat, "ChatService", _route_ran)
    assert _refused(lambda: _ask(new_session, key, agentId=agent)) == (409, chat_turn_keys.STILL_RUNNING)
    assert _refused(lambda: _ask(new_session, key, agentId=agent, model="claude-haiku-5-5")) == (
        409, chat_turn_keys.USED_FOR_ANOTHER)
    assert route.turns == [] and [r["state"] for r in _rows(new_session, ws)] == [RUNNING]


@pytest.mark.parametrize("changed", [
    {"message": {"content": "Roll for the door."}},
    {"endUserId": "player-8"},
    {"appContext": {"level": 3}},
    {"agentId": None},
    {"model": "claude-haiku-5-5"},
    {"id": str(uuid.UUID(int=7))},
])
def test_the_same_key_on_a_different_request_is_a_conflict(new_session, workspace, changed):
    ws, agent = workspace
    key = _key(new_session, ws)
    asked = {"message": {"content": ASKED}, "agentId": agent, "endUserId": "player-7", "appContext": {"level": 2}}
    _claim(new_session, key, KEY, ChatRequest(**asked))
    assert _refused(lambda: _claim(new_session, key, KEY, ChatRequest(**{**asked, **changed}))) == (
        409, chat_turn_keys.USED_FOR_ANOTHER)
    assert len(_rows(new_session, ws)) == 1


def test_the_hash_is_the_turn_without_its_history_or_timing_tag():
    asked = ChatRequest(message={"content": ASKED}, agentId=3, endUserId="player-7")
    resent = ChatRequest(message={"role": "user", "content": ASKED}, agentId=3, endUserId="player-7",
                         messages=[{"content": "Earlier."}], turnRef="vt-1-1")
    assert chat_turn_keys.request_hash(asked) == chat_turn_keys.request_hash(resent)
    assert len(chat_turn_keys.request_hash(asked)) == 64


def test_two_requests_racing_on_one_key_run_one_turn(new_session, workspace, monkeypatch):
    ws, agent = workspace
    key = _key(new_session, ws)
    request = ChatRequest(message={"content": ASKED}, agentId=agent)
    winner = _claim(new_session, key, KEY, request)
    looked = []
    real = chat_turn_keys._row

    def first_look_misses(db, workspace_id, key_id, idempotency_key):  # the loser looked before the winner committed
        looked.append(idempotency_key)
        return None if len(looked) == 1 else real(db, workspace_id, key_id, idempotency_key)

    monkeypatch.setattr(chat_turn_keys, "_row", first_look_misses)
    assert _refused(lambda: _claim(new_session, key, KEY, request)) == (409, chat_turn_keys.STILL_RUNNING)
    assert isinstance(winner, uuid.UUID) and len(looked) == 2
    assert [r["state"] for r in _rows(new_session, ws)] == [RUNNING]


# --- when a key's turn runs again --------------------------------------------------------------------------

def test_a_failed_or_dead_turn_may_run_again_under_its_key(new_session, workspace):
    ws, agent = workspace
    key = _key(new_session, ws)
    request = ChatRequest(message={"content": ASKED}, agentId=agent)
    first = _claim(new_session, key, KEY, request)
    s = new_session()
    try:
        assert chat_turn_keys.record_end(s, first, str(uuid.uuid4()), None, completed=False) == FAILED
    finally:
        s.close()
    second = _claim(new_session, key, KEY, request)
    _backdate(new_session, ws, seconds=chat_turns.INFLIGHT_TTL_S + 60)  # its process died: nothing marked it
    third = _claim(new_session, key, KEY, request)
    assert isinstance(second, uuid.UUID) and len({first, second, third}) == 3
    assert [r["state"] for r in _rows(new_session, ws)] == [RUNNING]


@pytest.mark.parametrize("reply", [
    {"parts": [{"type": "text", "text": NO_ANSWER_SENTENCES[0]}]},  # saved in place of an answer it never got
    {"parts": [{"type": "text", "text": "Not this time."}], "source": {"origin": TURN_ERROR_ORIGIN}},  # its note
])
def test_a_turn_that_completed_without_an_answer_ends_failed(new_session, workspace, reply):
    ws, agent = workspace
    key = _key(new_session, ws)
    held = _claim(new_session, key, KEY, ChatRequest(message={"content": ASKED}, agentId=agent))
    s = new_session()
    try:
        chats = ChatService(s)
        chat_id = str(chats.create_chat(user_id=chat.get_user_id(s, key), title="game", workspace_id=uuid.UUID(ws)).id)
        asked = chats.save_message(chat_id=chat_id, role="user", parts=[{"type": "text", "text": ASKED}], workspace_id=ws)
        chats.save_message(chat_id=chat_id, role="assistant", workspace_id=ws, **reply)
        assert chat_turn_keys.record_end(s, held, chat_id, asked.id, completed=True) == FAILED
    finally:
        s.close()
    (row,) = _rows(new_session, ws)
    assert (row["state"], row["answer_message_id"], str(row["chat_id"])) == (FAILED, None, chat_id)


def test_a_turn_refused_before_it_starts_leaves_its_key_free(new_session, workspace, route):
    ws, agent = workspace
    key = _key(new_session, ws)
    assert _refused(lambda: _ask(new_session, key)) == (422, chat_server_key.NAME_A_LEAN_AGENT)  # no agent named
    assert _refused(lambda: _ask(new_session, key, agentId=_agent_id(new_session, agent))) == (
        422, chat_server_key.NAME_IT_BY_PUBLIC_ID)  # named by integer
    assert _rows(new_session, ws) == [] and route.turns == []
    _ask(new_session, key, agentId=agent)  # the corrected request, under the same key
    assert len(route.turns) == 1 and [r["state"] for r in _rows(new_session, ws)] == [DONE]


def test_a_repeat_whose_answer_is_no_longer_kept_runs_as_new(new_session, workspace, route):
    ws, agent = workspace
    key = _key(new_session, ws)
    _ask(new_session, key, agentId=agent)
    _sql(new_session, "DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)", w=ws)  # its chat was cleared
    again, _ = _ask(new_session, key, agentId=agent)
    assert len(route.turns) == 2 and agent_execute.REPLAYED_HEADER not in again.headers
    assert [r["state"] for r in _rows(new_session, ws)] == [DONE]


def test_a_key_past_the_window_runs_as_new_and_its_expired_rows_are_deleted(new_session, workspace, route):
    ws, agent = workspace
    key, theirs = _key(new_session, ws), _key(new_session, ws)
    _ask(new_session, key, agentId=agent)
    _claim(new_session, key, "another-turn", ChatRequest(message={"content": "Open the map."}, agentId=agent))
    _claim(new_session, theirs, KEY, ChatRequest(message={"content": ASKED}, agentId=agent))
    _backdate(new_session, ws, hours=chat_turn_keys.KEY_WINDOW_HOURS + 1)
    again, _ = _ask(new_session, key, agentId=agent)
    assert len(route.turns) == 2 and agent_execute.REPLAYED_HEADER not in again.headers
    by_key = sorted((str(r["api_key_id"]) == str(key.api_key_id), r["idempotency_key"], r["state"])
                    for r in _rows(new_session, ws))
    assert by_key == [(False, KEY, RUNNING), (True, KEY, DONE)]  # "another-turn" went; another key's row stays


# --- everyone else ------------------------------------------------------------------------------------------

def test_no_header_and_a_persons_turn_run_as_before(new_session, workspace, monkeypatch):
    ws, agent = workspace
    key = _key(new_session, ws)
    person = NS(workspace_id=uuid.UUID(ws), auth_type="clerk", api_key_id=None,
                user=NS(id="user_1", email=None, clerk_user_id="user_1"))
    monkeypatch.setattr(chat, "ChatService", _route_ran)
    by_integer = _agent_id(new_session, agent)  # a person names the agent by integer, as before
    for ctx, header, named in ((key, None, agent), (person, KEY, by_integer), (person, "has space", by_integer)):
        with pytest.raises(_RouteRan):  # a person's header is ignored
            _call(new_session, ctx, header, agentId=named)
    assert _rows(new_session, ws) == []


@pytest.mark.parametrize("bad", ["", "has space", "tab\\there", "é", "x" * 256])
def test_a_malformed_key_is_refused_before_the_route_runs(new_session, workspace, monkeypatch, bad):
    ws, agent = workspace
    key = _key(new_session, ws)
    monkeypatch.setattr(chat, "ChatService", _route_ran)
    status, _ = _refused(lambda: _call(new_session, key, bad.replace("\\t", "\t"), agentId=agent))
    assert status == 422 and _rows(new_session, ws) == []


# --- the table, its migration and the route's header -------------------------------------------------------

def test_the_migration_chains_onto_the_head_and_creates_the_models_table():
    source = (VERSIONS / f"{MIGRATION}.py").read_text()
    assert f'revision = "{MIGRATION}"' in source and 'down_revision = "credit_ledger_sub_tenant_20261113"' in source
    assert "CREATE TABLE IF NOT EXISTS chat_turn_keys" in source
    assert f"CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_INDEX} ON chat_turn_keys (api_key_id, idempotency_key)" in source
    assert all(f"\n    {column.name} " in source for column in ChatTurnKey.__table__.columns)
    index = next(i for i in ChatTurnKey.__table__.indexes if i.name == UNIQUE_INDEX)
    assert index.unique and [c.name for c in index.columns] == ["api_key_id", "idempotency_key"]
    assert ChatTurnKey.__table__.c.idempotency_key.type.length == agent_execute.IDEMPOTENCY_KEY_MAX
    # prd251d_wave3 was written on the same head and merged the same morning: merge_heads_20261117
    # joins the two, and the spend cap (product_spend_cap_20261113) chains onto it as the one head.
    merge = (VERSIONS / "merge_heads_20261117.py").read_text()
    assert f'"{MIGRATION}"' in merge and '"prd251d_wave3"' in merge
    guard = (Path(__file__).resolve().parent / "test_prd209_alembic_single_head.py").read_text()
    assert 'EXPECTED_HEAD = "product_spend_cap_20261113"' in guard


def test_the_table_stays_behind_on_a_workspace_move_and_names_its_key_as_secret():
    from services.workspace_move.table_map import rule_for

    rule = rule_for("chat_turn_keys")
    assert rule is not None and not rule.moved and "api_key_id" in rule.secret_columns


def test_the_route_reads_the_key_from_the_idempotency_key_header():
    from main import app

    operation = app.openapi()["paths"]["/api/chat"]["post"]
    headers = {p["name"]: p for p in operation.get("parameters", []) if p["in"] == "header"}
    assert agent_execute.IDEMPOTENCY_HEADER in headers
    assert not headers[agent_execute.IDEMPOTENCY_HEADER].get("required", False)
    assert chat._chat_turn.__wrapped__.__name__ == "_chat_turn" and chat._after_the_turn.__wrapped__
