"""PRD-274 Wave 3 (§5.5, §8-17): a product's end users, scoped by workspace.

An end user is ``(workspace, end_user_id)``, asserted by a server key of that workspace, with
their facts in ``mem:{ws}:end_user:{id}``. The product records consent before anything is stored;
reads leave out what is private to a person (and, for a key, what an audience limits); a search
reads the end user's facts and the workspace's, never another end user's; an erase deletes the
namespace and the record. Real database for the records; the memory store is a fake.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from api import end_user_memory as eum
from api.widget_memory import MemoryCreate
from core.auth.scopes import MEMORY_READ, MEMORY_WRITE

ROWS = [
    {"id": "plain", "memory": "Prefers morning sessions.", "metadata": {"end_user_id": "player-7"}},
    {"id": "limited", "memory": "Owes the guild 40 gold.", "metadata": {"audience": ["gm"]}},
    {"id": "private", "memory": "A member's own note.", "metadata": {"scope": "private", "owner": "user:1"}},
]


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.end_users import EndUser

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    EndUser.__table__.create(eng, checkfirst=True)  # a test database built before the table
    with eng.begin() as conn:  # or before its erase mark
        conn.execute(text("ALTER TABLE end_users ADD COLUMN IF NOT EXISTS erasing_at TIMESTAMPTZ"))
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w3-end-users')"), {"id": ws})
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    for table in ("end_users", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


class _Store:
    def __init__(self):
        self.calls = []
        self._durable = self

    async def store_long_term_messages(self, user_id, messages, metadata=None, workspace_id=None):
        self.calls.append(("store", user_id, metadata))
        return {"success": True, "id": "m-1"}

    async def get_all_memories_scoped(self, user_id, limit=100, workspace_id=None):
        self.calls.append(("list", user_id))
        return [dict(r) for r in ROWS]

    async def search(self, query, user_id, limit=5, workspace_id=None, also_namespaces=()):
        self.calls.append(("search", user_id, tuple(also_namespaces)))
        return [dict(r, score=0.8) for r in ROWS]

    async def get_one(self, memory_id, user_id):
        self.calls.append(("get", user_id, memory_id))
        return next((dict(r) for r in ROWS if r["id"] == memory_id), None)

    async def delete(self, memory_ids, user_id, workspace_id=None):
        self.calls.append(("delete", user_id, tuple(memory_ids)))
        return True

    async def erase_namespace(self, user_id):
        self.calls.append(("erase", user_id))
        return 3


@pytest.fixture
def store(monkeypatch):
    fake = _Store()
    monkeypatch.setattr("api.widget_memory._get_memory_service", lambda: fake)
    return fake


def _key(new_session, ws):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme game", key_type="server",
                                            permissions=[MEMORY_READ, MEMORY_WRITE])
    finally:
        s.close()
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=uuid.UUID(made["id"]),
                           user=SimpleNamespace(id=f"sdk:{made['id']}", email=None, clerk_user_id=None))


def _member(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk", api_key_id=None,
                           user=SimpleNamespace(id="user_1", email=None, clerk_user_id="user_1"))


def _call(new_session, fn, *args, **kwargs):
    s = new_session()
    try:
        result = fn(*args, db=s, **kwargs)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result
    finally:
        s.close()


def test_the_namespace_sits_inside_the_workspace():
    from modules.memory.durable_store import workspace_from_namespace
    from modules.memory.unified_memory_service import MemoryNamespace

    ns = MemoryNamespace("ws-1").end_user("player:7")
    assert ns == "mem:ws-1:end_user:player:7" and workspace_from_namespace(ns) == "ws-1"


def test_consent_is_recorded_by_the_key_and_refused_when_false(new_session, workspace):
    from core.models.end_users import EndUser

    key = _key(new_session, workspace)
    made = _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True, retention_days=90),
                 end_user_id="player-7", ctx=key)
    again = _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    assert (made["retention_days"], again["retention_days"]) == (90, None)
    s = new_session()
    rows = s.query(EndUser).filter(EndUser.workspace_id == uuid.UUID(workspace)).all()
    s.close()
    assert len(rows) == 1 and rows[0].api_key_id == key.api_key_id
    with pytest.raises(HTTPException) as refused:
        _call(new_session, eum.record_consent, eum.EndUserConsent(consent=False), end_user_id="player-8", ctx=key)
    assert refused.value.status_code == 422


def test_a_fact_is_stored_only_with_consent_into_the_end_users_namespace(new_session, workspace, store):
    key = _key(new_session, workspace)
    body = MemoryCreate(content="Prefers morning sessions.", metadata={"written_by": "sdk:forged"})
    with pytest.raises(HTTPException) as no_consent:
        _call(new_session, eum.store_end_user_memory, body, end_user_id="player-7", ctx=key)
    assert no_consent.value.status_code == 409 and store.calls == []
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    _call(new_session, eum.store_end_user_memory, body, end_user_id="player-7", ctx=key)
    (_, namespace, metadata), = store.calls
    assert namespace == f"mem:{workspace}:end_user:player-7"
    assert metadata["end_user_id"] == "player-7" and metadata["written_by"] == key.user.id


def test_reads_leave_out_what_is_private_and_for_a_key_what_an_audience_limits(new_session, workspace, store):
    key = _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    by_key = _call(new_session, eum.list_end_user_memories, end_user_id="player-7", limit=50, ctx=key)
    by_member = _call(new_session, eum.list_end_user_memories, end_user_id="player-7", limit=50, ctx=_member(workspace))
    assert [m.id for m in by_key["memories"]] == ["plain"]
    assert [m.id for m in by_member["memories"]] == ["plain", "limited"]


def test_a_search_reads_the_end_user_and_the_workspace_never_another_end_user(new_session, workspace, store):
    key = _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    _call(new_session, eum.search_end_user_memories, q="sessions", end_user_id="player-7", limit=20, ctx=key)
    ((_, namespace, also),) = [c for c in store.calls if c[0] == "search"]
    assert namespace == f"mem:{workspace}:end_user:player-7" and also == (f"mem:{workspace}",)


def test_an_end_user_of_another_workspace_or_unknown_is_not_found(new_session, workspace, store):
    key = _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    stranger = SimpleNamespace(**{**key.__dict__, "workspace_id": uuid.uuid4()})
    for ctx, end_user_id in ((stranger, "player-7"), (key, "player-99")):
        with pytest.raises(HTTPException) as missing:
            _call(new_session, eum.list_end_user_memories, end_user_id=end_user_id, limit=50, ctx=ctx)
        assert missing.value.status_code == 404
    assert store.calls == []


def test_an_erase_deletes_the_namespace_and_the_record(new_session, workspace, store):
    from core.models.end_users import EndUser

    key = _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    erased = _call(new_session, eum.erase_end_user, end_user_id="player-7", ctx=key)
    assert erased == {"end_user_id": "player-7", "erased": True, "memories_deleted": 3, "chats_deleted": 0}
    assert store.calls == [("erase", f"mem:{workspace}:end_user:player-7")]
    s = new_session()
    assert s.query(EndUser).filter(EndUser.workspace_id == uuid.UUID(workspace)).count() == 0
    s.close()


def test_the_routes_carry_the_gate_and_take_only_a_clean_end_user_id():
    from fastapi.routing import APIRoute, iter_route_contexts

    from core.auth.server_key_gate import KEY_SCOPE_MARKER_ATTR
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    want = {("PUT", "/api/memory/end-users/{end_user_id}"): ("knowledge:create", MEMORY_WRITE),
            ("POST", "/api/memory/end-users/{end_user_id}/memories"): ("knowledge:create", MEMORY_WRITE),
            ("GET", "/api/memory/end-users/{end_user_id}/memories"): ("knowledge:read", MEMORY_READ),
            ("GET", "/api/memory/end-users/{end_user_id}/memories/search"): ("knowledge:read", MEMORY_READ),
            ("DELETE", "/api/memory/end-users/{end_user_id}"): ("knowledge:delete", MEMORY_WRITE),
            ("DELETE", "/api/memory/end-users/{end_user_id}/memories/{memory_id}"): ("knowledge:delete", MEMORY_WRITE)}
    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}
    for key, (permission, scope) in want.items():
        marked = [(getattr(d.call, PERMISSION_MARKER_ATTR, None), getattr(d.call, KEY_SCOPE_MARKER_ATTR))
                  for d in routes[key].dependant.dependencies if getattr(d.call, KEY_SCOPE_MARKER_ATTR, None)]
        assert marked == [(permission, scope)], key
    parameter = next(p for p in app.openapi()["paths"]["/api/memory/end-users/{end_user_id}"]["put"]["parameters"]
                     if p["name"] == "end_user_id")
    assert parameter["schema"]["pattern"] == "^[A-Za-z0-9._:@-]{1,128}$"


def test_a_key_deletes_only_a_fact_it_wrote_and_a_member_any(new_session, workspace, store, monkeypatch):
    key, other = _key(new_session, workspace), _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)
    mine = {"id": "mine", "memory": "x", "metadata": {"written_by": key.user.id}}
    theirs = {"id": "theirs", "memory": "y", "metadata": {"written_by": other.user.id}}
    owners = {"id": "owners", "memory": "z", "metadata": {}}

    async def one(memory_id, user_id):  # read by id: found however many facts the end user has
        return next((dict(r) for r in (mine, theirs, owners) if r["id"] == memory_id), None)

    monkeypatch.setattr(store, "get_one", one)
    done = _call(new_session, eum.delete_end_user_memory, memory_id="mine", end_user_id="player-7", ctx=key)
    assert done["deleted"] is True
    for memory_id in ("theirs", "owners", "missing"):
        with pytest.raises(HTTPException) as refused:
            _call(new_session, eum.delete_end_user_memory, memory_id=memory_id, end_user_id="player-7", ctx=key)
        assert refused.value.status_code == 404
    _call(new_session, eum.delete_end_user_memory, memory_id="owners", end_user_id="player-7", ctx=_member(workspace))
    namespace = f"mem:{workspace}:end_user:player-7"
    assert [c for c in store.calls if c[0] == "delete"] == [("delete", namespace, ("mine",)),
                                                           ("delete", namespace, ("owners",))]


def test_a_store_that_cant_be_read_is_a_503_for_a_key_not_a_404(new_session, workspace, store, monkeypatch):
    key = _key(new_session, workspace)
    _call(new_session, eum.record_consent, eum.EndUserConsent(consent=True), end_user_id="player-7", ctx=key)

    async def down(memory_id, user_id):
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(store, "get_one", down)
    with pytest.raises(HTTPException) as unavailable:
        _call(new_session, eum.delete_end_user_memory, memory_id="mine", end_user_id="player-7", ctx=key)
    assert unavailable.value.status_code == 503
    assert not [c for c in store.calls if c[0] == "delete"]


def test_the_store_reads_one_fact_by_id_and_only_from_its_namespace():
    from modules.memory.durable_store import DurableMemoryStore

    fact_id, namespace = str(uuid.uuid4()), "mem:ws:end_user:player-7"
    point = SimpleNamespace(id=fact_id, payload={"namespace": namespace, "content": "x",
                                                 "metadata": {"written_by": "sdk:k1"}})
    asked = []

    async def retrieve(collection, ids, with_payload=True):
        asked.append(tuple(ids))
        return [point] if ids == [fact_id] else []

    durable = object.__new__(DurableMemoryStore)
    durable._enabled, durable._bootstrap_done, durable._collection = True, True, "memories"
    durable._client = SimpleNamespace(retrieve=retrieve)
    found = asyncio.run(durable.get_one(fact_id, namespace))
    assert (found["id"], found["metadata"]) == (fact_id, {"written_by": "sdk:k1"})
    assert asyncio.run(durable.get_one(fact_id, "mem:ws:end_user:player-8")) is None  # another end user's
    assert asyncio.run(durable.get_one(str(uuid.uuid4()), namespace)) is None          # no such fact
    assert asyncio.run(durable.get_one("not-a-uuid", namespace)) is None               # never sent to Qdrant
    assert len(asked) == 3
