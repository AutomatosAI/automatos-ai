"""PRD-268 US-C02d (§6.2.3, §9, K17): every read action writes nothing.

The check-in lane (US-C02b) runs only what is labelled ``read``, so the lane is only as good
as the labels. This sweep calls every registered read action's handler (the one
``PLATFORM_HANDLERS`` names, its wrappers included) with parameters synthesised from its
JSON schema (its required fields, each with its minimal valid value). Each runs against a
session over an empty database that records every write, while httpx and requests answer
every request without sending it (``tests/helpers_write_spies.py``). A write fails the
sweep, naming the action and what it wrote. A handler that raises, or runs past its time,
still counts as checked; work it leaves running is waited for and counted as its own.

What counts as a write:

- on the session: ``add``, ``add_all``, ``delete``, ``merge``, ``flush``, ``commit``, the
  ``bulk_*`` calls, ``Query.update`` and ``Query.delete``, and an ``execute`` of an INSERT,
  UPDATE or DELETE (on the session or its connection), or of SQL that changes the schema,
  sends a notification or moves a sequence;
- over HTTP: every request, since each leaves the platform, except one to the platform's own
  vector store (``config.QDRANT_URL``). That is a datastore like Postgres, so only a change
  to it counts (a collection or an index created, an upsert, a delete, a payload set), and
  its reads (a GET, a retrieve, a search, a scroll, a count) do not.

What the sweep cannot see: a write reached only when rows exist (the database is empty), a
session the handler opens itself, Redis, and clients that are neither httpx nor requests.

A read action the sweep catches writing is relabelled ``write``, with its hierarchy-gate
entry, and never excepted (K17). It caught ``platform_field_query``: on first use it creates
the shared field's collection and indexes, and it reinforces every pattern it returns
(``record_access``). That action is now a write.

A read action with no platform handler (the workspace file tools) never runs in a check-in:
the lane refuses every tool that is not a platform action.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Set
from uuid import uuid4

import httpx
import pytest
import requests
from sqlalchemy import column, delete, insert, literal, select, table, text, update

from tests.helpers_write_spies import WriteLog, minimal_params, recording_http, spy_session

ORCH = Path(__file__).resolve().parents[1]
# Long enough for a handler that waits on a store that never answers to give up; a
# handler past it is cancelled and counts as checked.
HANDLER_SECONDS = 5.0
OUTCOME_CHARS = 120
# How long work a handler left running (a task it created) has to finish.
SETTLE_SECONDS = 1.0
RELABELLED = ("platform_field_query",)
VECTOR_STORE = "http://vectors.test:6333"
ELSEWHERE = "https://api.example.test/v1/things"
CARDS = table("cards", column("id"), column("title"))


# ── the sweep ───────────────────────────────────────────────────────────────

async def _settle(left: Set["asyncio.Task[Any]"]) -> None:
    """Wait for the work a handler left running, then cancel what is still going and let it
    unwind, all while the spies still record."""
    if not left:
        return
    _, pending = await asyncio.wait(left, timeout=SETTLE_SECONDS)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=SETTLE_SECONDS)
    for task in left:
        if task.done() and not task.cancelled():
            task.exception()


async def _call(handler: Callable[..., Any], params: Dict[str, Any]) -> WriteLog:
    """One handler's call against the spies: what it wrote, and how it ended."""
    log = WriteLog()
    with recording_http(log):
        before = asyncio.all_tasks()
        try:
            outcome = handler(spy_session(log), uuid4(), params)
            if inspect.isawaitable(outcome):
                await asyncio.wait_for(outcome, HANDLER_SECONDS)
        except asyncio.TimeoutError:
            log.outcome = "timed out"
        except (Exception, asyncio.CancelledError) as error:
            log.outcome = f"raised {type(error).__name__}: {error}"[:OUTCOME_CHARS]
        await _settle(asyncio.all_tasks() - before)
    return log


async def _sweep(handlers: Dict[str, Callable[..., Any]], schemas: Dict[str, Any]) -> Dict[str, WriteLog]:
    """Every handler called in turn, each with its own spies, by action name."""
    return {name: await _call(handlers[name], minimal_params(schemas.get(name) or {})) for name in sorted(handlers)}


def _wrote(logs: Dict[str, WriteLog]) -> Dict[str, list]:
    return {name: log.writes for name, log in logs.items() if log.writes}


def _run(*steps: Callable[..., Any]) -> WriteLog:
    """One stub handler that takes ``steps`` (each given the session), through the sweep."""
    async def handler(db, workspace_id, params):
        for step in steps:
            done = step(db)
            if inspect.isawaitable(done):
                await done
        return {"success": True}

    return asyncio.run(_sweep({"platform_probe": handler}, {}))["platform_probe"]


@pytest.fixture
def vector_store(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "QDRANT_URL", VECTOR_STORE, raising=False)
    return VECTOR_STORE


def _registry_reads() -> Dict[str, Any]:
    from modules.tools.discovery import get_action_registry

    return {action.name: action for action in get_action_registry().get_by_permission("read")}


def _platform_handlers() -> Dict[str, Callable[..., Any]]:
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    return dict(PLATFORM_HANDLERS)


def _gate():
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_read_action_writes_nothing():
    reads = _registry_reads()
    handlers = {name: handler for name, handler in _platform_handlers().items() if name in reads}

    logs = asyncio.run(_sweep(handlers, {name: action.parameters for name, action in reads.items()}))
    wrote = _wrote(logs)

    assert handlers and set(logs) == set(handlers)
    assert wrote == {}, "read actions that wrote; relabel each write (K17): " + "; ".join(
        f"{name} [{logs[name].outcome}]: {', '.join(writes)}" for name, writes in sorted(wrote.items()))


def test_every_read_action_the_sweep_does_not_call_is_refused_in_a_check_in():
    """A read action with no platform handler runs elsewhere (the workspace tools), and the
    lane refuses it before anything routes it, so the sweep covers all a check-in can run."""
    from modules.tools.discovery.check_in_lane import is_a_platform_action, offered_in_a_check_in

    reads = _registry_reads()
    unswept = sorted(name for name in reads if name not in _platform_handlers())

    assert [name for name in unswept if is_a_platform_action(name)] == []
    assert [name for name in unswept if offered_in_a_check_in(reads[name])] == []


# ── the spies see every kind of write ───────────────────────────────────────

class _Row:
    pass


@pytest.mark.parametrize("label,step", [
    ("add", lambda db: db.add(_Row())),
    ("add_all", lambda db: db.add_all([_Row()])),
    ("delete", lambda db: db.delete(_Row())),
    ("merge", lambda db: db.merge(_Row())),
    ("flush", lambda db: db.flush()),
    ("commit", lambda db: db.commit()),
    ("bulk_save_objects", lambda db: db.bulk_save_objects([_Row()])),
    ("Query.update", lambda db: db.query(_Row).filter(True).update({"title": "x"})),
    ("Query.delete", lambda db: db.query(_Row).filter_by(id=1).delete()),
    ("insert", lambda db: db.execute(insert(CARDS).values(title="x"))),
    ("update", lambda db: db.execute(update(CARDS).values(title="x"))),
    ("delete statement", lambda db: db.execute(delete(CARDS))),
    ("INSERT text", lambda db: db.execute(text("INSERT INTO cards (title) VALUES (:t)"), {"t": "x"})),
    ("UPDATE text", lambda db: db.execute(text("update cards set title = 'x' where id = 1"))),
    ("DELETE text", lambda db: db.execute(text("DELETE FROM cards WHERE id = 1"))),
    ("a CTE that deletes", lambda db: db.execute(text("WITH gone AS (DELETE FROM cards RETURNING id) SELECT 1"))),
    ("DDL", lambda db: db.execute(text("CREATE TABLE IF NOT EXISTS seen (id int)"))),
    ("a notification", lambda db: db.execute(text("SELECT pg_notify('board', 'changed')"))),
    ("a sequence", lambda db: db.execute(text("SELECT nextval('ticket_numbers')"))),
    ("scalar of an insert", lambda db: db.scalar(insert(CARDS).values(title="x").returning(CARDS.c.id))),
    ("the session's connection", lambda db: db.connection().execute(text("INSERT INTO cards DEFAULT VALUES"))),
    ("the engine's connection", lambda db: db.get_bind().connect().execute(update(CARDS).values(title="x"))),
    ("httpx", lambda db: httpx.Client().get(ELSEWHERE)),
    ("httpx async", lambda db: httpx.AsyncClient().post(ELSEWHERE, json={"a": 1})),
    ("requests", lambda db: requests.get(ELSEWHERE, timeout=1)),
])
def test_the_spies_record_each_kind_of_write(label, step):
    assert _run(step).writes, label


def test_a_requests_answer_streams_and_closes():
    def streamed(db):
        with requests.get(ELSEWHERE, stream=True, timeout=1) as response:
            assert (response.status_code, b"".join(response.iter_content(2)), response.json()) == (200, b"{}", {})

    log = _run(streamed)

    assert (log.writes, log.outcome) == (["HTTP GET api.example.test"], "returned")


@pytest.mark.parametrize("method,path", [
    ("PUT", "/collections/shared_field"),
    ("PUT", "/collections/shared_field/index"),
    ("PUT", "/collections/shared_field/points"),
    ("POST", "/collections/shared_field/points/delete"),
    ("POST", "/collections/shared_field/points/payload"),
    ("POST", "/collections/shared_field/snapshots"),
    ("DELETE", "/collections/shared_field"),
])
def test_a_change_to_the_vector_store_is_a_write(vector_store, method, path):
    log = _run(lambda db: httpx.AsyncClient(base_url=vector_store).request(method, path))

    assert log.writes == [f"vector store {method} {path}"]


@pytest.mark.parametrize("method,path", [
    ("GET", "/collections"),
    ("GET", "/collections/shared_field/exists"),
    ("POST", "/collections/shared_field/points"),
    ("POST", "/collections/shared_field/points/query"),
    ("POST", "/collections/shared_field/points/scroll"),
    ("POST", "/collections/shared_field/points/count"),
    ("POST", "/collections/shared_field/facet"),
])
def test_a_read_of_the_vector_store_is_not(vector_store, method, path):
    assert _run(lambda db: httpx.AsyncClient(base_url=vector_store).request(method, path)).writes == []


def test_every_request_elsewhere_is_a_write_even_a_get(vector_store):
    log = _run(lambda db: httpx.AsyncClient().get(f"{ELSEWHERE}?q=1"), lambda db: requests.head(ELSEWHERE, timeout=1))

    assert log.writes == ["HTTP GET api.example.test", "HTTP HEAD api.example.test"]


def test_reads_record_nothing_and_find_nothing():
    found = {}

    def reads(db):
        found["all"] = db.query(_Row).filter(True).join(_Row).order_by(None).limit(5).offset(1).all()
        found["first"] = db.query(_Row).filter_by(id=1).first()
        found["count"] = db.query(_Row).count()
        found["get"] = db.get(_Row, 1)
        found["rows"] = db.execute(select(literal(1))).fetchall()
        found["scalars"] = db.scalars(select(CARDS.c.id)).all()
        found["mapped"] = db.execute(text("SELECT updated_at, deleted_at FROM cards")).mappings().first()
        found["locked"] = db.execute(text("SELECT id FROM cards FOR UPDATE SKIP LOCKED")).scalar()
        found["subquery"] = db.execute(select(db.query(_Row).subquery())).all()
        found["compared"] = db.query(_Row).filter(CARDS.c.id == db.query(_Row).scalar_subquery()).all()
        with db.begin_nested():
            found["nested"] = list(db.query(_Row))
        db.rollback()
        db.expire_all()

    log = _run(reads)

    assert log.writes == [] and log.outcome == "returned"
    assert found == {"all": [], "first": None, "count": 0, "get": None, "rows": [], "scalars": [],
                     "mapped": None, "locked": None, "subquery": [], "compared": [], "nested": []}


def test_a_handler_that_raises_is_checked_and_its_writes_before_the_raise_count():
    def boom(db):
        raise ValueError("not found")

    raised = _run(boom)
    wrote_then_raised = _run(lambda db: db.commit(), boom)

    assert (raised.writes, raised.outcome) == ([], "raised ValueError: not found")
    assert (wrote_then_raised.writes, wrote_then_raised.outcome) == (["commit"], "raised ValueError: not found")


def test_a_handler_past_its_time_is_checked(monkeypatch):
    monkeypatch.setitem(globals(), "HANDLER_SECONDS", 0.05)

    log = _run(lambda db: asyncio.sleep(5))

    assert (log.writes, log.outcome) == ([], "timed out")


def test_work_a_handler_leaves_running_is_its_own():
    async def later(db):
        await asyncio.sleep(0)
        db.add(_Row())

    log = _run(lambda db: asyncio.get_running_loop().create_task(later(db)) and None)

    assert log.writes == ["add(_Row)"]


def test_the_sweep_names_each_action_that_wrote():
    async def reads(db, workspace_id, params):
        return {"success": True, "rows": db.query(_Row).all()}

    async def writes(db, workspace_id, params):
        db.add(_Row())
        db.commit()
        return {"success": True}

    logs = asyncio.run(_sweep({"platform_reads": reads, "platform_mislabelled": writes}, {}))

    assert _wrote(logs) == {"platform_mislabelled": ["add(_Row)", "commit"]}


# ── the parameters ──────────────────────────────────────────────────────────

def test_parameters_are_the_required_fields_at_their_minimal_values():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "code": {"type": "string", "minLength": 3},
            "kind": {"type": "string", "enum": ["research", "audit"]},
            "top_k": {"type": "integer"},
            "page": {"type": "integer", "minimum": 2},
            "ratio": {"type": "number", "exclusiveMinimum": 0},
            "strict": {"type": "boolean"},
            "ids": {"type": "array", "items": {"type": "integer"}},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "maybe": {"type": ["null", "string"]},
            "filter": {"type": "object", "properties": {"agent": {"type": "integer"}, "note": {"type": "string"}},
                       "required": ["agent"]},
            "optional": {"type": "string"},
        },
        "required": ["query", "code", "kind", "top_k", "page", "ratio", "strict", "ids", "tags", "maybe", "filter"],
    }

    assert minimal_params(schema) == {
        "query": "1", "code": "111", "kind": "research", "top_k": 1, "page": 2, "ratio": 1, "strict": False,
        "ids": [], "tags": ["1", "1"], "maybe": "1", "filter": {"agent": 1},
    }
    assert minimal_params({"type": "object", "properties": {"q": {"type": "string"}}}) == {}


# ── what the sweep caught is relabelled (K17) ───────────────────────────────

@pytest.mark.parametrize("name", RELABELLED)
def test_what_the_sweep_caught_is_a_write_the_hierarchy_gate_accounts_for(name):
    from modules.tools.discovery import get_action_registry

    gate = _gate()

    assert get_action_registry().get(name).permission_level == "write"
    assert gate.collect_registrations()[name].permission_level == "write"
    assert name in gate.ALLOW_LIST
    assert gate.main() == 0


@pytest.mark.parametrize("name", RELABELLED)
def test_what_the_sweep_caught_is_never_offered_or_run_in_a_check_in(name):
    from modules.policy.policy_document import RISK_INTERNAL_WRITE, classify_action
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.check_in_lane import may_clear_in_a_check_in, offered_in_a_check_in
    from modules.tools.execution.turn_account import is_read

    action = get_action_registry().get(name)

    assert not offered_in_a_check_in(action) and not may_clear_in_a_check_in(name)
    assert classify_action(name, permission_level=action.permission_level) == RISK_INTERNAL_WRITE
    assert is_read(name) is False
