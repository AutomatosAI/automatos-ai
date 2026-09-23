"""F103 (night 3) — a memory write the store did not answer is never "noted".

Night 3 lost durable-memory writes to Qdrant 408s inside the F105 event-loop
freezes (19 of them in 63 ms at one point), with nothing in the log naming what
was lost. Now a PUT that times out is tried once more after a short pause; a
second timeout is logged as "memory write lost: <namespace>, <chars>", and the
tool result says the memory was NOT saved — platform_store_memory and
record_memory never report success over a lost write.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from modules.memory import durable_store as ds

FACT = "Declan's wholesale orders ship on Thursdays."
WS = str(uuid.uuid4())
NS = f"mem:{WS}"


def _408():
    return UnexpectedResponse(408, "Request Timeout", b"", httpx.Headers())


class _Client:
    """Raises the queued outcomes in turn; None is a write that lands."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def upsert(self, collection_name, points):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


class _Embedder:
    async def generate_embedding(self, text):
        return [0.0, 0.1, 0.2]


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelno, record.getMessage()))


@pytest.fixture
def lines():
    handler = _Lines()
    ds.logger.addHandler(handler)
    previous = ds.logger.level
    ds.logger.setLevel(logging.INFO)
    yield handler.lines
    ds.logger.removeHandler(handler)
    ds.logger.setLevel(previous)


def _store(monkeypatch, *outcomes):
    monkeypatch.setattr(ds.config, "MEMORY_WRITE_RETRY_PAUSE_S", 0, raising=False)
    store = ds.DurableMemoryStore.__new__(ds.DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client, store._embedder = _Client(*outcomes), _Embedder()

    async def no_duplicate(user_id, content_hash):
        return None

    store._find_by_hash = no_duplicate
    return store


def _add(store):
    return asyncio.run(store.add(messages=[{"role": "user", "content": FACT}], user_id=NS, workspace_id=WS))


def _lost(lines):
    return [m for level, m in lines if level == logging.WARNING and m.startswith("memory write lost")]


@pytest.mark.parametrize("timeout", [
    _408(),
    ResponseHandlingException(httpx.ReadTimeout("timed out")),     # the client's own timeout, wrapped
    asyncio.TimeoutError(),
], ids=["qdrant-408", "client-timeout", "asyncio-timeout"])
def test_a_timeout_then_success_is_saved(monkeypatch, lines, timeout):
    store = _store(monkeypatch, timeout, None)
    added = _add(store)
    assert added["success"] is True and added["id"]
    assert store._client.calls == 2
    assert _lost(lines) == []


def test_two_timeouts_are_logged_and_not_saved(monkeypatch, lines):
    store = _store(monkeypatch, _408(), _408())
    added = _add(store)
    assert added == {"success": False, "error": "the memory store did not answer in time, twice"}
    assert store._client.calls == 2
    assert len(_lost(lines)) == 1
    assert _lost(lines)[0].startswith(f"memory write lost: {NS}, {len(FACT)} chars")


def test_a_healthy_put_is_unchanged(monkeypatch, lines):
    store = _store(monkeypatch)
    added = _add(store)
    assert added["success"] is True and added["id"] and "deduped" not in added
    assert store._client.calls == 1
    assert _lost(lines) == []


def test_any_other_failure_is_not_retried(monkeypatch):
    bad_request = UnexpectedResponse(400, "Bad Request", b"wrong vector size", httpx.Headers())
    refused = ResponseHandlingException(httpx.ConnectError("connection refused"))
    for error in (bad_request, refused):
        store = _store(monkeypatch, error)
        with pytest.raises(type(error)):
            _add(store)
        assert store._client.calls == 1


# ── what the model is told ──────────────────────────────────────────────────

def _memory_service(monkeypatch, store):
    """The real UnifiedMemoryService.store_long_term over the given store."""
    from modules.memory import unified_memory_service as ums

    monkeypatch.setattr(ds.config, "QDRANT_URL", "http://qdrant.test", raising=False)
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store

    async def no_cache(workspace_id):
        return None

    service._invalidate_search_cache = no_cache
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    return service


def _store_memory():
    from modules.tools.discovery.handlers_workspace import store_memory

    return asyncio.run(store_memory(None, WS, {"content": FACT}))


def test_the_tool_says_not_saved_after_two_timeouts(monkeypatch, lines):
    _memory_service(monkeypatch, _store(monkeypatch, _408(), _408()))
    result = _store_memory()
    assert result["success"] is False
    assert result["error"] == ("Memory NOT saved — the memory store did not answer in time, twice. "
                               "Tell the owner it was not stored.")


def test_the_tool_says_stored_after_a_timeout_and_a_retry(monkeypatch, lines):
    _memory_service(monkeypatch, _store(monkeypatch, _408(), None))
    result = _store_memory()
    assert result["success"] is True and result["message"].startswith("Stored in memory")


class _Executor:
    """The platform dispatcher as record_memory sees it: one result per action."""

    results: dict = {}

    def __init__(self, db):
        pass

    async def execute_tool(self, *, tool_name, parameters, **kw):
        return self.results[parameters["action"]]


LOST = {"success": False, "error": "Memory NOT saved — the memory store did not answer in time, twice. "
                                   "Tell the owner it was not stored."}


def _record_memory(monkeypatch, results, *, mission_field_id=None):
    from modules.tools.execution import unified_executor
    from services import session_tools as st

    monkeypatch.setattr(_Executor, "results", results)
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    ctx = st.SessionContext(task_id=119, agent_id=268, agent_name="TRACKER", workspace_id=WS,
                            mission_field_id=mission_field_id)
    return asyncio.run(st._run_record_memory(None, {"content": FACT}, ctx))


def test_record_memory_never_reports_a_lost_write_as_recorded(monkeypatch):
    alone = _record_memory(monkeypatch, {"platform_store_memory": LOST})
    assert alone["success"] is False and "Memory NOT saved" in alone["error"]

    # night-1 parity: a mission ticket also writes the shared field; that one landing
    # does not make the workspace-memory write a success
    in_a_mission = _record_memory(monkeypatch, {"platform_store_memory": LOST,
                                                "platform_field_inject": {"success": True}},
                                  mission_field_id="field-7")
    assert in_a_mission["success"] is False
    assert (in_a_mission["stored_durable"], in_a_mission["stored_field"]) == (False, True)
    assert "workspace memory did NOT keep it" in in_a_mission["error"]


def test_record_memory_that_lands_is_unchanged(monkeypatch):
    stored = {"success": True, "message": "Stored in memory"}
    alone = _record_memory(monkeypatch, {"platform_store_memory": stored})
    assert alone["success"] is True and alone["message"].startswith("Recorded in workspace memory")
    both = _record_memory(monkeypatch, {"platform_store_memory": stored, "platform_field_inject": {"success": True}},
                          mission_field_id="field-7")
    assert both["success"] is True and both["message"] == (
        "Recorded in workspace memory and this mission's shared field.")
