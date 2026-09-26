"""F105 (26 Sep): a database question never freezes the event loop.

DatabaseKnowledgeService.query_database is async, but three of its steps were
sync and ran on the loop. It sampled column values from the owner's database
(up to 40 columns, 5 s each), generated the SQL through
llm_provider.generate_response_sync, and ran the query on the owner's database
(statement timeout 30 s by default). execute_template ran its query the same
way. The backend's scheduler slipped by 1.12 to 1.67 s at each of the three
NL2SQL calls between 01:41 and 01:57Z (01:46:49, 01:47:19, 01:53:19).

A fake source, model and database hold their thread the way the real ones do;
the tests watch the loop around them. No database, no model.
"""
import asyncio
import contextvars
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SERVICE = "modules.nl2sql.service"
HOLD_SECONDS = 0.4
CREDS = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"}
_caller = contextvars.ContextVar("f105_nl2sql_caller", default=None)


class _Model:
    """generate_response_sync that holds its thread like a model call, and notes
    where it ran."""

    def __init__(self):
        self.calls = []

    def generate_response_sync(self, messages):
        self.calls.append(SimpleNamespace(thread=threading.get_ident(), caller=_caller.get()))
        time.sleep(HOLD_SECONDS)
        return SimpleNamespace(content="SQL:\nSELECT id FROM users\nEXPLANATION:\nEvery user's id.")


def _slow(result=None):
    def hold(*args, **kwargs):
        time.sleep(HOLD_SECONDS)
        return result
    return hold


def _source():
    return SimpleNamespace(
        dialect="postgresql", workspace_id="ws-c1", credential_id=1,
        schema_metadata={"tables": [{"name": "users", "columns": [{"name": "id", "type": "integer"}]}]},
        semantic_layer=None, query_timeout_seconds=30, max_rows_limit=1000,
    )


@pytest.fixture
def service(monkeypatch):
    from modules.nl2sql.service import DatabaseKnowledgeService

    svc = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
    svc.llm_provider = _Model()
    svc._get_source = AsyncMock(return_value=_source())
    svc._get_example_store = MagicMock(return_value=None)
    svc._calculate_confidence = MagicMock(return_value={"score": 1.0})
    svc._augment_schema_with_samples = _slow()
    svc._run_sql_with_guards = _slow((["id"], [{"id": 1}]))
    svc._decrypt_source_credentials = MagicMock(return_value=CREDS)

    credentials = MagicMock()
    credentials.get_credential.return_value = SimpleNamespace(encrypted_data=b"x")
    encryption = MagicMock()
    encryption.decrypt_dict.return_value = CREDS
    monkeypatch.setattr(f"{SERVICE}._emit_nl2sql_primitive", MagicMock())
    monkeypatch.setattr("core.database.database.SessionLocal", MagicMock())
    monkeypatch.setattr("core.credentials.service.CredentialStore", MagicMock(return_value=credentials))
    monkeypatch.setattr("core.credentials.encryption.EncryptionService", MagicMock(return_value=encryption))
    monkeypatch.setattr("modules.context.ContextService", MagicMock())
    return svc


async def _watched(step):
    """``await step()`` and the longest the loop stood still meanwhile."""
    gaps, done = [], asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    result = await step()
    done.set()
    await beat
    return result, max(gaps)


def _ask(service):
    return service.query_database(
        source_id="1", natural_language_query="How many users do I have?", user_id="u-1",
        agent_id="a-1", workspace_id="ws-c1", auto_train=False,
    )


@pytest.mark.asyncio
async def test_a_database_question_keeps_the_loop_running(service):
    result, stood_still = await _watched(lambda: _ask(service))

    assert stood_still < 0.3, f"the loop stood still for {stood_still:.2f} s during one database question"
    assert (result["success"], result["data"]) == (True, [{"id": 1}])
    assert len(service.llm_provider.calls) == 1


@pytest.mark.asyncio
async def test_the_model_call_runs_on_a_thread_with_the_callers_context(service):
    token = _caller.set("req-6058c308ef31")
    try:
        await _ask(service)
    finally:
        _caller.reset(token)

    (call,) = service.llm_provider.calls
    assert call.thread != threading.get_ident()
    assert call.caller == "req-6058c308ef31"


@pytest.mark.asyncio
async def test_a_query_template_keeps_the_loop_running(service, monkeypatch):
    template = SimpleNamespace(sql_template="SELECT id FROM users WHERE id = :id",
                               visualization_type="table", usage_count=0)
    lookup = MagicMock()
    lookup.filter.return_value = lookup
    lookup.first.return_value = template
    db = MagicMock()
    db.query.return_value = lookup
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    monkeypatch.setattr("core.models.database_knowledge.DatabaseQueryTemplate", MagicMock())

    result, stood_still = await _watched(lambda: service.execute_template(
        source_id=1, template_id=3, parameters={"id": 1}, workspace_id="ws-c1", max_rows=100))

    assert stood_still < 0.3, f"the loop stood still for {stood_still:.2f} s during one template query"
    assert (result["success"], result["data"]) == (True, [{"id": 1}])
