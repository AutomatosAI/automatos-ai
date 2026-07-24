"""PRD-142 Wave 1 · WS-D · W1-S9 — no connection may sit 'idle in transaction'.

Background (PRD-135): a single DB session is held open across long ``await``
points on four hot surfaces — the coordinator's mission tick, the mission
reconciler's verification loop, the heartbeat tick, and the harness tick. Each
one issues a SELECT (SQLAlchemy opens a transaction on first query) and *then*
awaits an LLM call or an ``asyncio.gather`` of agent coroutines. For the whole
duration of that await the backing connection is 'idle in transaction': it
holds row locks and blocks DDL. The smoking gun was a 9-hour idle SELECT on
``agents`` that wedged a migration.

The structural fix is a single helper, ``end_open_transaction(db)``, called
*immediately before* each long await. It commits the session, which ends the
open transaction and returns the connection to an idle (not idle-in-tx) state.
Because pending writes are flushed and committed at that point, this shifts the
write boundary to incremental commits — a deliberate atomicity change, covered
per surface below.

These tests drive the helper and each surface with recording fakes — no real DB
and no real LLM. The ordering tests are the heart of W1-S9: they assert the
commit happens *between* the read and the await, which is the property that
clears idle-in-transaction.
"""
import os
import sys
import types
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# The heartbeat tool loop imports consumers.chatbot.personality, whose package
# __init__ eagerly pulls the chatbot stack → RAG → camelot, an optional PDF dep
# not installed in the unit-test env. Stub it so the import chain resolves.
sys.modules.setdefault("camelot", types.ModuleType("camelot"))


def _stub(name, **attrs):
    """Register a stub module in sys.modules (only if absent) with attrs set."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# heartbeat_service / harness_service import APScheduler at module load. It's a
# real prod dep but isn't installed in the unit-test env, and these tests never
# touch the scheduler — stub the tree so the modules import.
if "apscheduler" not in sys.modules:
    _stub("apscheduler")
    _stub("apscheduler.schedulers")
    _stub("apscheduler.schedulers.asyncio", AsyncIOScheduler=type("AsyncIOScheduler", (), {}))
    _stub("apscheduler.jobstores")
    _stub("apscheduler.jobstores.memory", MemoryJobStore=type("MemoryJobStore", (), {}))
    _stub("apscheduler.triggers")
    _stub("apscheduler.triggers.cron", CronTrigger=type("CronTrigger", (), {}))

# Importing database.py builds the SQLAlchemy engine, which refuses to construct
# without POSTGRES_* creds. These tests never touch a real DB; setdefault means
# a real .env still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.database.database as dbmod  # noqa: E402


class _RecordingSession:
    """Records the transaction-lifecycle calls made against it, in order."""

    def __init__(self):
        self.calls = []

    def commit(self):
        self.calls.append("commit")

    def rollback(self):
        self.calls.append("rollback")

    def close(self):
        self.calls.append("close")


# ---------------------------------------------------------------------------
# Task #60 — the shared helper
# ---------------------------------------------------------------------------

def test_end_open_transaction_commits_the_session():
    """The helper must end the open transaction by committing — nothing else."""
    rec = _RecordingSession()
    dbmod.end_open_transaction(rec)
    assert rec.calls == ["commit"]


def test_end_open_transaction_is_idempotent_after_commit():
    """Calling it twice just commits twice; a commit with no active tx is a
    harmless no-op at the driver level, so the helper need not guard."""
    rec = _RecordingSession()
    dbmod.end_open_transaction(rec)
    dbmod.end_open_transaction(rec)
    assert rec.calls == ["commit", "commit"]


# ---------------------------------------------------------------------------
# Task #61 — the reconciler verification loop (the documented 9hr leak)
#
# _verify_completed_tasks() SELECTs the completed tasks and the assigned
# agent's model, *then* awaits VerificationService.verify_task() — an LLM call.
# Pre-fix, the session held that read transaction open for the whole LLM call,
# leaving the connection idle-in-transaction (PRD-135: the 9hr idle SELECT on
# agents). The fix commits *between* the reads and the await. These fakes record
# the order of DB and verify operations so we can assert that boundary.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from modules.coordination import reconciler as rec  # noqa: E402
from modules.coordination.reconciler import MissionReconciler  # noqa: E402


class _OrderingQuery:
    """Chainable query stub returning a fixed task list."""

    def __init__(self, tasks):
        self._tasks = tasks

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return self._tasks


class _OrderingDB:
    """Fake Session recording read/commit/flush order in a shared event log."""

    def __init__(self, tasks, events):
        self._tasks = tasks
        self.events = events

    def query(self, *a, **k):
        self.events.append("select")
        return _OrderingQuery(self._tasks)

    def commit(self):
        self.events.append("commit")

    def flush(self):
        self.events.append("flush")

    def rollback(self):
        self.events.append("rollback")


@pytest.mark.asyncio
async def test_reconciler_commits_after_reads_before_verify(monkeypatch):
    """The read transaction must be committed *before* the verify_task await.

    Asserts the ordering select < commit < verify (and that the agent-model
    read is also committed before the await). This is the property that clears
    'idle in transaction' across the long LLM call.
    """
    events: list[str] = []

    task = SimpleNamespace(
        id="task-1",
        title="Do the thing",
        description="describe it",
        output="the output",
        verification_criteria=None,
        task_type="research",
        state="completed",
        sequence_number=1,
        tokens_used=0,
        output_metadata=None,
        assigned_agent_id="agent-1",
    )
    run = SimpleNamespace(
        id="run-1",
        config=None,  # → skip_verification False → real verify path
        tokens_used=0,
        token_budget_estimate=None,
    )
    db = _OrderingDB([task], events)

    # The assigned-agent model read is the documented SELECT-on-agents. Record
    # it on the shared log so we can assert it too is committed before verify.
    def _fake_executor_model(_db, _task):
        events.append("select_agent")
        return "model-x"

    monkeypatch.setattr(
        MissionReconciler, "_get_executor_model", staticmethod(_fake_executor_model)
    )

    # State-machine writes and event emission are no-ops for this ordering test.
    monkeypatch.setattr(rec, "transition_task", lambda **k: None)
    monkeypatch.setattr(rec, "sync_board_status", lambda *a, **k: None)
    monkeypatch.setattr(rec, "emit_event", lambda **k: None)

    async def _fake_apply_pass(_db, _task):
        events.append("apply")

    monkeypatch.setattr(
        MissionReconciler, "_apply_verdict_pass", staticmethod(_fake_apply_pass)
    )

    class _FakeVerificationService:
        async def verify_task(self, **kwargs):
            events.append("verify")
            return SimpleNamespace(
                verdict=rec.VERDICT_PASS,
                scores={},
                reasoning="ok",
                confidence=1.0,
                deterministic_passed=True,
                tokens_used=0,
            )

    monkeypatch.setattr(rec, "VerificationService", _FakeVerificationService)

    await MissionReconciler._verify_completed_tasks(db, run)

    assert "commit" in events, "session was never committed before the LLM await"
    assert "verify" in events
    assert events.index("select") < events.index("commit"), events
    assert events.index("select_agent") < events.index("commit"), events
    assert events.index("commit") < events.index("verify"), events


# ---------------------------------------------------------------------------
# Task #62 — the coordinator mission tick (parallel agent I/O)
#
# _process_run() prepares tasks (transition to RUNNING, build prompts, activate
# agents — all writes) and *then* awaits asyncio.gather() over the agents' LLM
# calls. Pre-fix it only flush()ed before the gather, so the session held the
# prep transaction open for the entire parallel phase. The fix commits before
# the gather. The fake gather records when the parallel phase begins so we can
# assert the commit precedes it.
# ---------------------------------------------------------------------------

from services import coordinator_service as csmod  # noqa: E402


class _CoordQuery:
    def __init__(self, agents, task):
        self._agents = agents
        self._task = task

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._agents

    def first(self):
        return self._task


class _CoordDB:
    def __init__(self, agents, task, events):
        self._agents = agents
        self._task = task
        self.events = events

    def query(self, *a, **k):
        self.events.append("select")
        return _CoordQuery(self._agents, self._task)

    def commit(self):
        self.events.append("commit")

    def flush(self):
        self.events.append("flush")

    def rollback(self):
        self.events.append("rollback")

    def refresh(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_coordinator_commits_prep_before_parallel_gather(monkeypatch):
    """Phase-1 prep must be committed before the asyncio.gather of agent I/O.

    Asserts commit < agent_io, i.e. the prep transaction is closed before the
    parallel LLM phase — clearing idle-in-transaction across the gather.
    """
    events: list[str] = []

    agent = SimpleNamespace(id="agent-1")
    task = SimpleNamespace(id="task-1")
    run = SimpleNamespace(
        workspace_id="ws-1",
        config={"field_id": "f1"},  # has field → skips field creation/awaits
        id="run-1",
        state=csmod.RunState.RUNNING.value,  # non-terminal, non-verifying
    )
    db = _CoordDB([agent], task, events)
    svc = csmod.CoordinatorService()

    # No shared-context backend → field exists-check branch is skipped.
    monkeypatch.setattr(svc, "_get_field", lambda: None)

    # One ready task dispatched to one agent.
    dispatched = SimpleNamespace(dispatched=True, task_id="task-1", agent_id="agent-1")
    monkeypatch.setattr(
        csmod.MissionDispatcher, "dispatch_ready",
        staticmethod(lambda _db, _run, _agents: [dispatched]),
    )

    async def _fake_prepare(_db, _run, _task, _agent_id):
        return {
            "factory": object(),
            "agent_id": _agent_id,
            "prompt": "do it",
            "task": _task,
            "attachment_ids": [],
            "mode_caps": {},
            "agent_runtime": None,
        }

    monkeypatch.setattr(svc, "_prepare_task", _fake_prepare)

    async def _fake_run_agent_io(*a, **k):
        events.append("agent_io")  # the parallel LLM phase is now running
        return {"status": "ok"}

    monkeypatch.setattr(svc, "_run_agent_io", _fake_run_agent_io)

    async def _fake_record(*a, **k):
        events.append("record")

    monkeypatch.setattr(svc, "_record_task_result", _fake_record)

    async def _fake_reconcile(_db, _run):
        events.append("reconcile")

    monkeypatch.setattr(csmod.MissionReconciler, "reconcile", staticmethod(_fake_reconcile))

    await svc._process_run(db, run)

    assert "commit" in events, "prep was never committed before the gather"
    assert "agent_io" in events
    assert events.index("commit") < events.index("agent_io"), events


# ---------------------------------------------------------------------------
# Task #63 — the heartbeat tool loop
#
# _orchestrator_tick_llm() builds context (reads) and then loops calling
# llm.generate_response() — an LLM await — once per tool iteration. Pre-fix the
# session held the build_context reads (and each prior tool's writes) open
# across every generate_response. The fix commits at the top of each loop pass.
# ---------------------------------------------------------------------------


class _SimpleDB:
    """Fake Session recording commit/close in a shared event log."""

    def __init__(self, events):
        self.events = events

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")

    def flush(self):
        self.events.append("flush")

    def close(self):
        self.events.append("close")


@pytest.mark.asyncio
async def test_heartbeat_commits_before_each_generate(monkeypatch):
    """The session must be committed before each generate_response LLM await.

    Asserts select(build_context) < commit < generate — the read transaction is
    closed before the LLM call, clearing idle-in-transaction in the tool loop.
    """
    events: list[str] = []

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _SimpleDB(events))
    monkeypatch.setattr(
        "consumers.chatbot.personality.load_orchestrator_settings",
        lambda _ws: {},
    )
    monkeypatch.setattr("core.services.auto_cadence.build_cadence_block", lambda _c: "")
    # config.AGENT_HEARTBEAT_MAX_TOOL_ITERATIONS reads a system_setting from the
    # DB; there's none in this unit env. Give the loop a small fixed budget.
    monkeypatch.setattr(
        "core.llm.manager.get_system_setting", lambda category, key, default=None: 2
    )

    class _FakeContext:
        system_prompt = "system"
        tools = []

    class _FakeContextService:
        def __init__(self, _db):
            pass

        async def build_context(self, **kwargs):
            events.append("select")  # build_context issues the reads
            return _FakeContext()

    monkeypatch.setattr("modules.context.ContextService", _FakeContextService)

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

        async def generate_response(self, messages, tools=None):
            events.append("generate")
            return SimpleNamespace(content="done", tool_calls=[], usage={})

    monkeypatch.setattr("core.llm.manager.LLMManager", _FakeLLM)

    class _FakeExecutor:
        def __init__(self, _db, _ws):
            pass

    monkeypatch.setattr(
        "modules.tools.discovery.platform_executor.PlatformActionExecutor", _FakeExecutor
    )

    from services.heartbeat_service import HeartbeatService

    svc = HeartbeatService()
    result = {"findings": [], "actions_taken": [], "tokens_used": 0}
    await svc._orchestrator_tick_llm("ws-1", {}, result)

    assert "commit" in events, "session was never committed before the LLM await"
    assert "generate" in events
    assert events.index("select") < events.index("commit"), events
    assert events.index("commit") < events.index("generate"), events


# ---------------------------------------------------------------------------
# Task #64 — the harness weekly pipeline
#
# _harness_tick() reads dormancy/config and then awaits five phases (collect,
# diagnose, prescribe, apply, baseline) on one shared session. Pre-fix that
# session stayed idle-in-transaction across every phase. The fix commits before
# each phase. We assert a commit immediately precedes each phase.
# ---------------------------------------------------------------------------


class _GetQuery:
    def __init__(self, obj):
        self._obj = obj

    def get(self, *a, **k):
        return self._obj

    # §12.3: the tick now reads the autonomy level via
    # db.query(Workspace).filter(...).first() — serve the same workspace.
    def filter(self, *a, **k):
        return self

    def first(self):
        return self._obj


class _HarnessDB:
    def __init__(self, ws, events):
        self._ws = ws
        self.events = events

    def query(self, *a, **k):
        return _GetQuery(self._ws)

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")

    def flush(self):
        self.events.append("flush")

    def close(self):
        pass


@pytest.mark.asyncio
async def test_harness_commits_before_each_phase(monkeypatch):
    """A commit must precede every one of the five awaited phases.

    Asserts the exact commit/phase interleaving, proving the connection is never
    idle-in-transaction across a phase await.
    """
    events: list[str] = []
    ws = SimpleNamespace(settings={})

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _HarnessDB(ws, events))

    from services.harness_service import HarnessService

    svc = HarnessService()

    monkeypatch.setattr(
        svc, "_sufficiency_breakdown",
        lambda _db, _ws: {
            "agents_ok": True, "data_ok": True,
            "active_agents": 1, "min_required_agents": 1,
            "heartbeat_days_available": 7, "min_required_days": 1,
        },
    )
    monkeypatch.setattr(svc, "_workspace_allows_auto_apply", lambda _s: True)
    monkeypatch.setattr(svc, "_write_last_run", lambda *a, **k: None)

    async def _collect(_ws, _db):
        events.append("collect")
        return {}

    async def _diagnose(_ws, _metrics, _db):
        events.append("diagnose")
        return {}

    async def _prescribe(_ws, _diag, _metrics, _db):
        events.append("prescribe")
        return []

    async def _apply(_ws, _rx, _db, allow_auto_apply=True, max_risk=None):
        events.append("apply")
        return {"applied": [], "queued": []}

    async def _baseline(_ws, _metrics, _diag, _rx, _changelog, _db):
        events.append("baseline")
        return ({"iteration": 1}, {})

    monkeypatch.setattr(svc, "_phase_collect", _collect)
    monkeypatch.setattr(svc, "_phase_diagnose", _diagnose)
    monkeypatch.setattr(svc, "_phase_prescribe", _prescribe)
    monkeypatch.setattr(svc, "_phase_apply", _apply)
    monkeypatch.setattr(svc, "_phase_baseline", _baseline)

    await svc._harness_tick("ws-1")

    assert events == [
        "commit", "collect",
        "commit", "diagnose",
        "commit", "prescribe",
        "commit", "apply",
        "commit", "baseline",
    ], events
