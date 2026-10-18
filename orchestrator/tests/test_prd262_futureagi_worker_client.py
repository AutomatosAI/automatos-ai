"""PRD-262 US-006 (the API side): the FutureAGI client sends the worker's token
and holds no database session while it waits on the worker.

- every worker call carries X-Internal-Token (config.AGENT_OPT_WORKER_TOKEN);
- a chat turn reads only the cached "any prompt has scoring on" flag: while it
  is fresh and off, nothing is scheduled and the database is not touched; a stale
  flag is re-read in a thread, never on the event loop;
- live scoring and run_assessment check no session out while the worker is
  awaited (live scoring up to 120 s; an optimisation polls for up to 25 minutes),
  and write their results in a new session.

The database is a stand-in that counts the sessions checked out at each moment.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from core.models.system_prompts import SystemPrompt, SystemPromptEvalRun, SystemPromptVersion
from core.services import futureagi_service as fas

ROOT = Path(__file__).resolve().parents[2]


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    def __init__(self, db: "_Database"):
        self._db = db

    def query(self, model):
        return _Query(self._db.rows.get(model, ()))

    def add(self, obj):
        self._db.added.append(obj)


class _Database:
    """A get_db_session stand-in: counts the sessions checked out right now."""

    def __init__(self, rows=None):
        self.rows = rows or {}
        self.added: list = []
        self.checked_out = 0
        self.opened = 0
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def __call__(self):
        with self._lock:
            self.checked_out += 1
            self.opened += 1
        try:
            yield _Session(self)
        finally:
            with self._lock:
                self.checked_out -= 1


def _forbidden(*_args, **_kwargs):
    raise AssertionError("the database was touched")


@pytest.fixture()
def service(monkeypatch):
    svc = fas.FutureAGIService()
    svc._available = True
    monkeypatch.setattr(fas, "WORKER_URL", "http://agent-opt-worker.test:8080")
    return svc


@pytest.fixture()
def database(monkeypatch):
    db = _Database()
    monkeypatch.setattr("core.database.database.get_db_session", db)
    return db


# --------------------------------------------------------------------------- #
# The token
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token,sent", [("worker-secret", "worker-secret"), ("", None)])
def test_every_worker_call_carries_the_token(service, monkeypatch, token, sent):
    monkeypatch.setattr(fas.config, "AGENT_OPT_WORKER_TOKEN", token)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("X-Internal-Token")))
        return httpx.Response(200, json={"status": "running"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    asyncio.run(service._call_worker("/score", {"input_text": "q", "output_text": "a"}))
    asyncio.run(service._call_worker_get("/optimize/job-1"))
    assert seen == [("POST", "/score", sent), ("GET", "/optimize/job-1", sent)]


# --------------------------------------------------------------------------- #
# The per-turn hook
# --------------------------------------------------------------------------- #


def test_a_turn_with_scoring_known_off_schedules_nothing_and_touches_no_database(service, monkeypatch):
    monkeypatch.setattr("core.database.database.get_db_session", _forbidden)
    monkeypatch.setattr(fas, "_any_prompt_scoring_enabled", _forbidden)
    service._scoring_flag = fas._ScoringFlag(enabled=False, read_at=time.monotonic())

    async def turn():
        service.schedule_live_eval("What is 2+2?", "4", "system prompt")
        return set(service._live_tasks)

    assert asyncio.run(turn()) == set()


def test_a_stale_flag_is_reread_off_the_event_loop(service, monkeypatch):
    reads: list[int] = []

    def read_flag() -> bool:
        reads.append(threading.get_ident())
        return False

    monkeypatch.setattr(fas, "_any_prompt_scoring_enabled", read_flag)
    service._scoring_flag = fas._ScoringFlag(enabled=False, read_at=time.monotonic() - fas.SCORING_FLAG_TTL_SECONDS - 1)

    async def turn() -> int:
        service.schedule_live_eval("What is 2+2?", "4")
        await asyncio.gather(*list(service._live_tasks))
        return threading.get_ident()

    loop_thread = asyncio.run(turn())
    assert reads and all(thread != loop_thread for thread in reads)
    assert service._scoring_flag.enabled is False
    assert service._scoring_flag.fresh(time.monotonic())


def test_a_toggle_drops_the_cached_flag(service):
    service._scoring_flag = fas._ScoringFlag(enabled=False, read_at=time.monotonic())
    service.forget_scoring_flag()
    assert service._scoring_flag is None


def test_an_empty_turn_or_an_unconfigured_worker_schedules_nothing(service):
    async def turns():
        service.schedule_live_eval("", "4")
        service.schedule_live_eval("q", "")
        service._available = False
        service.schedule_live_eval("q", "a")
        return set(service._live_tasks)

    assert asyncio.run(turns()) == set()


def test_live_scoring_holds_no_session_while_the_worker_scores(service, database, monkeypatch):
    prompt = SimpleNamespace(id=uuid.uuid4(), slug="chatbot-friendly", futureagi_eval_enabled=True)
    version = SimpleNamespace(id=uuid.uuid4(), prompt_id=prompt.id, status="active")
    database.rows = {SystemPrompt: [prompt], SystemPromptVersion: [version]}
    during_worker: list[int] = []

    async def worker(path, payload, timeout=fas.WORKER_TIMEOUT):
        during_worker.append(database.checked_out)
        assert path == "/score" and payload["metrics"] == fas.FutureAGIService.LIVE_METRICS
        return {"scores": {"is_helpful": {"score": 1.0}}}

    monkeypatch.setattr(service, "_call_worker", worker)
    asyncio.run(service.eval_live_traffic("What is 2+2?", "4", "system prompt"))

    assert during_worker == [0]
    assert database.checked_out == 0
    assert database.opened == 2  # the flag read, then the write
    [run] = database.added
    assert isinstance(run, SystemPromptEvalRun)
    assert (run.prompt_id, run.version_id, run.run_type, run.status) == (prompt.id, version.id, "live", "completed")


# --------------------------------------------------------------------------- #
# run_assessment
# --------------------------------------------------------------------------- #


def _run(run_type: str, settings=None):
    version = SimpleNamespace(id=uuid.uuid4(), content="You are helpful.", eval_scores=None)
    run = SimpleNamespace(
        id=uuid.uuid4(), version_id=version.id, run_type=run_type, metadata_=settings or {},
        status="pending", started_at=None, completed_at=None, error_message=None, scores=None,
    )
    return run, version


@pytest.fixture()
def no_sleep(monkeypatch, database):
    """asyncio.sleep returns at once, and records the sessions checked out while polling."""
    during_sleep: list[int] = []

    async def sleep(_seconds, *_args, **_kwargs):
        during_sleep.append(database.checked_out)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return during_sleep


def test_no_session_is_checked_out_while_optimize_prompt_polls(service, database, no_sleep, monkeypatch):
    run, version = _run("optimize", {"num_iterations": 1})
    database.rows = {SystemPromptEvalRun: [run], SystemPromptVersion: [version]}
    during_worker: list[int] = []
    polls = iter([{"status": "running"}, {"status": "running"}, {
        "status": "completed", "optimized_prompt": "You are very helpful.", "final_score": 0.9,
        "rounds_completed": 1, "duration_seconds": 3.0,
    }])

    async def dataset(limit=20):
        during_worker.append(database.checked_out)
        return [{"input": "q", "output": "a"}]

    async def start(path, payload, timeout=fas.WORKER_TIMEOUT):
        during_worker.append(database.checked_out)
        assert path == "/optimize"
        return {"job_id": "job-1"}

    async def poll(path, timeout=15):
        during_worker.append(database.checked_out)
        assert path == "/optimize/job-1"
        return next(polls)

    monkeypatch.setattr(service, "_collect_optimization_dataset", dataset)
    monkeypatch.setattr(service, "_call_worker", start)
    monkeypatch.setattr(service, "_call_worker_get", poll)
    asyncio.run(service.run_assessment(str(run.id)))

    assert len(during_worker) == 5 and set(during_worker) == {0}
    assert no_sleep and set(no_sleep) == {0}
    assert database.checked_out == 0 and database.opened == 2
    assert run.status == "completed" and run.started_at and run.completed_at
    assert run.scores["optimized_prompt"] == "You are very helpful."


def test_an_assessment_writes_its_scores_in_a_new_session(service, database, monkeypatch):
    run, version = _run("assess", {"metrics": ["is_helpful"]})
    database.rows = {SystemPromptEvalRun: [run], SystemPromptVersion: [version]}
    scores = {"is_helpful": {"score": 1.0, "passed": True}}
    during_worker: list[int] = []

    async def dataset(limit=20):
        return []

    async def assess(path, payload, timeout=fas.WORKER_TIMEOUT):
        during_worker.append(database.checked_out)
        assert path == "/assess" and payload["metrics"] == ["is_helpful"]
        return {"scores": scores, "metrics_run": 1}

    monkeypatch.setattr(service, "_collect_optimization_dataset", dataset)
    monkeypatch.setattr(service, "_call_worker", assess)
    asyncio.run(service.run_assessment(str(run.id)))

    assert during_worker == [0]
    assert run.status == "completed" and version.eval_scores == scores
    assert database.opened == 2 and database.checked_out == 0


def test_a_worker_failure_marks_the_run_failed_in_a_new_session(service, database, monkeypatch):
    run, version = _run("safety")
    database.rows = {SystemPromptEvalRun: [run], SystemPromptVersion: [version]}

    async def broken(path, payload, timeout=fas.WORKER_TIMEOUT):
        assert database.checked_out == 0
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(service, "_call_worker", broken)
    asyncio.run(service.run_assessment(str(run.id)))

    assert run.status == "failed" and run.error_message == "worker exploded" and run.completed_at
    assert database.checked_out == 0


def test_a_missing_version_fails_the_run_without_calling_the_worker(service, database, monkeypatch):
    run, _ = _run("assess")
    database.rows = {SystemPromptEvalRun: [run]}
    monkeypatch.setattr(service, "_call_worker", _forbidden)
    asyncio.run(service.run_assessment(str(run.id)))
    assert run.status == "failed" and run.error_message == "Version not found"


# --------------------------------------------------------------------------- #
# The callers
# --------------------------------------------------------------------------- #


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    node = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return ast.get_source_segment(source, node)


def test_the_chat_turn_calls_the_cached_hook_only():
    hook = _function_source(ROOT / "orchestrator" / "consumers" / "chatbot" / "service.py", "_post_response")
    assert "futureagi_service.schedule_live_eval(" in hook
    assert "eval_live_traffic" not in hook and "is_available" not in hook


def test_the_admin_toggle_drops_the_cached_flag():
    toggle = _function_source(ROOT / "orchestrator" / "api" / "admin_prompts.py", "toggle_futureagi")
    assert "futureagi_service.forget_scoring_flag()" in toggle
