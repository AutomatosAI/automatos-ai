"""PRD-262 US-044: the worker's task_executions write, without SQLAlchemy and off the loop.

The workspace worker wrote each task's lifecycle to task_executions through a
SQLAlchemy engine, synchronously, on the event loop that also serves its health,
canvas and file routes. SQLAlchemy (27.7 MB installed) existed in the worker for
that one UPDATE. It now runs through psycopg2 (task_executions_db.py), in a thread
(main.py, asyncio.to_thread), with bound %(name)s parameters, after dropping a
SQLAlchemy driver suffix from DATABASE_URL that libpq would refuse.

The last test runs the statement against the CI Postgres in a scratch schema, so
the CASE/COALESCE text is proven against a real server, through a
postgresql+psycopg2:// URL.
"""
from __future__ import annotations

import importlib.util
import re
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import psycopg2
import pytest

from tests.helpers_workspace_worker import WORKER_DIR, load_worker_main

TASK_ID = "6f1c2a4e-9b0d-4c55-8e21-0d6c3b7a9f10"
WORKER_ID = "worker-7-1760000000"


def _load_db_module():
    spec = importlib.util.spec_from_file_location("workspace_worker_task_executions_db", WORKER_DIR / "task_executions_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


db = _load_db_module()


@pytest.mark.parametrize(("url", "dsn"), [
    ("postgresql+psycopg2://u:p@db:5432/x", "postgresql://u:p@db:5432/x"),
    ("postgres+psycopg2://u:p@db/x?sslmode=require", "postgresql://u:p@db/x?sslmode=require"),
    ("postgresql+asyncpg://u:p@db/x", "postgresql://u:p@db/x"),
    ("postgresql://u:p@db:5432/x", "postgresql://u:p@db:5432/x"),
    ("postgres://u:p@db:5432/x", "postgres://u:p@db:5432/x"),
    ("  postgresql+psycopg2://u:p+q@db/x  ", "postgresql://u:p+q@db/x"),
])
def test_a_sqlalchemy_driver_scheme_is_dropped_for_libpq(url, dsn):
    assert db.libpq_dsn(url) == dsn


@pytest.mark.parametrize(("status", "started", "finished"), [
    ("running", True, False),
    ("completed", False, True),
    ("failed", False, True),
    ("timed_out", False, True),
    ("cancelled", False, True),
    ("queued", False, False),
])
def test_each_status_moves_only_its_own_timestamp(status, started, finished):
    now = datetime.now(timezone.utc)
    params = db.update_parameters(TASK_ID, status, WORKER_ID, None, now)
    assert (params["started"], params["finished"]) == (started, finished)
    assert params["now"] is now and params["status"] == status and params["worker_id"] == WORKER_ID


@pytest.mark.parametrize(("error", "bound"), [(None, None), ("", None), ("boom", "boom")])
def test_an_empty_error_leaves_the_message_alone(error, bound):
    assert db.update_parameters(TASK_ID, "failed", WORKER_ID, error, datetime.now(timezone.utc))["error"] == bound


def test_the_statement_is_fixed_text_with_bound_named_parameters():
    sql = db.UPDATE_TASK_EXECUTION
    placeholders = set(re.findall(r"%\((\w+)\)s", sql))
    params = db.update_parameters(TASK_ID, "running", WORKER_ID, "x", datetime.now(timezone.utc))
    assert placeholders == set(params)
    assert sql.count("%") == len(re.findall(r"%\(\w+\)s", sql)), "a bare % would break psycopg2's formatting"
    assert "{" not in sql and ":" not in sql, "no format fields and no SQLAlchemy-style :params"
    assert re.search(r"WHERE id = %\(task_id\)s", sql)


class _Recorder:
    """psycopg2.connect, recorded: the DSN, the timeout, the statement, the commit and the close."""

    def __init__(self):
        self.calls, self.executed, self.committed, self.closed = [], [], False, False

    def connect(self, dsn, **kwargs):
        self.calls.append((dsn, kwargs))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.committed = exc_type is None
        return False

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def close(self):
        self.closed = True


def test_one_status_change_is_one_committed_statement_on_a_closed_connection(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(db.psycopg2, "connect", recorder.connect)
    assert db.record_task_status("postgresql+psycopg2://u:p@db/x", TASK_ID, "completed", WORKER_ID, "boom") is True
    assert recorder.calls == [("postgresql://u:p@db/x", {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS})]
    ((sql, params),) = recorder.executed
    assert sql == db.UPDATE_TASK_EXECUTION
    assert (params["task_id"], params["status"], params["finished"], params["error"]) == (TASK_ID, "completed", True, "boom")
    assert recorder.committed and recorder.closed


def test_a_database_error_is_logged_and_reported_not_raised(monkeypatch):
    def refuse(dsn, **kwargs):
        raise psycopg2.OperationalError("connection refused")

    logged = []

    class _Logger:
        def exception(self, message, *args):
            logged.append(message % args)

    monkeypatch.setattr(db.psycopg2, "connect", refuse)
    monkeypatch.setattr(db, "logger", _Logger())
    assert db.record_task_status("postgresql://u:p@db/x", TASK_ID, "running", WORKER_ID) is False
    assert logged == ["DB update failed for task 6f1c2a4e (running)"]


class _RefusingRecorder(_Recorder):
    """A connection whose statement psycopg2 refuses before sending it."""

    def execute(self, sql, params):
        raise ValueError("A string literal cannot contain NUL (0x00) characters.")


def test_a_parameter_psycopg2_refuses_is_logged_and_reported_not_raised(monkeypatch):
    """psycopg2 raises ValueError, not psycopg2.Error, for a NUL byte in a string: an error
    message from a command's binary output. The SQLAlchemy write caught it with every other
    exception, and a status write still must not fail the task."""
    recorder, logged = _RefusingRecorder(), []

    class _Logger:
        def exception(self, message, *args):
            logged.append(message % args)

    monkeypatch.setattr(db.psycopg2, "connect", recorder.connect)
    monkeypatch.setattr(db, "logger", _Logger())
    assert db.record_task_status("postgresql://u:p@db/x", TASK_ID, "failed", WORKER_ID, "bad\x00output") is False
    assert logged == ["DB update failed for task 6f1c2a4e (failed)"]
    assert not recorder.committed and recorder.closed


@pytest.mark.asyncio
async def test_the_worker_writes_in_a_thread_not_on_its_event_loop(monkeypatch, tmp_path):
    worker_main = load_worker_main(monkeypatch, tmp_path)
    calls = []

    def record(url, task_id, status, worker_id, error=None):
        calls.append((threading.get_ident(), url, task_id, status, worker_id, error))
        return True

    monkeypatch.setattr(worker_main, "record_task_status", record)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@db/x")
    worker = worker_main.WorkspaceWorker()
    await worker._update_db_execution(TASK_ID, "failed", error="boom")
    ((thread, url, task_id, status, worker_id, error),) = calls
    assert thread != threading.get_ident(), "the UPDATE ran on the event loop's thread"
    assert (url, task_id, status, worker_id, error) == (
        "postgresql+psycopg2://u:p@db/x", TASK_ID, "failed", worker._worker_id, "boom")


@pytest.mark.asyncio
async def test_no_database_url_skips_the_write(monkeypatch, tmp_path):
    worker_main = load_worker_main(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_main, "record_task_status", lambda *a, **k: pytest.fail("wrote without a DATABASE_URL"))
    monkeypatch.setenv("DATABASE_URL", "  ")
    await worker_main.WorkspaceWorker()._update_db_execution(TASK_ID, "running")


def test_the_worker_no_longer_imports_sqlalchemy():
    for path in WORKER_DIR.glob("*.py"):
        assert not re.search(r"^\s*(?:from|import)\s+sqlalchemy\b", path.read_text(), re.M), path.name


# ── against the CI Postgres ─────────────────────────────────────────────────

TABLE = """
CREATE TABLE task_executions (
    id UUID PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    worker_id VARCHAR(255)
)
"""


@pytest.fixture
def scratch_schema():
    """A schema holding task_executions as the platform defines its columns; dropped after."""
    from core.database.database import get_database_url

    try:
        url = get_database_url()
        admin = psycopg2.connect(db.libpq_dsn(url), connect_timeout=5)
    except Exception as exc:  # noqa: BLE001 - no database configured or reachable: skip, as the other Postgres tests do
        pytest.skip(f"needs the Postgres test database: {exc}")
    admin.autocommit = True
    schema = f"prd262_us044_{uuid.uuid4().hex[:8]}"
    with admin.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
        cur.execute(f'SET search_path TO "{schema}"')
        cur.execute(TABLE)
        cur.execute("INSERT INTO task_executions (id, status) VALUES (%s, 'queued')", (TASK_ID,))
    try:
        yield url, schema, admin
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
        admin.close()


def _row(admin) -> tuple:
    with admin.cursor() as cur:
        cur.execute("SELECT status, worker_id, started_at, completed_at, error_message FROM task_executions WHERE id = %s",
                    (TASK_ID,))
        return cur.fetchone()


def test_the_lifecycle_lands_in_postgres_through_a_sqlalchemy_style_url(scratch_schema):
    url, schema, admin = scratch_schema
    separator = "&" if "?" in url else "?"
    worker_url = re.sub(r"^postgres(?:ql)?(?:\+\w+)?://", "postgresql+psycopg2://", url)
    worker_url += f"{separator}options={quote(f'-csearch_path={schema}')}"

    assert db.record_task_status(worker_url, TASK_ID, "running", WORKER_ID) is True
    status, worker_id, started, completed, error = _row(admin)
    assert (status, worker_id, completed, error) == ("running", WORKER_ID, None, None)
    assert started is not None

    assert db.record_task_status(worker_url, TASK_ID, "failed", WORKER_ID, "boom") is True
    assert _row(admin)[:2] == ("failed", WORKER_ID)
    assert _row(admin)[2] == started, "finishing must not move started_at"
    assert _row(admin)[3] is not None and _row(admin)[4] == "boom"

    assert db.record_task_status(worker_url, TASK_ID, "cancelled", WORKER_ID, None) is True
    assert _row(admin)[4] == "boom", "a change with no error keeps the earlier message"

    # The real psycopg2 refuses a NUL byte with a ValueError: reported, not raised.
    assert db.record_task_status(worker_url, TASK_ID, "failed", WORKER_ID, "bad\x00output") is False
    assert _row(admin)[0] == "cancelled" and _row(admin)[4] == "boom"
