"""PRD-278 W3 (US-303): the workspace worker's one database connection is secured by the
same two settings as the platform's, read in its own config module.

``worker_config.database_tls_params()`` reads ``DATABASE_SSLMODE`` and
``DATABASE_SSLROOTCERT``; the worker reads them once at start and hands them to
``task_executions_db.record_task_status``, which passes them to ``psycopg2.connect`` as
libpq's ``sslmode`` and ``sslrootcert`` keywords, over the same keys in the address. libpq
then checks the server's certificate against that CA, and its host name for verify-full.
Neither set: no keyword, as before. verify-ca or verify-full without a CA file, and a CA
file that is missing or holds no certificate, stop the worker at start. No database: psycopg2.connect is recorded; the CA files are
made at run time (``tests/helpers_prd278_w3_ca.py``).
"""
from __future__ import annotations

import importlib.util

import pytest

from tests.helpers_prd278_w3_ca import write_ca_bundle
from tests.helpers_workspace_worker import WORKER_DIR, load_worker_main

TASK_ID = "6f1c2a4e-9b0d-4c55-8e21-0d6c3b7a9f10"
WORKER_ID = "worker-7-1760000000"
# The path the chart mounts the CA at; read only by the tests that never open it.
CA_IN_POD = "/etc/automatos/database-ca/ca.crt"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"workspace_worker_{name}_prd278", WORKER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker_config = _load("worker_config")
db = _load("task_executions_db")


def _address() -> str:
    """A worker DATABASE_URL in SQLAlchemy's driver form, built from parts."""
    return "postgresql+psycopg2" + "://" + "worker" + "@" + "db.example.invalid" + "/" + "app"


@pytest.fixture
def tls_env(monkeypatch):
    monkeypatch.delenv("DATABASE_SSLMODE", raising=False)
    monkeypatch.delenv("DATABASE_SSLROOTCERT", raising=False)

    def apply(sslmode=None, sslrootcert=None):
        for name, value in (("DATABASE_SSLMODE", sslmode), ("DATABASE_SSLROOTCERT", sslrootcert)):
            if value is not None:
                monkeypatch.setenv(name, value)

    return apply


@pytest.fixture
def ca_file(tmp_path) -> str:
    return str(write_ca_bundle(tmp_path / "ca.crt"))


class _Recorder:
    """psycopg2.connect, recorded, with a connection that commits one statement."""

    def __init__(self):
        self.calls = []

    def connect(self, dsn, **kwargs):
        self.calls.append((dsn, kwargs))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params):
        return None

    def close(self):
        return None


def test_neither_setting_gives_no_keyword(tls_env):
    assert worker_config.database_tls_params() == {}


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
def test_a_mode_that_checks_no_certificate_needs_no_ca(tls_env, mode):
    tls_env(sslmode=mode)
    assert worker_config.database_tls_params() == {"sslmode": mode}


def test_verify_full_with_a_ca_path_gives_both_keywords(tls_env, ca_file):
    tls_env(sslmode="VERIFY-FULL ", sslrootcert=ca_file)
    assert worker_config.database_tls_params() == {"sslmode": "verify-full", "sslrootcert": ca_file}


def test_the_write_connects_with_verify_full_and_the_ca_path(tls_env, monkeypatch, ca_file):
    tls_env(sslmode="verify-full", sslrootcert=ca_file)
    recorder = _Recorder()
    monkeypatch.setattr(db.psycopg2, "connect", recorder.connect)
    assert db.record_task_status(_address(), TASK_ID, "running", WORKER_ID,
                                 tls=worker_config.database_tls_params()) is True
    ((dsn, kwargs),) = recorder.calls
    assert dsn == db.libpq_dsn(_address())
    assert kwargs == {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS,
                      "sslmode": "verify-full", "sslrootcert": ca_file}


def test_the_settings_win_over_an_sslmode_in_the_address():
    """psycopg2 merges keywords over the DSN's own keys: the setting is the one libpq sees."""
    import psycopg2.extensions

    merged = psycopg2.extensions.make_dsn(db.libpq_dsn(_address()) + "?sslmode=disable",
                                          sslmode="verify-full", sslrootcert=CA_IN_POD)
    parsed = psycopg2.extensions.parse_dsn(merged)
    assert parsed["sslmode"] == "verify-full" and parsed["sslrootcert"] == CA_IN_POD


@pytest.mark.parametrize(("sslmode", "sslrootcert", "message"), [
    ("verify", None, "DATABASE_SSLMODE must be one of"),
    (None, CA_IN_POD, "DATABASE_SSLROOTCERT is used only with"),
    ("require", CA_IN_POD, "DATABASE_SSLROOTCERT is used only with"),
    ("verify-full", None, "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
    ("verify-ca", None, "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
    (" VERIFY-FULL ", "   ", "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
])
def test_values_that_cannot_work_stop_the_worker_at_start(tls_env, monkeypatch, tmp_path, sslmode, sslrootcert, message):
    tls_env(sslmode=sslmode, sslrootcert=sslrootcert)
    worker_main = load_worker_main(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message) as refused:
        worker_main.WorkspaceWorker()
    assert CA_IN_POD not in str(refused.value)


@pytest.mark.parametrize("content", [None, b"", b"this file holds no certificate\n"],
                         ids=["missing", "empty", "no-certificate"])
def test_an_unusable_ca_file_stops_the_worker_at_start(tls_env, monkeypatch, tmp_path, content):
    path = tmp_path / "ca.crt"
    if content is not None:
        path.write_bytes(content)
    tls_env(sslmode="verify-full", sslrootcert=str(path))
    worker_main = load_worker_main(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="DATABASE_SSLROOTCERT must name a readable file") as refused:
        worker_main.WorkspaceWorker()
    assert str(path) not in str(refused.value)
    assert refused.value.__cause__ is None


@pytest.mark.asyncio
async def test_the_worker_hands_its_settings_to_every_write(tls_env, monkeypatch, tmp_path, ca_file):
    tls_env(sslmode="verify-full", sslrootcert=ca_file)
    worker_main = load_worker_main(monkeypatch, tmp_path)
    seen = []

    def record(url, task_id, status, worker_id, error=None, tls=None):
        seen.append(tls)
        return True

    monkeypatch.setattr(worker_main, "record_task_status", record)
    monkeypatch.setenv("DATABASE_URL", _address())
    await worker_main.WorkspaceWorker()._update_db_execution(TASK_ID, "running")
    assert seen == [{"sslmode": "verify-full", "sslrootcert": ca_file}]
