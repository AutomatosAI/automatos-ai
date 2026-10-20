"""PRD-266 US-005a (K11) — ``docker compose exec backend python -m scripts.mint_pairing_code``.

* In paired mode it mints a code for the local operator in ``DEFAULT_WORKSPACE_ID`` through
  ``services.mobile.pairing`` and prints the code, its expiry and the universal link
  (``{FRONTEND_URL}/go/pair?api=…&code=…``; ``api`` is ``BACKEND_URL`` while
  ``PUBLIC_API_HOST`` is the hosted default). The row holds only the code's hash, and the
  code then pairs a browser through the public ``POST /pair``; ``--name`` labels it. A link
  whose API address is loopback says a phone cannot open it and how to fix that.
* It refuses (exit 2, nothing opened) in saas, without ``DEFAULT_WORKSPACE_ID``, and when
  the mode is not ``paired``, saying how to turn it on. A missing operator row or a database
  error is exit 1; the error is printed by type only, and the code never reaches a log.

Against Postgres (``db_session``) for the mint; the refusals never open a session.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from config import HOSTED_PUBLIC_API_HOST, config
from core.auth.pairing import hash_secret, normalize_pairing_code
from core.models.mobile_devices import NAME_LENGTH, MobileDevice
from scripts import mint_pairing_code as script

STUDIO = "https://studio.prd266.test"
BACKEND = "http://192.168.1.20:8000"
CODE_LINE = re.compile(r"^Pairing code: ([A-Z0-9]{4}-[A-Z0-9]{4})$", re.M)
DEVICE_NAME = "Studio on the laptop"
BOUND_SECRET = "prd266-bound-parameter-never-printed"


class _BorrowedSession:
    """The test's transactional session, lent to the script: ``close()`` is counted, not done."""

    def __init__(self, session):
        self._session = session
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        self.closes += 1


class _FailingSession:
    """A session whose first statement fails with an error that carries a bound parameter."""

    def __init__(self):
        self.rollbacks = 0
        self.closes = 0

    def execute(self, *args, **kwargs):
        raise OperationalError("SELECT id FROM users WHERE email = %(value)s", {"value": BOUND_SECRET},
                               Exception(BOUND_SECRET))

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1


def _never_opened():
    raise AssertionError("a refused run must not open a database session")


def _run(*argv: str):
    out, err = io.StringIO(), io.StringIO()
    code = script.main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def paired(monkeypatch, db_session, seed_workspace):
    """A local install in paired mode, its workspace and its operator: ``(workspace_id, user_pk)``."""
    tag = uuid.uuid4().hex[:12]
    email = f"prd266-console-{tag}@t.test"
    user_pk = db_session.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-console-{tag}", "e": email},
    ).scalar()
    workspace_id = uuid.UUID(seed_workspace())
    for name, value in {
        "AUTH_EDITION": "local", "LOCAL_REMOTE_ACCESS": "paired", "DEFAULT_WORKSPACE_ID": str(workspace_id),
        "LOCAL_OPERATOR_EMAIL": email, "FRONTEND_URL": STUDIO, "PUBLIC_API_HOST": HOSTED_PUBLIC_API_HOST,
        "BACKEND_URL": BACKEND,
    }.items():
        monkeypatch.setattr(config, name, value)
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(script, "SessionLocal", lambda: borrowed)
    return SimpleNamespace(workspace_id=workspace_id, user_pk=user_pk, session=borrowed)


def _rows(db, paired) -> list:
    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.workspace_id == paired.workspace_id,
                                         MobileDevice.user_id == paired.user_pk).all()


# --------------------------------------------------------------------------- the mint


def test_it_prints_the_code_its_expiry_and_the_universal_link(db_session, paired):
    exit_code, out, err = _run("--name", DEVICE_NAME)

    assert (exit_code, err) == (0, "")
    code = CODE_LINE.search(out).group(1)
    assert re.search(r"^Expires at:\s+\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00 \(in 10 minutes", out, re.M)
    link = f"{STUDIO}/go/pair?api=http%3A%2F%2F192.168.1.20%3A8000&code={code}"
    assert re.search(rf"^Pair link:\s+{re.escape(link)}$", out, re.M)
    [row] = _rows(db_session, paired)
    assert (row.kind, row.status, row.name) == ("paired", "pending", DEVICE_NAME)
    assert row.pairing_code_hash == hash_secret(normalize_pairing_code(code))
    assert code not in repr(row.to_dict()) and row.token_hash is None
    assert paired.session.closes == 1


def test_a_link_on_a_routable_address_has_no_hint(paired):
    assert "PUBLIC_API_HOST" not in _run()[1]


@pytest.mark.parametrize("backend", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_a_loopback_link_says_a_phone_cannot_open_it_and_how_to_fix_that(paired, monkeypatch, backend):
    monkeypatch.setattr(config, "BACKEND_URL", backend)

    exit_code, out, _err = _run()

    assert exit_code == 0
    assert "a phone cannot open it" in out and "set PUBLIC_API_HOST" in out


def test_the_operators_own_host_names_the_link_and_needs_no_hint(paired, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_API_HOST", "go.prd266.test")
    monkeypatch.setattr(config, "BACKEND_URL", "http://localhost:8000")

    out = _run()[1]

    code = CODE_LINE.search(out).group(1)
    assert f"{STUDIO}/go/pair?api=https%3A%2F%2Fgo.prd266.test&code={code}" in out
    assert "PUBLIC_API_HOST" not in out


def test_the_console_code_pairs_a_browser_through_the_public_route(db_session, paired, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.database.database import get_db

    monkeypatch.setattr(mobile_devices, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 3, 2, 600)))
    app = FastAPI()
    app.include_router(mobile_devices.router)
    app.dependency_overrides[get_db] = lambda: db_session
    code = CODE_LINE.search(_run("--name", DEVICE_NAME)[1]).group(1)

    response = TestClient(app).post("/api/v1/mobile/pair", json={
        "code": code, "platform": "web", "install_id": "prd266-console-browser"})

    assert response.status_code == 200, response.text
    assert response.json()["token"].startswith("amd_")
    [row] = _rows(db_session, paired)
    assert (row.status, row.platform, row.name) == ("active", "web", DEVICE_NAME)


def test_the_code_never_reaches_a_log(paired, caplog):
    caplog.set_level(logging.DEBUG)

    code = CODE_LINE.search(_run()[1]).group(1)

    assert all(code not in record.getMessage() for record in caplog.records)


def test_a_name_longer_than_the_column_is_refused_before_anything_runs(monkeypatch):
    monkeypatch.setattr(script, "SessionLocal", _never_opened)

    with pytest.raises(SystemExit):
        script.main(["--name", "x" * (NAME_LENGTH + 1)], out=io.StringIO(), err=io.StringIO())


# --------------------------------------------------------------------------- the refusals


def test_it_refuses_in_saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(script, "SessionLocal", _never_opened)

    exit_code, out, err = _run()

    assert (exit_code, out) == (script.EXIT_REFUSED, "")
    assert "AUTH_EDITION=local" in err


@pytest.mark.parametrize("mode", ["off", "open"])
def test_it_refuses_when_the_mode_is_not_paired_and_says_how_to_turn_it_on(monkeypatch, mode):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(uuid.uuid4()))
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)
    monkeypatch.setattr(script, "SessionLocal", _never_opened)

    exit_code, out, err = _run()

    assert (exit_code, out) == (script.EXIT_REFUSED, "")
    assert f"LOCAL_REMOTE_ACCESS={mode}" in err and "set LOCAL_REMOTE_ACCESS=paired" in err


@pytest.mark.parametrize("workspace", [None, "", "not-a-uuid"])
def test_it_refuses_without_a_local_workspace(monkeypatch, workspace):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", workspace)
    monkeypatch.setattr(script, "SessionLocal", _never_opened)

    exit_code, _out, err = _run()

    assert exit_code == script.EXIT_REFUSED and "DEFAULT_WORKSPACE_ID" in err


def test_a_missing_operator_row_fails_and_writes_nothing(db_session, paired, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", f"prd266-nobody-{uuid.uuid4().hex[:8]}@t.test")

    exit_code, out, err = _run()

    assert (exit_code, out) == (script.EXIT_FAILED, "")
    assert "LOCAL_OPERATOR_EMAIL" in err
    assert _rows(db_session, paired) == []


def test_a_database_error_is_printed_by_its_type_only(monkeypatch):
    failing = _FailingSession()
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(uuid.uuid4()))
    monkeypatch.setattr(script, "SessionLocal", lambda: failing)

    exit_code, out, err = _run()

    assert (exit_code, out) == (script.EXIT_FAILED, "")
    assert "OperationalError" in err and BOUND_SECRET not in err
    assert (failing.rollbacks, failing.closes) == (1, 1)


def test_the_docstring_names_the_console_command():
    assert "docker compose exec backend python -m scripts.mint_pairing_code" in script.__doc__
