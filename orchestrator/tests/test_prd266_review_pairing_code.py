"""PRD-266 US-019d (D13) — App Review's code: ``scripts.mint_pairing_code --review --ttl-hours N``.

* ``--review`` needs ``--ttl-hours N``, a whole number of hours from 1 to
  ``REVIEW_CODE_MAX_HOURS`` (336, two weeks); ``--ttl-hours`` without ``--review`` is refused.
  Both are refused before anything opens; an ordinary code still lasts ten minutes.
* The review code is single-use like every pairing code (the public ``POST /pair``,
  unchanged), and the script refuses it in saas and outside paired mode as it refuses any code.
* Minting it writes ONE ``audit_logs`` row (the local operator, the expiry, the device name;
  never the code or its hash) in the code's commit; when that row cannot be written the
  code is rolled back with it.
* The lifetime override is the script's alone: ``POST /pairing-codes`` still issues a
  600-second code whatever its body says, and no route or setting passes it.
* The code is printed once, to the console, and never logged.

Against Postgres (CI) in a session whose commits and rollbacks stay inside a savepoint, so
``AuditService.log``'s commit and a refused audit's rollback never take the seed.
"""
from __future__ import annotations

import ast
import inspect
import io
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from config import HOSTED_PUBLIC_API_HOST, config
from core.auth.pairing import PAIRING_CODE_TTL_SECONDS, hash_secret, normalize_pairing_code
from core.models.mobile_devices import MobileDevice
from core.workspaces.audit import AuditLog
from scripts import mint_pairing_code as script

ORCH = Path(__file__).resolve().parents[1]
STUDIO = "https://studio.prd266.test"
BACKEND = "http://192.168.1.20:8000"
CODE_LINE = re.compile(r"^Pairing code: ([A-Z0-9]{4}-[A-Z0-9]{4})$", re.M)
DEVICE_NAME = "App Review iPhone"
PAIRING_CODES = "/api/v1/mobile/pairing-codes"
PAIR = "/api/v1/mobile/pair"
# Seconds of slack between the code's minting and the test reading the clock.
CLOCK_SLACK = 60


class _BorrowedSession:
    """The test's session, lent to the script: ``close()`` is counted, not done."""

    def __init__(self, session):
        self._session = session
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        self.closes += 1


class _RefusingAudit:
    """``AuditService`` when its insert fails: it rolls the session back and answers ``None``."""

    def __init__(self, db):
        self.db = db

    def log(self, **_fields):
        self.db.rollback()
        return None


def _never_opened():
    raise AssertionError("a refused run must not open a database session")


def _run(*argv: str):
    out, err = io.StringIO(), io.StringIO()
    code = script.main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _review(hours: int = 24, *extra: str):
    return _run("--review", "--ttl-hours", str(hours), *extra)


@pytest.fixture
def db_session(test_engine):
    """A session whose commit releases a savepoint and whose rollback returns to one."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def paired(monkeypatch, db_session, seed_workspace):
    """A local install in paired mode, its workspace and its operator, committed to a savepoint."""
    tag = uuid.uuid4().hex[:12]
    email = f"prd266-review-{tag}@t.test"
    user_pk = db_session.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-review-{tag}", "e": email},
    ).scalar()
    workspace_id = uuid.UUID(seed_workspace())
    db_session.commit()
    for name, value in {
        "AUTH_EDITION": "local", "LOCAL_REMOTE_ACCESS": "paired", "DEFAULT_WORKSPACE_ID": str(workspace_id),
        "LOCAL_OPERATOR_EMAIL": email, "FRONTEND_URL": STUDIO, "PUBLIC_API_HOST": HOSTED_PUBLIC_API_HOST,
        "BACKEND_URL": BACKEND,
    }.items():
        monkeypatch.setattr(config, name, value)
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(script, "SessionLocal", lambda: borrowed)
    return SimpleNamespace(workspace_id=workspace_id, user_pk=user_pk, email=email, session=borrowed)


@pytest.fixture
def refused_before_opening(monkeypatch):
    """A local install in paired mode whose script may not open a session."""
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(uuid.uuid4()))
    monkeypatch.setattr(script, "SessionLocal", _never_opened)
    return config


def _rows(db, paired) -> list:
    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.workspace_id == paired.workspace_id,
                                         MobileDevice.user_id == paired.user_pk).all()


def _audits(db, paired) -> list:
    db.expire_all()
    return db.query(AuditLog).filter(AuditLog.workspace_id == paired.workspace_id,
                                     AuditLog.action == script.REVIEW_CODE_ACTION).all()


def _seconds_left(expires_at: datetime) -> float:
    deadline = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=timezone.utc)
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def _app(db_session, ctx=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(mobile_devices.router)
    app.dependency_overrides[get_db] = lambda: db_session
    if ctx is not None:
        app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app)


def _operator_ctx(paired):
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=paired.email, email=paired.email, system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": paired.user_pk})
    return RequestContext(workspace_id=paired.workspace_id, user=user)


# --------------------------------------------------------------------------- the bounds


def test_the_longest_review_code_is_two_weeks():
    assert script.REVIEW_CODE_MAX_HOURS == 336


@pytest.mark.parametrize("hours", [1, 336])
def test_a_review_code_lasts_the_hours_asked_within_the_bounds(db_session, paired, hours):
    exit_code, out, err = _review(hours, "--name", DEVICE_NAME)

    assert (exit_code, err) == (script.EXIT_OK, "")
    unit = "hour" if hours == 1 else "hours"
    assert re.search(rf"^Expires at:\s+\S+ \(in {hours} {unit}; it works once\)$", out, re.M)
    assert "This is App Review's code" in out
    [row] = _rows(db_session, paired)
    assert (row.kind, row.status, row.name) == ("paired", "pending", DEVICE_NAME)
    assert abs(_seconds_left(row.pairing_expires_at) - hours * 3600) < CLOCK_SLACK


@pytest.mark.parametrize("hours", ["0", "337", "-1", "1.5", "two"])
def test_hours_outside_one_to_336_are_refused_before_anything_opens(refused_before_opening, hours):
    with pytest.raises(SystemExit) as refused:
        _run("--review", "--ttl-hours", hours)

    assert refused.value.code == script.EXIT_REFUSED


def test_ttl_hours_without_review_is_refused(refused_before_opening):
    with pytest.raises(SystemExit) as refused:
        _run("--ttl-hours", "24")

    assert refused.value.code == script.EXIT_REFUSED


def test_review_without_ttl_hours_is_refused(refused_before_opening):
    with pytest.raises(SystemExit) as refused:
        _run("--review")

    assert refused.value.code == script.EXIT_REFUSED


def test_an_ordinary_code_still_lasts_ten_minutes_and_is_not_audited(db_session, paired):
    exit_code, out, _err = _run("--name", DEVICE_NAME)

    assert exit_code == script.EXIT_OK
    assert "(in 10 minutes; it works once)" in out and "App Review" not in out
    [row] = _rows(db_session, paired)
    assert abs(_seconds_left(row.pairing_expires_at) - PAIRING_CODE_TTL_SECONDS) < CLOCK_SLACK
    assert _audits(db_session, paired) == []


# --------------------------------------------------------------------------- the refusals


def test_a_review_code_is_refused_in_saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(script, "SessionLocal", _never_opened)

    exit_code, out, err = _review()

    assert (exit_code, out) == (script.EXIT_REFUSED, "")
    assert err.strip() == script.SAAS_REFUSAL


@pytest.mark.parametrize("mode", ["off", "open"])
def test_a_review_code_is_refused_outside_paired_mode(refused_before_opening, monkeypatch, mode):
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)

    exit_code, out, err = _review()

    assert (exit_code, out) == (script.EXIT_REFUSED, "")
    assert err.strip() == script.NOT_PAIRED_REFUSAL.format(mode=mode)


# --------------------------------------------------------------------------- the audit row


def test_minting_a_review_code_writes_one_audit_row_without_the_code(db_session, paired):
    out = _review(48, "--name", DEVICE_NAME)[1]

    code = CODE_LINE.search(out).group(1)
    [row] = _rows(db_session, paired)
    [audit] = _audits(db_session, paired)
    assert (audit.user_id, audit.actor_type) == (paired.user_pk, "user")
    assert (audit.resource_type, audit.resource_name) == (script.REVIEW_CODE_RESOURCE, DEVICE_NAME)
    details = dict(audit.details)
    assert datetime.fromisoformat(details.pop("expires_at")) == row.pairing_expires_at
    assert details == {"ttl_hours": 48, "device_name": DEVICE_NAME, "source": script.REVIEW_CODE_SOURCE}
    written = json.dumps({column.name: str(getattr(audit, column.name)) for column in AuditLog.__table__.columns})
    for secret in (code, normalize_pairing_code(code), hash_secret(normalize_pairing_code(code))):
        assert secret not in written


def test_a_review_code_without_a_name_records_no_device_name(db_session, paired):
    assert _review()[0] == script.EXIT_OK

    [audit] = _audits(db_session, paired)
    assert audit.details["device_name"] is None and audit.resource_name is None


def test_an_audit_row_that_cannot_be_written_takes_the_code_with_it(db_session, paired, monkeypatch):
    monkeypatch.setattr(script, "AuditService", _RefusingAudit)

    exit_code, out, err = _review(24, "--name", DEVICE_NAME)

    assert (exit_code, out) == (script.EXIT_FAILED, "")
    assert err.strip() == script.AUDIT_FAILED
    assert _rows(db_session, paired) == [] and _audits(db_session, paired) == []
    assert paired.session.closes == 1


# --------------------------------------------------------------------------- single use, never logged


def test_the_review_code_pairs_once_through_the_unchanged_public_route(db_session, paired, monkeypatch):
    from api import mobile_devices

    monkeypatch.setattr(mobile_devices, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 3, 2, 600)))
    code = CODE_LINE.search(_review(24, "--name", DEVICE_NAME)[1]).group(1)
    client = _app(db_session)
    body = {"code": code, "platform": "ios", "install_id": "prd266-review-phone"}

    first = client.post(PAIR, json=body)
    second = client.post(PAIR, json={**body, "install_id": "prd266-review-other"})

    assert first.status_code == 200, first.text
    assert second.status_code == 401 and second.json()["detail"] == {"code": "invalid_pairing_code"}
    [row] = _rows(db_session, paired)
    assert (row.status, row.pairing_code_hash, row.pairing_expires_at) == ("active", None, None)


def test_the_code_is_printed_once_and_never_logged(paired, caplog):
    caplog.set_level(logging.DEBUG)

    _exit, out, err = _review(24, "--name", DEVICE_NAME)

    [code] = CODE_LINE.findall(out)
    assert code not in err
    assert all(code not in record.getMessage() for record in caplog.records)
    assert all(hash_secret(normalize_pairing_code(code)) not in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- the override is the script's


def test_studio_still_issues_a_600_second_code_whatever_the_body_says(db_session, paired):
    response = _app(db_session, _operator_ctx(paired)).post(
        PAIRING_CODES, json={"ttl_seconds": 1_209_600, "ttl_hours": 336, "review": True,
                             "expires_at": (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()})

    assert response.status_code == 200, response.text
    expires_at = datetime.fromisoformat(response.json()["expires_at"])
    assert abs(_seconds_left(expires_at) - PAIRING_CODE_TTL_SECONDS) < CLOCK_SLACK
    [row] = _rows(db_session, paired)
    assert abs(_seconds_left(row.pairing_expires_at) - PAIRING_CODE_TTL_SECONDS) < CLOCK_SLACK
    assert _audits(db_session, paired) == []


SKIPPED_PARTS = {"tests", "alembic", ".venv", "venv", "node_modules", "__pycache__"}


def _passes_a_lifetime(node: ast.AST) -> bool:
    """Is ``node`` a call of ``create_pairing_code`` (any receiver) with ``ttl_seconds``?"""
    if not isinstance(node, ast.Call):
        return False
    name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
    return name == "create_pairing_code" and any(keyword.arg == "ttl_seconds" for keyword in node.keywords)


def _callers_passing_a_lifetime() -> set:
    """Every module (tests aside) that calls ``create_pairing_code`` with ``ttl_seconds``."""
    callers = set()
    for path in ORCH.rglob("*.py"):
        relative = path.relative_to(ORCH)
        source = path.read_text(encoding="utf-8")
        if SKIPPED_PARTS & set(relative.parts) or "create_pairing_code" not in source:
            continue
        if any(_passes_a_lifetime(node) for node in ast.walk(ast.parse(source))):
            callers.add(relative.as_posix())
    return callers


def test_only_the_console_script_passes_a_lifetime():
    assert _callers_passing_a_lifetime() == {"scripts/mint_pairing_code.py"}


def test_no_route_or_setting_can_set_the_lifetime():
    from api import mobile_devices

    assert set(inspect.signature(mobile_devices.create_pairing_code).parameters) == {
        "request", "response", "ctx", "db"}
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert not [name for name in surface if "PAIRING" in name or "REVIEW_CODE" in name]
    assert not [name for name in dir(config) if "PAIRING" in name or "REVIEW_CODE" in name]


def test_the_docstring_and_the_docs_name_the_review_flags():
    assert "--review --ttl-hours N" in script.__doc__
    for doc in ("QUICKSTART.md", "SETUP.md"):
        assert "--review --ttl-hours" in (ORCH.parent / doc).read_text(encoding="utf-8")
