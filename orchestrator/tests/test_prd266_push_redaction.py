"""PRD-266 US-014b (§4.4, D20) — ``push_redact_questions``: questions on the lock screen, generic.

``PUT /api/v1/mobile/push-settings`` ``{redact_questions: bool}`` (``api/mobile_push.py``):

* ``workspace:manage``: the owner (``workspaces.owner_id``) and an admin may set it, read
  through the real role resolver; an editor and a viewer are 403 and write nothing.
* It merges ``workspace.settings['push']['redact_questions']`` as new dicts (every other key
  kept, ``flag_modified``), commits and answers ``{push: {redact_questions}}``; the caller's
  workspace only, so another workspace's settings never change; a body other than one
  boolean is 422; a workspace that is not there is 404; a failed commit is 503 and changes
  nothing.
* ``GET /api/workspaces/current`` (never edited: ``api/workspaces.py`` is over 800 lines)
  shows it under ``settings.push``; a fresh workspace has none, so redaction is off.
* The fan-out honours it: with it on, a question pushed through ``fanout.deliver`` (a fake
  sender) carries no question and no options, the generic line, category ``automatos.open``;
  with it off again, the question and its options come back.
* A plain ``def`` route in the committed route manifest.

The route tests run on Postgres (CI) inside the test's rolled-back transaction, joined by
SAVEPOINT so the route's commit never takes the seed with it.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.mobile_push as mobile_push  # noqa: E402
import api.workspaces as workspaces_api  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.approval_grants import KIND_QUESTION, SUBJECT_TOOL_CALL, ApprovalGrant  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from services.push import fanout  # noqa: E402
from services.push.fanout import PushEvent  # noqa: E402
from services.push.payload import QUESTION_LINE, redact_questions_enabled, verify_signature  # noqa: E402
from services.push.sender import TICKET_OK, PushTicket  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
PUSH_SETTINGS = "/api/v1/mobile/push-settings"
CURRENT = "/api/workspaces/current"
MANIFEST = ORCH / "reports" / "route-manifest.json"
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
QUESTION_MD = "Ship the **Lisbon** deck?"
QUESTION_TEXT = "Ship the Lisbon deck?"
OPTIONS = ["Yes, ship it", "Hold it"]
KEPT = {"voice_live": {"enabled": True}, "push": {"later_key": "kept"}}


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _CommitFails(_BorrowedSession):
    """The test's session whose commit fails, as a database that went away mid-request."""

    def commit(self):
        raise OperationalError("COMMIT", {}, Exception("connection lost"))


class _FakeSender:
    """Accepts every message with a fresh ticket id, and keeps what it was sent."""

    def __init__(self):
        self.sent = []

    def send(self, messages, /):
        self.sent.extend(messages)
        return [PushTicket(to=m.to, status=TICKET_OK, ticket_id=str(uuid.uuid4())) for m in messages]

    def receipts(self, ticket_ids, /):
        return {}


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The PUT commits, and its failure path rolls back; here either reaches the last savepoint
    only, and every seed commits (``_settle``), so no route can take the seed with it.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    """Release the seed's savepoint: a route's rollback never reaches it."""
    db.commit()


@pytest.fixture(autouse=True)
def saas(monkeypatch):
    """SaaS, with the GET's unrelated reads (the Socials master switch, AI providers) quiet."""
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda _category, _key: None)
    monkeypatch.setattr(workspaces_api, "ai_providers_or_none", lambda _db, _workspace_id: [])
    monkeypatch.setattr(fanout, "_now", lambda: T0)
    return config


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
    _settle(db_session)
    return workspace_id


def _client(db, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(mobile_push.router)
    app.include_router(workspaces_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


# --------------------------------------------------------------------------- seeds


def _person(db) -> SimpleNamespace:
    """A Clerk user: its users.id and the identity the role resolver reads."""
    tag = uuid.uuid4().hex[:12]
    clerk_id = f"clerk_prd266_redact_{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, clerk_user_id, is_active) "
             "VALUES (:u, :e, :c, true) RETURNING id"),
        {"u": f"prd266-redact-{tag}", "e": f"prd266-redact-{tag}@t.test", "c": clerk_id},
    ).scalar()
    _settle(db)
    return SimpleNamespace(pk=pk, clerk_id=clerk_id)


def _owner(db, workspace_id) -> SimpleNamespace:
    person = _person(db)
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": person.pk, "ws": workspace_id})
    _settle(db)
    return person


def _member(db, workspace_id, role, person=None) -> SimpleNamespace:
    person = person or _person(db)
    db.execute(text(
        "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
        "VALUES (CAST(:ws AS uuid), :uid, :role, true)"),
        {"ws": workspace_id, "uid": person.pk, "role": role})
    _settle(db)
    return person


def _ctx(person, workspace_id) -> RequestContext:
    user = UserContext(id=person.clerk_id, clerk_user_id=person.clerk_id, system_role="user")
    return RequestContext(workspace_id=uuid.UUID(workspace_id), user=user, auth_type="clerk")


def _set_settings(db, workspace_id, settings) -> None:
    db.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:ws AS uuid)"),
               {"s": json.dumps(settings), "ws": workspace_id})
    _settle(db)


def _settings(db, workspace_id):
    db.expire_all()
    return db.get(Workspace, uuid.UUID(workspace_id)).settings


def _put(db, person, workspace_id, redact):
    return _client(db, _ctx(person, workspace_id)).put(PUSH_SETTINGS, json={"redact_questions": redact})


# --------------------------------------------------------------------------- who may set it


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_the_owner_and_an_admin_may_set_it(db_session, ws, role):
    person = _owner(db_session, ws) if role == "owner" else _member(db_session, ws, "admin")

    on = _put(db_session, person, ws, True)
    assert on.status_code == 200, on.text
    assert on.json() == {"push": {"redact_questions": True}}
    assert _settings(db_session, ws) == {"push": {"redact_questions": True}}

    off = _put(db_session, person, ws, False)
    assert off.status_code == 200, off.text
    assert off.json() == {"push": {"redact_questions": False}}
    assert redact_questions_enabled(_settings(db_session, ws)) is False


@pytest.mark.parametrize("role", ["editor", "viewer"])
def test_an_editor_and_a_viewer_get_403_and_nothing_is_written(db_session, ws, role):
    _set_settings(db_session, ws, KEPT)
    person = _member(db_session, ws, role)

    response = _put(db_session, person, ws, True)

    assert response.status_code == 403, response.text
    assert _settings(db_session, ws) == KEPT


def test_a_person_with_no_role_in_the_workspace_gets_403(db_session, ws):
    response = _put(db_session, _person(db_session), ws, True)

    assert response.status_code == 403, response.text
    assert _settings(db_session, ws) is None


# --------------------------------------------------------------------------- what it writes


def test_it_merges_into_the_settings_and_keeps_every_other_key(db_session, ws):
    _set_settings(db_session, ws, KEPT)
    person = _owner(db_session, ws)

    assert _put(db_session, person, ws, True).status_code == 200

    assert _settings(db_session, ws) == {
        "voice_live": {"enabled": True},
        "push": {"later_key": "kept", "redact_questions": True},
    }


def test_another_workspaces_settings_never_change(db_session, ws, seed_workspace):
    other = seed_workspace()
    _settle(db_session)
    _set_settings(db_session, other, KEPT)
    person = _owner(db_session, ws)
    _member(db_session, other, "admin", person)  # the same person manages both

    assert _put(db_session, person, ws, True).status_code == 200

    assert _settings(db_session, ws) == {"push": {"redact_questions": True}}
    assert _settings(db_session, other) == KEPT


@pytest.mark.parametrize("body", [
    {"redact_questions": "true"},
    {"redact_questions": 1},
    {"redact_questions": None},
    {},
    {"redact_questions": True, "options": False},
])
def test_a_body_other_than_one_boolean_is_422(db_session, ws, body):
    person = _owner(db_session, ws)

    response = _client(db_session, _ctx(person, ws)).put(PUSH_SETTINGS, json=body)

    assert response.status_code == 422, response.text
    assert _settings(db_session, ws) is None


def test_a_workspace_that_is_not_there_is_404(db_session):
    # Only the super-admin passes the gate without a role in the workspace it names.
    user = UserContext(id="prd266-redact-su", clerk_user_id="clerk_prd266_redact_su", system_role="super_admin")
    ctx = RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="clerk")

    response = _client(db_session, ctx).put(PUSH_SETTINGS, json={"redact_questions": True})

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": mobile_push.WORKSPACE_NOT_FOUND}


def test_a_failed_commit_is_503_and_changes_nothing(db_session, ws):
    _set_settings(db_session, ws, KEPT)
    person = _owner(db_session, ws)

    response = _client(_CommitFails(db_session), _ctx(person, ws)).put(PUSH_SETTINGS, json={"redact_questions": True})

    assert response.status_code == 503, response.text
    assert response.json() == {"detail": mobile_push.SETTINGS_UNAVAILABLE}
    assert _settings(db_session, ws) == KEPT


# --------------------------------------------------------------------------- the workspace answer


def test_get_current_workspace_shows_it_and_the_default_is_off(db_session, ws):
    person = _owner(db_session, ws)
    client = _client(db_session, _ctx(person, ws))

    fresh = client.get(CURRENT)
    assert fresh.status_code == 200, fresh.text
    assert "push" not in fresh.json()["settings"]
    assert redact_questions_enabled(fresh.json()["settings"]) is False

    assert client.put(PUSH_SETTINGS, json={"redact_questions": True}).status_code == 200
    shown = client.get(CURRENT)
    assert shown.status_code == 200, shown.text
    assert shown.json()["settings"]["push"]["redact_questions"] is True


# --------------------------------------------------------------------------- the fan-out honours it


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture
def preferences_table(db_session):
    """``notification_preferences`` as production has it (prd128's DDL; IF NOT EXISTS).

    CI's test database has no such table (no model; ``init_test_db`` does not build it), and
    the fan-out reads the push switches from it.
    """
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


def _phone(db, workspace_id, user_pk) -> MobileDevice:
    tag = uuid.uuid4().hex
    device = MobileDevice(
        workspace_id=uuid.UUID(workspace_id), user_id=user_pk, kind="clerk", status="active",
        install_id=f"prd266-redact-{tag[:12]}", platform="ios", locale="en-IE",
        expo_push_token=f"ExponentPushToken[{tag}]", push_enabled=True,
        push_key=f"prd266-redact-key-{tag}", events=[],
    )
    db.add(device)
    _settle(db)
    return device


def _question(db, workspace_id) -> int:
    grant = ApprovalGrant(workspace_id=uuid.UUID(workspace_id), subject_type=SUBJECT_TOOL_CALL,
                          subject_id=f"call-{uuid.uuid4().hex[:8]}", kind=KIND_QUESTION,
                          question_md=QUESTION_MD, options=list(OPTIONS))
    db.add(grant)
    _settle(db)
    return grant.id


def _pushed(db, workspace_id, grant_id):
    """The one message ``fanout.deliver`` sends for the question, through a fake sender."""
    event = PushEvent(workspace_id=workspace_id, event_type="question_pending", notification_id="n-1",
                      link_type="question", link_id=str(grant_id))
    sender = _FakeSender()
    fanout.deliver(event, sender=sender, session_factory=lambda: _BorrowedSession(db))
    (message,) = sender.sent
    return message


def test_a_question_pushed_with_redaction_on_is_generic(db_session, ws, preferences_table):
    person = _owner(db_session, ws)
    phone = _phone(db_session, ws, person.pk)
    grant_id = _question(db_session, ws)

    before = _pushed(db_session, ws, grant_id)
    assert before.body == QUESTION_TEXT and before.data["options"] == OPTIONS  # the default: off

    assert _put(db_session, person, ws, True).status_code == 200
    redacted = _pushed(db_session, ws, grant_id)
    assert "question" not in redacted.data and "options" not in redacted.data
    assert redacted.body == QUESTION_LINE and redacted.category_id == "automatos.open"
    assert redacted.data["grant_id"] == str(grant_id)
    assert verify_signature(phone.push_key, redacted.data, T0) is True

    assert _put(db_session, person, ws, False).status_code == 200
    again = _pushed(db_session, ws, grant_id)
    assert again.data["question"] == QUESTION_TEXT and again.data["options"] == OPTIONS


# --------------------------------------------------------------------------- shape


def test_the_route_is_a_plain_def_in_the_route_manifest():
    assert not inspect.iscoroutinefunction(mobile_push.put_push_settings)
    routes = json.loads(MANIFEST.read_text())["routes"]
    assert {"method": "PUT", "path": PUSH_SETTINGS} in routes
