"""PRD-266 US-014a (§4.4, K7, K12) — the phone's push switches in /api/notification-preferences.

``api/notifications.py``'s preference routes over ``services/push/preferences.py``, on
Postgres through a ``TestClient``. Proven here:

* GET with no rows shows today's nine rows, plus a ``push`` row, enabled, for each of the
  five push events and ``daily_digest``, and the ``push_types`` map built from
  ``PUSH_TYPE_EVENTS`` (with ``digest``: ``daily_digest``).
* PUT a push row off for question_pending: GET shows it off, and the fan-out skips that
  person (``fanout.eligible_devices``; ``deliver`` with a fake sender) while another editor's
  phone still receives the question.
* PUT a push row for task_complete is 422 and writes nothing; so is a ``daily_digest`` row
  that is not a push row (K12: the digest's switch is a push row only).
* PUT daily_digest off: GET shows it off, every other switch still on.
* A viewer may write their own rows (``members:read``); someone outside the workspace may not.
* A person with only a push row still gets the bell row for a dispatch (US-010d's in_app
  default): the push row is a switch, never a delivery (K7).
* ``with_push_defaults`` is pure: a new list, its input untouched, a push row already there
  kept as it is.
* The web settings tab renders its nine events only, none of them a push switch event.

CI's test database has no ``notification_preferences`` or ``notifications`` (no model;
``init_test_db`` does not build them), so the fixture runs the real ``prd128_notifications``
revision's DDL inside the test's transaction, which is rolled back with it.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import os
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from config import config  # noqa: E402
from core.auth.hybrid import DEFAULT_NOTIFICATION_PREFERENCES  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.services.notification_dispatcher import NotificationDispatcher  # noqa: E402
from services.push import fanout  # noqa: E402
from services.push.fanout import PushEvent  # noqa: E402
from services.push.payload import PUSH_TYPE_EVENTS  # noqa: E402
from services.push.preferences import PUSH_SWITCH_EVENTS, with_push_defaults  # noqa: E402
from services.push.sender import TICKET_OK, PushTicket  # noqa: E402

PREFERENCES = "/api/notification-preferences"
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
SETTINGS_TAB = ORCH.parent / "frontend" / "components" / "settings" / "NotificationsSettingsTab.tsx"
HAND_OVER = "services.push.fanout.hand_over"
PUSH = "push"


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeSender:
    """Accepts every message with a fresh ticket id, and remembers who it was sent to."""

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

    The PUT and ``deliver`` commit; on this session a commit releases the last savepoint
    only, and every seed commits (``_settle``), so no rollback ever takes the seed.
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
    db.commit()


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture(autouse=True)
def preferences_table(db_session):
    """``notification_preferences`` and ``notifications`` as production has them (prd128's DDL)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    return config


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    _settle(db_session)
    return workspace_id


@pytest.fixture
def make_client(db_session):
    """A client on the preference routes, ``ctx`` standing in for the request context."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import notifications
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    def _make(ctx):
        app = FastAPI()
        app.include_router(notifications.preferences_router)
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


# --------------------------------------------------------------------------- seeds


def _person(db) -> tuple:
    """A users row with a Clerk id: ``(pk, email, clerk_user_id)``."""
    tag = uuid.uuid4().hex[:12]
    email, clerk_id = f"prd266-push-prefs-{tag}@t.test", f"user_prd266pp{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, clerk_user_id) "
             "VALUES (:u, :e, true, :c) RETURNING id"),
        {"u": f"prd266-push-prefs-{tag}", "e": email, "c": clerk_id},
    ).scalar()
    _settle(db)
    return pk, email, clerk_id


def _clerk(db, workspace_id, role="editor", *, member=True):
    """A Clerk caller (a member holding ``role`` unless ``member`` is False): ``(ctx, pk)``."""
    from core.auth.dependencies import RequestContext, UserContext

    pk, email, clerk_id = _person(db)
    if member:
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :uid, :role, true)"),
                   {"ws": str(workspace_id), "uid": pk, "role": role})
        _settle(db)
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk"), pk


def _phone(db, workspace_id, user_pk) -> str:
    """An active Clerk iPhone with push on; returns its Expo token."""
    tag = uuid.uuid4().hex
    token = f"ExponentPushToken[{tag}]"
    db.add(MobileDevice(
        workspace_id=workspace_id, user_id=user_pk, kind="clerk", status="active",
        install_id=f"prd266-push-prefs-{tag[:12]}", platform="ios", expo_push_token=token,
        push_enabled=True, push_key=f"prd266-push-prefs-key-{tag}", events=[],
    ))
    _settle(db)
    return token


def _pref(event_type, enabled, destination=PUSH) -> dict:
    return {"event_type": event_type, "destination": destination, "enabled": enabled,
            "channel_connection_id": None}


def _get(client) -> dict:
    response = client.get(PREFERENCES)
    assert response.status_code == 200, response.text
    return response.json()


def _put(client, rows):
    return client.put(PREFERENCES, json={"preferences": rows})


def _switches(body) -> dict:
    """GET's push rows, ``{event_type: enabled}``; each event has one at most."""
    rows = [row for row in body["preferences"] if row["destination"] == PUSH]
    assert len(rows) == len({row["event_type"] for row in rows})
    return {row["event_type"]: row["enabled"] for row in rows}


def _stored(db, workspace_id, user_pk) -> set:
    rows = db.execute(text(
        "SELECT event_type, destination, enabled FROM notification_preferences "
        "WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid"),
        {"ws": str(workspace_id), "uid": user_pk}).fetchall()
    return {tuple(row) for row in rows}


# --------------------------------------------------------------------------- GET


def test_get_with_no_rows_shows_the_six_push_rows_on_and_the_push_types_map(ws, db_session, make_client):
    ctx, _pk = _clerk(db_session, ws)

    body = _get(make_client(ctx))

    assert len(PUSH_SWITCH_EVENTS) == 6
    assert _switches(body) == dict.fromkeys(PUSH_SWITCH_EVENTS, True)
    assert body["push_types"] == {
        "needs_answer": ["question_pending", "approval_pending", "mission_plan_ready"],
        "review": ["review_pending"],
        "auto_message": ["auto_message"],
        "digest": ["daily_digest"],
    }
    assert {key: tuple(events) for key, events in body["push_types"].items() if key != "digest"} == dict(PUSH_TYPE_EVENTS)
    today = {(row["event_type"], row["destination"]) for row in body["preferences"] if row["destination"] != PUSH}
    assert today == set(DEFAULT_NOTIFICATION_PREFERENCES)


# --------------------------------------------------------------------------- PUT


def test_a_push_row_off_for_question_pending_shows_off_and_the_fan_out_skips_that_person(saas, ws, db_session,
                                                                                       make_client):
    ctx, quiet = _clerk(db_session, ws, "editor")
    _other, loud = _clerk(db_session, ws, "editor")
    _phone(db_session, ws, quiet)
    loud_token = _phone(db_session, ws, loud)
    client = make_client(ctx)

    response = _put(client, [_pref("question_pending", False)])

    assert response.status_code == 200 and response.json()["updated_count"] == 1
    switches = _switches(_get(client))
    assert switches["question_pending"] is False
    assert all(switches[event] for event in PUSH_SWITCH_EVENTS if event != "question_pending")
    event = PushEvent(workspace_id=str(ws), event_type="question_pending", link_type="question", link_id="0")
    assert [device.user_id for device in fanout.eligible_devices(db_session, event, [quiet, loud])] == [loud]
    sender = _FakeSender()
    counts = fanout.deliver(event, sender=sender, session_factory=lambda: _BorrowedSession(db_session))
    assert {message.to for message in sender.sent} == {loud_token}
    assert (counts["recipients"], counts["devices"], counts["sent"]) == (2, 1, 1)


@pytest.mark.parametrize("refused", [_pref("task_complete", True), _pref("daily_digest", False, "in_app")])
def test_a_row_that_is_not_a_push_switch_is_422_and_writes_nothing(ws, db_session, make_client, refused):
    ctx, pk = _clerk(db_session, ws)

    response = _put(make_client(ctx), [_pref("review_pending", False), refused])

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"].endswith(f": {refused['event_type']}")
    assert _stored(db_session, ws, pk) == set()


def test_daily_digest_off_shows_off(ws, db_session, make_client):
    ctx, pk = _clerk(db_session, ws)
    client = make_client(ctx)

    assert _put(client, [_pref("daily_digest", False)]).status_code == 200

    switches = _switches(_get(client))
    assert switches["daily_digest"] is False
    assert all(switches[event] for event in PUSH_SWITCH_EVENTS if event != "daily_digest")
    assert _stored(db_session, ws, pk) == {("daily_digest", PUSH, False)}


def test_a_viewer_may_write_their_own_rows_and_an_outsider_may_not(ws, db_session, make_client):
    viewer_ctx, viewer = _clerk(db_session, ws, "viewer")
    outsider_ctx, outsider = _clerk(db_session, ws, member=False)
    rows = [_pref("review_pending", False), _pref("auto_message", False)]

    response = _put(make_client(viewer_ctx), rows)

    assert response.status_code == 200 and response.json()["updated_count"] == 2
    assert _stored(db_session, ws, viewer) == {("review_pending", PUSH, False), ("auto_message", PUSH, False)}
    assert _put(make_client(outsider_ctx), rows).status_code == 403
    assert _stored(db_session, ws, outsider) == set()


# --------------------------------------------------------------------------- the bell row


def test_a_person_with_only_a_push_row_still_gets_the_bell_row(ws, db_session, make_client):
    ctx, pk = _clerk(db_session, ws, "editor")
    assert _put(make_client(ctx), [_pref("question_pending", False)]).status_code == 200
    assert _stored(db_session, ws, pk) == {("question_pending", PUSH, False)}
    handed = MagicMock(return_value=False)

    with patch.object(NotificationDispatcher, "_load_auto_reporting", lambda self: {}), \
            patch(HAND_OVER, new=handed):
        result = asyncio.run(NotificationDispatcher(db_session, ws).dispatch(
            event_type="question_pending", title="Auto asks", link_type="question", link_id="0", user_id=pk,
        ))

    assert result == {"dispatched_to": ["in_app"]}
    bell = db_session.execute(text(
        "SELECT user_id, event_type FROM notifications WHERE workspace_id = CAST(:ws AS uuid)"),
        {"ws": str(ws)}).fetchall()
    assert [tuple(row) for row in bell] == [(pk, "question_pending")]
    handed.assert_called_once()


# --------------------------------------------------------------------------- the pure part


def test_with_push_defaults_is_a_new_list_and_keeps_a_push_row_already_there():
    rows = [_pref("review_pending", False), _pref("task_complete", True, "in_app")]
    before = copy.deepcopy(rows)

    out = with_push_defaults(rows)

    assert rows == before
    assert out is not rows and out[0] is not rows[0] and out[:2] == rows
    added = [row["event_type"] for row in out[2:]]
    assert added == [event for event in PUSH_SWITCH_EVENTS if event != "review_pending"]
    assert all(row["destination"] == PUSH and row["enabled"] is True for row in out[2:])


def test_the_web_settings_tab_renders_its_nine_events_none_a_push_switch():
    source = SETTINGS_TAB.read_text(encoding="utf-8")
    block = source.split("const EVENT_TYPES", 1)[1].split("const DESTINATIONS", 1)[0]
    events = re.findall(r"id: '([a-z_]+)'", block)

    assert len(events) == len(set(events)) == len(DEFAULT_NOTIFICATION_PREFERENCES)
    assert not set(events) & set(PUSH_SWITCH_EVENTS)
