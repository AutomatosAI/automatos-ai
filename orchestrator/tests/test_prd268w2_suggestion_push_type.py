"""PRD-268 US-C09a (§6.5.3, K9, K16, K19) — the fourth push type, ``suggestion``, shipped off.

``services/push/payload.py`` gains the type, ``services/push/fanout.py`` holds it behind
``config.SUGGESTION_PUSH_ENABLED``. Proven here:

* ``suggestion_ready`` raises the ``suggestion`` push: channel ``suggestions``, iOS category
  ``automatos.suggestion`` (Start… and Open on the phone), default priority (never high), and
  the AI badge in the title as every push.
* The body is the generic line ("Auto has a suggestion", in the phone's language: English and
  Portuguese (Brazil), K16) unless the workspace's ``settings.push.suggestion_titles`` is true;
  then it is the suggestion's title, one plain line of at most 120 characters. The title is
  never signed or carried in the payload, and only ever a suggestion push's body.
* The signed field set and its order are PRD-266's (``question`` is ``''``); the vector file's
  new ``suggestion`` case recomputes with stdlib ``hmac``, verifies with ``verify_signature``,
  and a copy with any signed field changed does not.
* On Postgres: while the switch is off (its shipped default) ``eligible_devices`` returns no
  phone for a suggestion and ``deliver`` sends nothing, though the same phone is still pushed an
  ``auto_message``, and the bell row is still written; with the switch on the person's phone
  gets one signed suggestion push, its ticket kept as ``suggestion``. The fan-out reads the
  title only with the workspace switch on, and never another person's suggestion.
* ``GET /api/notification-preferences`` lists the fourth switch (``push_types.suggestion``), on
  with no row; a push row off for ``suggestion_ready`` keeps that person's phone out.

CI's test database has no ``notification_preferences`` or ``notifications`` (no model), so the
fixture runs the real ``prd128_notifications`` revision's DDL inside the test's transaction,
which is rolled back with it (as ``test_prd266_push_preferences.py`` does).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from config import config
from core.models.mobile_push import PUSH_TYPES
from modules.proactive.locale import LOCALES
from services.push import payload as push_payload
from services.push.payload import (
    PUSH_TYPE_EVENTS,
    SIGNED_TEXT_FIELDS,
    SUGGESTION_LINES,
    SUGGESTION_TITLE_MAX_CHARS,
    build_payload,
    canonical_message,
    push_type_for,
    suggestion_titles_enabled,
    to_message,
    verify_signature,
)
from services.push.sender import PRIORITY_DEFAULT, PRIORITY_HIGH, TICKET_OK, PushTicket

ORCH = Path(__file__).resolve().parents[1]
VECTORS = ORCH / "tests" / "fixtures" / "prd266_push_signature_vectors.json"
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
PREFERENCES = "/api/notification-preferences"
DELIVER_OFF_LOOP = "services.push.fanout._deliver_off_loop"
EVENT = "suggestion_ready"
LINE_EN, LINE_PT = "Auto has a suggestion", "O Auto tem uma sugestão"
PUSH_KEY = "pk-test9"  # under 10 characters: gitleaks' generic-api-key never matches it
WS = uuid.UUID("5b0c4c8e-2f6a-4d3e-9a7b-1c2d3e4f5a6b")
SUGGESTION_ID = uuid.UUID("8a4e1f2b-6c3d-4e5f-9a1b-2c3d4e5f6a7b")
NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)
TITLE = "Renew the Stripe key before Friday"
PRD266_SIGNED_FIELDS = ("type", "workspace_id", "notification_id", "grant_id", "link_type", "link_id",
                        "ticket_number", "question")


def _built(**kwargs) -> dict:
    return build_payload(PUSH_KEY, event_type=EVENT, workspace_id=WS, notification_id=1006,
                         link_type="suggestion", link_id=SUGGESTION_ID, now=NOW, **kwargs)


def _device(locale="en") -> SimpleNamespace:
    return SimpleNamespace(expo_push_token="ExpoTok9", locale=locale)


def _case() -> dict:
    return next(case for case in json.loads(VECTORS.read_text(encoding="utf-8"))["cases"]
                if case["name"] == "suggestion")


# --------------------------------------------------------------------------- the type


def test_suggestion_ready_raises_the_fourth_push_type():
    assert PUSH_TYPE_EVENTS["suggestion"] == (EVENT,) and push_type_for(EVENT) == "suggestion"
    assert set(PUSH_TYPE_EVENTS) == set(PUSH_TYPES) == {"needs_answer", "review", "auto_message", "suggestion"}
    assert push_payload.CHANNEL_BY_TYPE["suggestion"] == "suggestions"
    assert push_payload.GENERIC_LINES[EVENT] == LINE_EN
    assert push_payload.SUGGESTION_CATEGORY == "automatos.suggestion"


def test_a_suggestions_payload_and_message():
    built = _built()
    message = to_message(_device(), built, event_type=EVENT, agent_name="Auto")

    assert built["type"] == "suggestion" and (built["link_type"], built["link_id"]) == ("suggestion", str(SUGGESTION_ID))
    assert "question" not in built and "options" not in built and "grant_id" not in built
    assert (message.channel_id, message.category_id) == ("suggestions", "automatos.suggestion")
    assert message.priority == PRIORITY_DEFAULT != PRIORITY_HIGH
    assert message.title == "Auto · AI"
    assert to_message(_device("pt-BR"), built, event_type=EVENT).title == "Auto · IA"
    assert dict(message.data) == built and verify_signature(PUSH_KEY, built, NOW)


# --------------------------------------------------------------------------- the body


@pytest.mark.parametrize("locale, line", [("en", LINE_EN), ("en-GB", LINE_EN), (None, LINE_EN), ("de", LINE_EN),
                                          ("pt-BR", LINE_PT), ("pt", LINE_PT), ("pt_PT", LINE_PT)])
def test_the_body_is_the_generic_line_in_the_phones_language_by_default(locale, line):
    assert to_message(_device(locale), _built(), event_type=EVENT).body == line


def test_the_line_has_a_string_in_every_language_the_server_writes():
    assert set(SUGGESTION_LINES) == set(LOCALES) and all(SUGGESTION_LINES.values())
    assert SUGGESTION_LINES["en"] == LINE_EN


def test_the_titles_switch_is_off_unless_it_is_true():
    assert suggestion_titles_enabled({"push": {"suggestion_titles": True}}) is True
    for settings in ({}, None, [], {"push": {}}, {"push": {"suggestion_titles": "true"}},
                     {"push": {"suggestion_titles": 1}}, {"push": "on"}, {"suggestion_titles": True}):
        assert suggestion_titles_enabled(settings) is False


def test_with_titles_on_the_body_is_the_title_as_one_plain_line_and_never_signed():
    built = _built()
    message = to_message(_device(), built, event_type=EVENT, suggestion_title=f"  **{TITLE}**\n")

    assert message.body == TITLE
    assert TITLE not in json.dumps(dict(message.data)) and TITLE not in canonical_message(built).decode("utf-8")
    assert to_message(_device(), built, event_type=EVENT, suggestion_title=" \n\x00 ").body == LINE_EN


def test_a_title_past_120_characters_ends_in_an_ellipsis():
    body = to_message(_device(), _built(), event_type=EVENT, suggestion_title="t" * 300).body
    assert len(body) == SUGGESTION_TITLE_MAX_CHARS == 120
    assert body.endswith(push_payload.ELLIPSIS) and body[:-1] == "t" * 119


def test_a_title_is_only_ever_a_suggestion_pushs_body():
    built = build_payload(PUSH_KEY, event_type="auto_message", workspace_id=WS, now=NOW)
    assert to_message(_device(), built, event_type="auto_message", suggestion_title=TITLE).body == "Auto messaged you"


# --------------------------------------------------------------------------- the signature


def test_the_signed_fields_and_their_order_are_prd266s():
    assert SIGNED_TEXT_FIELDS == PRD266_SIGNED_FIELDS
    signed = json.loads(canonical_message(_built()))
    assert signed[SIGNED_TEXT_FIELDS.index("question") + 1] == "" and signed[-2] == []


def test_the_new_vector_recomputes_with_stdlib_hmac_and_verifies():
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    case, key = _case(), vectors["key"]
    payload = case["payload"]
    array = ["v2", *(payload.get(name, "") for name in PRD266_SIGNED_FIELDS), [], payload["iat"]]
    message = json.dumps(array, separators=(",", ":"), ensure_ascii=False)
    sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)

    assert message == case["message"] and sig == case["sig"] == payload["sig"]
    assert payload["type"] == "suggestion" and "question" not in payload
    assert verify_signature(key, payload, at) is True


@pytest.mark.parametrize("field, value", [
    ("type", "auto_message"), ("workspace_id", str(uuid.uuid4())), ("notification_id", "1007"),
    ("link_type", "chat"), ("link_id", str(uuid.uuid4())), ("question", "Start it?"), ("iat", 1792540801),
])
def test_a_tampered_copy_of_the_new_vector_fails(field, value):
    key = json.loads(VECTORS.read_text(encoding="utf-8"))["key"]
    payload = _case()["payload"]
    at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
    assert verify_signature(key, {**payload, field: value}, at) is False


# --------------------------------------------------------------------------- on Postgres


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeSender:
    """Accepts every message with a fresh ticket id, and remembers what was sent."""

    def __init__(self):
        self.sent = []

    def send(self, messages, /):
        self.sent.extend(messages)
        return [PushTicket(to=m.to, status=TICKET_OK, ticket_id=str(uuid.uuid4())) for m in messages]

    def receipts(self, ticket_ids, /):
        return {}


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here): the
    route and ``deliver`` commit a savepoint only, and the whole test is rolled back."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture
def ws(seed_workspace, db_session, monkeypatch) -> uuid.UUID:
    """A SaaS workspace with ``notification_preferences`` and ``notifications`` as production has them."""
    for table in ("proactive_suggestions", "mobile_devices"):
        if not db_session.execute(text("SELECT to_regclass(:t) IS NOT NULL"), {"t": table}).scalar():
            pytest.skip(f"the suggestion push tests need Postgres with {table}")
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    workspace_id = uuid.UUID(seed_workspace())
    db_session.commit()
    return workspace_id


def _person(db, workspace_id) -> tuple:
    """A member with a Clerk id and an active iPhone that registered the suggestion category:
    ``(users.id, request context, Expo token)``."""
    from core.auth.dependencies import RequestContext, UserContext
    from core.models.mobile_devices import MobileDevice

    tag = uuid.uuid4().hex[:12]
    clerk_id, email = f"user_prd268w2sp{tag}", f"prd268w2-push-{tag}@t.test"
    pk = db.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) "
                         "VALUES (:u, :e, true, :c) RETURNING id"),
                    {"u": f"prd268w2-push-{tag}", "e": email, "c": clerk_id}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, 'editor', true)"), {"ws": str(workspace_id), "uid": pk})
    token = f"ExponentPushToken[{tag}]"
    db.add(MobileDevice(workspace_id=workspace_id, user_id=pk, kind="clerk", status="active",
                        install_id=f"prd268w2-push-{tag}", platform="ios", locale="en", expo_push_token=token,
                        push_enabled=True, push_key=f"prd268w2-push-key-{tag}", events=[],
                        push_categories=["suggestion"]))
    db.commit()
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id)
    return pk, RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk"), token


def _suggestion(db, workspace_id, user_id, title=TITLE) -> str:
    """An open suggestion for today, for ``user_id``."""
    from core.models.proactive import ProactiveSuggestion

    suggestion_id = uuid.uuid4()
    db.add(ProactiveSuggestion(
        id=suggestion_id, workspace_id=workspace_id, user_id=user_id, source_type="check_in", title=title,
        proposed_action={"kind": "chat", "payload": {}}, subject_key=f"title:{title.lower()}:{suggestion_id}",
        severity="today", created_at=datetime.now(timezone.utc), last_seen_at=datetime.now(timezone.utc)))
    db.commit()
    return str(suggestion_id)


def _event(workspace_id, user_id, suggestion_id, event_type=EVENT, link_type="suggestion"):
    from services.push.fanout import PushEvent

    return PushEvent(workspace_id=str(workspace_id), event_type=event_type, notification_id=None,
                     link_type=link_type, link_id=suggestion_id, user_id=user_id, agent_name="Auto")


def _deliver(db, event) -> list:
    from services.push import fanout

    sender = _FakeSender()
    fanout.deliver(event, sender=sender, session_factory=lambda: _BorrowedSession(db))
    return sender.sent


def _switches(body) -> dict:
    """GET's push rows, ``{event_type: enabled}``."""
    return {row["event_type"]: row["enabled"] for row in body["preferences"] if row["destination"] == "push"}


def _show_titles(db, workspace_id, on: bool) -> None:
    db.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:ws AS uuid)"),
               {"s": json.dumps({"push": {"suggestion_titles": on}}), "ws": str(workspace_id)})
    db.commit()


def test_while_the_switch_is_off_no_phone_is_chosen_and_nothing_is_sent(ws, db_session):
    from services.push import fanout

    assert config.SUGGESTION_PUSH_ENABLED is False  # K9: it ships off
    pk, _ctx, token = _person(db_session, ws)
    event = _event(ws, pk, _suggestion(db_session, ws, pk))

    assert fanout.eligible_devices(db_session, event, [pk]) == []
    sender = _FakeSender()
    counts = fanout.deliver(event, sender=sender, session_factory=lambda: _BorrowedSession(db_session))
    assert sender.sent == [] and (counts["recipients"], counts["devices"], counts["sent"]) == (1, 0, 0)
    chat = _event(ws, pk, str(uuid.uuid4()), "auto_message", "chat")
    assert [device.expo_push_token for device in fanout.eligible_devices(db_session, chat, [pk])] == [token]


def test_with_the_switch_on_the_persons_phone_gets_one_signed_suggestion_push(ws, db_session, monkeypatch):
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)
    pk, _ctx, token = _person(db_session, ws)
    event = _event(ws, pk, _suggestion(db_session, ws, pk))

    sent = _deliver(db_session, event)

    assert [message.to for message in sent] == [token]
    message = sent[0]
    assert (message.body, message.channel_id, message.category_id) == (LINE_EN, "suggestions", "automatos.suggestion")
    assert message.data["type"] == "suggestion" and message.data["link_id"] == event.link_id
    assert db_session.execute(text("SELECT push_type FROM mobile_push_tickets WHERE workspace_id = CAST(:ws AS uuid)"),
                              {"ws": str(ws)}).scalars().all() == ["suggestion"]


def test_the_fan_out_reads_the_title_only_with_titles_on_and_only_the_persons_own_open_one(ws, db_session,
                                                                                            monkeypatch):
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)
    pk, _ctx, _token = _person(db_session, ws)
    other, _other_ctx, _other_token = _person(db_session, ws)
    mine, theirs = _suggestion(db_session, ws, pk), _suggestion(db_session, ws, other, "Their private plan")

    def bodies(suggestion_id):
        return {m.body for m in _deliver(db_session, _event(ws, pk, suggestion_id))}

    assert bodies(mine) == {LINE_EN}
    _show_titles(db_session, ws, True)
    assert bodies(mine) == {TITLE}
    assert "Their private plan" not in bodies(theirs)  # another person's suggestion: never its title
    db_session.execute(text("UPDATE proactive_suggestions SET status = 'hidden' WHERE id = CAST(:id AS uuid)"),
                       {"id": mine})
    db_session.commit()
    assert TITLE not in bodies(mine)  # no longer open: never its title


@pytest.mark.parametrize("on", [False, True])
def test_the_bell_row_is_written_and_the_push_handed_over_only_with_the_switch_on(ws, db_session, monkeypatch, on):
    from core.services.notification_dispatcher import NotificationDispatcher

    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", on)
    pk, _ctx, _token = _person(db_session, ws)
    suggestion_id = _suggestion(db_session, ws, pk)
    handed = MagicMock()

    with patch.object(NotificationDispatcher, "_load_auto_reporting", lambda self: {}), \
            patch(DELIVER_OFF_LOOP, new=handed):
        result = asyncio.run(NotificationDispatcher(db_session, ws).dispatch(
            event_type=EVENT, title="Auto has a suggestion", message="1 suggestion for today",
            link_type="suggestion", link_id=suggestion_id, agent_name="Auto", user_id=pk,
        ))

    assert result == {"dispatched_to": ["in_app", "push"] if on else ["in_app"]}
    bell = db_session.execute(text("SELECT user_id, event_type, link_id FROM notifications "
                                   "WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(ws)}).fetchall()
    assert [tuple(row) for row in bell] == [(pk, EVENT, suggestion_id)]
    assert handed.call_count == (1 if on else 0)


def test_the_preferences_get_lists_the_fourth_switch_and_its_row_keeps_the_phone_out(ws, db_session, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import notifications
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db
    from services.push import fanout

    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)
    pk, ctx, _token = _person(db_session, ws)
    app = FastAPI()
    app.include_router(notifications.preferences_router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    client = TestClient(app)

    body = client.get(PREFERENCES).json()
    assert body["push_types"]["suggestion"] == [EVENT] and _switches(body)[EVENT] is True
    off = {"event_type": EVENT, "destination": "push", "enabled": False, "channel_connection_id": None}
    assert client.put(PREFERENCES, json={"preferences": [off]}).status_code == 200
    assert _switches(client.get(PREFERENCES).json())[EVENT] is False
    event = _event(ws, pk, _suggestion(db_session, ws, pk))
    assert fanout.eligible_devices(db_session, event, [pk]) == []
