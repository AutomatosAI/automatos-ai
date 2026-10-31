"""PRD-268 US-G03 (§6.4, K11) — the daily brief's push carries its message's id, so the phone opens
Auto's thread at the brief.

The scheduled brief posts ``[text, brief@1]`` into the person's Auto thread and is announced
(``auto_message``); its push already links the thread (``link_type`` chat, ``link_id`` the chat).
Proven here:

* ``build_payload`` carries a ``message_id`` it is given, as a string, and leaves it out when it is
  given none (never null). It is not signed: the signed fields and their order stay PRD-266's
  (K9), so every phone's signature check is unchanged; the vector file's ``daily-brief`` case
  recomputes with stdlib ``hmac`` over the PRD-266 array alone and verifies.
* The fan-out's message for one phone carries the event's ``message_id`` in its data, and none
  when the event has none; the dispatcher hands a ``message_id`` to the push only (the bell row
  links the chat, as before), and none when it is given none.
* On Postgres, through the real ``deliver_background_message``: a scheduled brief's post tells its
  owner with the posted message's id, and that dispatch, replayed through the real dispatcher and
  the fan-out's message builder, is a signed push whose data holds the chat's and the message's
  ids and never a word of the brief. A digest post that holds no ``brief@1`` part carries no id.

Ids only, never the brief's text (PRD-266's push privacy rules). The real-DB tests skip without
Postgres and the PRD-205 schema; ``NotificationDispatcher.dispatch`` is recorded there (CI's test
database has no ``notifications`` table), and ``send_soon`` runs the notice inline (a sync test
runs no loop).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from core.services.notification_dispatcher import NotificationDispatcher
from services.push import fanout
from services.push.fanout import PushEvent
from services.push.payload import (
    AUTO_MESSAGE_LINE,
    SIGNED_TEXT_FIELDS,
    build_payload,
    canonical_message,
    verify_signature,
)

VECTORS = Path(__file__).resolve().parent / "fixtures" / "prd266_push_signature_vectors.json"
PRD266_SIGNED_FIELDS = ("type", "workspace_id", "notification_id", "grant_id", "link_type", "link_id",
                        "ticket_number", "question")
PUSH_KEY = "pk-test7"  # under 10 characters: gitleaks' generic-api-key never matches it
TOKEN = "ExpoTok7"
WS = uuid.UUID("5b0c4c8e-2f6a-4d3e-9a7b-1c2d3e4f5a6b")
CHAT = uuid.UUID("0d9f2c51-7a43-4b8e-9c61-2e5f7a1b3c4d")
MESSAGE = uuid.UUID("3c1e5b7a-9d2f-4a6b-8c0e-1f3a5b7c9d2e")
NOW = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
SAT = date(2026, 10, 24)
ROW_ID = uuid.UUID("6f1c2d3e-4a5b-4c6d-8e9f-0a1b2c3d4e5f")
HAND_OVER = "services.push.fanout.hand_over"
NO_ROUTES: dict[str, Any] = {"enabled": True, "routes": {}, "quiet_hours": {"enabled": False}}
PROSE = "Two things need you: the oat milk order and a question about Sunday opening."
ROW_TITLE = "Answer Sunday opening"
BRIEF = {"v": 1, "date": SAT.isoformat(), "prose": PROSE,
         "sections": {"needs_you": [{"title": ROW_TITLE, "why": "Bo asked at 07:40"}]}}
# Captured before any test records it: the replay runs the real dispatch.
REAL_DISPATCH = NotificationDispatcher.dispatch


def _brief_push(**kwargs: Any) -> dict:
    return build_payload(PUSH_KEY, event_type="auto_message", workspace_id=WS, notification_id=1007,
                         link_type="chat", link_id=CHAT, now=NOW, **kwargs)


def _phone() -> SimpleNamespace:
    return SimpleNamespace(push_key=PUSH_KEY, workspace_id=WS, expo_push_token=TOKEN, locale="en")


def _pushed(event: PushEvent):
    """The fan-out's signed message for one phone, as ``deliver`` builds it."""
    return fanout._message(_phone(), event, fanout._Context(grant=None, ticket_number=None, redact=False), NOW)


def _words_in(pushed: Any) -> Optional[str]:
    """The first of the brief's words found in the push's title, body or data; None when none is."""
    shown = " ".join([pushed.title, pushed.body, json.dumps(dict(pushed.data), ensure_ascii=False)])
    return next((words for words in (PROSE, ROW_TITLE, "oat milk", "Sunday") if words in shown), None)


def _mock_db() -> MagicMock:
    """The prd128 style: the first execute() is the preferences SELECT (none), every later one the
    bell row's insert returning ``ROW_ID``."""
    db = MagicMock()
    prefs, insert = MagicMock(), MagicMock()
    prefs.fetchall.return_value = []
    insert.scalar.return_value = ROW_ID
    db.execute.side_effect = [prefs] + [insert] * 20
    return db


def _handed_over(workspace_id: Any, **event: Any) -> tuple[PushEvent, MagicMock]:
    """One real dispatch on a mock session, ``hand_over`` patched: the push event it handed over,
    and the session (its bell row's insert is in its calls)."""
    db, handed = _mock_db(), MagicMock(return_value=True)
    with patch.object(NotificationDispatcher, "_load_auto_reporting", lambda self: NO_ROUTES), \
            patch.object(NotificationDispatcher, "_is_quiet_hours", staticmethod(lambda s: False)), \
            patch(HAND_OVER, new=handed):
        result = asyncio.run(REAL_DISPATCH(NotificationDispatcher(db, workspace_id), **event))
    assert result == {"dispatched_to": ["in_app", "push"]}
    handed.assert_called_once()
    return handed.call_args.args[1], db


def _bell_rows(db: MagicMock) -> list[dict]:
    return [call.args[1] for call in db.execute.call_args_list if "INSERT INTO notifications" in str(call.args[0])]


# --------------------------------------------------------------------------- the payload


def test_a_payload_carries_the_message_id_it_is_given_as_a_string():
    built = _brief_push(message_id=MESSAGE)

    assert built["message_id"] == str(MESSAGE) and isinstance(built["message_id"], str)
    assert (built["type"], built["link_type"], built["link_id"]) == ("auto_message", "chat", str(CHAT))
    assert verify_signature(PUSH_KEY, built, NOW) is True


@pytest.mark.parametrize("message_id", [None, ""])
def test_a_payload_without_a_message_id_leaves_it_out(message_id):
    built = _brief_push(message_id=message_id)

    assert "message_id" not in built and None not in built.values()
    assert built == _brief_push()


def test_the_message_id_is_not_signed_so_every_phones_check_is_unchanged():
    with_id, without = _brief_push(message_id=MESSAGE), _brief_push()

    assert SIGNED_TEXT_FIELDS == PRD266_SIGNED_FIELDS
    assert canonical_message(with_id) == canonical_message(without)
    assert str(MESSAGE) not in canonical_message(with_id).decode("utf-8")
    assert with_id["sig"] == without["sig"]
    assert {key: value for key, value in with_id.items() if key != "message_id"} == without


def test_the_vector_files_daily_brief_case_recomputes_over_the_prd266_array_and_verifies():
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    case = next(case for case in vectors["cases"] if case["name"] == "daily-brief")
    payload, key = case["payload"], vectors["key"]
    array = ["v2", *(payload.get(name, "") for name in PRD266_SIGNED_FIELDS), [], payload["iat"]]
    message = json.dumps(array, separators=(",", ":"), ensure_ascii=False)
    sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

    assert message == case["message"] and sig == case["sig"] == payload["sig"]
    assert (payload["type"], payload["link_type"], payload["message_id"]) == ("auto_message", "chat", str(MESSAGE))
    assert verify_signature(key, payload, datetime.fromtimestamp(payload["iat"], tz=timezone.utc)) is True


# --------------------------------------------------------------------------- the fan-out and the dispatcher


def test_the_fan_outs_message_carries_the_events_message_id_and_none_without_one():
    event = PushEvent(workspace_id=str(WS), event_type="auto_message", notification_id="n-1",
                      link_type="chat", link_id=str(CHAT), message_id=str(MESSAGE))

    pushed, plain = _pushed(event), _pushed(replace(event, message_id=None))

    assert pushed.data["message_id"] == str(MESSAGE) and pushed.data["link_id"] == str(CHAT)
    assert pushed.body == AUTO_MESSAGE_LINE and verify_signature(PUSH_KEY, pushed.data, NOW) is True
    assert "message_id" not in plain.data and plain.data["link_id"] == str(CHAT)


def test_the_dispatcher_hands_the_message_id_to_the_push_only():
    event, db = _handed_over(WS, event_type="auto_message", title=AUTO_MESSAGE_LINE, message=PROSE,
                             link_type="chat", link_id=str(CHAT), agent_name="Auto", user_id=7,
                             message_id=str(MESSAGE))

    assert (event.message_id, event.link_type, event.link_id) == (str(MESSAGE), "chat", str(CHAT))
    assert event.notification_id == str(ROW_ID)
    pushed = _pushed(event)
    assert pushed.data["message_id"] == str(MESSAGE)
    assert _words_in(pushed) is None  # the bell's excerpt never reaches the push
    (bell,) = _bell_rows(db)
    assert "message_id" not in bell and str(MESSAGE) not in bell.values()
    assert (bell["link_type"], bell["link_id"]) == ("chat", str(CHAT))


def test_a_dispatch_without_a_message_id_hands_over_none():
    event, _ = _handed_over(WS, event_type="auto_message", title=AUTO_MESSAGE_LINE, link_type="chat",
                            link_id=str(CHAT), user_id=7)

    assert event.message_id is None
    assert "message_id" not in _pushed(event).data


# --------------------------------------------------------------------------- on Postgres


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source, parts FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the brief push tests need Postgres with the PRD-205 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def owner(engine, new_session):
    """A workspace and its owner: ``(workspace id, clerk id, users.id)``. Swept on teardown."""
    ws, clerk = str(uuid.uuid4()), f"user_prd268w2bp_{uuid.uuid4().hex[:10]}"
    s = new_session()
    pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    s.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), :n, :o)"),
              {"id": ws, "n": "prd268w2-brief-push", "o": pk})
    s.commit()
    yield ws, clerk, int(pk)
    s = new_session.sweep()
    for stmt in ("DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
                 "DELETE FROM users WHERE clerk_user_id = :c"):
        s.execute(text(stmt), {"w": ws, "c": clerk})
    s.commit()


@pytest.fixture
def told(monkeypatch, new_session) -> list:
    """Every dispatch's arguments, recorded; the notice runs on the test's own sessions."""
    from services.notification_events import auto_message

    calls: list = []

    async def record(self, **kwargs):
        calls.append({"workspace_id": self.workspace_id, **kwargs})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", record)
    monkeypatch.setattr(auto_message, "_open_session", new_session)
    return calls


def _replayed(call: dict) -> PushEvent:
    """A recorded dispatch, run again through the real dispatcher: the push event it hands over."""
    arguments = {key: value for key, value in call.items() if key != "workspace_id"}
    return _handed_over(call["workspace_id"], **arguments)[0]


def test_a_scheduled_briefs_push_carries_its_message_id_and_never_its_words(owner, told, new_session):
    from services.brief.model import VIA_SCHEDULE
    from services.mobile import digest
    from services.mobile.digest import DuePerson

    ws, _, user_pk = owner
    s = new_session()
    message = digest.post_brief_message(s, DuePerson(uuid.UUID(ws), user_pk, SAT), BRIEF, VIA_SCHEDULE)
    s.commit()

    assert message is not None and [part["type"] for part in message.parts] == ["text", "brief@1"]
    (call,) = told
    assert (call["link_type"], call["link_id"], call["message_id"]) == ("chat", str(message.chat_id), str(message.id))
    pushed = _pushed(_replayed(call))
    assert pushed.data["message_id"] == str(message.id) and pushed.data["link_id"] == str(message.chat_id)
    assert pushed.data["type"] == "auto_message" and verify_signature(PUSH_KEY, pushed.data, NOW) is True
    assert _words_in(pushed) is None and pushed.body == AUTO_MESSAGE_LINE


def test_a_digest_post_without_a_brief_carries_no_message_id(owner, told, new_session):
    from services.chat_messenger import deliver_background_message

    ws, clerk, _ = owner
    s = new_session()
    message = deliver_background_message(s, workspace_id=ws, text=PROSE, clerk_user_id=clerk,
                                         source={"origin": "daily_digest", "local_date": SAT.isoformat()})
    s.commit()

    assert message is not None
    (call,) = told
    assert call["link_id"] == str(message.chat_id) and call["message_id"] is None
    pushed = _pushed(_replayed(call))
    assert "message_id" not in pushed.data and _words_in(pushed) is None
