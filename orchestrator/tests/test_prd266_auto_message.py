"""PRD-266 US-012b (§4.4, D22, K11) — 'Auto messaged you': Auto's unprompted posts tell the chat's owner.

On Postgres with committing sessions: the posts go through the real
``services.chat_messenger.deliver_background_message``; ``NotificationDispatcher.dispatch`` records
each call and writes the bell row through the dispatcher's own ``_insert_in_app`` (so the
ten-minute window reads exactly what the dispatcher writes); ``send_soon`` runs inline (a sync test
runs no loop). CI's test DB has no ``notifications`` table (scripts/init_test_db.py builds from the
models), so the fixture makes it as alembic/versions/prd128_notifications.py does and drops it
after. Time passes by moving the workspace's bell rows back. Proven here:

* A watcher post tells the chat's owner only, linked to the chat, the excerpt as its message; a
  post without a person (an agent's scheduled task) tells the owner of the chat it landed in; on
  a running loop the notice runs when the loop does.
* A question's answer, a click's result, and a post without an origin tell nobody.
* Three Mission narration lines in ten minutes tell once; a daily digest right after tells once
  more (it is never held back); a scheduled-task post eleven minutes later tells once more. The
  window is per chat; two digests in a row both tell.
* None, a mock (with or without Message's spec), and an unsaved Message tell nobody.
* An ``announce`` or a dispatcher that raises never reaches ``deliver_background_message``'s
  caller, which still gets the saved message; a message expired by its commit still tells.
* A chat outside the message's workspace tells nobody; a long message is cut to 200 characters.
* The event is valid, a push type, and its origins and window are K11's.
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

# The table as alembic/versions/prd128_notifications.py makes it: a test DB built from the
# models has none. On a migrated DB the real one is used and only this test's rows are swept.
PRD128_NOTIFICATIONS = """
    CREATE TABLE notifications (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        event_type VARCHAR(50) NOT NULL, title VARCHAR(255) NOT NULL,
        message TEXT, link_type VARCHAR(30), link_id TEXT, agent_id INTEGER, agent_name VARCHAR(100),
        status VARCHAR(20) NOT NULL DEFAULT 'ok', read_at TIMESTAMPTZ, dismissed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the auto_message tests need Postgres with the PRD-205 schema: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag):
    clerk = f"user_prd266am_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return clerk, int(pk)


@pytest.fixture
def people(engine, new_session):
    """A workspace, its owner and another member: ``(workspace id, owner, other)``, each person
    ``(clerk id, users.id)``; the owner also has an ordinary chat. Swept on teardown."""
    ws = str(uuid.uuid4())
    s = new_session()
    built = not s.execute(text("SELECT to_regclass('notifications') IS NOT NULL")).scalar()
    if built:
        s.execute(text(PRD128_NOTIFICATIONS))
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd266-auto-message')"),
              {"id": ws})
    owner, other = _person(s, "owner"), _person(s, "other")
    s.commit()
    yield ws, owner, other
    s = new_session.sweep()
    s.execute(text("DROP TABLE notifications") if built else
              text("DELETE FROM notifications WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    for stmt in ("DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM users WHERE clerk_user_id IN (:a, :b)",
                 "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"):
        s.execute(text(stmt), {"w": ws, "a": owner[0], "b": other[0]})
    s.commit()


@pytest.fixture
def told(monkeypatch, new_session):
    """Every dispatch, recorded, its bell row written as the dispatcher writes it; the notice runs
    on the test's own sessions."""
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.notification_events import auto_message

    calls = []

    async def record(self, **kwargs):
        calls.append({"workspace_id": self.workspace_id, **kwargs})
        self._insert_in_app(
            user_id=kwargs.get("user_id"), event_type=kwargs["event_type"], title=kwargs["title"],
            message=kwargs.get("message"), link_type=kwargs.get("link_type"),
            link_id=kwargs.get("link_id"), agent_id=kwargs.get("agent_id"),
            agent_name=kwargs.get("agent_name"), status=kwargs.get("status", "ok"))
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", record)
    monkeypatch.setattr(auto_message, "_open_session", new_session)
    return calls


def _post(new_session, ws, person, origin, words="Watched it. 8.4/10, passed.", chat_id=None, **session):
    """One background post through the producers' seam, committed as the Mission narration commits."""
    from services.chat_messenger import deliver_background_message

    s = new_session(**session)
    message = deliver_background_message(
        s, workspace_id=ws, text=words, source={"origin": origin} if origin else None,
        chat_id=chat_id, clerk_user_id=person[0])
    s.commit()
    return message


def _pass_minutes(new_session, ws, minutes):
    """Time passes: the workspace's bell rows move back by ``minutes``."""
    s = new_session()
    s.execute(text("UPDATE notifications SET created_at = created_at - make_interval(mins => :m) "
                   "WHERE workspace_id = CAST(:w AS uuid)"), {"m": minutes, "w": ws})
    s.commit()


def _bell(new_session, ws, user_pk):
    return new_session().execute(text(
        "SELECT event_type, title, message, link_type, link_id, agent_name FROM notifications "
        "WHERE workspace_id = CAST(:w AS uuid) AND user_id = :u"), {"w": ws, "u": user_pk}).fetchall()


def _user_chat(new_session, ws, person):
    chat_id = str(uuid.uuid4())
    s = new_session()
    s.execute(text(
        "INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind) "
        "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), 'Lisbon deck', 'private', 'user')"),
        {"id": chat_id, "u": person[1], "w": ws})
    s.commit()
    return chat_id


# --------------------------------------------------------------------------- who is told


def test_a_watcher_post_tells_the_chats_owner_only(people, told, new_session):
    ws, owner, other = people

    message = _post(new_session, ws, owner, "watcher", words="Watch verdict\n\n8.4/10, passed.")

    assert told == [{
        "workspace_id": ws, "event_type": "auto_message", "title": "Auto messaged you",
        "message": "Watch verdict 8.4/10, passed.", "link_type": "chat", "link_id": str(message.chat_id),
        "agent_name": "Auto", "user_id": owner[1],
    }]
    assert _bell(new_session, ws, owner[1]) == [(
        "auto_message", "Auto messaged you", "Watch verdict 8.4/10, passed.", "chat",
        str(message.chat_id), "Auto")]
    assert _bell(new_session, ws, other[1]) == []


def test_a_post_without_a_person_tells_the_chats_owner(people, told, new_session):
    ws, _, other = people
    their_chat = _user_chat(new_session, ws, other)

    _post(new_session, ws, (None, None), "scheduled_task", chat_id=their_chat)   # an agent's own task

    assert [(c["user_id"], c["link_id"]) for c in told] == [(other[1], their_chat)]


def test_a_post_on_a_running_loop_tells_when_the_loop_runs_the_notice(people, told, new_session):
    ws, owner, _ = people

    async def on_a_loop():
        message = _post(new_session, ws, owner, "watcher")
        before = len(told)
        for _ in range(3):
            await asyncio.sleep(0)
        return message, before

    message, before = asyncio.run(on_a_loop())

    assert before == 0 and [c["link_id"] for c in told] == [str(message.chat_id)]


def test_a_question_answer_and_a_click_result_tell_nobody(people, told, new_session):
    from services.click_results import ORIGIN as CLICK_RESULT

    ws, owner, _ = people

    for origin in ("question_answer", CLICK_RESULT, None):
        assert _post(new_session, ws, owner, origin) is not None   # posted, never announced

    assert told == []


# --------------------------------------------------------------------------- the window


def test_narration_is_told_once_in_ten_minutes_the_digest_always_and_a_later_post_again(
        people, told, new_session):
    ws, owner, _ = people

    _post(new_session, ws, owner, "mission", words="Mission started: Lisbon deck")
    _pass_minutes(new_session, ws, 4)
    _post(new_session, ws, owner, "mission", words="Step 1 of 3 done")
    _pass_minutes(new_session, ws, 5)
    _post(new_session, ws, owner, "mission", words="Step 2 of 3 done")
    assert [c["message"] for c in told] == ["Mission started: Lisbon deck"]

    _post(new_session, ws, owner, "daily_digest", words="Your Read for Friday")
    _pass_minutes(new_session, ws, 11)
    _post(new_session, ws, owner, "scheduled_task", words="Weekly numbers: £4,210")

    assert [c["message"] for c in told] == [
        "Mission started: Lisbon deck", "Your Read for Friday", "Weekly numbers: £4,210"]
    assert len({c["link_id"] for c in told}) == 1           # one Auto thread throughout


def test_two_digests_in_a_row_both_tell(people, told, new_session):
    ws, owner, _ = people

    _post(new_session, ws, owner, "daily_digest", words="Your Read for Friday")
    _post(new_session, ws, owner, "daily_digest", words="Your Read for Saturday")

    assert [c["message"] for c in told] == ["Your Read for Friday", "Your Read for Saturday"]


def test_the_window_is_per_chat(people, told, new_session):
    ws, owner, _ = people
    deck_chat = _user_chat(new_session, ws, owner)

    auto_thread = _post(new_session, ws, owner, "watcher").chat_id
    _post(new_session, ws, owner, "mission", words="Mission started", chat_id=deck_chat)

    assert [c["link_id"] for c in told] == [str(auto_thread), deck_chat]


# --------------------------------------------------------------------------- what is announced


def test_a_mock_none_or_unsaved_message_tells_nobody(people, monkeypatch):
    from core.models.core import Message
    from services.notification_events import auto_message

    ws, _, _ = people
    scheduled = []
    monkeypatch.setattr(auto_message, "send_soon", lambda coro, what: scheduled.append(what))
    unsaved = Message(id=uuid.uuid4(), chat_id=uuid.uuid4(), workspace_id=uuid.UUID(ws), role="assistant",
                      parts=[{"type": "text", "text": "never written"}])

    for message in (None, MagicMock(), MagicMock(spec=Message), unsaved):
        auto_message.announce(message, {"origin": "watcher"})

    assert scheduled == []


def test_a_message_expired_by_its_commit_still_tells(people, told, new_session):
    ws, owner, _ = people

    message = _post(new_session, ws, owner, "watcher", expire_on_commit=True)

    assert [(c["link_id"], c["message"]) for c in told] == [
        (str(message.chat_id), "Watched it. 8.4/10, passed.")]


def test_a_long_message_is_cut_to_200_characters(people, told, new_session):
    ws, owner, _ = people

    _post(new_session, ws, owner, "scheduled_task", words="Revenue " * 60)

    excerpt = told[0]["message"]
    assert len(excerpt) == 200 and excerpt.endswith("…") and excerpt.startswith("Revenue Revenue")


def test_a_chat_outside_the_messages_workspace_tells_nobody(people, told, new_session):
    from services.notification_events.auto_message import AutoMessage, dispatch_auto_message

    ws, owner, _ = people
    chat_id = _user_chat(new_session, ws, owner)

    asyncio.run(dispatch_auto_message(AutoMessage(
        workspace_id=str(uuid.uuid4()), chat_id=chat_id, excerpt="elsewhere", origin="watcher")))

    assert told == []


# --------------------------------------------------------------------------- never in the way


def test_an_announce_that_raises_never_reaches_the_caller(people, monkeypatch, new_session):
    from core.models.core import Message
    from services.notification_events import auto_message

    ws, owner, _ = people

    def boom(message, source):
        raise RuntimeError("the notice is down")

    monkeypatch.setattr(auto_message, "announce", boom)

    message = _post(new_session, ws, owner, "watcher")

    assert isinstance(message, Message)
    assert new_session().query(Message).filter(Message.id == message.id).count() == 1


def test_a_dispatcher_that_raises_never_reaches_the_caller(people, monkeypatch, new_session):
    from core.models.core import Message
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.notification_events import auto_message

    ws, owner, _ = people

    async def boom(self, **kwargs):
        raise RuntimeError("the dispatcher is down")

    monkeypatch.setattr(NotificationDispatcher, "dispatch", boom)
    monkeypatch.setattr(auto_message, "_open_session", new_session)

    message = _post(new_session, ws, owner, "mission")

    assert isinstance(message, Message) and message.parts == [
        {"type": "text", "text": "Watched it. 8.4/10, passed."}]


def test_deliver_background_message_still_returns_the_saved_message(people, told, new_session, monkeypatch):
    from services import chat_messenger

    ws, owner, _ = people
    returned = []
    real_post = chat_messenger.post_background_message

    def post(db, **kwargs):
        returned.append(real_post(db, **kwargs))
        return returned[-1]

    monkeypatch.setattr(chat_messenger, "post_background_message", post)

    message = _post(new_session, ws, owner, "watcher")

    assert message is returned[0] and len(told) == 1


# --------------------------------------------------------------------------- the event


def test_auto_message_is_a_valid_event_and_a_push_type_with_k11s_origins_and_window():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES
    from services.notification_events.auto_message import ANNOUNCED_ORIGINS, AUTO_MESSAGE_COALESCE_SECONDS
    from services.push.payload import PUSH_TYPE_EVENTS

    assert "auto_message" in VALID_EVENT_TYPES and PUSH_TYPE_EVENTS["auto_message"] == ("auto_message",)
    assert ANNOUNCED_ORIGINS == {"watcher", "scheduled_task", "mission", "daily_digest"}
    assert AUTO_MESSAGE_COALESCE_SECONDS == 600
