"""PRD-205 S1 (+S7 backend) -- ChatMessenger + chat_changed emitter.

Covers: Clerk-string resolution (#513), workspace-mismatch rejection with
Auto-thread fallback, the exact AI-SDK parts shape parity with in-turn
assistant writes, source stamping (label default + link merge), the
fail-soft wrapper contract (a raising messenger never propagates), the
injectable session factory, and the S7 ``chat_changed`` NOTIFY: payload
shape, channel, post-commit emission, fail-softness, and
``frame_for_payload`` tolerance (no task_id/status keys -- must pass the
workspace gate untouched).

The chat_changed notify is CAPTURED at the emitter seam (never LISTEN'd);
DB tests are live-Postgres (PRD-204 stage-1 pattern) with clean skip.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url


# ---------------------------------------------------------------------------
# S7 backend -- pure unit tests (mock db; no Postgres needed)
# ---------------------------------------------------------------------------


def test_notify_chat_event_payload_shape_and_channel():
    from services.board_events import NOTIFY_CHANNEL, notify_chat_event

    db = MagicMock()
    ws = uuid.uuid4()
    chat = uuid.uuid4()
    notify_chat_event(db, workspace_id=ws, chat_id=chat, user_id=7)

    assert db.execute.call_count == 1
    _, params = db.execute.call_args[0]
    assert params["chan"] == NOTIFY_CHANNEL == "board_events"
    payload = json.loads(params["payload"])
    assert payload == {
        "workspace_id": str(ws),
        "chat_id": str(chat),
        "user_id": 7,
        "event": "chat_changed",
    }


def test_notify_chat_event_is_fail_soft():
    from services.board_events import notify_chat_event

    db = MagicMock()
    db.execute.side_effect = RuntimeError("pg down")
    # Must not raise -- NOTIFY buys latency, not correctness.
    notify_chat_event(db, workspace_id=uuid.uuid4(), chat_id=uuid.uuid4(), user_id=1)


def test_frame_for_payload_passes_chat_changed_without_task_keys():
    """The SSE workspace gate keys ONLY on workspace_id -- a chat payload
    (no task_id/status) must pass for its workspace and drop for others."""
    from services.board_events import frame_for_payload

    ws = str(uuid.uuid4())
    payload = json.dumps(
        {"workspace_id": ws, "chat_id": str(uuid.uuid4()), "user_id": 3,
         "event": "chat_changed"}
    )
    frame = frame_for_payload(payload, ws)
    assert frame is not None
    assert "chat_changed" in frame  # the payload's event field is forwarded

    assert frame_for_payload(payload, str(uuid.uuid4())) is None  # other tenant


# ---------------------------------------------------------------------------
# Fail-soft wrapper -- no DB needed
# ---------------------------------------------------------------------------


def test_deliver_never_propagates_a_raising_session_factory():
    from services.chat_messenger import deliver_background_message

    def _boom():
        raise RuntimeError("no database for you")

    out = deliver_background_message(
        workspace_id=str(uuid.uuid4()),
        text="verdict prose",
        source={"origin": "watcher"},
        session_factory=_boom,
    )
    assert out is None


def test_deliver_never_propagates_an_inner_failure_and_closes_session():
    from services.chat_messenger import deliver_background_message

    session = MagicMock()
    session.query.side_effect = RuntimeError("session poisoned")

    out = deliver_background_message(
        workspace_id="not-even-a-uuid",  # trips the boundary ValueError too
        text="prose",
        source={"origin": "watcher"},
        session_factory=lambda: session,
    )
    assert out is None
    assert session.close.called  # owned session is always released


# ---------------------------------------------------------------------------
# DB fixtures (PRD-204 stage-1 pattern; skip cleanly without Postgres)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"prd205 S1 suite needs a migrated Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seeded(new_session):
    ws_id = str(uuid.uuid4())
    tag = uuid.uuid4().hex[:10]
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": f"prd205-s1-{tag}"},
    )
    user_id = s.execute(
        text(
            "INSERT INTO users (email, username, clerk_user_id) "
            "VALUES (:e, :u, :c) RETURNING id"
        ),
        {
            "e": f"prd205-s1-{tag}@test.local",
            "u": f"prd205-s1-{tag}",
            "c": f"user_prd205s1_{tag}",
        },
    ).scalar()
    s.commit()
    s.close()

    yield {
        "ws_id": ws_id,
        "user_id": user_id,
        "clerk": f"user_prd205s1_{tag}",
        "email": f"prd205-s1-{tag}@test.local",
        "tag": tag,
    }

    s = new_session()
    for stmt in (
        "DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
    s.commit()
    s.close()


@pytest.fixture
def capture_chat_notify(monkeypatch):
    """Capture the chat_changed emitter at the seam (never LISTEN)."""
    import services.board_events as be

    fired = []

    def _capture(db, *, workspace_id, chat_id, user_id):
        fired.append(
            {"workspace_id": str(workspace_id), "chat_id": str(chat_id),
             "user_id": user_id}
        )

    monkeypatch.setattr(be, "notify_chat_event", _capture)
    return fired


# ---------------------------------------------------------------------------
# S1 mechanics on the live schema
# ---------------------------------------------------------------------------


def test_clerk_resolution_auto_thread_parts_parity_and_source(
    new_session, seeded, capture_chat_notify
):
    from services.chat_messenger import (
        AUTO_BACKGROUND_LABEL,
        post_background_message,
    )

    s = new_session()
    try:
        msg = post_background_message(
            s,
            workspace_id=seeded["ws_id"],
            text="Watched it. 8.4/10, passed.",
            source={"origin": "watcher"},
            clerk_user_id=seeded["clerk"],
            link_type="watch",
            link_id="w-123",
        )
        assert msg is not None
        assert msg.role == "assistant"
        # EXACT AI-SDK shape parity with the in-turn assistant save
        # (consumers/chatbot/service.py: [{'type': 'text', 'text': ...}]).
        assert msg.parts == [{"type": "text", "text": "Watched it. 8.4/10, passed."}]
        assert msg.source["origin"] == "watcher"
        assert msg.source["label"] == AUTO_BACKGROUND_LABEL
        assert msg.source["link_type"] == "watch"
        assert msg.source["link_id"] == "w-123"

        # Landed in the (created-on-demand) Auto thread of the resolved user.
        row = s.execute(
            text(
                "SELECT c.kind, c.user_id, c.title FROM chats c "
                "JOIN messages m ON m.chat_id = c.id WHERE m.id = CAST(:m AS uuid)"
            ),
            {"m": str(msg.id)},
        ).fetchone()
        assert row.kind == "auto"
        assert row.user_id == seeded["user_id"]
        assert row.title == "Auto"

        # chat_changed fired post-commit with the chat owner's integer id.
        assert len(capture_chat_notify) == 1
        assert capture_chat_notify[0]["user_id"] == seeded["user_id"]
        assert capture_chat_notify[0]["workspace_id"] == seeded["ws_id"]
    finally:
        s.close()


def test_valid_origin_chat_is_used_directly(new_session, seeded, capture_chat_notify):
    from consumers.chatbot.service import ChatService
    from services.chat_messenger import post_background_message

    s = new_session()
    try:
        origin = ChatService(s).create_chat(
            user_id=seeded["user_id"],
            title=f"origin {seeded['tag']}",
            workspace_id=uuid.UUID(seeded["ws_id"]),
        )
        msg = post_background_message(
            s,
            workspace_id=seeded["ws_id"],
            text="verdict in the originating conversation",
            source={"origin": "watcher"},
            chat_id=str(origin.id),
            clerk_user_id=seeded["clerk"],
        )
        assert msg is not None
        assert str(msg.chat_id) == str(origin.id)
        # No Auto thread was needed.
        n_auto = s.execute(
            text(
                "SELECT COUNT(*) FROM chats WHERE workspace_id = CAST(:w AS uuid) "
                "AND kind = 'auto'"
            ),
            {"w": seeded["ws_id"]},
        ).scalar()
        assert n_auto == 0
        assert capture_chat_notify[-1]["chat_id"] == str(origin.id)
    finally:
        s.close()


def test_foreign_workspace_chat_falls_back_to_auto_thread(
    new_session, seeded, capture_chat_notify
):
    """A chat_id from ANOTHER workspace must never be posted into."""
    from consumers.chatbot.service import ChatService
    from services.chat_messenger import post_background_message

    other_ws = str(uuid.uuid4())
    s = new_session()
    try:
        s.execute(
            text(
                "INSERT INTO workspaces (id, name) "
                "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
            ),
            {"id": other_ws, "n": f"prd205-foreign-{seeded['tag']}"},
        )
        s.commit()
        foreign_chat = ChatService(s).create_chat(
            user_id=seeded["user_id"],
            title=f"foreign {seeded['tag']}",
            workspace_id=uuid.UUID(other_ws),
        )

        msg = post_background_message(
            s,
            workspace_id=seeded["ws_id"],  # posting into THIS workspace
            text="must not land in the foreign chat",
            source={"origin": "watcher"},
            chat_id=str(foreign_chat.id),  # ...but the chat lives elsewhere
            clerk_user_id=seeded["clerk"],
        )
        assert msg is not None
        assert str(msg.chat_id) != str(foreign_chat.id)
        assert str(msg.workspace_id) == seeded["ws_id"]

        # Foreign chat stayed empty.
        n = s.execute(
            text("SELECT COUNT(*) FROM messages WHERE chat_id = CAST(:c AS uuid)"),
            {"c": str(foreign_chat.id)},
        ).scalar()
        assert n == 0
    finally:
        s2 = new_session()
        for stmt in (
            "DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
            "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
            "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
        ):
            s2.execute(text(stmt), {"w": other_ws})
        s2.commit()
        s2.close()
        s.close()


def test_no_chat_and_unresolvable_user_drops_message(
    new_session, seeded, capture_chat_notify
):
    from services.chat_messenger import post_background_message

    s = new_session()
    try:
        before = s.execute(
            text("SELECT COUNT(*) FROM messages WHERE workspace_id = CAST(:w AS uuid)"),
            {"w": seeded["ws_id"]},
        ).scalar()
        msg = post_background_message(
            s,
            workspace_id=seeded["ws_id"],
            text="nowhere to go",
            source={"origin": "watcher"},
            clerk_user_id="user_does_not_exist_anywhere",
        )
        assert msg is None
        after = s.execute(
            text("SELECT COUNT(*) FROM messages WHERE workspace_id = CAST(:w AS uuid)"),
            {"w": seeded["ws_id"]},
        ).scalar()
        assert after == before
        assert capture_chat_notify == []
    finally:
        s.close()


def test_email_fallback_resolution(new_session, seeded, capture_chat_notify):
    from services.chat_messenger import post_background_message

    s = new_session()
    try:
        msg = post_background_message(
            s,
            workspace_id=seeded["ws_id"],
            text="resolved via email",
            source={"origin": "scheduled_task"},
            clerk_user_id=seeded["email"],
        )
        assert msg is not None
        owner = s.execute(
            text(
                "SELECT c.user_id FROM chats c JOIN messages m ON m.chat_id = c.id "
                "WHERE m.id = CAST(:m AS uuid)"
            ),
            {"m": str(msg.id)},
        ).scalar()
        assert owner == seeded["user_id"]
    finally:
        s.close()


def test_empty_text_posts_nothing(new_session, seeded, capture_chat_notify):
    from services.chat_messenger import post_background_message

    s = new_session()
    try:
        assert (
            post_background_message(
                s,
                workspace_id=seeded["ws_id"],
                text="   ",
                source={"origin": "scheduled_task"},
                clerk_user_id=seeded["clerk"],
            )
            is None
        )
        assert capture_chat_notify == []
    finally:
        s.close()


def test_deliver_with_injected_factory_end_to_end(engine, seeded, capture_chat_notify):
    """The wrapper on the injected test factory: full path, own session."""
    from services.chat_messenger import deliver_background_message

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    msg = deliver_background_message(
        workspace_id=seeded["ws_id"],
        text="delivered through the wrapper",
        source={"origin": "watcher"},
        clerk_user_id=seeded["clerk"],
        link_type="watch",
        link_id=str(uuid.uuid4()),
        session_factory=factory,
    )
    assert msg is not None
    assert len(capture_chat_notify) == 1

    # Verify from a FRESH session that the write is committed and visible.
    s = factory()
    try:
        n = s.execute(
            text(
                "SELECT COUNT(*) FROM messages WHERE workspace_id = CAST(:w AS uuid) "
                "AND source IS NOT NULL"
            ),
            {"w": seeded["ws_id"]},
        ).scalar()
        assert n == 1
    finally:
        s.close()
