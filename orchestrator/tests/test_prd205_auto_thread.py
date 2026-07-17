"""PRD-205 S2 -- the per-user Auto thread (find-or-create).

Covers: idempotent find-or-create, race-safety against the partial unique
index (IntegrityError -> re-select adopts the winner), ordinary-chat
behavior (history list ordered by updated_at, deletion + recreation),
and the integer-user-id boundary (#513 lesson).

Live-Postgres suite (PRD-204 stage-1 pattern): probes the migrated schema
and skips cleanly without it. No notification/dispatch code path here.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"prd205 S2 suite needs a migrated Postgres: {exc}")
    yield eng
    eng.dispose()


# ``new_session`` comes from tests/conftest.py -- the shared tracking
# factory (leaked-session guard); teardown sweeps run via new_session.sweep().


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
        {"id": ws_id, "n": f"prd205-s2-{tag}"},
    )
    user_id = s.execute(
        text(
            "INSERT INTO users (email, username, clerk_user_id) "
            "VALUES (:e, :u, :c) RETURNING id"
        ),
        {
            "e": f"prd205-s2-{tag}@test.local",
            "u": f"prd205-s2-{tag}",
            "c": f"user_prd205s2_{tag}",
        },
    ).scalar()
    s.commit()
    s.close()

    yield {"ws_id": ws_id, "user_id": user_id, "tag": tag}

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
    s.commit()
    s.close()


def test_find_or_create_is_idempotent(new_session, seeded):
    from services.chat_messenger import find_or_create_auto_chat

    s = new_session()
    try:
        first = find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])
        second = find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])
        assert first.id == second.id
        assert first.kind == "auto"
        assert first.title == "Auto"
        assert first.visibility == "private"
        count = s.execute(
            text(
                "SELECT COUNT(*) FROM chats WHERE workspace_id = CAST(:w AS uuid) "
                "AND user_id = :u AND kind = 'auto'"
            ),
            {"w": seeded["ws_id"], "u": seeded["user_id"]},
        ).scalar()
        assert count == 1
    finally:
        s.close()


def test_create_race_adopts_the_winner(new_session, seeded, monkeypatch):
    """Simulate the lost race: the initial SELECT misses an existing thread,
    the INSERT trips the partial unique index, and the recovery re-select
    adopts the winner instead of raising."""
    import services.chat_messenger as cm

    s = new_session()
    try:
        winner = cm.find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])

        real_find = cm._find_auto_chat
        calls = {"n": 0}

        def _first_call_misses(db, workspace_id, user_int_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # pretend the concurrent create isn't visible yet
            return real_find(db, workspace_id, user_int_id)

        monkeypatch.setattr(cm, "_find_auto_chat", _first_call_misses)

        adopted = cm.find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])
        assert adopted.id == winner.id
        assert calls["n"] >= 2  # miss, insert collides, recovery re-select
    finally:
        s.close()


def test_rejects_clerk_string_user_id(new_session, seeded):
    """#513 lesson: never let a Clerk subject string near chats.user_id."""
    from services.chat_messenger import find_or_create_auto_chat

    s = new_session()
    try:
        with pytest.raises(ValueError):
            find_or_create_auto_chat(s, seeded["ws_id"], "user_2abcDEF")
    finally:
        s.close()


def test_history_and_get_chat_surface_kind(new_session, seeded):
    """S7 passthrough: /history rows and GET /{chat_id} carry ``chats.kind``
    so the UI can mark the Auto thread ('user' for ordinary chats)."""
    import asyncio
    from types import SimpleNamespace

    try:
        from api.chat import get_chat, get_chat_history
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.chat not importable in this env: {e}")
    from consumers.chatbot.service import ChatService
    from services.chat_messenger import find_or_create_auto_chat

    s = new_session()
    try:
        svc = ChatService(s)
        regular = svc.create_chat(
            user_id=seeded["user_id"],
            title=f"regular-kind {seeded['tag']}",
            workspace_id=uuid.UUID(seeded["ws_id"]),
        )
        auto = find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])

        # get_user_id's fast path takes an integer ctx.user.id as-is, so a
        # SimpleNamespace principal exercises the real endpoint bodies.
        ctx = SimpleNamespace(
            workspace_id=uuid.UUID(seeded["ws_id"]),
            user=SimpleNamespace(id=seeded["user_id"]),
        )

        rows = asyncio.run(get_chat_history(limit=50, ctx=ctx, db=s))
        by_id = {row["id"]: row for row in rows}
        assert by_id[str(auto.id)]["kind"] == "auto"
        assert by_id[str(regular.id)]["kind"] == "user"

        payload = asyncio.run(get_chat(str(auto.id), ctx=ctx, db=s))
        assert payload["kind"] == "auto"
        assert payload["userId"] == seeded["user_id"]
    finally:
        s.close()


def test_auto_thread_is_an_ordinary_chat_in_history_and_deletable(new_session, seeded):
    from consumers.chatbot.service import ChatService
    from services.chat_messenger import find_or_create_auto_chat

    s = new_session()
    try:
        svc = ChatService(s)
        regular = svc.create_chat(
            user_id=seeded["user_id"],
            title=f"regular {seeded['tag']}",
            workspace_id=uuid.UUID(seeded["ws_id"]),
        )
        auto = find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])

        # Created later -> newer updated_at -> first in the history list.
        history = svc.get_chat_history(
            user_id=seeded["user_id"], workspace_id=uuid.UUID(seeded["ws_id"])
        )
        ids = [c.id for c in history]
        assert auto.id in ids and regular.id in ids
        assert ids.index(auto.id) < ids.index(regular.id)

        # A message to the regular chat bumps it back above (updated_at sort).
        svc.save_message(
            chat_id=str(regular.id),
            role="user",
            parts=[{"type": "text", "text": "bump"}],
            workspace_id=seeded["ws_id"],
        )
        history = svc.get_chat_history(
            user_id=seeded["user_id"], workspace_id=uuid.UUID(seeded["ws_id"])
        )
        ids = [c.id for c in history]
        assert ids.index(regular.id) < ids.index(auto.id)

        # Deletion allowed; the next find-or-create mints a fresh thread.
        assert svc.delete_chat(str(auto.id)) is True
        recreated = find_or_create_auto_chat(s, seeded["ws_id"], seeded["user_id"])
        assert recreated.id != auto.id
        assert recreated.kind == "auto"
    finally:
        s.close()
