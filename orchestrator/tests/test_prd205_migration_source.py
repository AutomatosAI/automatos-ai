"""PRD-205 S3 -- migration + messages.source surfacing.

Covers:
- the migration file's chaining contract: single parent on
  prd204_watch_registry (PR #551 owns the heads join -- authoring a second
  join of the same parents re-forks the graph);
- messages.source round-trip through the models (stamped row returns the
  dict, old rows return None);
- GET /{chat_id}/messages surfaces ``source`` (null for legacy rows);
- chats.kind defaults to 'user' on the existing create path.

DB tests follow the PRD-204 stage-1 pattern: live Postgres with the
migrated schema, probe + clean skip without it. Notification dispatch is
not on any path here, so no dispatcher patching is needed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url


# ---------------------------------------------------------------------------
# Migration chaining -- pure, no DB
# ---------------------------------------------------------------------------


def _load_migration_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "alembic"
        / "versions"
        / "prd205_auto_speaks.py"
    )
    spec = importlib.util.spec_from_file_location("prd205_auto_speaks_mig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_chains_single_parent_on_prd204_watch_registry():
    mod = _load_migration_module()
    assert mod.revision == "prd205_auto_speaks"
    # Single parent, NOT a join: PR #551 owns the prd204_watch_registry x
    # w3_post201_merge_heads join; a duplicate join re-forks the head graph.
    assert mod.down_revision == "prd204_watch_registry"
    assert isinstance(mod.down_revision, str)


# ---------------------------------------------------------------------------
# DB fixtures (PRD-204 stage-1 pattern; skip cleanly without Postgres)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT source FROM messages LIMIT 1"))
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT origin_chat_id FROM watches LIMIT 1"))
            c.execute(
                text("SELECT origin_chat_id, created_by FROM agent_scheduled_tasks LIMIT 1")
            )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"prd205 S3 suite needs a migrated Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seeded(new_session):
    """workspace + user + chat; yields ids; cleans up FK-safe."""
    ws_id = str(uuid.uuid4())
    tag = uuid.uuid4().hex[:10]
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": f"prd205-s3-{tag}"},
    )
    user_id = s.execute(
        text(
            "INSERT INTO users (email, username, clerk_user_id) "
            "VALUES (:e, :u, :c) RETURNING id"
        ),
        {
            "e": f"prd205-{tag}@test.local",
            "u": f"prd205-{tag}",
            "c": f"user_prd205_{tag}",
        },
    ).scalar()
    s.commit()
    s.close()

    yield {"ws_id": ws_id, "user_id": user_id, "tag": tag}

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


# ---------------------------------------------------------------------------
# messages.source round-trip + /messages surfacing
# ---------------------------------------------------------------------------


def test_source_roundtrip_and_null_for_in_turn_writes(new_session, seeded):
    from consumers.chatbot.service import ChatService

    s = new_session()
    try:
        svc = ChatService(s)
        chat = svc.create_chat(
            user_id=seeded["user_id"],
            title=f"s3 roundtrip {seeded['tag']}",
            workspace_id=uuid.UUID(seeded["ws_id"]),
        )
        # chats.kind defaults to 'user' on the untouched create path.
        kind = s.execute(
            text("SELECT kind FROM chats WHERE id = CAST(:c AS uuid)"),
            {"c": str(chat.id)},
        ).scalar()
        assert kind == "user"

        plain = svc.save_message(
            chat_id=str(chat.id),
            role="assistant",
            parts=[{"type": "text", "text": "in-turn write"}],
            workspace_id=seeded["ws_id"],
        )
        assert plain.source is None

        stamped_source = {
            "origin": "watcher",
            "label": "Auto \u00b7 background",
            "link_type": "watch",
            "link_id": str(uuid.uuid4()),
        }
        stamped = svc.save_message(
            chat_id=str(chat.id),
            role="assistant",
            parts=[{"type": "text", "text": "background write"}],
            workspace_id=seeded["ws_id"],
            source=stamped_source,
        )
        s.expire_all()
        got = svc.get_message(str(chat.id), str(stamped.id))
        assert got.source == stamped_source
    finally:
        s.close()


def test_get_chat_messages_endpoint_returns_source_field(new_session, seeded):
    from api.chat import get_chat_messages
    from consumers.chatbot.service import ChatService

    s = new_session()
    try:
        svc = ChatService(s)
        chat = svc.create_chat(
            user_id=seeded["user_id"],
            title=f"s3 endpoint {seeded['tag']}",
            workspace_id=uuid.UUID(seeded["ws_id"]),
        )
        svc.save_message(
            chat_id=str(chat.id),
            role="user",
            parts=[{"type": "text", "text": "hello"}],
            workspace_id=seeded["ws_id"],
        )
        svc.save_message(
            chat_id=str(chat.id),
            role="assistant",
            parts=[{"type": "text", "text": "verdict prose"}],
            workspace_id=seeded["ws_id"],
            source={"origin": "scheduled_task", "label": "Auto \u00b7 background"},
        )

        ctx = SimpleNamespace(
            workspace_id=uuid.UUID(seeded["ws_id"]),
            user=SimpleNamespace(id=int(seeded["user_id"])),
        )
        rows = asyncio.run(get_chat_messages(str(chat.id), ctx=ctx, db=s))

        assert len(rows) == 2
        assert all("source" in r for r in rows)
        assert rows[0]["source"] is None  # legacy/in-turn rows read back null
        assert rows[1]["source"]["origin"] == "scheduled_task"
        assert rows[1]["source"]["label"] == "Auto \u00b7 background"
    finally:
        s.close()
