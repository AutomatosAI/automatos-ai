"""PRD-266 US-007b — the agent-info frame the web reads, and a chat list that pages.

1. The agent-info frame. ``_stream_response_with_agent_scoped`` builds
   ``{"type": "agent-info", "agent": {…}}`` and hands the mapping itself to
   ``StreamingHandler.format_aisdk_data``, which put it under ``type``:
   ``d:{"type": {"type": "agent-info", …}}``. ``frontend/lib/chat/hooks.ts`` reads
   ``data.type === 'agent-info' && data.agent``, so the agent never reached the web.
   The fix is in the handler (the 365-line emitter is not touched): a mapping that
   carries its own ``type`` is the frame. Every other frame is byte-identical.
   ``AGENT_INFO_LINE`` is the same line
   ``frontend/lib/chat/__tests__/prd266-agent-info-frame.test.ts`` feeds the hook.

2. ``GET /api/chat/history?starting_after=<chat id>`` returns the caller's chats after
   that one, in the list's own order (most recent activity first, ties by id), with
   ``limit`` honoured. A cursor that is not the caller's chat in this workspace is 404;
   no cursor is the list as it was. These run against Postgres (skipped without it).
"""
from __future__ import annotations

import inspect
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from consumers.chatbot.streaming import StreamingHandler
from core.database.database import get_database_url

AGENT = {"id": 7, "name": "Auto", "type": "auto", "skills": ["web_search"]}
AGENT_INFO_LINE = (
    'd:{"type": "agent-info", "agent": {"id": 7, "name": "Auto", "type": "auto", "skills": ["web_search"]}}'
)
ROW_KEYS = {
    "id", "userId", "title", "createdAt", "updatedAt", "visibility", "lastContext", "lastMessagePreview", "kind",
}
PAGE = 3
BASE_TIME = datetime(2026, 10, 21, 9, 0, 0)
# The caller's chats in the workspace, by minutes after BASE_TIME: a tie at 50 straddles pages 1 and 2.
CALLER_MINUTES = (70, 60, 50, 50, 40, 30, 20)


# ---------------------------------------------------------------------------
# 1. The agent-info frame
# ---------------------------------------------------------------------------

def test_the_agent_info_frame_is_the_shape_hooks_ts_reads():
    line = StreamingHandler().format_aisdk_data({"type": "agent-info", "agent": AGENT})

    assert line == AGENT_INFO_LINE + "\n"
    frame = json.loads(line[len("d:"):])
    assert frame["type"] == "agent-info" and frame["agent"] == AGENT


def test_the_emitter_still_hands_the_frame_itself_to_the_handler():
    from consumers.chatbot.service import StreamingChatService

    source = inspect.getsource(StreamingChatService._stream_response_with_agent_scoped)
    assert '"type": "agent-info"' in source
    assert "format_aisdk_data(_agent_info)" in source


def test_every_other_format_aisdk_data_frame_is_unchanged():
    handler = StreamingHandler()

    assert handler.format_aisdk_data("progress", {"text": "Reading the board"}) == (
        'd:{"type": "progress", "data": {"text": "Reading the board"}}\n'
    )
    assert handler.format_aisdk_data("mission-suggestion", {"type": "plan", "goal": "x"}) == (
        'd:{"type": "mission-suggestion", "data": {"type": "plan", "goal": "x"}}\n'
    )
    assert handler.format_aisdk_data("done") == 'd:{"type": "done"}\n'
    assert handler.format_aisdk_reasoning("hmm") == 'd:{"type": "reasoning", "data": {"delta": "hmm"}}\n'


# ---------------------------------------------------------------------------
# 2. The chat list pages (Postgres)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT id, updated_at, kind FROM chats LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the chat-history suite needs Postgres with the chats table: {exc}")
    yield eng
    eng.dispose()


def _insert_user(s, clerk_id: str) -> int:
    row = s.execute(
        text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": clerk_id, "e": f"{clerk_id}@test.local", "c": clerk_id},
    ).first()
    return int(row[0])


def _add_chat(s, user_id: int, workspace_id: uuid.UUID, title: str, minutes: int):
    from core.models import Chat

    chat = Chat(
        id=uuid.uuid4(), user_id=user_id, workspace_id=workspace_id, title=title, visibility="private",
        kind="user", created_at=BASE_TIME, updated_at=BASE_TIME + timedelta(minutes=minutes),
    )
    s.add(chat)
    return chat


def _seed_chats(s, ws, other_ws, caller: int, other: int, tag: str) -> SimpleNamespace:
    mine = [_add_chat(s, caller, ws, f"prd266-{tag}-{i}", m) for i, m in enumerate(CALLER_MINUTES)]
    others = _add_chat(s, other, ws, f"prd266-{tag}-other-user", 55)
    elsewhere = _add_chat(s, caller, other_ws, f"prd266-{tag}-other-ws", 65)
    s.commit()
    ordered = sorted(mine, key=lambda c: (c.updated_at, c.id.int), reverse=True)
    return SimpleNamespace(
        expected=[str(c.id) for c in ordered],
        cursors={
            "unknown": str(uuid.uuid4()),
            "not_an_id": "not-a-chat-id",
            "other_users": str(others.id),
            "other_workspace": str(elsewhere.id),
        },
    )


@pytest.fixture
def seeded(engine, new_session):
    ws, other_ws = uuid.uuid4(), uuid.uuid4()
    tag = uuid.uuid4().hex[:10]
    clerk_ids = (f"user_prd266_caller_{tag}", f"user_prd266_other_{tag}")
    s = new_session()
    for w in (ws, other_ws):
        s.execute(
            text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
            {"id": str(w), "n": "prd266-chat-history"},
        )
    caller, other = (_insert_user(s, c) for c in clerk_ids)
    s.commit()
    chats = _seed_chats(s, ws, other_ws, caller, other, tag)
    s.close()

    ctx = SimpleNamespace(workspace_id=ws, user=SimpleNamespace(id=caller))
    yield SimpleNamespace(ctx=ctx, db=new_session(), expected=chats.expected, cursors=chats.cursors)

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM chats WHERE workspace_id IN (CAST(:w AS uuid), CAST(:o AS uuid))",
        "DELETE FROM users WHERE clerk_user_id IN (:c1, :c2)",
        "DELETE FROM workspaces WHERE id IN (CAST(:w AS uuid), CAST(:o AS uuid))",
    ):
        s.execute(text(stmt), {"w": str(ws), "o": str(other_ws), "c1": clerk_ids[0], "c2": clerk_ids[1]})
    s.commit()
    s.close()


def _history(seeded, **params):
    from api.chat import get_chat_history

    return get_chat_history(ctx=seeded.ctx, db=seeded.db, **params)


def test_starting_after_pages_over_three_pages_in_the_lists_own_order(seeded):
    first = _history(seeded, limit=PAGE)
    second = _history(seeded, limit=PAGE, starting_after=first[-1]["id"])
    third = _history(seeded, limit=PAGE, starting_after=second[-1]["id"])

    assert [len(page) for page in (first, second, third)] == [PAGE, PAGE, 1]
    assert [row["id"] for row in first + second + third] == seeded.expected
    assert _history(seeded, limit=PAGE, starting_after=third[-1]["id"]) == []


@pytest.mark.parametrize("whose", ["unknown", "not_an_id", "other_users", "other_workspace"])
def test_a_cursor_that_is_not_the_callers_chat_here_is_404(seeded, whose):
    with pytest.raises(HTTPException) as refused:
        _history(seeded, limit=PAGE, starting_after=seeded.cursors[whose])
    assert refused.value.status_code == 404


def test_no_cursor_is_todays_answer(seeded):
    rows = _history(seeded, limit=50)

    assert [row["id"] for row in rows] == seeded.expected
    assert [row["updatedAt"] for row in rows] == sorted((row["updatedAt"] for row in rows), reverse=True)
    assert all(set(row) == ROW_KEYS for row in rows)
    assert [row["id"] for row in _history(seeded, limit=2)] == seeded.expected[:2]
