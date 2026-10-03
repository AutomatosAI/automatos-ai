"""PRD-252 R2 — a review is Reject (with the owner's words) or Approve (with a
note kept on the ticket); R1 — the activity feed links a ticket the way the board
opens one.

Night 1 (F038): ``approve`` threw its body away, so a note written while approving
was lost. The board's Reject sent no note at all, so a redo was told only "The
owner sent it back without a note." Now the reject note leads the redo's brief,
word for word, on both claim paths, and the approve note is kept in the ticket's
notes, only when the approval stands.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

BRIEF = "Write the welcome email for a new wholesale café."
WHATS_WRONG = "Name the café: The Salt Loft. Keep it under 120 words."
APPROVE_NOTE = "Good. Send it on Monday, not Friday."


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the review-note tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def ticket(engine, new_session, monkeypatch):
    import api.board_tasks as bt

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd252')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Words', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"w": ws}).scalar()
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, raw_prompt, status, result, assigned_agent_id, review_mode) "
        "VALUES (CAST(:w AS uuid), 'Welcome email', :p, 'review', 'Draft 1: Hello!', :a, 'human') RETURNING id"),
        {"w": ws, "a": agent, "p": BRIEF}).scalar()
    s.commit()

    async def _no_notice(*a, **k):
        return None

    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _no_notice)
    ctx = NS(workspace_id=UUID(ws), user=NS(id=7, clerk_user_id=None, email="owner@cafe.test"))
    yield NS(id=task, ws=ws, new=new_session, ctx=ctx)
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _call(endpoint, ticket, body):
    try:
        return asyncio.run(endpoint(ticket.id, _Req(body), ctx=ticket.ctx, db=ticket.new()))
    except HTTPException as refused:
        return refused


def _row(ticket, column):
    value = ticket.new().execute(text(f"SELECT {column} FROM board_tasks WHERE id = :i"), {"i": ticket.id}).scalar()
    return json.loads(value) if isinstance(value, str) and column != "status" else value


def _notes(ticket):
    return [(n.get("note"), n.get("by")) for n in (_row(ticket, "runtime_ref") or {}).get("session_notes") or []]


def _session_prompt(ticket):
    """The CLI host's claim prompt, read without consuming anything."""
    from core.models.core import BoardTask
    from services.cli_host_service import _ticket_prompt

    return _ticket_prompt(ticket.new().get(BoardTask, ticket.id))


def _dispatch_prompt(ticket):
    """The API dispatcher's claim prompt (this consumes the waiting feedback)."""
    from config import config
    from services.board_dispatcher import _claim_and_sweep

    claimed = _claim_and_sweep(ticket.new, config, "w-prd252")["claimed"]
    return next(c["prompt"] for c in claimed if c["task_id"] == ticket.id)


def test_the_reject_note_leads_the_redo_word_for_word(ticket):
    import api.board_tasks as bt

    assert _call(bt.reject_task, ticket, {"feedback": WHATS_WRONG})["status"] == "assigned"

    for prompt in (_session_prompt(ticket), _dispatch_prompt(ticket)):
        assert prompt.startswith(BRIEF)
        redo = prompt[prompt.index("## Redo: your last attempt was sent back"):]
        # The owner's words open the redo, before the draft they correct (they were
        # the last line of a list under it).
        assert redo.split("\n")[1:3] == ["The owner sent it back with these words:", WHATS_WRONG]
        assert redo.index(WHATS_WRONG) < redo.index("Your last attempt:\nDraft 1: Hello!")


def test_a_send_back_without_a_note_opens_with_no_owner_words(ticket):
    import api.board_tasks as bt

    assert _call(bt.reject_task, ticket, {})["status"] == "assigned"
    redo = _session_prompt(ticket)
    assert "The owner sent it back with these words:" not in redo
    assert "The owner sent it back without a note." in redo


def test_the_approve_note_is_kept_on_the_ticket(ticket):
    import api.board_tasks as bt

    assert _call(bt.approve_task, ticket, {"note": f"  {APPROVE_NOTE}  "})["status"] == "done"
    assert _notes(ticket) == [(f"Approved: {APPROVE_NOTE}", "you")]   # night 1: the note was thrown away


def test_an_approval_without_a_note_adds_none(ticket):
    import api.board_tasks as bt

    assert _call(bt.approve_task, ticket, {})["status"] == "done"
    assert _notes(ticket) == []


def test_an_approval_whose_action_fails_keeps_no_note(ticket, monkeypatch):
    import api.board_tasks as bt

    async def _fails(db, ctx, action):
        raise HTTPException(status_code=500, detail="Approval action failed: planner unavailable")

    s = ticket.new()
    s.execute(text("UPDATE board_tasks SET planning_data = CAST(:pd AS jsonb) WHERE id = :i"),
              {"pd": json.dumps({"approval_action": {"type": "create_blog", "topic": "Oat milk"}}), "i": ticket.id})
    s.commit()
    monkeypatch.setattr(bt, "_run_approval_action", _fails)

    refused = _call(bt.approve_task, ticket, {"note": APPROVE_NOTE})

    assert isinstance(refused, HTTPException) and refused.status_code == 500
    assert _row(ticket, "status") == "review"     # back to review, to be approved again
    assert _notes(ticket) == []                   # no "Approved:" on a ticket that was not


def test_a_note_that_is_not_text_is_refused(ticket):
    import api.board_tasks as bt

    refused = _call(bt.approve_task, ticket, {"note": ["not", "text"]})

    assert isinstance(refused, HTTPException) and refused.status_code == 422
    assert _row(ticket, "status") == "review"


def test_a_long_approve_note_keeps_its_prefix_within_the_notes_bound(monkeypatch):
    import services.cli_host_service as host
    from services.ticket_verdict import APPROVAL_NOTE_PREFIX, _keep_note

    kept = {}
    monkeypatch.setattr(host, "append_session_note", lambda db, **kw: kept.update(kw))

    _keep_note(None, task_id=1, workspace_id="ws", note="x" * 5000)

    assert kept["note"].startswith(APPROVAL_NOTE_PREFIX) and kept["by"] == "you"
    assert len(kept["note"]) == host.MAX_ASK_QUESTION_KEPT   # nothing left for append_session_note to cut


def test_the_feed_links_a_ticket_the_way_the_board_opens_one():
    """F218: the feed built ?task=<id>; the board opens a ticket for ?task_id=<id>."""
    from services.activity_service import ActivityService

    svc = ActivityService.__new__(ActivityService)
    svc._build_feed_item = lambda **kw: dict(kw)
    ticket = NS(id=1169, status="review", assigned_agent_id=None, error_message=None, result=None,
                description="d", title="Welcome email", started_at=None, created_at=None, completed_at=None,
                source_type="user", orchestration_run_id=None)

    item = svc._board_feed_item(ticket, {})

    assert item["source_url"] == "/command-center?tab=board&task_id=1169"
    assert item["source_id"] == "1169" and item["status"] == "pending"
