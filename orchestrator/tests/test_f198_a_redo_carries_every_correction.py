"""F198 — a redo carries every correction on the ticket, and the draft it corrects.

Night 6 (#1120, rounds 1-4): "each Reject only carries my latest note: round 1
had the voice and no café, round 2 the café and no voice, round 3 the voice and
no café again"; a later redo wiped a correct total (#1152). A Reject overwrote
the ticket's one feedback field and wiped the draft, so each redo started over
with only the newest note. Every note is now kept on the ticket, the draft sent
back is kept (from review too), and both claim paths hand the redo both.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

VOICE = "Use the brand voice guide."
CAFE = "Name the café: The Salt Loft."


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the redo tests need a reachable Postgres: {exc}")
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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f198')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Words', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"w": ws}).scalar()
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, raw_prompt, status, assigned_agent_id, review_mode) "
        "VALUES (CAST(:w AS uuid), 'Welcome email', 'Write the welcome email for a new wholesale café.', "
        "'review', :a, 'human') RETURNING id"), {"w": ws, "a": agent}).scalar()
    s.commit()
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    yield NS(id=task, ws=ws, new=new_session, ctx=NS(workspace_id=UUID(ws), user=NS(id=7, email="owner@cafe.test")))
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _sent_back(ticket, draft, note=None):
    """The redo finished with ``draft``; the owner sends it back with ``note``."""
    import api.board_tasks as bt

    s = ticket.new()
    s.execute(text("UPDATE board_tasks SET status = 'review', result = :r WHERE id = :i"), {"r": draft, "i": ticket.id})
    s.commit()
    body = {"feedback": note} if note else {}
    assert asyncio.run(bt.reject_task(ticket.id, _Req(body), ctx=ticket.ctx, db=ticket.new()))["status"] == "assigned"


def _session_prompt(ticket):
    """The CLI host's claim prompt, read without consuming anything."""
    from core.models.core import BoardTask
    from services.cli_host_service import _ticket_prompt

    return _ticket_prompt(ticket.new().get(BoardTask, ticket.id))


def _dispatch_prompt(ticket):
    """The API dispatcher's claim prompt (this consumes the waiting feedback)."""
    from config import config
    from services.board_dispatcher import _claim_and_sweep

    claimed = _claim_and_sweep(ticket.new, config, "w-f198")["claimed"]
    return next(c["prompt"] for c in claimed if c["task_id"] == ticket.id)


def test_every_correction_and_the_last_draft_reach_the_redo(ticket):
    _sent_back(ticket, "Draft 1: Hello! Welcome aboard.", VOICE)
    _dispatch_prompt(ticket)                                    # round 1 is redone
    _sent_back(ticket, "Draft 2: Welcome, dear café, to our roastery family.", CAFE)

    for prompt in (_session_prompt(ticket), _dispatch_prompt(ticket)):
        assert prompt.startswith("Write the welcome email for a new wholesale café.")
        assert f"1. {VOICE}\n2. {CAFE}" in prompt              # night: only the café note
        assert "Your last attempt:\nDraft 2: Welcome, dear café, to our roastery family." in prompt
        assert "keep everything else as it was" in prompt


def test_a_send_back_without_a_note_still_carries_the_corrections(ticket):
    _sent_back(ticket, "Draft 1", VOICE)
    _dispatch_prompt(ticket)
    _sent_back(ticket, "Draft 2 with the voice")

    prompt = _session_prompt(ticket)

    assert f"1. {VOICE}" in prompt and "This time it came back without a new note." in prompt   # night: no redo at all
    assert "Your last attempt:\nDraft 2 with the voice" in prompt


def test_every_send_back_keeps_the_draft_and_the_note_on_the_ticket(ticket):
    _sent_back(ticket, "Draft 1", VOICE)

    data = ticket.new().execute(text("SELECT planning_data FROM board_tasks WHERE id = :i"), {"i": ticket.id}).scalar()
    data = (data if isinstance(data, dict) else json.loads(data)) if data else {}
    runs = [(r["status"], r["result"], r["why"]) for r in data.get("previous_runs") or []]
    assert runs == [("review", "Draft 1", "sent back")]                        # night: nothing kept from review
    assert [c["note"] for c in data.get("owner_corrections") or []] == [VOICE]
