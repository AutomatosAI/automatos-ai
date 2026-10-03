"""PRD-252 R2 — Discuss: talk a ticket through, then "Update ticket and re-queue".

Reject sends work back with a note, and after the third note the owner and the
agent are still talking past each other. Discuss opens a chat with the ticket in
the page's context; the brief the owner agrees there goes back onto the ticket
as its brief, with the owner's correction, and the ticket returns to its agent.
Acceptance: the re-queued ticket's next run works from the agreed brief.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

AGREED = "Write to Priya Shah at Gull & Anchor: two short paragraphs, the Thursday delivery, no prices."


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Discuss tests need a reachable Postgres: {exc}")
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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd252-discuss')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Words', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"w": ws}).scalar()
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, raw_prompt, description, status, assigned_agent_id, review_mode, "
        "result) VALUES (CAST(:w AS uuid), 'Welcome email', 'Write the welcome email.', 'Write the welcome email.', "
        "'review', :a, 'human', 'Draft 1: Hello!') RETURNING id"), {"w": ws, "a": agent}).scalar()
    s.commit()
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    yield NS(id=task, ws=ws, new=new_session, ctx=NS(workspace_id=UUID(ws), user=NS(id=7, email="owner@cafe.test")))
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _rebrief(ticket, brief=AGREED):
    from api.board_task_rebrief import RebriefBody, rebrief_task

    return rebrief_task(ticket.id, RebriefBody(brief=brief), ctx=ticket.ctx, db=ticket.new())


def _row(ticket):
    from core.models.core import BoardTask

    return ticket.new().get(BoardTask, ticket.id)


def test_the_agreed_brief_goes_onto_the_ticket_and_back_to_its_agent(ticket):
    from services.ticket_redo import BRIEF_AGREED, REBRIEFED

    out = _rebrief(ticket)

    row = _row(ticket)
    assert out["status"] == row.status == "assigned"
    assert row.raw_prompt == row.description == AGREED                    # a claim works from raw_prompt
    assert row.planning_data["previous_briefs"][-1]["description"] == "Write the welcome email."
    assert row.planning_data["owner_corrections"][-1]["note"] == BRIEF_AGREED
    assert row.planning_data["previous_runs"][-1] | {"why": REBRIEFED} == row.planning_data["previous_runs"][-1]
    assert row.planning_data["previous_runs"][-1]["result"] == "Draft 1: Hello!"
    assert (row.result, row.review_feedback) == (None, BRIEF_AGREED)


def test_the_next_run_works_from_the_agreed_brief(ticket):
    """Acceptance: the re-queued ticket's prompt is the brief, with the owner's word on it."""
    from services.cli_host_service import _ticket_prompt
    from services.ticket_redo import AGREED_BRIEF_BLOCK

    _rebrief(ticket)

    prompt = _ticket_prompt(_row(ticket))
    assert prompt.startswith(AGREED) and prompt.endswith(AGREED_BRIEF_BLOCK)


def test_a_brief_agreed_after_send_backs_is_not_mixed_with_the_old_draft(ticket):
    """Review of #861: the redo after a re-brief still carried the draft sent back
    before the discussion, and "keep everything else as it was"."""
    from services.cli_host_service import _ticket_prompt
    from services.ticket_redo import AGREED_BRIEF_BLOCK

    _sent_back(ticket, "Draft 1: Hello!", "Too long.")
    _rebrief(ticket)
    prompt = _ticket_prompt(_row(ticket))
    assert prompt.startswith(AGREED) and AGREED_BRIEF_BLOCK in prompt
    assert "Draft 1" not in prompt and "Too long." not in prompt and "last attempt" not in prompt

    _sent_back(ticket, "Draft 2 from the agreed brief", "Sign it Sam.")      # a send-back after the brief
    prompt = _ticket_prompt(_row(ticket))
    assert "Draft 2 from the agreed brief" in prompt and "Sign it Sam." in prompt
    assert "Draft 1" not in prompt and "Too long." not in prompt


@pytest.mark.parametrize("source", ["orchestration", "orchestration_task", "mission"])
def test_a_missions_tickets_are_not_rebriefed_on_the_board(ticket, source):
    s = ticket.new()
    s.execute(text("UPDATE board_tasks SET source_type = :s WHERE id = :i"), {"s": source, "i": ticket.id})
    s.commit()

    with pytest.raises(HTTPException) as refused:
        _rebrief(ticket)

    assert refused.value.status_code == 409 and _row(ticket).description == "Write the welcome email."


def test_a_blank_brief_is_refused_at_the_door():
    from pydantic import ValidationError

    from api.board_task_rebrief import RebriefBody

    assert RebriefBody(brief="  Two paragraphs.  ").brief == "Two paragraphs."
    with pytest.raises(ValidationError):
        RebriefBody(brief="   ")


def test_a_running_ticket_is_not_rebriefed_under_its_run(ticket):
    s = ticket.new()
    s.execute(text("UPDATE board_tasks SET status = 'in_progress' WHERE id = :i"), {"i": ticket.id})
    s.commit()

    with pytest.raises(HTTPException) as refused:
        _rebrief(ticket)

    assert refused.value.status_code == 409 and "running" in refused.value.detail


def _sent_back(ticket, draft, note=None):
    import api.board_tasks as bt

    s = ticket.new()
    s.execute(text("UPDATE board_tasks SET status = 'review', result = :r WHERE id = :i"), {"r": draft, "i": ticket.id})
    s.commit()
    note = note or f"Not yet: {draft}"
    asyncio.run(bt.reject_task(ticket.id, _Req({"feedback": note}), ctx=ticket.ctx, db=ticket.new()))


def test_the_board_counts_the_send_backs_since_the_brief_was_agreed(ticket):
    from services.board_task_view import board_dict

    for draft in ("Draft 1", "Draft 2", "Draft 3"):
        _sent_back(ticket, draft)
    assert board_dict(_row(ticket))["times_sent_back"] == 3                # the panel now suggests Discuss

    _rebrief(ticket)
    assert board_dict(_row(ticket))["times_sent_back"] == 0                # a new brief starts the count again
    _sent_back(ticket, "Draft 4")
    assert board_dict(_row(ticket))["times_sent_back"] == 1


def test_a_chat_opened_from_a_ticket_or_a_mission_tells_auto_to_read_it_first():
    from services.page_context import render_page_preamble, sanitize_page_context

    ticket = render_page_preamble(sanitize_page_context(
        {"page": "chat", "route": "/chat", "selected": {"type": "board_task", "id": "612"}}))
    mission = render_page_preamble(sanitize_page_context(
        {"page": "chat", "route": "/chat", "selected": {"type": "mission", "id": "b6af0355"}}))

    assert "platform_get_task (task_id 612)" in ticket and "Update ticket and re-queue" in ticket
    assert "one fenced block (```)" in ticket                              # where the dialog finds the brief
    assert "platform_get_mission (mission_id b6af0355)" in mission            # D4: a mission is Auto's to discuss
