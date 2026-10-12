"""P256-FIX-N13-7 (F414, night 13): one card number everywhere, and no instruction the owner did not give.

Night 13 (Auto on Haiku 5.5): Auto wrote and answered with a ticket's board number (#1394,
its workspace_seq) while a surface showed its row id (2680), so "did that go through?" got
yes, then no, on a number the owner could not find (asks 409/412). A brief recorded "as you
asked" for an instruction the owner never gave, and platform_update_mission_plan changed
#0936's steps with no card (ask 404).

- The ticket tools read "#1394" and bare 1394 as the card numbered 1394, even when another
  card's row id is 1394; what the model reads names it by number alone; the approval card's
  digest of a call names its ticket by number, never by row id (``card_ticket_numbers``).
- A brief, card note, memory or mission goal Auto writes from a person's chat keeps an
  attribution only when one of the owner's own messages in this chat said it
  (``said_instructions``); otherwise it is taken off, before the owner-only gate.
- A plan edit from the chat that changes a mission's steps or checks waits for the owner's
  click (``owner_only.PLAN_EDITS``), its card naming each change (``card_question_plans``).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery import owner_only, said_instructions
from modules.tools.discovery.said_instructions import without_unsaid_attributions

PLAN = "platform_update_mission_plan"
WS = uuid4()
MOVED_ASIDE = 500_000


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


# ── One card number: the board's, wherever the card is named ─────────────────────────

@pytest.fixture
def board(db_session, seed_workspace):
    """Two cards: 'Price list' numbered with the other card's row id, as #1394 and 2680 ran close."""
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    other = BoardTask(workspace_id=ws, title="Old supplier note", status="done", description="Old.")
    price = BoardTask(workspace_id=ws, title="Price list", status="review", description="The price list.")
    db_session.add_all([other, price])
    db_session.flush()
    renumber = text("UPDATE board_tasks SET workspace_seq = :seq WHERE id = :id")
    db_session.execute(renumber, {"seq": other.id + MOVED_ASIDE, "id": other.id})   # its own number out of the way
    db_session.execute(renumber, {"seq": other.id, "id": price.id})
    db_session.expire_all()
    return NS(db=db_session, ws=ws, price=price, other=other, number=f"#{other.id:04d}")


def test_did_1394_go_through_resolves_the_card_the_board_numbers_1394(board):
    from services.ticket_refs import ticket_id_named

    for said in (board.number, str(board.other.id), board.other.id):   # "#1394", "1394", 1394
        assert ticket_id_named(board.db, board.ws, said)[0] == board.price.id, said


def test_the_model_reads_the_card_by_its_number_alone(board):
    from modules.tools.discovery.handlers_board_tasks import get_board_task
    from modules.tools.formatting.tickets_by_number import by_number_only

    answer = asyncio.run(get_board_task(board.db, board.ws, {"task_id": board.number}))
    seen = by_number_only(answer)

    assert answer["success"] is True and answer["task"]["number"] == board.number
    assert answer["task"]["id"] == board.price.id                       # the platform keeps the id
    assert "id" not in seen["task"] and seen["task"]["number"] == board.number
    assert "id" not in seen["frontend_data"]["task_card"]


def test_the_approval_card_names_the_ticket_by_number_never_its_row_id(board):
    from modules.tools.discovery.card_ticket_numbers import numbers_on_the_card
    from modules.tools.formatting.result_formatter import ToolResultFormatter
    from services.ticket_numbers import TicketId

    move = "platform_update_task_status"
    shown = numbers_on_the_card(board.db, board.ws, move, {"task_id": TicketId(board.price.id), "status": "done"})
    ask = {"success": False, "requires_confirmation": True, "grant_id": 409, "action": "platform_update_task_status",
           "params": {"task_id": board.price.id, "status": "done"}, "card_numbers": shown}
    card = ToolResultFormatter.format_for_frontend(ask, "platform_update_task_status")["tool_approval"]

    assert shown == {"task_id": board.number}
    assert card["params"] == {"task_id": board.number, "status": "done"}       # was: task_id 2680
    # A scheduled task's task_id is no ticket: it is shown as given.
    assert numbers_on_the_card(board.db, board.ws, "platform_cancel_scheduled_task", {"task_id": board.other.id}) == {}


def test_a_card_without_numbers_shows_the_call_as_given():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    ask = {"success": False, "requires_confirmation": True, "grant_id": 5, "action": "platform_update_agent",
           "params": {"agent_id": 12, "description": "Runs the desk"}}
    card = ToolResultFormatter.format_for_frontend(ask, "platform_update_agent")["tool_approval"]

    assert card["params"] == {"agent_id": 12, "description": "Runs the desk"}


# ── No instruction the owner did not give ──────────────────────────────────────────

def test_a_note_with_as_you_asked_and_no_owner_turn_that_said_it_is_rewritten():
    note = "Sent back for a redo, as you asked. The redo keeps the October prices."
    owner = ["What's on the board today?"]

    assert without_unsaid_attributions(note, owner) == "Sent back for a redo. The redo keeps the October prices."
    assert without_unsaid_attributions("Per your instruction, close it today.", []) == "Close it today."
    assert without_unsaid_attributions("Priced at £12 as you asked me to.", []) == "Priced at £12."


def test_the_attribution_stays_when_the_owners_message_in_this_chat_said_it():
    brief = "Draft a reply to Declan about the late order, as you asked."
    owner = ["Morning", "Can you get the team to draft a reply to Declan about his late order?"]

    assert without_unsaid_attributions(brief, owner) == brief
    assert without_unsaid_attributions("Nothing claimed here.", []) == "Nothing claimed here."


class _Executor:
    """PlatformActionExecutor._run_cleared's shape: the gates have cleared, the handler runs."""
    db = object()
    workspace_id = WS

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(None, WS, params)


def _run(action, params, caller_context):
    handler = AsyncMock(return_value={"success": True})
    reply = asyncio.run(_Executor()._run_cleared(action, params, caller_context, None, handler))
    return reply, handler


def test_autos_brief_and_memory_are_written_without_the_unsaid_instruction(monkeypatch):
    monkeypatch.setattr(said_instructions, "chat_owner_turns", lambda db, ws, params: ["Show me the board"])
    _, created = _run("platform_create_task", {"title": "Reorder", "description": "Reorder oat milk, as you asked."},
                      _owners_chat())
    _, stored = _run("platform_store_memory", {"content": "Per your instruction, Friday is delivery day."},
                     _owners_chat())

    assert created.call_args.args[2]["description"] == "Reorder oat milk."
    assert stored.call_args.args[2]["content"] == "Friday is delivery day."


def test_an_agents_run_is_not_read_against_a_chat(monkeypatch):
    monkeypatch.setattr(said_instructions, "chat_owner_turns", lambda *a: pytest.fail("no chat to read"))
    params = {"title": "Reorder", "description": "Reorder oat milk, as you asked."}

    _, handler = _run("platform_create_task", params, {"board_task_id": 2318})

    assert handler.call_args.args[2] == params


# ── A plan edit from the chat waits for the owner's click ────────────────────────────

@pytest.fixture
def asks(monkeypatch):
    asked = []
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: None)
    monkeypatch.setattr(owner_only, "platform_ask", lambda db, ws, action, params, ctx: asked.append(action) or {
        "success": False, "requires_confirmation": True, "owner_only": True, "action": action})
    return asked


def test_update_mission_plan_changing_steps_raises_one_grant_and_runs_nothing(asks):
    params = {"mission_id": "#0936", "task_edits": [{"task_id": "#0936.2", "agent_id": 12}]}

    reply, handler = _run(PLAN, params, _owners_chat())

    assert reply["requires_confirmation"] is True and asks == [PLAN]
    handler.assert_not_called()


def test_switching_each_steps_check_waits_too_and_a_plan_edit_that_changes_nothing_does_not(asks):
    assert owner_only.is_owner_only(PLAN, {"mission_id": "#0936", "check_each_step": True})
    assert owner_only.is_owner_only(PLAN, '{"mission_id": "#0936", "task_edits": [{"title": "Draft"}]}')
    assert not owner_only.is_owner_only(PLAN, {"mission_id": "#0936"})
    assert not owner_only.is_owner_only(PLAN, {"mission_id": "#0936", "task_edits": []})
    assert PLAN not in owner_only.OWNER_ONLY_ACTIONS and owner_only.VERBS[PLAN]   # params-conditional, as cli creates


def test_an_agents_plan_edit_runs_without_a_card(asks):
    reply, handler = _run(PLAN, {"mission_id": "#0936", "task_edits": [{"title": "Draft"}]}, {"board_task_id": 936})

    assert reply == {"success": True} and asks == []
    handler.assert_called_once()


def test_the_plan_cards_question_names_each_change(monkeypatch):
    from modules.tools.discovery import card_question_rows
    from modules.tools.discovery.card_question import platform_question

    monkeypatch.setattr(card_question_rows, "mission_lines", lambda db, ws, params: ["- mission: 'Spring menu' (#0936)"])
    params = {"mission_id": "#0936", "check_each_step": True,
              "task_edits": [{"task_id": "#0936.2", "title": "Price the menu"}, {"sequence_number": 3, "agent_role": "Writer"}]}

    asked = platform_question(None, WS, PLAN, params, "change a mission's plan on #0936")

    assert asked.splitlines() == [
        "Change a mission's plan on #0936:",
        "- mission: 'Spring menu' (#0936)",
        "- step #0936.2 title: Price the menu",
        "- step 3 agent: Writer",
        "- every step waits in Review for your check before the next starts",
    ]
