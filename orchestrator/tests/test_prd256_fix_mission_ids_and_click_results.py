"""PRD-256 FX-009 (night 12, B3, M5, E6, F388): mission tools take the number the owner
uses, a click's result reaches the next turn, and approve/cancel mission are first-class.

- Card 1942 asked the owner about mission_id '220', a task card's number, and card 1965
  about '1065'. The owner clicked; the resumed call failed ("#0220 … is a task card, not
  a mission"). The card is now raised only on the mission the call will act on.
- After a click, the grant kept only success/error and the history kept text parts only,
  so the mission the click created never reached the next turn: the model guessed again.
- platform_approve_mission and platform_cancel_mission were not pinned first-class.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
APPROVE, CANCEL, GET = "platform_approve_mission", "platform_cancel_mission", "platform_get_mission"
GOAL = "Plan the spring menu"
SPRING_PARTY = "Plan the spring menu launch party"


def _mission(db, ws, goal):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    run = OrchestrationRun(workspace_id=ws, goal=goal, state="awaiting_approval", created_by="user_test", config={})
    db.add(run)
    db.flush()
    card = create_mission_board_task(db, run)
    step = OrchestrationTask(run_id=run.id, title="Cost the dishes", description="Do it.", sequence_number=1,
                             state="pending", state_type="initial")
    db.add(step)
    db.flush()
    create_task_board_task(db, run, step)
    return run, card


@pytest.fixture
def board(db_session, seed_workspace):
    """A mission awaiting approval (its card and a step's card) and a plain task card."""
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    run, card = _mission(db_session, ws, GOAL)
    task = BoardTask(workspace_id=ws, title="Order the oat milk", status="review", description="Order it.")
    db_session.add(task)
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, card=card, task=task, seq=card.workspace_seq,
              number=f"#{card.workspace_seq:04d}", task_seq=task.workspace_seq)


def _owners_chat():
    return {"conversation_id": str(uuid4()), "turn_id": "t-1", "driving_user_id": "1"}


def _grants(board):
    from core.models.approval_grants import ApprovalGrant

    return board.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == board.ws).count()


# ── The mission the owner names: its number, its uuid, its title ────────────────

def test_the_missions_ticket_number_resolves_to_the_mission(board):
    from modules.tools.execution.subject_targets import resolve_targets

    for said in (board.number, f"{board.seq:04d}", str(board.seq), board.seq):   # '#0992', '0992', '992'
        found, missing = resolve_targets(board.db, board.ws, {"mission_id": said}, APPROVE)
        assert missing == [], said
        assert [t.ident for t in found] == [str(board.run.id)], said
        assert found[0].called == f"mission {board.number}" and found[0].name == board.card.title


def test_the_missions_title_resolves_to_the_mission(board):
    from modules.tools.discovery.handlers_missions import get_mission

    for said in (GOAL, GOAL.upper(), board.card.title, "spring menu"):
        out = asyncio.run(get_mission(board.db, board.ws, {"mission_id": said}))
        assert out["success"] is True and out["mission"]["id"] == board.run.id, said


def test_a_title_two_missions_share_is_refused_naming_both(board):
    from modules.tools.discovery.handlers_missions import cancel_mission

    party, party_card = _mission(board.db, board.ws, SPRING_PARTY)
    out = asyncio.run(cancel_mission(board.db, board.ws, {"mission_id": "spring menu"}))

    assert out["success"] is False and "More than one mission" in out["error"]
    assert board.number in out["error"] and f"#{party_card.workspace_seq:04d}" in out["error"]
    board.db.refresh(party)
    assert party.state == "awaiting_approval"                       # nothing was done
    exact = asyncio.run(_title_of(board, GOAL))
    assert exact == board.run.id                                     # a whole title is still one mission


async def _title_of(board, said):
    from modules.tools.discovery.handlers_missions import get_mission

    return (await get_mission(board.db, board.ws, {"mission_id": said}))["mission"]["id"]


def test_a_title_no_mission_has_names_none(board):
    from modules.tools.execution.subject_targets import missing_targets_error, resolve_targets

    found, missing = resolve_targets(board.db, board.ws, {"mission_id": "the autumn menu"}, CANCEL)
    assert found == [] and "No mission titled 'the autumn menu'" in missing_targets_error(CANCEL, missing)["error"]


def test_another_workspaces_mission_is_never_found(board, seed_workspace):
    from modules.tools.execution.subject_targets import missing_targets_error, resolve_targets

    theirs = UUID(seed_workspace())
    run, _card = _mission(board.db, theirs, "Their winter menu")
    by_id = resolve_targets(board.db, board.ws, {"mission_id": str(run.id)}, APPROVE)
    by_title = resolve_targets(board.db, board.ws, {"mission_id": "Their winter menu"}, APPROVE)

    assert by_id[0] == [] and missing_targets_error(APPROVE, by_id[1])["error"].startswith(f"No mission {run.id}")
    assert by_title[0] == [] and "No mission titled" in missing_targets_error(APPROVE, by_title[1])["error"]


# ── The card is raised only on a resolved mission ───────────────────────────────

@pytest.mark.parametrize("action", [APPROVE, CANCEL])
def test_a_task_cards_number_is_refused_without_a_card(board, action):
    """Night 12, card 1942: mission_id '220' was a task card's number."""
    from modules.tools.discovery.owner_only import platform_ask

    reply = platform_ask(board.db, board.ws, action, {"mission_id": str(board.task_seq)}, _owners_chat())

    assert reply["success"] is False and "is a task card, not a mission" in reply["error"]
    assert "requires_confirmation" not in reply and "question_md" not in reply
    assert _grants(board) == 0                                      # no card for the owner to click


def test_the_ask_on_a_missions_number_names_the_mission(board):
    from modules.tools.discovery.owner_only import platform_ask

    reply = platform_ask(board.db, board.ws, APPROVE, {"mission_id": board.number}, _owners_chat())

    assert reply["requires_confirmation"] is True
    assert f"'{board.card.title}' (mission {board.number})" in reply["act"]
    assert _grants(board) == 1


def test_the_click_runs_on_the_mission_the_card_showed(board):
    """A title read again at the click could name another mission: the grant holds the mission's own id."""
    from core.models.approval_grants import ApprovalGrant
    from modules.tools.discovery.owner_only import platform_ask

    reply = platform_ask(board.db, board.ws, CANCEL, {"mission_id": "spring menu"}, _owners_chat())
    _mission(board.db, board.ws, "Spring menu")                     # now the words name this one in full

    grant = board.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == board.ws).one()
    assert reply["params"]["mission_id"] == str(board.run.id)
    assert grant.details["params"]["mission_id"] == str(board.run.id)
    assert f"(mission {board.number})" in reply["act"]


def test_a_step_cancelled_alone_is_refused_before_any_card(board):
    from modules.tools.discovery.owner_only import platform_ask

    reply = platform_ask(board.db, board.ws, CANCEL, {"mission_id": f"{board.number}.1"}, _owners_chat())

    assert reply["success"] is False and "A step stops with its mission" in reply["error"]
    assert _grants(board) == 0


# ── The click's result reaches the next turn ─────────────────────────────────────

@pytest.fixture
def chat(board):
    from core.models.core import Chat

    person = board.db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                              {"e": f"owner-{uuid4().hex[:8]}@cafe.test", "u": f"owner-{uuid4().hex[:8]}"}).scalar()
    row = Chat(id=uuid4(), user_id=person, workspace_id=board.ws, title="Spring menu", visibility="private")
    board.db.add(row)
    board.db.flush()
    return row


def _granted(board, chat, action, params, asked):
    from core.services.approval_grants import grant_grant
    from modules.tools.execution.tool_grants import issue_tool_grant

    context = {"conversation_id": str(chat.id), "turn_id": "t-1", "driving_user_id": str(chat.user_id)}
    grant = issue_tool_grant(board.db, board.ws, action=action, params=params, caller_context=context,
                             question_md=asked)
    grant_grant(grant, granted_by=f"user:{chat.user_id}")
    return grant


def _clicked(board, grant, result):
    from api.approval_grants import _resume_tool_call
    from services.click_results import said_in_the_chat

    executor = AsyncMock()
    executor.execute_tool = AsyncMock(return_value=result)
    with patch("modules.tools.execution.unified_executor.UnifiedToolExecutor", return_value=executor):
        asyncio.run(_resume_tool_call(board.db, grant))
    said_in_the_chat(board.db, grant)


def _next_turn_reads(board, chat):
    """The conversation the next turn reads, as the model is given it (api/chat._turn_history)."""
    from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
    from consumers.chatbot.service import ChatService

    history = [{"role": m.role, "parts": m.parts} for m in ChatService(board.db).get_messages_by_chat_id(str(chat.id))]
    return "\n".join(get_prompt_analyzer()._extract_message_content(message) for message in history)


def test_after_a_granted_create_mission_the_next_turn_reads_its_uuid(board, chat):
    made = uuid4()
    grant = _granted(board, chat, "platform_create_mission", {"goal": GOAL}, f"Start a mission:\n- goal: {GOAL}")

    _clicked(board, grant, {"success": True, "mission_id": made, "number": "#0992", "goal": GOAL,
                            "state": "awaiting_approval"})

    ids = grant.details["executed_result"]["ids"]
    assert ids["mission_id"] == str(made) and ids["number"] == "#0992" and ids["goal"] == GOAL
    said = _next_turn_reads(board, chat)
    assert f"Your click ran: start a mission → mission_id {made}, number #0992" in said


def test_a_click_that_did_not_go_through_says_why(board, chat):
    grant = _granted(board, chat, APPROVE, {"mission_id": board.number}, "Approve a mission's plan.")

    _clicked(board, grant, {"success": False, "error": "Mission is not awaiting approval"})

    assert "ids" not in grant.details["executed_result"]
    said = _next_turn_reads(board, chat)
    assert "Your click ran: approve a mission's plan → it did not go through: Mission is not awaiting approval" \
        in said


def test_a_ticket_result_names_the_ticket_and_its_row():
    from services.click_results import result_ids

    ids = result_ids({"success": True, "task": {"id": 2150, "number": "#0892", "title": "Design the menu card"}})
    assert ids == {"task_id": "2150", "number": "#0892", "title": "Design the menu card"}


def test_an_ask_with_no_chat_tells_no_chat(monkeypatch):
    from services import chat_messenger, click_results

    told = []
    monkeypatch.setattr(chat_messenger, "deliver_background_message", lambda db, **kw: told.append(kw))
    for details in ({"lane": "agent", "executed_result": {"success": True}},
                    {"lane": "board", "board_task_id": 7, "executed_result": {"resumed_via": "board_task_requeue"}},
                    {"lane": "chat", "conversation_id": "not-a-chat", "executed_result": {"success": True}}):
        click_results.said_in_the_chat(None, NS(id=1, workspace_id=uuid4(), details=details))
    assert told == []


def test_the_approval_route_tells_the_chat_after_the_click_is_committed():
    tree = ast.parse((ORCH / "api" / "approval_grants.py").read_text(encoding="utf-8"))
    route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "grant_approval")
    calls = sorted((call.lineno, getattr(call.func, "id", None) or getattr(call.func, "attr", None))
                   for call in ast.walk(route) if isinstance(call, ast.Call))
    names = [name for _, name in calls]
    resumed = names.index("_requeue_subject")
    assert names.index("said_in_the_chat") > names.index("commit", resumed)


# ── First-class: approve and cancel mission are pinned ───────────────────────────

@pytest.mark.parametrize("name", [APPROVE, CANCEL])
def test_approve_and_cancel_mission_are_pinned_first_class(name):
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.tool_router import _first_class_names, _promotion_pins

    registry = get_action_registry()
    action = registry.get(name)
    assert action.promoted and name in _promotion_pins()
    assert action.permission_level == "write" and not action.admin_only and not action.super_admin_only
    promoted = {a.name for a in registry.get_all() if a.promoted}
    assert name in _first_class_names(None, promoted)
    schema = action.to_openai_schema()["function"]["parameters"]
    assert schema["required"] == ["mission_id"]
    assert all(spec.get("type") == "string" for spec in schema["properties"].values())   # strict: no catch-all


def test_get_mission_reads_a_title_as_the_tools_describe_it():
    from modules.tools.discovery.actions_missions import MISSION_REF_TEXT

    assert "title" in MISSION_REF_TEXT and "#0188" in MISSION_REF_TEXT
