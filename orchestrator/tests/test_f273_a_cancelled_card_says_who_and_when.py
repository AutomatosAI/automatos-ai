"""F273 (night 7b) — a cancelled card says who cancelled it, and when.

Every cancel of night 7b held: two runs of "Weekly Instagram posts" (#0189 cancelled
0.1 s after it started, #0190 a second into step 1), the mission #0191 with its three
step cards, the Inbox card #0193 and the Support Agent's running task #0197. But none
of those cards said who cancelled it or when: "no note, no field". The mission page
said "Cancelled by user local@automatos.local" and the playbook's runs said who, and
Auto's own changes leave a note on the card ("…, in chat."). A cancel made on the
board left nothing in the card's notes, and #0191's own card had no record of who.

Each card a cancel stops now says so among its notes, the list the ticket view shows:
who ("you"; Auto, in chat), when, and what went with it. F245's cancel is unchanged.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.core import BoardTask
from core.models.orchestration_enums import TaskState
from services import coordinator_service as cs
from tests.test_f116_a_cancelled_run_stops_its_sessions import _recipe, _run, _step
from tests.test_f245_cancel_stops_a_mission import _mission

OWNER = "user:2"
YOU = "you"
EARLIER_NOTE = {"note": "Use the new oat flat white price.", "by": "you", "at": "2026-10-03T18:40:00+00:00"}


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture(autouse=True)
def no_narration(monkeypatch):
    """The launching thread's lines are their own suite's business (F245's tests)."""
    monkeypatch.setattr(cs, "_narrate_run_terminal", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_narrate_mission", lambda *a, **k: None)


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    """The owner's workspace on the local edition. A playbook run told to stop is
    recorded, not looked for on this worker (api/recipe_executor)."""
    import api.recipe_executor as executor

    told = []
    monkeypatch.setattr(executor, "request_execution_cancel", lambda execution_id: told.append(execution_id) or True)
    ws = UUID(seed_workspace())
    owner = NS(workspace_id=ws, user_id="2", auth_type="anonymous", user=NS(id="2"))
    return NS(db=db_session, ws=ws, owner=owner, stopped=told)


def _card(board, **fields):
    task = BoardTask(**{"workspace_id": board.ws, "priority": "medium", "source_type": "user", **fields})
    board.db.add(task)
    board.db.flush()
    return task


def _row(board, task_id):
    """The card as it now stands in the database."""
    board.db.flush()
    board.db.expire_all()
    return board.db.get(BoardTask, task_id)


def _notes(board, task_id):
    """The card's notes as the ticket view lists them: who, and what."""
    return [(n["by"], n["note"]) for n in (_row(board, task_id).runtime_ref or {}).get("session_notes", [])]


def _said_when(board, task_id, before, after):
    """The last note's time is the cancel's, and the record's."""
    ref = _row(board, task_id).runtime_ref
    at = ref["session_notes"][-1]["at"]
    return at == ref["cancelled"]["at"] and before <= datetime.fromisoformat(at) <= after


def _cancel_on_the_board(board, task_id):
    from api.board_tasks import cancel_task

    return asyncio.run(cancel_task(task_id, ctx=board.owner, db=board.db))


def _drag_to_cancelled(board, task_id):
    from api.board_tasks import update_task_status

    return asyncio.run(update_task_status(task_id, _Request({"status": "cancelled"}), ctx=board.owner, db=board.db))


@pytest.mark.parametrize("cancel", [_cancel_on_the_board, _drag_to_cancelled], ids=["Cancel", "a drag"])
@pytest.mark.parametrize("status, title", [
    ("inbox", "Price list for The Lantern Room"),                     # #0193
    ("in_progress", "Club FAQ page for the website (wait for me)"),    # #0197, 4 s into its run
], ids=["#0193 in the Inbox", "#0197 running"])
def test_a_card_cancelled_on_the_board_says_you_did_it_and_when(board, cancel, status, title):
    card = _card(board, title=title, status=status, runtime_ref={"session_notes": [EARLIER_NOTE]})
    before = datetime.now(timezone.utc)

    out = cancel(board, card.id)

    after = datetime.now(timezone.utc)
    assert out["status"] == "cancelled"
    assert _notes(board, card.id) == [(YOU, EARLIER_NOTE["note"]), (YOU, "Cancelled this.")]   # night 7b: none
    assert _said_when(board, card.id, before, after)
    assert _row(board, card.id).runtime_ref["cancelled"]["by"] == OWNER   # F245's record, for the banner


def _running_playbook(board):
    """A run of "Weekly Instagram posts" a second into step 1: its card, and the
    session ticket of its first step."""
    recipe = _recipe(board.db, board.ws)
    run = _run(board.db, board.ws, recipe.id)
    card = _card(board, title="Recipe: Weekly Instagram posts", status="in_progress", source_type="recipe",
                 source_id=run, review_mode="auto")
    return NS(recipe=recipe, run=run, card=card.id, step=_step(board.db, board.ws, run, 1, "in_progress"))


def _from_the_card(board, pb):
    return _cancel_on_the_board(board, pb.card)


def _from_the_playbooks_page(board, pb):
    from api.workflow_recipes import cancel_execution

    return asyncio.run(cancel_execution(str(pb.recipe.id), pb.run, ctx=board.owner, db=board.db))


@pytest.mark.parametrize("cancel", [_from_the_card, _from_the_playbooks_page], ids=["the card", "the playbook's page"])
def test_a_playbook_runs_card_and_its_step_say_who_cancelled_the_run(board, cancel):
    """#0189 and #0190: only the run said who; the card and its step said nothing."""
    pb = _running_playbook(board)
    before = datetime.now(timezone.utc)

    cancel(board, pb)

    after = datetime.now(timezone.utc)
    assert board.stopped == [pb.run]                                    # F245: the run stopped
    assert _notes(board, pb.card) == [(YOU, "Cancelled the playbook run.")]
    assert _notes(board, pb.step) == [(YOU, "Cancelled with its playbook run.")]
    assert _said_when(board, pb.card, before, after) and _said_when(board, pb.step, before, after)


def _mission_from_the_board(board, run, card):
    _cancel_on_the_board(board, card.id)


def _mission_from_its_page(board, run, card):
    """POST /api/missions/{id}/cancel names the person by their id, as here."""
    cs.CoordinatorService().cancel_mission(board.db, run.id, "2")
    board.db.commit()


UNFINISHED = (TaskState.PENDING, TaskState.QUEUED, TaskState.RUNNING, TaskState.COMPLETED)


@pytest.mark.parametrize("cancel", [_mission_from_the_board, _mission_from_its_page],
                         ids=["on the board", "on the mission's page"])
def test_a_cancelled_mission_says_who_on_its_card_and_its_steps(board, cancel):
    """#0191 and #0191.1 to .3: Cancelled, and not a word of who or when."""
    run, card, steps = _mission(board.db, board.ws)
    before = datetime.now(timezone.utc)

    cancel(board, run, card)

    after = datetime.now(timezone.utc)
    mission_card = _row(board, card.id)
    assert mission_card.status == "cancelled"                           # F245
    assert mission_card.runtime_ref["cancelled"]["by"] == OWNER          # night 7b: no record at all
    assert _notes(board, card.id) == [(YOU, "Cancelled the mission.")] and _said_when(board, card.id, before, after)
    for state in UNFINISHED:
        step_card = steps[state][1]
        assert _notes(board, step_card.id) == [(YOU, "Cancelled with its mission.")], state
        assert _said_when(board, step_card.id, before, after), state
    assert _notes(board, steps[TaskState.VERIFIED][1].id) == []          # finished work: Done, no cancel note


def test_auto_cancelling_a_mission_in_chat_says_auto_did_it_in_chat(board):
    """As Auto's other changes say on the card ("…, in chat."), while the record still
    names the person whose chat it was."""
    from core.llm.usage_context import LANE_CHAT, usage_scope
    from core.models.core import Agent
    from modules.tools.discovery.handlers_missions import cancel_mission

    auto = Agent(name="Auto", agent_type="chatbot", description="", status="active", configuration={},
                 model_config=None, workspace_id=board.ws, created_by="test", owner_type="workspace",
                 owner_id=str(board.ws))
    board.db.add(auto)
    board.db.flush()
    run, card, steps = _mission(board.db, board.ws)

    with usage_scope(request_type=LANE_CHAT, execution_id=f"chat:{uuid.uuid4()}", agent_id=auto.id):
        reply = asyncio.run(cancel_mission(board.db, board.ws, {"mission_id": str(run.id),
                                                                "_created_by": "owner@cafe.test"}))

    assert reply["success"] is True
    assert _notes(board, card.id) == [("Auto", "Cancelled the mission, in chat.")]
    assert _notes(board, steps[TaskState.QUEUED][1].id) == [("Auto", "Cancelled with its mission, in chat.")]
    assert _row(board, card.id).runtime_ref["cancelled"]["by"] == "user:owner@cafe.test"


def test_a_card_the_platform_stopped_says_what_stopped_it_and_why(board):
    """F224: a playbook run that failed stops the session working its step. The card
    says the run stopped it, and why, rather than naming a person."""
    from services.board_task_bridge import complete_recipe_board_task

    recipe = _recipe(board.db, board.ws)
    run = _run(board.db, board.ws, recipe.id)
    working = _step(board.db, board.ws, run, 1, "in_progress")
    board.db.commit()

    complete_recipe_board_task(board.db, run, success=False, error_message="the model refused")

    assert _notes(board, working) == [("the playbook run", "Cancelled this because the run failed.")]
