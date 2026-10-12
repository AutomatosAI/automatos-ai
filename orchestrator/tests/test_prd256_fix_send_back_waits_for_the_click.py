"""P256-FIX-T3 (Gerard, 11 Oct, benchmark N5-a): sending a card back to its agent waits for the owner's click.

Eval N5-a on Haiku 5.5: "send #1219 back" → platform_update_task_status {status: 'assigned'} on a
Review card ran at once ("I've sent #1219 back … It's now in Assigned"), and no card:
``owner_only.is_owner_only`` read a card move as owner-only only when it closed the card. A worked
card (Review or Done) moved back to Assigned, In progress or the Inbox is now owner-only, its status
read from the board (``owner_only.sends_back``, ``send_backs``); the card says what the click does,
and the click runs the move once. Giving a new inbox card to an agent stays free.

Each call runs through the executor's whole stack (the question guard, the owner's-words guard, the
session ticket, the role gates, the hierarchy check, the rate limit, the ask) onto the real status
tool, whose send-back is the board's Reject (``ticket_moves``).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

shop = f1094.board      # #1094's Postgres: a workspace, its Content Creator, Auto's real tools
MOVE = "platform_update_task_status"
EDIT = "platform_update_task"
NOTE = "Lead with the Guji's price, and drop the line about Tuesday."
CLICKER = "user:7"
DRAFT = "Hello Larder & Loaf, our wholesale list for October…"


@pytest.fixture
def desk(shop, monkeypatch):
    """The shop's executor, as the chat's: Auto is the running agent, the dial is on (an
    owner-only call still asks), and the owner's words are not read (the click decides)."""
    import modules.tools.discovery.follows_the_owner as guard
    from core.models.core import Agent
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=shop.ws, created_by="test", owner_type="workspace", owner_id=str(shop.ws),
                 is_system_agent=True)
    shop.db.add(auto)
    shop.db.flush()
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: None)
    executor = PlatformActionExecutor(shop.db, shop.ws)
    executor._full_autonomy = lambda: True
    return NS(**vars(shop), executor=executor, auto=auto.id)


def _card(desk, status):
    worked = status in ("review", "done")
    return f1094._ticket(desk, status=status, assigned_agent_id=desk.agent, result=DRAFT if worked else None,
                         completed_at=datetime.now(timezone.utc) if worked else None)


def _number(task):
    return f"#{task.workspace_seq:04d}"


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _call(desk, params, caller_context, action=MOVE):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(desk.executor.execute(action, {**params, "_agent_id": desk.auto}, caller_context))


def _grants(desk):
    from core.models.approval_grants import ApprovalGrant

    return desk.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == desk.ws).all()


def _asked(reply):
    return (reply["success"] is False and reply.get("requires_confirmation") is True
            and reply.get("owner_only") is True and isinstance(reply.get("grant_id"), int))


def _status(desk, task):
    desk.db.refresh(task)
    return task.status


# ── N5-a: a Review card sent back from chat ─────────────────────────────────────────

def test_a_review_card_sent_back_from_chat_raises_one_card_and_stays_in_review(desk):
    from consumers.chatbot.receipts import WAITING, receipt

    task = _card(desk, "review")
    params = {"task_id": _number(task), "status": "assigned", "note": NOTE}

    reply = _call(desk, params, _owners_chat())

    assert _asked(reply), reply
    assert len(_grants(desk)) == 1
    assert _status(desk, task) == "review"                                     # N5-a: Assigned, no card
    act = f"send '{task.title}' (ticket {_number(task)}) back to Content Creator with your note: {NOTE}"
    assert act in reply["message"] and reply["question_md"].lower().startswith(act.lower())
    assert receipt(MOVE, params, reply)["status"] == WAITING                  # FX-004: waiting, never a failure


def test_the_click_moves_it_to_assigned_once_with_the_note(desk):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant
    from services.ticket_redo import times_sent_back

    task = _card(desk, "review")
    params = {"task_id": _number(task), "status": "assigned", "note": NOTE}
    asked = _call(desk, params, _owners_chat())
    grant_grant(desk.db.get(ApprovalGrant, asked["grant_id"]), granted_by=CLICKER)
    desk.db.flush()

    ran = _call(desk, params, _owners_chat())

    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"], ran
    assert ran["sent_back"] is True
    assert _status(desk, task) == "assigned" and task.review_feedback == NOTE
    assert task.planning_data["owner_corrections"][-1]["note"] == NOTE
    assert times_sent_back(task) == 1
    task.status = "review"                                                      # answered again: back in Review
    desk.db.flush()
    again = _call(desk, params, _owners_chat())                                 # one click, one run
    assert _asked(again) and again["grant_id"] != asked["grant_id"], again
    assert _status(desk, task) == "review" and times_sent_back(task) == 1


def test_a_send_back_that_also_gives_the_card_names_the_agent_it_goes_to(desk):
    """The click signs what the card shows: platform_update_task's agent_id beside the send-back."""
    from sqlalchemy import text

    other = desk.db.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Wholesale Desk', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(desk.ws)}).scalar()
    task = _card(desk, "review")

    reply = _call(desk, {"task_id": _number(task), "status": "assigned", "note": NOTE, "agent_id": other},
                  _owners_chat(), action=EDIT)

    assert _asked(reply), reply
    assert "back to Wholesale Desk with your note" in reply["message"]


def test_a_done_card_reopened_to_in_progress_waits_for_the_click(desk):
    task = _card(desk, "done")

    reply = _call(desk, {"task_id": _number(task), "status": "in_progress", "note": NOTE}, _owners_chat())

    assert _asked(reply), reply
    assert len(_grants(desk)) == 1 and _status(desk, task) == "done"


def test_the_edit_tools_send_back_waits_too(desk):
    """F279's call: platform_update_task {send_back: true}, the owner's words as its note."""
    task = _card(desk, "review")

    reply = _call(desk, {"task_id": _number(task), "send_back": True, "note": NOTE}, _owners_chat(), action=EDIT)

    assert _asked(reply), reply
    assert "back to Content Creator with your note" in reply["message"]
    assert _status(desk, task) == "review"


def test_a_paraphrased_note_raises_the_card_before_the_signature_rule(desk, monkeypatch):
    """D1 (FX-003): an owner-only call's note is on the card the owner clicks, so the words rule waits for it."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import NamedCard, OwnerTurn

    task = _card(desk, "review")
    said = f"Send {_number(task)} back: lead with the price."
    turn = OwnerTurn(latest=said, earlier="",
                     cards=(NamedCard(ref=_number(task), seq=task.workspace_seq, step=None, task=task),))
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: turn if ctx else None)
    monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: (said,))

    reply = _call(desk, {"task_id": _number(task), "status": "assigned", "note": NOTE}, _owners_chat())

    assert _asked(reply), reply
    assert "signed as the owner's" not in str(reply.get("error", ""))
    assert _status(desk, task) == "review"


# ── What stays free ─────────────────────────────────────────────────────────────────

def test_a_new_inbox_card_given_to_its_agent_moves_at_once(desk):
    """The hand-off lane: inbox → assigned is no send-back."""
    from core.models.approval_grants import GrantStatus

    task = _card(desk, "inbox")

    reply = _call(desk, {"task_id": _number(task), "status": "assigned"}, _owners_chat())

    assert reply["success"] is True and "requires_confirmation" not in reply, reply
    # The owner's own move is recorded as consent (board_consent.record_operator_consent: a grant already
    # granted, so the dispatcher does not ask again); what must not exist is a card waiting for a click.
    assert [g for g in _grants(desk) if g.status == GrantStatus.PENDING] == [] and _status(desk, task) == "assigned"


def test_an_agents_own_send_back_is_unchanged(desk):
    """No driving user (an agent's run, a playbook step): there is no click to ask for."""
    task = _card(desk, "review")

    reply = _call(desk, {"task_id": _number(task), "status": "assigned", "note": NOTE}, {"board_task_id": 85})

    assert reply["success"] is True and "requires_confirmation" not in reply, reply
    assert _grants(desk) == [] and _status(desk, task) == "assigned"


# ── The rule ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("card_status, new_status, sent_back", [
    ("review", "assigned", True), ("review", "in_progress", True), ("review", "inbox", True),
    ("done", "assigned", True), ("done", "in_progress", True), ("done", "inbox", True),
    ("inbox", "assigned", False), ("assigned", "in_progress", False), ("in_progress", "assigned", False),
    ("review", "done", False), ("review", "blocked", False), ("blocked", "assigned", False),
])
def test_sends_back_reads_a_worked_card_moved_back_for_rework(card_status, new_status, sent_back):
    from modules.tools.discovery.owner_only import sends_back

    assert sends_back(card_status, new_status) is sent_back


def test_the_board_is_read_in_the_callers_workspace_only(desk, seed_workspace):
    """Another workspace's Review card, named by its number, sends nothing back here."""
    from uuid import UUID

    from modules.tools.discovery.owner_only import is_owner_only

    task = _card(desk, "review")
    params = {"task_id": _number(task), "status": "assigned"}
    assert is_owner_only(MOVE, params, db=desk.db, workspace_id=desk.ws) is True
    assert is_owner_only(MOVE, params, db=desk.db, workspace_id=UUID(seed_workspace())) is False
    assert is_owner_only(MOVE, params) is False                                 # no board: by its status alone


def test_a_rebrief_and_a_missions_own_card_are_not_read_as_a_send_back(desk):
    """FX-013: a new brief drops the call's status; a mission's card is decided on its page."""
    from api.board_tasks import MISSION_CARD_SOURCE
    from modules.tools.discovery.owner_only import is_owner_only

    task = _card(desk, "review")
    rebrief = {"task_id": _number(task), "description": "Write it for the October list.", "status": "in_progress"}
    assert is_owner_only(EDIT, rebrief, db=desk.db, workspace_id=desk.ws) is False
    task.source_type = MISSION_CARD_SOURCE
    desk.db.flush()
    moved_back = {"task_id": _number(task), "status": "assigned"}
    assert is_owner_only(MOVE, moved_back, db=desk.db, workspace_id=desk.ws) is False
