"""P256-FIX-N17-1 (F438, nights 16–17): a Re-brief from the owner's chat waits for their click.

Night 16, ask 118: "I turned down the autumn launch copy card … Get it rewritten … and bring it
back to me." Auto's platform_update_task {task_id: '#1713', agent_name: 'HARBOUR LOG', note,
description} on a card in Review ran at once: #1713 went back to Assigned and re-ran, its receipt
said 'agent set', and no card was raised. Night 17 (#1751, no agent): the same. A new brief on a
worked card is the board's Re-brief, which sends the card back to its agent (FX-013 drops any
status beside it); ``send_backs.moved_to`` read it as no move at all. Now the Re-brief is the
send-back it is: one approval card that says 'a new brief' and names the agent, the click
re-briefs the card once, and the receipt reads 'sent back to its agent'.

Each call runs through the executor's whole stack onto the real edit tool (send_back_waits' desk).
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from tests import test_prd256_fix_send_back_waits_for_the_click as waits

shop, desk = waits.shop, waits.desk
EDIT = waits.EDIT
OLD_BRIEF = "Write the autumn launch copy for the newsletter."
NEW_BRIEF = "Redraft the autumn launch copy: lead with the Guji, two short paragraphs, no exclamation marks."
OWNERS_WORDS = "I turned down the autumn launch copy card. Get it rewritten and bring it back to me."


@pytest.fixture
def here(desk, monkeypatch):
    """send_back_waits' desk, with the ticket notes (written in a session of their own,
    ticket_changes) in the test's session, whose rows another session can't see; the
    dispatcher is not woken."""
    import api.board_tasks as board
    import core.database.database as database

    @contextmanager
    def the_tests():
        yield desk.db

    monkeypatch.setattr(database, "get_db_session", the_tests)
    monkeypatch.setattr(board, "notify_task_available", lambda *a, **k: None)
    return desk


def _card(desk, status):
    task = waits._card(desk, status)
    task.description = OLD_BRIEF
    desk.db.flush()
    return task


def _granted(desk, reply):
    from core.models.approval_grants import ApprovalGrant

    return dict(desk.db.get(ApprovalGrant, reply["grant_id"]).details["params"])


def _click(desk, reply):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    grant_grant(desk.db.get(ApprovalGrant, reply["grant_id"]), granted_by=waits.CLICKER)
    desk.db.flush()


def _pending(desk):
    from core.models.approval_grants import GrantStatus

    return [grant for grant in waits._grants(desk) if grant.status == GrantStatus.PENDING]


# ── Night 16 ask 118 and night 17 #1751 ─────────────────────────────────────────────

def test_ask_118_a_rebrief_naming_its_own_agent_raises_one_card(here):
    from consumers.chatbot.receipts import WAITING, receipt

    task = _card(here, "review")
    params = {"task_id": waits._number(task), "agent_name": "Content Creator", "note": OWNERS_WORDS,
              "description": NEW_BRIEF}

    reply = waits._call(here, params, waits._owners_chat(), action=EDIT)

    assert waits._asked(reply), reply
    assert len(waits._grants(here)) == 1
    assert (waits._status(here, task), task.description) == ("review", OLD_BRIEF)        # F438: Assigned, no card
    assert "a new brief" in reply["message"] and "back to Content Creator" in reply["message"]
    assert "send back to its agent for a redo" in reply["question_md"] and "- agent:" not in reply["question_md"]
    granted = _granted(here, reply)
    assert granted["agent_id"] == here.agent and "agent_name" not in granted             # the click runs on the id
    assert receipt(EDIT, params, reply)["status"] == WAITING


def test_night_17_the_click_rebriefs_it_once(here):
    from consumers.chatbot.receipts import receipt
    from modules.tools.execution.call_effects import done_effects

    task = _card(here, "review")
    params = {"task_id": waits._number(task), "description": NEW_BRIEF, "note": OWNERS_WORDS}
    asked = waits._call(here, params, waits._owners_chat(), action=EDIT)
    _click(here, asked)

    ran = waits._call(here, params, waits._owners_chat(), action=EDIT)

    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"], ran
    assert ran["sent_back"] is True
    assert (waits._status(here, task), task.description) == ("assigned", NEW_BRIEF)
    assert task.planning_data["previous_briefs"][-1]["description"] == OLD_BRIEF
    assert "send_back" in done_effects(EDIT, params, ran)
    assert "sent back to its agent" in receipt(EDIT, params, ran)["effect"]
    task.status = "review"                                                               # answered again
    here.db.flush()
    again = waits._call(here, params, waits._owners_chat(), action=EDIT)                 # one click, one run
    assert waits._asked(again) and again["grant_id"] != asked["grant_id"], again
    assert waits._status(here, task) == "review"


@pytest.mark.parametrize("status", ["review", "done"])
def test_a_done_card_rebriefed_waits(here, status):
    task = _card(here, status)

    reply = waits._call(here, {"task_id": waits._number(task), "description": NEW_BRIEF}, waits._owners_chat(),
                        action=EDIT)

    assert waits._asked(reply), reply
    assert len(waits._grants(here)) == 1
    assert (waits._status(here, task), task.description) == (status, OLD_BRIEF)


def test_a_rebrief_with_a_non_closing_status_waits_and_shows_no_status_move(here):
    """FX-013 drops the status beside a Re-brief, so the card shows the send-back, not a move."""
    task = _card(here, "review")

    reply = waits._call(here, {"task_id": waits._number(task), "description": NEW_BRIEF, "status": "in_progress",
                               "note": OWNERS_WORDS}, waits._owners_chat(), action=EDIT)

    assert waits._asked(reply), reply
    assert "- status:" not in reply["question_md"] and "send back to its agent for a redo" in reply["question_md"]
    assert "a new brief" in reply["message"]
    assert (waits._status(here, task), task.description) == ("review", OLD_BRIEF)


# ── What stays free ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["inbox", "assigned"])
def test_a_new_brief_on_a_card_nobody_has_worked_on_is_a_plain_edit(here, status):
    task = _card(here, status)

    reply = waits._call(here, {"task_id": waits._number(task), "description": NEW_BRIEF}, waits._owners_chat(),
                        action=EDIT)

    assert reply["success"] is True and "requires_confirmation" not in reply, reply
    assert _pending(here) == []
    assert (waits._status(here, task), task.description) == (status, NEW_BRIEF)


def test_an_agents_own_rebrief_is_unchanged(here):
    """No driving user (an agent's run, a playbook step): there is no click to ask for."""
    task = _card(here, "review")

    reply = waits._call(here, {"task_id": waits._number(task), "description": NEW_BRIEF}, {"board_task_id": 85},
                        action=EDIT)

    assert reply["success"] is True and "requires_confirmation" not in reply, reply
    assert waits._grants(here) == []
    assert (waits._status(here, task), task.description) == ("assigned", NEW_BRIEF)


@pytest.mark.parametrize("status, waits_for_it", [
    ("review", True), ("done", True),
    ("inbox", False), ("assigned", False), ("blocked", False), ("failed", False), ("in_progress", False),
])
def test_what_waits_is_read_from_the_cards_status(here, status, waits_for_it):
    from modules.tools.discovery.owner_only import is_owner_only

    task = _card(here, status)
    rebrief = {"task_id": waits._number(task), "description": NEW_BRIEF}

    assert is_owner_only(EDIT, rebrief, db=here.db, workspace_id=here.ws) is waits_for_it


def test_no_board_a_missions_card_or_a_brief_of_status_orders_is_no_rebrief(here):
    from api.board_tasks import MISSION_CARD_SOURCE
    from modules.tools.discovery.owner_only import is_owner_only

    task = _card(here, "review")
    number = waits._number(task)
    assert is_owner_only(EDIT, {"task_id": number, "description": NEW_BRIEF}) is False      # no board
    assert is_owner_only(EDIT, {"task_id": number, "description": "Mark the card as done."},
                         db=here.db, workspace_id=here.ws) is False                        # only a status order
    task.source_type = MISSION_CARD_SOURCE
    here.db.flush()
    assert is_owner_only(EDIT, {"task_id": number, "description": NEW_BRIEF}, db=here.db, workspace_id=here.ws) is False


# ── What the model is told, and what the receipts say ───────────────────────────────

@pytest.mark.parametrize("said", ["Send #0022 back: lead with the Guji.",
                                  "Update #0022 with this brief: lead with the Guji."])
def test_the_send_back_and_rebrief_calls_say_they_wait_for_the_click(said):
    from types import SimpleNamespace as NS

    from modules.tools.discovery.follows_the_owner import _send_back_call, right_call
    from modules.tools.discovery.owner_turn import NamedCard, OwnerTurn

    card = NamedCard(ref="#0022", seq=22, step=None,
                     task=NS(id=922, title="Autumn launch copy", status="review", source_type="user", description=""))

    assert "waits for the owner's click" in right_call(OwnerTurn(latest=said, earlier="", cards=(card,)), card)
    assert "waits for the owner's click" in _send_back_call("#0022")


def test_the_edit_tool_says_its_rebrief_waits_for_the_click():
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get(EDIT)
    assert "waits for the owner's click" in action.parameters["properties"]["description"]["description"]
    assert "waits for the owner's click" in action.description


def test_a_rebrief_says_sent_back_and_a_card_given_again_says_so():
    from consumers.chatbot.receipts import receipt
    from modules.tools.execution.call_effects import GIVEN_AGAIN, SENT_BACK, result_effects

    given = {"success": True, "task_id": 5, "status": "assigned", "runs_again": True}
    assert result_effects(given) == (GIVEN_AGAIN,) and SENT_BACK not in result_effects(given)
    effect = receipt("platform_assign_task", {"task_id": 5, "agent_name": "OPS"}, given)["effect"]
    assert "given to another agent; it runs again from its brief" in effect and "sent back" not in effect
    rebriefed = receipt(EDIT, {"task_id": 5, "description": NEW_BRIEF}, {"success": True, "task_id": 5,
                                                                         "sent_back": True})
    assert rebriefed["effect"] == "sent back to its agent"
