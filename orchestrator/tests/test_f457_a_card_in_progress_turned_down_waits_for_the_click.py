"""F457 (night 18, ask 837): turning down a card its agent is working on waits for the owner's click.

"Turn #1836 down, it is the wrong job." became platform_update_task_status {status: 'assigned', note}
on a card In progress, and it ran with no card: ``owner_only.sends_back`` read only a worked card
(Review or Done) moved back as a send-back. The receipt was a bare write · done, and Auto said
"nothing has started" while the board's dispatcher started the card over. Ask 777 ("send #1846 back
to RESEARCHER with this note"), the same call on a Review card, raised its card and waited.

Now a card In progress taken back to Assigned or the Inbox is a send-back too: it raises one card
and moves on the click, once. A card whose run is still going is refused before any card, in the
board's words (the board would refuse the click). A new brief on an In progress card stays the
handler's to refuse at once (the board's Re-brief refuses every In progress card). A mission's own
card is still decided on its page, and an agent's own move is unchanged. The move's answer says the
card is queued to run again, so Auto can't say "nothing has started".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests import test_prd256_fix_send_back_waits_for_the_click as t3

shop = t3.shop      # #1094's Postgres: a workspace, its Content Creator, Auto's real tools
desk = t3.desk      # the shop's executor, as the chat's

MOVE, EDIT, NOTE = t3.MOVE, t3.EDIT, "it is the wrong job."


def _at_work(desk, *, running=False):
    """A card its agent is working on; ``running``: a live claim holds it (``_running_now``)."""
    task = t3._card(desk, "in_progress")
    if running:
        task.lease_until = datetime.now(timezone.utc) + timedelta(minutes=10)
        desk.db.flush()
    return task


def _turned_down(task):
    return {"task_id": t3._number(task), "status": "assigned", "note": NOTE}


def _click(desk, reply):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    grant_grant(desk.db.get(ApprovalGrant, reply["grant_id"]), granted_by=t3.CLICKER)
    desk.db.flush()


# ── The rule ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("new_status, sent_back", [
    ("assigned", True), ("inbox", True), ("in_progress", False), ("review", False), ("blocked", False),
])
def test_a_card_in_progress_taken_back_is_a_send_back(new_status, sent_back):
    from modules.tools.discovery.owner_only import sends_back

    assert sends_back("in_progress", new_status) is sent_back


# ── Night 18, ask 837 ───────────────────────────────────────────────────────────────

def test_turning_down_a_card_in_progress_raises_one_card_and_moves_nothing(desk):
    from consumers.chatbot.receipts import WAITING, receipt

    task = _at_work(desk)
    params = _turned_down(task)

    reply = t3._call(desk, params, t3._owners_chat())

    assert t3._asked(reply), reply                                              # ask 837: write · done, no card
    assert len(t3._grants(desk)) == 1 and t3._status(desk, task) == "in_progress"
    act = f"send '{task.title}' (ticket {t3._number(task)}) back to Content Creator with your note: {NOTE}"
    assert act in reply["message"]
    assert receipt(MOVE, params, reply)["status"] == WAITING
    assert desk.woken == []                                                     # the dispatcher was not woken


@pytest.mark.parametrize("status", ["review", "in_progress"])
def test_the_same_words_wait_on_a_review_card_and_a_card_in_progress(desk, status):
    """Ask 777 (Review) and ask 837 (In progress): one rule, one card each."""
    task = t3._card(desk, status)

    reply = t3._call(desk, _turned_down(task), t3._owners_chat())

    assert t3._asked(reply), reply
    assert t3._status(desk, task) == status


def test_the_click_moves_it_once_and_says_it_is_queued_to_run_again(desk):
    from modules.tools.discovery.ticket_moves import QUEUED, QUEUED_TO_RUN

    task = _at_work(desk)
    params = _turned_down(task)
    asked = t3._call(desk, params, t3._owners_chat())
    _click(desk, asked)

    ran = t3._call(desk, params, t3._owners_chat())

    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"], ran
    assert t3._status(desk, task) == "assigned"
    assert ran[QUEUED] == [task.id] and QUEUED_TO_RUN in ran["message"]         # never "nothing has started"
    assert desk.woken == [task.id]
    task.status = "in_progress"                                                 # the dispatcher took it again
    desk.db.flush()
    again = t3._call(desk, params, t3._owners_chat())                           # one click, one run
    assert t3._asked(again) and again["grant_id"] != asked["grant_id"], again


def test_the_edit_tools_send_back_on_a_card_in_progress_waits_too(desk):
    """platform_update_task {send_back} is a move to Assigned through the status tool (ticket_edits)."""
    task = _at_work(desk)

    reply = t3._call(desk, {"task_id": t3._number(task), "send_back": True, "note": NOTE}, t3._owners_chat(),
                     action=EDIT)

    assert t3._asked(reply), reply
    assert t3._status(desk, task) == "in_progress"


# ── Refused before any card ─────────────────────────────────────────────────────────

def test_a_card_whose_run_is_going_is_refused_before_any_card(desk):
    """The board refuses the move while the run holds the card: a card would be a click that does nothing (F444)."""
    task = _at_work(desk, running=True)

    reply = t3._call(desk, _turned_down(task), t3._owners_chat())

    assert reply["success"] is False and not reply.get("requires_confirmation"), reply
    assert "is running" in reply["error"] and "Nothing was asked or done" in reply["error"]
    assert '"cancelled"' in reply["error"]                                      # what turns it down instead
    assert t3._grants(desk) == [] and t3._status(desk, task) == "in_progress"


def test_a_bulk_take_back_leaves_off_the_card_still_running(desk):
    idle, running = _at_work(desk), _at_work(desk, running=True)
    ok, left = t3._number(idle), t3._number(running)

    reply = t3._call(desk, {"task_ids": [ok, left], "status": "assigned", "note": NOTE}, t3._owners_chat())

    assert t3._asked(reply), reply
    assert reply["params"]["task_ids"] == [ok]                                  # the click takes back only #ok
    assert [item["card"] for item in reply["left_out"]] == [left]
    assert f"Left off this card: {left}, which can't go back to an agent" in reply["message"]
    assert "is running" in reply["left_out"][0]["why"]
    assert t3._status(desk, idle) == "in_progress" and t3._status(desk, running) == "in_progress"


def test_a_missions_step_in_progress_is_refused_before_any_card(desk):
    """Code review: the status tool refuses Assigned on any mission ticket (``mission_runs_it``), so the
    take-back of a step in progress is refused in its words, never a card whose click it would refuse."""
    task = _at_work(desk)
    task.source_type = "orchestration_task"
    desk.db.flush()

    reply = t3._call(desk, _turned_down(task), t3._owners_chat())

    assert reply["success"] is False and not reply.get("requires_confirmation"), reply
    assert "a step of the mission" in reply["error"] and "Nothing was asked or done" in reply["error"]
    assert t3._grants(desk) == [] and t3._status(desk, task) == "in_progress"


def test_a_new_brief_on_a_card_in_progress_is_refused_at_once_with_no_card(desk):
    """The board's Re-brief refuses every In progress card, so that call is no send-back."""
    task = _at_work(desk)

    reply = t3._call(desk, {"task_id": t3._number(task), "description": t3.NEW_BRIEF}, t3._owners_chat(), action=EDIT)

    assert reply["success"] is False and not reply.get("requires_confirmation"), reply
    assert "re-brief it once it stops" in reply["error"]
    assert t3._grants(desk) == []


# ── What stays as it was ────────────────────────────────────────────────────────────

def test_a_missions_own_card_in_progress_is_decided_on_its_page(desk):
    from api.board_tasks import MISSION_CARD_SOURCE
    from modules.tools.discovery.owner_only import is_owner_only

    task = _at_work(desk)
    assert is_owner_only(MOVE, _turned_down(task), db=desk.db, workspace_id=desk.ws) is True
    task.source_type = MISSION_CARD_SOURCE
    desk.db.flush()
    assert is_owner_only(MOVE, _turned_down(task), db=desk.db, workspace_id=desk.ws) is False


def test_an_agents_own_move_runs_at_once_and_says_it_is_queued(desk):
    """No driving user (an agent's run, a playbook step): there is no click to ask for."""
    from modules.tools.discovery.ticket_moves import QUEUED

    task = _at_work(desk)

    reply = t3._call(desk, _turned_down(task), {"board_task_id": 85})

    assert reply["success"] is True and "requires_confirmation" not in reply, reply
    assert t3._grants(desk) == [] and t3._status(desk, task) == "assigned"
    assert reply[QUEUED] == [task.id]


def test_a_move_to_the_inbox_is_not_queued_to_run(desk):
    """Only Assigned with an agent is the dispatcher's: an Inbox card waits."""
    from modules.tools.discovery.ticket_moves import QUEUED

    task = _at_work(desk)

    reply = t3._call(desk, {"task_id": t3._number(task), "status": "inbox"}, {"board_task_id": 85})

    assert reply["success"] is True and t3._status(desk, task) == "inbox", reply
    assert QUEUED not in reply and desk.woken == []
