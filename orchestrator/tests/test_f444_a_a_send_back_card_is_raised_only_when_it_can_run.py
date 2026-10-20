"""P256-FIX-N17-2 (F444, night 17): a send-back's card is raised only when its click can run.

Night 17: #1767 was filed by platform_create_task with approval_action create_blog and no
agent, so it was born in Review. "Turn #1767 down" became platform_update_task_status
{status: 'assigned', note}; card #2827 ("send … back to its agent") was granted, and the
board's Reject refused the click ("Cannot reject a task with no assigned agent"): the grant
was given back, the card never moved, and the owner learned it only by asking. L468's retry
raised #2832 on the same path. Now each card the call hands to the Reject is read against
the Reject's rule (``services.send_back_rules``) before any card is raised: an approval card
is refused naming the cancel that turns it down, a card with no agent naming how to give it
one, and a bulk move leaves off only the cards that cannot go back.

The latent sibling: {agent, status: 'assigned', note} on an answered card ran it again for
the new agent and never reached the Reject, so the owner's note never reached the redo. The
card is now only given, and the Reject sends it back with the note.

Each call runs through the executor's whole stack (send_back_waits' desk).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094
from tests import test_f438_a_rebrief_or_reassign_waits_for_the_click as f438
from tests import test_prd256_fix_send_back_waits_for_the_click as waits

shop, desk, here = waits.shop, waits.desk, f438.here
MOVE, EDIT, NOTE = waits.MOVE, waits.EDIT, waits.NOTE
TOPIC = "Autumn launch: the Christmas Blend"
NO_AGENT = "Cannot reject a task with no assigned agent"


def _approval_card(desk, agent=None):
    """Night 17's #1767: approval_action create_blog, no agent, born in Review."""
    return f1094._ticket(desk, status="review", title="Autumn launch blog", assigned_agent_id=agent,
                         result=waits.DRAFT, completed_at=datetime.now(timezone.utc),
                         planning_data={"approval_action": {"type": "create_blog", "topic": TOPIC}})


def _no_agent_card(desk, status="review"):
    return f1094._ticket(desk, status=status, title="Wholesale reply", result=waits.DRAFT,
                         completed_at=datetime.now(timezone.utc))


def _refused_before_any_card(reply):
    return reply["success"] is False and "requires_confirmation" not in reply and "grant_id" not in reply


# ── Night 17: #1767, an approval card with no agent ─────────────────────────────────

def test_an_approval_cards_send_back_is_refused_before_any_card_naming_cancel(desk):
    task = _approval_card(desk)
    number = waits._number(task)

    reply = waits._call(desk, {"task_id": number, "status": "assigned", "note": NOTE}, waits._owners_chat())

    assert _refused_before_any_card(reply), reply                                # night 17: card #2827
    assert waits._grants(desk) == [] and waits._status(desk, task) == "review"
    error = reply["error"]
    assert f"Ticket {number} ('Autumn launch blog') is a card for the owner's approval" in error
    assert f"starts a blog mission on '{TOPIC}'" in error and "Nothing was asked or done" in error
    assert f'platform_update_task_status {{task_id: "{number}", status: "cancelled"}}' in error
    assert 'status: "cancelled"' in error and "agent_id" not in error            # never offers a redo


def test_send_back_to_a_named_agent_on_an_approval_card_is_refused_before_any_grant(desk):
    """L468: 'Send #1767 back to WRITER…'; an agent named or not, an agent's redo can't change what Approve runs."""
    task = _approval_card(desk)
    number = waits._number(task)

    named = waits._call(desk, {"task_id": number, "status": "assigned", "note": NOTE, "agent_name": "Content Creator"},
                        waits._owners_chat(), action=EDIT)
    unnamed = waits._call(desk, {"task_id": number, "status": "assigned", "note": NOTE}, waits._owners_chat())

    for reply in (named, unnamed):
        assert _refused_before_any_card(reply) and "a card for the owner's approval" in reply["error"], reply
    assert waits._grants(desk) == [] and waits._status(desk, task) == "review"


def test_a_publish_or_owners_decision_card_is_refused_with_what_its_approval_does(desk):
    publish = _approval_card(desk, agent=desk.agent)
    publish.planning_data = {"approval_action": {"type": "publish_blog", "post_id": "p-1"}}
    decision = _no_agent_card(desk)
    decision.planning_data = {"owners_decision": True}
    desk.db.flush()

    for task, does in ((publish, "publishes its blog post"), (decision, "records the owner's decision")):
        reply = waits._call(desk, {"task_id": waits._number(task), "status": "assigned", "note": NOTE},
                            waits._owners_chat())
        assert _refused_before_any_card(reply) and f"approving it {does}" in reply["error"], reply
    assert waits._grants(desk) == []


def test_a_done_approval_card_is_refused_with_no_cancel_to_offer(desk):
    """Its approval was given already: sending it back would bring it to Review, where approving runs it again."""
    task = _approval_card(desk, agent=desk.agent)
    task.status = "done"
    desk.db.flush()

    reply = waits._call(desk, {"task_id": waits._number(task), "status": "assigned", "note": NOTE},
                        waits._owners_chat())

    assert _refused_before_any_card(reply), reply
    assert "a card for the owner's approval, and it is Done" in reply["error"]
    assert "cancelled" not in reply["error"] and "file a new card" in reply["error"]
    assert waits._grants(desk) == [] and waits._status(desk, task) == "done"


# ── A plain card with no agent ──────────────────────────────────────────────────────

def test_a_send_back_on_a_plain_card_with_no_agent_is_refused_naming_how_to_give_it(desk):
    task = _no_agent_card(desk)
    number = waits._number(task)

    reply = waits._call(desk, {"task_id": number, "status": "assigned", "note": NOTE}, waits._owners_chat())

    assert _refused_before_any_card(reply), reply
    assert waits._grants(desk) == [] and waits._status(desk, task) == "review"
    error = reply["error"]
    assert f"Ticket {number} ('Wholesale reply') can't go back to an agent: {NO_AGENT}." in error
    assert "Name the agent it goes to: platform_update_task" in error and "agent_id" in error
    assert 'status "cancelled"' in error


def test_a_send_back_that_names_its_agent_keeps_the_note_on_the_click(here):
    """The latent sibling: the agent the call gives the card counts before the card, and on the
    click the card is only given, so the board's Reject sends it back with the owner's note."""
    from services.ticket_redo import SENT_BACK, times_sent_back

    task = _no_agent_card(here)
    params = {"task_id": waits._number(task), "status": "assigned", "note": NOTE, "agent_id": here.agent}
    asked = waits._call(here, params, waits._owners_chat(), action=EDIT)
    assert waits._asked(asked) and "back to Content Creator with your note" in asked["message"], asked
    f438._click(here, asked)

    ran = waits._call(here, params, waits._owners_chat(), action=EDIT)

    assert ran["success"] is True and ran["sent_back"] is True, ran               # night 17: runs_again, no note
    assert (waits._status(here, task), task.assigned_agent_id) == ("assigned", here.agent)
    assert task.review_feedback == NOTE and task.planning_data["owner_corrections"][-1]["note"] == NOTE
    assert task.planning_data["previous_runs"][-1]["why"] == SENT_BACK and times_sent_back(task) == 1


# ── The Re-brief: a send-back since N17-1, read against the same rule ───────────────

def test_a_rebrief_of_a_card_with_no_agent_is_refused_before_any_card(desk):
    """N17-1 made a new brief on a worked card a send-back that waits for the click; the click's
    Re-brief refuses a card with no agent, so the card is never raised for it."""
    task = _no_agent_card(desk)

    reply = waits._call(desk, {"task_id": waits._number(task), "description": f438.NEW_BRIEF}, waits._owners_chat(),
                        action=EDIT)

    assert _refused_before_any_card(reply) and NO_AGENT in reply["error"], reply
    assert waits._grants(desk) == [] and waits._status(desk, task) == "review"


def test_a_rebrief_that_gives_the_card_to_another_agent_rebriefs_it_with_that_agent_on_the_click(here):
    """Before, the card was given the platform_assign_task way (it ran again from its old brief)
    and the new brief landed as a plain edit; now it is given, then re-briefed."""
    from sqlalchemy import text

    other = here.db.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Wholesale Desk', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(here.ws)}).scalar()
    task = f438._card(here, "review")
    params = {"task_id": waits._number(task), "agent_id": other, "description": f438.NEW_BRIEF}
    asked = waits._call(here, params, waits._owners_chat(), action=EDIT)
    assert waits._asked(asked) and "back to Wholesale Desk" in asked["message"], asked
    f438._click(here, asked)

    ran = waits._call(here, params, waits._owners_chat(), action=EDIT)

    assert ran["success"] is True and ran["sent_back"] is True, ran
    assert (waits._status(here, task), task.assigned_agent_id, task.description) == ("assigned", other,
                                                                                     f438.NEW_BRIEF)
    assert task.planning_data["previous_briefs"][-1]["description"] == f438.OLD_BRIEF


# ── What stays as it was ────────────────────────────────────────────────────────────

def test_review_to_inbox_on_a_no_agent_card_still_raises_its_card_and_runs_on_the_click(here):
    """Scope: a move to the Inbox is not the Reject; its card is raised and its click moves it."""
    task = _no_agent_card(here)
    params = {"task_id": waits._number(task), "status": "inbox"}
    asked = waits._call(here, params, waits._owners_chat())
    assert waits._asked(asked), asked
    f438._click(here, asked)

    ran = waits._call(here, params, waits._owners_chat())

    assert ran["success"] is True, ran
    assert waits._status(here, task) == "inbox"


def test_a_send_back_on_a_card_with_its_agent_still_raises_one_card(desk):
    task = waits._card(desk, "review")

    reply = waits._call(desk, {"task_id": waits._number(task), "status": "assigned", "note": NOTE},
                        waits._owners_chat())

    assert waits._asked(reply) and "left_out" not in reply, reply
    assert "Left off" not in reply["message"] and len(waits._grants(desk)) == 1


# ── A bulk send-back ────────────────────────────────────────────────────────────────

def test_a_bulk_send_back_drops_only_the_card_that_cannot_go_back(desk):
    sendable, stuck = waits._card(desk, "review"), _approval_card(desk)
    ok, left = waits._number(sendable), waits._number(stuck)

    reply = waits._call(desk, {"task_ids": [ok, left], "status": "assigned", "note": NOTE}, waits._owners_chat())

    assert waits._asked(reply), reply
    assert reply["params"]["task_ids"] == [ok]                                    # the click sends back only #ok
    assert f"Left off this card: {left}, which is a card for the owner's approval" in reply["message"]
    assert [item["card"] for item in reply["left_out"]] == [left]
    granted = f438._granted(desk, reply)
    assert granted["task_ids"] == [ok]
    assert waits._status(desk, sendable) == "review" and waits._status(desk, stuck) == "review"


def test_a_bulk_move_whose_only_send_back_cannot_go_is_refused_saying_the_rest_did_not_move(desk):
    """The card is raised for send-backs only: an Inbox card given to its agent is free, so the
    refusal says it was not moved and how to move it, rather than asking the owner to click for it."""
    stuck, fresh = _no_agent_card(desk), f1094._ticket(desk, status="inbox", assigned_agent_id=desk.agent)

    reply = waits._call(desk, {"task_ids": [waits._number(stuck), waits._number(fresh)], "status": "assigned",
                               "note": NOTE}, waits._owners_chat())

    assert _refused_before_any_card(reply), reply
    assert NO_AGENT in reply["error"] and "other cards were not moved either" in reply["error"]
    assert waits._grants(desk) == [] and waits._status(desk, fresh) == "inbox"


def test_a_bulk_send_back_none_of_whose_cards_can_go_back_raises_no_card(desk):
    first, second = _approval_card(desk), _no_agent_card(desk)

    reply = waits._call(desk, {"task_ids": [waits._number(first), waits._number(second)], "status": "assigned",
                               "note": NOTE}, waits._owners_chat())

    assert _refused_before_any_card(reply), reply
    assert reply["error"].startswith("Nothing was asked or done: no card this call names can go back to an agent.")
    assert waits._number(first) in reply["error"] and waits._number(second) in reply["error"]
    assert waits._grants(desk) == []


# ── One rule for the board's Reject and the chat's card ─────────────────────────────

def test_send_back_keeps_the_boards_codes(desk):
    """send_back_refusal is the Reject's own rule: the board's codes and words, unchanged."""
    from types import SimpleNamespace

    from api.board_tasks import send_back
    from services.send_back_rules import send_back_refusal

    inbox = _no_agent_card(desk, status="inbox")
    no_agent = _no_agent_card(desk)
    gone = f1094._ticket(desk, status="review", source_type="recipe", source_id="exec-gone", result=waits.DRAFT,
                         completed_at=datetime.now(timezone.utc))
    ctx = SimpleNamespace(workspace_id=desk.ws, user_id=None)
    expected = {inbox.id: 422, no_agent.id: 422, gone.id: 409}

    for task in (inbox, no_agent, gone):
        refused = send_back_refusal(desk.db, task)
        with pytest.raises(HTTPException) as raised:
            send_back(desk.db, ctx, task, NOTE, by="user:7")
        assert (raised.value.status_code, raised.value.detail) == refused
        assert refused[0] == expected[task.id]
    assert send_back_refusal(desk.db, inbox)[1] == "Only a ticket in review or done can be sent back (currently: inbox)"
    assert send_back_refusal(desk.db, no_agent)[1] == NO_AGENT
    assert send_back_refusal(desk.db, waits._card(desk, "review")) is None


def test_an_agent_the_edit_gives_counts_before_the_card():
    """The card is read as the edit leaves it (board_drag_rules.as_patched): the agent it gives counts."""
    from types import SimpleNamespace

    from modules.tools.discovery.send_back_checks import _as_given

    card = SimpleNamespace(status="review", assigned_agent_id=None, title="Wholesale reply")
    assert _as_given({"agent_id": 41}, card).assigned_agent_id == 41
    assert _as_given({"agent_id": "#41"}, card).assigned_agent_id == 41
    assert _as_given({}, card) is card
