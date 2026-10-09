"""F319 (night 9b): Auto's claim line never says the opposite of what a call did.

- Chat ba4628f8: "I've sent card 67.1 back to the Analyst with your instructions." after
  platform_update_task_status {task_id: 67.1, status: "assigned", note} went through
  (#1922, times_sent_back 2), and the reply ended "Just to be clear: I didn't send
  anything in this reply." The reply was cut into sentences at "67.", which made "I've
  sent card 67." an email claim no card move backs.
- Chat a0475f96: "I have already assigned the Shopify Inventory Watchdog (agent ID 342) to
  both steps" after two platform_update_playbook_step calls with agent_id 342 went through,
  then "Just to be clear: I didn't assign anything in this reply." No action with "assign"
  in its name had run. The false nudge made the model call the steps again and repeat its
  paragraph.
- Chat 6d4e45fe: 'Card #0044, "Minimum wholesale order and cut-off," has been cancelled.'
  after the move to done was refused: the quoted title hid the claim, so nothing nudged
  or corrected it, and #1898 stayed in Review.
"""
from __future__ import annotations

import pytest

from modules.tools.execution.action_claims import claimed_action_not_done
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

STEP_SENT_BACK = ("I've sent card 67.1 back to the Analyst with your instructions. It's now in progress and the "
                  "Analyst will work on providing the information you requested.")
STEPS_ASSIGNED = ("I apologize for the confusion. I have already assigned the Shopify Inventory Watchdog (agent ID "
                  "342) to both steps of your \"Monday green stock\" playbook and triggered the playbook to run once.")
CANCELLED = 'No problem, Gerard! Card #0044, "Minimum wholesale order and cut-off," has been cancelled.'
OK = {"success": True}


def _did(*calls):
    """The turn's calls as the chat records them: (tool, its arguments, its result)."""
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return tracker.succeeded


def _step(agent_id, index, result=OK):
    return ("platform_execute", {"action": "platform_update_playbook_step",
                                 "params": {"playbook_id": 115, "step_index": index, "agent_id": agent_id}}, result)


def test_a_step_sent_back_by_its_number_backs_the_send_back():
    done = _did(("platform_update_task_status", {"task_id": 67.1, "status": "assigned",
                                                 "note": "Tell me how much Guji and Nariño green we have."}, OK))

    assert claimed_action_not_done(STEP_SENT_BACK, done, promises=True) is None     # night 9b: "sent"
    assert claimed_action_not_done(STEP_SENT_BACK, set(), promises=True) == "sent back"


def test_an_agent_written_onto_playbook_steps_backs_the_assignment():
    refused = {"success": False, "error": "Agent 12 not found in this workspace"}
    done = _did(_step(12, 0, refused), _step(12, 1, refused), _step(342, 0), _step(342, 1),
                ("platform_execute", {"action": "platform_execute_playbook", "params": {"playbook_id": 115}}, OK))

    assert claimed_action_not_done(STEPS_ASSIGNED, done, promises=True) is None     # night 9b: "assigned"
    assert claimed_action_not_done(STEPS_ASSIGNED, _did(_step(12, 0, refused)), promises=True) == "assigned"


@pytest.mark.parametrize("word", ["send back", "rejected"])
def test_a_send_back_said_in_the_boards_words_backs_it(word):
    done = _did(("platform_execute", {"action": "platform_update_task_status",
                                      "params": {"task_id": "0081", "status": word, "note": "Only September."}}, OK))

    assert claimed_action_not_done("I've sent card #0081 back to the Shopify Business Analyst.", done,
                                   promises=True) is None


def test_a_card_a_call_says_it_sent_back_backs_it_whatever_the_call():
    """The chat wraps the call's answer as raw_result; the answer says the card went back."""
    done = _did(("platform_execute", {"action": "platform_review_mission_step", "params": {"step": "0067.1"}},
                 {"success": True, "raw_result": {"success": True, "status": "in_progress", "sent_back": True}}))

    assert claimed_action_not_done("I've sent #0067.1 back to the Analyst.", done, promises=True) is None


def test_a_cancel_told_with_the_cards_title_is_checked():
    refused = {"success": False, "error": "The owner said cancel, not approve. Nothing was done."}
    done = _did(("platform_update_task_status", {"status": "done", "task_id": "0044"}, refused))

    assert claimed_action_not_done(CANCELLED, done, promises=True) == "deleted"     # night 9b: never caught
    cancelled = _did(("platform_update_task_status", {"task_id": "0044", "status": "cancelled"}, OK))
    assert claimed_action_not_done(CANCELLED, cancelled, promises=True) is None


def test_a_reply_that_did_nothing_is_still_corrected():
    assert claimed_action_not_done("I've assigned a task to the Shopify Operations Manager.", set(),
                                   promises=True) == "assigned"
    assert claimed_action_not_done("I've sent card 67.1 back to the Analyst.", set(), promises=True) == "sent back"


# ── Chat 8578eeaf: a card named as a source is not a card to copy ─────────────────────

REORDER = ("My Business Analyst just worked out we're about 97 kg short of Kirinyaga green before the Christmas "
           "roast (card 1970). Can you get the Operations Manager to draft a reorder email to Maya at Tidewater, "
           "enough to cover it, in whole sacks? Draft only, don't send anything.")


@pytest.fixture
def turn(monkeypatch):
    """The owner's words this turn and the card they name, as owner_turn would read them.
    PRD-256 US-004: Auto's last reply is no longer patched: the guard read it only to judge a
    "yes" as an approval, which is the owner's click now."""
    from types import SimpleNamespace as NS

    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import NamedCard, OwnerTurn

    def _set(latest):
        task = NS(id=1970, title="Will the Kirinyaga last through Christmas?", status="done", source_type="user",
                  description="")
        card = NamedCard(ref="#0113", seq=113, step=None, task=task)
        monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: OwnerTurn(latest=latest, earlier="",
                                                                               cards=(card,)))

    monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: ())
    return _set


def _refused(action, **params):
    from uuid import uuid4

    from modules.tools.discovery.follows_the_owner import refusal_for

    return refusal_for(None, uuid4(), action, params, {"conversation_id": str(uuid4())})


def test_new_work_for_another_agent_is_made_when_a_card_is_only_its_source(turn):
    turn(REORDER)

    assert _refused("platform_create_task", title="Draft reorder email for Kirinyaga AA",
                    assigned_agent_name="Shopify Operations Manager",
                    description="Draft an email to Maya at Tidewater to reorder Kirinyaga AA, 97 kg short, in whole "
                                "sacks. Do not send it.") is None              # night 9b: refused as a copy


def test_a_new_card_for_a_card_the_owner_acts_on_is_still_a_copy(turn):
    turn("Give card 1970 to the Shopify Operations Manager.")

    assert "already on the owner's board" in _refused("platform_create_task", title="Kirinyaga reorder",
                                                      assigned_agent_name="Shopify Operations Manager")
