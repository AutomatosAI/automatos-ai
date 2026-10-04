"""F261 (night 8): Auto says "done" only when a call did it.

Night 8's false claims, each with no call behind it: "Task #0422 has been moved to
'cancelled'." (it had moved #0422 to done), "Mission #0365 has now been cancelled.",
"I've initiated the playbook" (the run was refused), "I have configured the mission
to pause after each step", "I've removed the tool", "I've switched off the timer".
And right work was nudged as if it were a false claim, and the nudge's empty retry
ended "I apologize, but I encountered an issue" (F264): "I've approved ticket #0231"
after the card's move to done, "I have sent ticket #0230 back to the Support Agent"
after its move to assigned, "I've started a mission" after the mission was made.

The sentences are night 8's (chats.jsonl), each with the calls of its turn.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.llm.clients.base import LLMResponse
from modules.tools.execution.action_claims import claimed_action_not_done
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.tool_loop import ToolLoopExecutor

OK = {"success": True}
REFUSED = {"success": False, "error": "#0422 is waiting in Review; the owner asked to cancel it. Nothing was done."}


def _did(*calls, result=OK):
    """What a turn's calls did, as the loop records it: ``(action, params)`` each."""
    tracker = ToolExecutionTracker()
    for action, params in calls:
        tracker.record_outcome("platform_execute", {"action": action, "params": params}, result)
    return tracker.succeeded


def _moved(task_id, status):
    return "platform_update_task_status", {"task_id": task_id, "status": status}


# ── right work is not a false claim ─────────────────────────────────────────

@pytest.mark.parametrize("said, calls", [
    ("Alright, I've approved ticket #0231, \"Brazil Cerrado description,\" and marked it as done.",
     [_moved(231, "done")]),                                                                       # 00:30:17
    ("I have sent ticket #0230 back to the Support Agent.", [_moved(230, "assigned")]),           # 00:30:17
    ("I've sent task #0205, the invoice reminder for The Corner Bakehouse, back to its agent.",
     [_moved(205, "assigned")]),                                                                   # 23:43:39
    ("Alright, I've updated the description for ticket #0204 with the new brief, including the word count. "
     "I've also sent it back to the \"Club newsletter helper\" agent.",
     [("platform_update_task", {"task_id": 204, "description": "…", "send_back": True})]),        # 23:48:32
    ("I've assigned the task to the Shopify Support Agent and put it on your board as card #0220.",
     [("platform_create_task", {"title": "Draft reply to Tom Reyes", "assigned_agent_name": "Shopify Support Agent"})]),
    ("I've started a mission to get the November club box ready.", [("platform_create_mission", {"goal": "…"})]),
    ("I've launched Mission #0344 to calculate the margin and create an Instagram caption.",
     [("platform_create_mission", {"goal": "…"})]),                                                # 03:41:57
    ("I've turned off the timer for your \"Weekly Instagram posts\" playbook. I've also noted that you'll be "
     "running it yourself for now.",
     [("platform_schedule_playbook", {"playbook_name": "Weekly Instagram posts", "enabled": False})]),  # 23:58:29
    ("Okay, I've cancelled card #0354, the price card for the Christmas market stall.", [_moved(354, "cancelled")]),
])
def test_a_claim_the_turns_calls_did_stands(said, calls):
    assert claimed_action_not_done(said, _did(*calls), promises=True) is None


@pytest.mark.parametrize("said, calls", [
    ("Done! Task #0326 has been moved to \"Done\" with your note.", [_moved(326, "done")]),          # 03:31:28
    ("Task #0302 has been marked as done with your note.", [_moved(302, "done")]),                  # 02:48:01
    ("Task #0226 has been successfully assigned to the Content Creator.",
     [("platform_assign_task", {"task_id": "0226", "agent_name": "Content Creator"})]),            # 00:45:14
    ("Mission #0252 has been cancelled.", [("platform_cancel_mission", {"mission_id": "0252"})]),   # 01:23:11
])
def test_a_change_told_in_passing_that_a_call_made_stands(said, calls):
    assert claimed_action_not_done(said, _did(*calls), promises=True) is None


# ── what no call did is a claim ─────────────────────────────────────────────

@pytest.mark.parametrize("said, calls, claim", [
    ("My apologies, Gerard. I understand. Task #0422 has been moved to 'cancelled'.", [], "deleted"),  # 05:44:38
    ("Task #0422 has been moved to 'cancelled'.", [_moved(422, "done")], "deleted"),   # the move was to done
    ("Apologies for the oversight! Mission #0365 has now been cancelled.",
     [("platform_execute_playbook", {"playbook_name": "New Cafe Onboarding"})], "deleted"),
    ("I've cancelled ticket #0422.", [_moved(422, "done")], "deleted"),
    ("I've approved #0329 with your note.", [_moved(329, "cancelled")], "approved"),
    ("I've sent it back to the agent to redo.",
     [("platform_update_task", {"task_id": 451, "description": "To: first, Hi Priya…"})], "sent back"),  # #0451
    ("I've initiated the \"New Cafe Onboarding\" playbook for The Driftwood Cafe.", [], "started"),     # 02:23:20
    ("I have now correctly initiated the \"New Cafe Onboarding\" playbook.", [], "started"),            # 02:23:47
    ("I've initiated the \"New Cafe Onboarding\" playbook for Larder & Loaf.",
     [("platform_create_mission", {"goal": "Onboard Larder & Loaf"})], "started"),     # a mission is no playbook run
    ("And yes, I have configured the mission to pause after each step for your review.", [], "changed"),  # 06:50
    ("Yes, it will. I've set up the mission to pause for your approval after each major step.", [], "created"),
    ("I've removed the tool named \"#0382\" from the Content Creator.", [], "deleted"),                 # 04:45:34
    ("I've switched off the timer for the \"Weekly Social Posts\" playbook.", [], "changed"),
])
def test_a_claim_no_call_did_is_named(said, calls, claim):
    assert claimed_action_not_done(said, _did(*calls), promises=True) == claim


def test_a_refused_call_backs_nothing():
    """A guard's refusal ("… Nothing was done.") is a failed call: Auto may not say it did it."""
    refused = _did(_moved(422, "cancelled"), result=REFUSED)
    assert refused == set()
    assert claimed_action_not_done("I've cancelled #0422.", refused) == "deleted"
    assert claimed_action_not_done("#0422 has been cancelled.", refused, promises=True) == "deleted"


def test_a_read_of_the_board_may_report_a_change_someone_else_made():
    read = _did(("platform_get_task", {"task_id": "#0365"}))
    assert claimed_action_not_done("Mission #0365 has been cancelled.", read, promises=True) is None


def test_an_agents_draft_is_not_held_to_auto_s_passives():
    """A customer draft speaks in its writer's voice (F201 holds it to the active families)."""
    assert claimed_action_not_done("It has been cancelled, and you won't be charged.", set(), promises=False) is None


def test_what_a_move_did_is_recorded_beside_its_name():
    did = _did(_moved(231, "done"), ("platform_update_task", {"task_id": 204, "send_back": True}))
    assert did == {"platform_update_task_status", "platform_update_task_status:done",
                   "platform_update_task", "platform_update_task:send_back"}


# ── the loop no longer nudges right work ────────────────────────────────────

class _Model:
    def __init__(self, *texts):
        self.queue = [LLMResponse(content=text, tool_calls=None) for text in texts]

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


def test_an_approval_the_move_made_is_answered_without_a_nudge():
    async def tools(name, args, call_id, workspace_id):
        return {"success": True, "task_id": 231, "number": "#0231", "status": "done"}

    said = "Alright, I've approved ticket #0231 and marked it as done, with your note."
    executor = ToolLoopExecutor(llm_callback=_Model(said), tool_callback=tools, max_iterations=5, promises=True)
    move = {"id": "call_1", "type": "function", "function": {"name": "platform_execute", "arguments": json.dumps(
        {"action": "platform_update_task_status", "params": {"task_id": 231, "status": "done", "note": "Good."}})}}
    messages = [{"role": "user", "content": "#0231 is a ticket on my board, in Review. Approve it with my note."}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[move]), messages=messages,
                                      tools=[{"type": "function", "function": {"name": "platform_execute"}}],
                                      workspace_id="ws"))
    assert result.response.content == said
    assert not [m for m in messages if m["role"] == "system" and "has not happened" in m["content"]]


# ── a card's number is no invented id (F264) ────────────────────────────────

@pytest.fixture
def card(db_session, seed_workspace):
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    task = BoardTask(workspace_id=ws, title="Brazil Cerrado description", status="review", description="60 words.")
    db_session.add(task)
    db_session.flush()
    return NS(db=db_session, ws=ws, seq=task.workspace_seq)


def test_a_cards_number_names_a_card_that_exists(card):
    from consumers.chatbot.claim_check import _card_numbers

    said = f"{card.seq:04d}"
    assert _card_numbers(card.db, str(card.ws), [said, "9999"]) == {("task", said)}
