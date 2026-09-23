"""F108 (night 3) — a reply may not say an action is done when no action did it.

The persona's morning report: "I've approved the mission. It's now running" —
it wasn't; "I've noted that the Taster plan is now £14" with no tool used;
"I've put your newsletter on the board" naming the owner's own ticket. F099's
check (a source that did not run) and #746's (actions told in prose with no
tool call at all) both missed these: other tools had run, or the claim was one
plain sentence. Now a claim of a done action with no action that does it
succeeding this turn gets the notice, on both reply paths.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.tool_loop import CLAIMED_ACTION_NOTICE, ToolLoopExecutor, claimed_action_not_done

NIGHT_3 = [
    ("I've approved the mission. It's now running.", "approved", "platform_approve_mission"),
    ("I've noted that the Taster plan is now £14.", "noted", "platform_store_memory"),
    ("I've put your newsletter on the board.", "put on the board", "platform_create_task"),
    ("Done — I've created a new agent called REPORT GENERATOR.", "created", "platform_create_agent"),
    ("I've emailed Declan the invoice.", "sent", "GMAIL_SEND_EMAIL"),
    ("I've cancelled the Friday schedule.", "deleted", "platform_cancel_scheduled_task"),
    ("I've renamed task 12 for you.", "changed", "platform_update_task"),
]


@pytest.mark.parametrize("reply, claim, backing", NIGHT_3, ids=[c for _, c, _ in NIGHT_3])
def test_a_claim_with_no_action_behind_it_is_named(reply, claim, backing):
    assert claimed_action_not_done(reply, set()) == claim
    assert claimed_action_not_done(reply, {"platform_list_tasks", "search_knowledge"}) == claim  # other tools ran
    assert claimed_action_not_done(reply, {backing}) is None


@pytest.mark.parametrize("reply", [
    "I haven't approved it yet — shall I?",
    "I approved it yesterday when you asked.",
    "As I've noted before, the price is £12.",
    "Want me to put it on the board?",
    "I can create an agent for that if you like.",
    "The mission is waiting for your approval.",
    "",
])
def test_a_denial_an_offer_or_a_back_reference_is_not_a_claim(reply):
    assert claimed_action_not_done(reply, set()) is None


def test_a_write_that_failed_does_not_back_a_claim():
    """F103: "Memory NOT saved" must not let the reply say it noted it."""
    tracker = ToolExecutionTracker()
    store = {"action": "platform_store_memory", "params": {"content": "Taster plan is £14"}}
    tracker.record_outcome("platform_execute", store, {"success": False, "error": "Memory NOT saved — …"})
    assert tracker.succeeded == set()
    assert claimed_action_not_done("I've noted that the Taster plan is now £14.", tracker.succeeded) == "noted"
    tracker.record_outcome("platform_execute", store, {"success": True, "message": "Stored in memory"})
    assert tracker.succeeded == {"platform_store_memory"}
    assert claimed_action_not_done("I've noted that the Taster plan is now £14.", tracker.succeeded) is None


def test_the_notice_says_what_did_not_happen():
    assert CLAIMED_ACTION_NOTICE.format(claim="approved").startswith(
        "This reply says something was approved, but no action that does that ran in this reply")


# ── the loop records what succeeded ─────────────────────────────────────────

class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


def _call(action):
    return {"id": f"call_{action}", "type": "function",
            "function": {"name": "platform_execute", "arguments": '{"action": "%s", "params": {}}' % action}}


def test_the_loop_records_the_actions_that_succeeded():
    async def tools(name, args, call_id, workspace_id):
        return {"success": args["action"] != "platform_approve_mission", "error": "not yours to approve"}

    # the claim is nudged once (test_f108_b); the retry says what happened
    executor = ToolLoopExecutor(llm_callback=_Model(LLMResponse(content="I've approved the mission.", tool_calls=None),
                                                    LLMResponse(content="The approval was refused.", tool_calls=None)),
                                tool_callback=tools, max_iterations=5)
    first = LLMResponse(content="", tool_calls=[_call("platform_get_mission"), _call("platform_approve_mission")])
    asyncio.run(executor.run(initial_response=first, messages=[{"role": "user", "content": "approve it"}],
                             tools=[{"type": "function", "function": {"name": "platform_execute"}}], workspace_id="ws"))
    assert executor.tracker.succeeded == {"platform_get_mission"}      # the refused approval is not "done"
    assert claimed_action_not_done("I've approved the mission.", executor.tracker.succeeded) == "approved"


# ── both reply paths ask ────────────────────────────────────────────────────

def test_the_chat_notice_covers_a_claimed_action():
    from consumers.chatbot.service import unexecuted_claims_notice

    tools = [{"type": "function", "function": {"name": "platform_execute"}}]
    reply = "I've approved the mission. It's now running."
    assert unexecuted_claims_notice(reply, tools, {"platform_get_mission"}, any_tool_ran=True,
                                    done={"platform_get_mission"}) == CLAIMED_ACTION_NOTICE.format(claim="approved")
    assert unexecuted_claims_notice(reply, tools, {"platform_approve_mission"}, any_tool_ran=True,
                                    done={"platform_approve_mission"}) is None
    assert unexecuted_claims_notice(reply, None, set(), any_tool_ran=False, done=set()) is None   # no tools offered


def test_both_reply_paths_pass_what_succeeded():
    from consumers.chatbot import service

    loop = inspect.getsource(service.StreamingChatService._stream_tool_loop)
    assert "done=executor.tracker.succeeded" in loop
    turn = inspect.getsource(service.StreamingChatService._stream_response_with_agent_scoped)
    assert "done={name for name, _args in _prefetched}" in turn
