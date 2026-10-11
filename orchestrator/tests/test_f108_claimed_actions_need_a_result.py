"""F108 (night 3) — a reply may not say an action is done when no action did it.

The persona's morning report: "I've approved the mission. It's now running" —
it wasn't; "I've noted that the Taster plan is now £14" with no tool used;
"I've put your newsletter on the board" naming the owner's own ticket. F099's
check (a source that did not run) and #746's (actions told in prose with no
tool call at all) both missed these: other tools had run, or the claim was one
plain sentence. Now a claim of a done action with no action that does it
succeeding this turn gets the notice, on both reply paths.

PRD-256 FX-007: the claim is read by the receipts' rule (``claims_backed``), not a regex
family: ``nudged`` is what the loop's nudge names, ``line`` the not-done line above the
answer (tests/helpers_receipts_rule.py).
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.helpers_receipts_rule import call, line, nudged

EMAIL_SENT = ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {"to": "declan@example.com"}},
              {"successful": True})
READS = ("platform_list_tasks", call("search_knowledge", {"query": "newsletter"}))
# The claim's verb names it when its family is known; a verb in no family ("put") is no nudge's
# (P256-FIX-RVW-7: the line above the answer still says it, test_prd256_fix_stream_truth).
NIGHT_3 = [
    ("I've approved the mission. It's now running.", "approved", "platform_approve_mission"),
    ("I've noted that the Taster plan is now £14.", "noted", "platform_store_memory"),
    ("I've put your newsletter on the board.", None, "platform_create_task"),
    ("Done — I've created a new agent called REPORT GENERATOR.", "created", "platform_create_agent"),
    ("I've emailed Declan the invoice.", "emailed", EMAIL_SENT),
    ("I've cancelled the Friday schedule.", "cancelled", "platform_cancel_scheduled_task"),
    ("I've renamed task 12 for you.", "renamed", "platform_update_task"),
]


@pytest.mark.parametrize("reply, claim, backing", NIGHT_3, ids=[c for _, c, _ in NIGHT_3])
def test_a_claim_with_no_action_behind_it_is_named(reply, claim, backing):
    assert nudged(reply) == claim
    assert nudged(reply, *READS) == claim                                   # other tools ran: reads back nothing
    assert nudged(reply, backing) is None


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
    assert nudged(reply) is None and line(reply) is None


def test_a_write_that_failed_does_not_back_a_claim():
    """F103: "Memory NOT saved" must not let the reply say it noted it."""
    store = {"content": "Taster plan is £14"}
    refused = call("platform_store_memory", store, {"success": False, "error": "Memory NOT saved — …"})
    stored = call("platform_store_memory", store, {"success": True, "message": "Stored in memory"})
    said = "I've noted that the Taster plan is now £14."

    assert nudged(said, refused) == "noted"
    assert nudged(said, refused, stored) is None


def test_the_notice_says_what_did_not_happen():
    # F314 (night 9): in Auto's own plain words, never "This reply says something was approved …";
    # FX-006/FX-007: named when another write went through, plain when nothing did
    said = "I've approved the mission."
    assert line(said, "platform_store_memory") == (
        "Just to be clear: I haven't approved anything in this reply. Ask me again if you want it done.")
    assert line(said) == ("Just to be clear: I haven't done that yet, and nothing has changed. "
                          "Ask me again if you want it done.")


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
    from consumers.chatbot.receipts import unbacked_claim

    assert unbacked_claim("I've approved the mission.", executor.tracker.outcomes) == "approved"


# ── both reply paths ask ────────────────────────────────────────────────────

def test_the_chat_notice_covers_a_claimed_action():
    """PRD-256 FX-005: the claim is the receipts' line, the one producer of it; the turn's
    notice never says it again (test_prd256_fix_stream_truth covers both reply paths)."""
    from consumers.chatbot.receipts import DONE, NOTHING_DONE_LINE, READ, WRITE, honesty_lines
    from consumers.chatbot.service import unexecuted_claims_notice

    tools = [{"type": "function", "function": {"name": "platform_execute"}}]
    reply = "I've approved the mission. It's now running."
    read = {"action": "platform_get_mission", "kind": READ, "status": DONE}
    approved = {"action": "platform_approve_mission", "kind": WRITE, "status": DONE}
    assert honesty_lines([read], reply) == [NOTHING_DONE_LINE]
    assert honesty_lines([read, approved], reply) == []
    assert unexecuted_claims_notice(reply, tools, {"platform_get_mission"}, any_tool_ran=True) is None
    assert unexecuted_claims_notice(reply, None, set(), any_tool_ran=False) is None   # no tools offered


def test_both_reply_paths_leave_the_claim_to_the_receipts():
    from consumers.chatbot import service

    loop = inspect.getsource(service.StreamingChatService._stream_tool_loop)
    turn = inspect.getsource(service.StreamingChatService._stream_response_with_agent_scoped)
    for path in (loop, turn):
        call = path[path.index("unexecuted_claims_notice("):]
        assert "done=" not in call[:call.index(")\n")]
    assert service.StreamingChatService._stream_tool_loop.__code__.co_qualname == \
        "the_loop_writes_receipts.<locals>.wrapped"
