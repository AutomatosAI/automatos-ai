"""PRD-256 P256-FIX-RVW-7: the in-loop nudge reads Auto's chat turn and an agent's run apart.

FX-007 put the receipts rule behind F108's nudge (tool_loop._recover_claimed_action) on every
ToolLoopExecutor run, agent runs included (agent_factory's loop). The base had held an agent's
draft only to its first-person action claims (``promises=False``): a customer draft speaks in
its writer's voice. After FX-007, "Hi Maya, your refund has been processed." or "It has been
cancelled, and you won't be charged." in an agent's draft, with tools offered and no call, got
"no tool call in this turn did that … Make the call now", inviting the agent to perform the
cancellation the draft only describes. And any "I've <verb>ed" outside a family ("I've prepared
a summary", "I've kept it short") cost Auto and agents an extra model call.

Now:
- an agent's run (``promises`` False, or the turn's lane when None) is nudged only for its
  first-person claims of a known family ("I've cancelled your subscription");
- Auto's chat turn (``promises`` True, or the chat lane) is nudged for a passive too;
- a participle in no family is never nudged; "Done.", "is now live" and "it's been done" still are
  (as "done"). The line above the answer is unchanged: it still reads every claim.
- F201's customer-draft check (services/draft_guides.py) is unchanged.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from consumers.chatbot.receipts import NOTHING_DONE_LINE
from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_BOARD_TASK, LANE_CHAT, usage_scope
from modules.tools.execution.nudges import is_nudge
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.helpers_receipts_rule import line, nudged

TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {}}}]
CANCELLED = "It has been cancelled, and you won't be charged."
PROCESSED = "Hi Maya, your refund has been processed."
OWN_CLAIM = "I've cancelled your subscription, so you won't be charged again."
PREPARED = "I've prepared a summary for you."
NO_FAMILY = [PREPARED, "I've kept it short, as you asked.", "I've put your newsletter on the board."]
ONLY_DONE = ["Done.", "The new page is now live.", "It's been done.", "It's now on your board."]


class _Model:
    """The model's replies, in order; it records each call so an extra one shows."""

    def __init__(self, *texts):
        self.queue = [LLMResponse(content=text, tool_calls=None) for text in texts]
        self.calls = 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


async def _no_tool(name, args, call_id, workspace_id):
    raise AssertionError("no tool is called in these runs")


def _run(said, *retries, promises=None):
    """One loop run over a reply with no call; the model's calls and the nudges it was sent."""
    model = _Model(*retries)
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_no_tool, max_iterations=5, promises=promises)
    messages = [{"role": "user", "content": "Draft a reply to Maya about her cancelled subscription."}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content=said, tool_calls=None),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model.calls, [m["content"] for m in messages if is_nudge(m)]


# ── the rule ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("draft", [CANCELLED, PROCESSED, "It's been cancelled for you.", "Your order is now live."])
def test_an_agents_passive_is_its_writers_voice(draft):
    assert nudged(draft, promises=False) is None


def test_an_agents_first_person_claim_of_a_family_is_still_nudged():
    assert nudged(OWN_CLAIM, promises=False) == "cancelled"
    assert nudged(f"{CANCELLED} I've also emailed the receipt.", promises=False) == "emailed"


def test_autos_passive_is_still_nudged():
    assert nudged(CANCELLED) == "cancelled"


@pytest.mark.parametrize("said", NO_FAMILY)
def test_a_participle_in_no_family_is_never_nudged(said):
    assert nudged(said) is None and nudged(said, promises=False) is None
    assert line(said) == NOTHING_DONE_LINE                     # the line above the answer is unchanged


def test_the_reply_written_below_is_no_claim_at_all():
    said = "I've written the reply below for you to check."
    assert nudged(said) is None and nudged(said, promises=False) is None and line(said) is None


@pytest.mark.parametrize("said", ONLY_DONE)
def test_a_claim_that_says_only_done_is_nudged_as_done(said):
    assert nudged(said) == "done"
    assert nudged(said, "platform_create_task") is None         # any done write backs it


def test_the_first_claim_the_nudge_can_name_is_named():
    assert nudged(f"{PREPARED} I've also approved the mission.") == "approved"


def test_the_turns_lane_decides_when_the_run_does_not_say():
    with usage_scope(request_type=LANE_CHAT, execution_id="chat:rvw7"):
        assert nudged(CANCELLED, promises=None) == "cancelled"
    with usage_scope(request_type=LANE_BOARD_TASK, execution_id="board_task:rvw7"):
        assert nudged(CANCELLED, promises=None) is None
        assert nudged(OWN_CLAIM, promises=None) == "cancelled"


# ── the loop ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("draft", [CANCELLED, PROCESSED])
def test_an_agents_customer_draft_costs_no_extra_model_call(draft):
    result, calls, nudges = _run(draft, promises=False)
    assert calls == 0 and nudges == [] and result.response.content == draft


def test_an_agent_run_in_its_own_lane_is_not_nudged_for_its_draft():
    with usage_scope(request_type=LANE_BOARD_TASK, execution_id="board_task:rvw7"):
        result, calls, nudges = _run(CANCELLED)
    assert calls == 0 and nudges == [] and result.response.content == CANCELLED


def test_an_agents_own_claim_is_nudged_once():
    plain = "I haven't cancelled the subscription: I have no tool for it."
    result, calls, nudges = _run(OWN_CLAIM, plain, promises=False)
    assert calls == 1 and len(nudges) == 1 and "says something was cancelled" in nudges[0]
    assert result.response.content == plain


def test_autos_passive_is_nudged_once_in_the_chat_lane():
    plain = "It hasn't been cancelled yet: shall I cancel it?"
    with usage_scope(request_type=LANE_CHAT, execution_id="chat:rvw7"):
        result, calls, nudges = _run(CANCELLED, plain)
    assert calls == 1 and "says something was cancelled" in nudges[0] and result.response.content == plain


def test_i_ve_prepared_a_summary_costs_no_extra_model_call():
    result, calls, nudges = _run(PREPARED, promises=True)
    assert calls == 0 and nudges == [] and result.response.content == PREPARED


def test_the_nudge_passes_the_runs_voice():
    source = inspect.getsource(ToolLoopExecutor._recover_claimed_action)
    assert "unbacked_claim(text, self.tracker.outcomes, promises=self.promises)" in source


# ── F201 is unchanged ───────────────────────────────────────────────────────

def test_the_customer_draft_check_still_reads_the_draft():
    """services/draft_guides.check_before_sending is not this story's: it still reads a passive
    and a verb in no family in the draft before it is sent."""
    from services.draft_guides import check_before_sending

    brief = "Email from Rosie Tanner, club member: she was charged twice. Please draft a reply."
    assert "says something was cancelled" in check_before_sending(brief, CANCELLED, ["platform_load_skill"])
    assert "says something was done" in check_before_sending(brief, PROCESSED, [])
    assert check_before_sending(brief, CANCELLED, ["platform_cancel_subscription"]) is None
