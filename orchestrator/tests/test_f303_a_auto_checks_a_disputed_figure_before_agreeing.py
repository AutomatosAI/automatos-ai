"""F303 (night 9, L106): told "The Analyst counted 11 club cancellations April to
September on card 1886. You told me 4 a minute ago. Which is right, and why did you
say 4?", Auto said the Analyst was correct and that its own 4 "was based on an
incomplete query", with no tool call at all (chat ec665ae6). Now the turn tells Auto
to check again before it agrees or disagrees, and a reply that says a figure is right
or wrong with no count, query or read behind it is nudged once, then corrected.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.claim_check import not_done
from consumers.chatbot.figure_disputes import RECHECK_NOTE, disputes_a_figure, rechecks_disputed_figures
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.execution.action_claims import claimed_action_not_done
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187

OWNER = ("The Analyst counted 11 club cancellations April to September on card 1886. You told me 4 a minute ago. "
         "Which is right, and why did you say 4?")
GAVE_WAY = ("You are right to call that out! I apologize for the discrepancy.\n\nThe Analyst's count of 11 club "
            "cancellations between April and September on card 1886 is correct.\n\nMy previous answer of 4 was "
            "based on an incomplete query of the database.")


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.mark.parametrize("said", [OWNER, "Where did 87 come from? Two cards this afternoon said 63 active.",
                                  "That's wrong, the shop system has 118 kg."])
def test_a_figure_set_against_autos_is_a_dispute(said):
    assert disputes_a_figure(said) is True


@pytest.mark.parametrize("said", ["How many club cancellations were there April to September?",
                                  "A café wants 10 kg next week: what do we charge?", "Which is right?"])
def test_a_plain_question_is_not(said):
    assert disputes_a_figure(said) is False


def test_the_turn_is_told_to_check_again_before_agreeing():
    messages = [{"role": "user", "content": OWNER}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        yield "retrieval"

    async def run(chat, said):
        return [f async for f in rechecks_disputed_figures(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run(NS(widget_mode=False), OWNER)) == ["retrieval"]
    assert messages[-1] == {"role": "system", "content": RECHECK_NOTE}
    assert "never guess why an earlier figure was different" in RECHECK_NOTE
    asyncio.run(run(NS(widget_mode=True), OWNER))
    assert len(messages) == 2                                            # a widget visitor's turn: no note


def test_giving_way_with_nothing_checked_is_a_claim():
    assert claimed_action_not_done(GAVE_WAY, set(), promises=True) == "re-checked"
    assert claimed_action_not_done(GAVE_WAY, {"platform_get_task"}, promises=True) == "re-checked"   # the card
    assert claimed_action_not_done(GAVE_WAY, {"platform_query_data"}, promises=True) is None       # a recount
    assert claimed_action_not_done("Is 11 right? I can count it again.", set(), promises=True) is None


def test_the_reply_is_nudged_to_check_then_corrected_if_it_still_gives_way():
    model = f187._Model(GAVE_WAY)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(GAVE_WAY), owner=OWNER)

    (sent,) = model.sent                                                  # the loop's one nudge
    assert "something was re-checked" in sent[-1]["content"]
    assert final["_f187"].claim == "re-checked"
    assert final["_f187"].correction is None and not_done("re-checked") == (      # PRD-256: said from receipts
        "Just to be clear: I didn't re-check that figure in this reply, so I can't yet say which one is right. "
        "Ask me again if you want it done.")


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first.__wrapped__.__wrapped__.__wrapped__  # under PRD-256, F241, F307
    assert inner.__code__ is rechecks_disputed_figures(lambda: None).__code__
