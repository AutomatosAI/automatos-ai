"""F324 (night 9b): "Can you make sure the whole team knows that, so I don't have to tell each of
them?" got "I've stored the information … in my memory. This will ensure that all agents,
including the Support Agent, are aware of this change going forward" (5e404e17), with only
platform_store_memory behind it; Support #1962 and the Analyst #1963 then said nothing was
changing. Now the turn is told that Auto's memory is its own and the owner's documents are what
every agent reads, so Auto says so and asks to write a note; and a reply that says the agents
know, with no call that put it where they read it, is nudged once and then corrected.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.claim_check import not_done
from consumers.chatbot.team_corrections import TELL_THE_TEAM_NOTE, asks_to_tell_the_team, tells_the_team_honestly
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.execution.action_claims import claimed_action_not_done
from modules.tools.execution.shop_and_team_claims import TEAM_LABEL, TEAM_LINE
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187

OWNER = ("My Support Agent just told me nothing is changing for Quay Coffee House's terms. I told you this "
         "afternoon they move to 30-day terms from November. Can you make sure the whole team knows that, so I "
         "don't have to tell each of them?")
SAID = ("You're right, Gerard. My apologies for the oversight. I've stored the information that Quay Coffee House "
        "moves to 30-day payment terms from November in my memory. This will ensure that all agents, including the "
        "Support Agent, are aware of this change going forward.")
TOLD = "This will ensure that all agents, including the Support Agent, are aware of this change going forward."
HONEST = ("I've kept this in my own memory, but your agents read your documents, not my memory. Shall I write it "
          "into a short note in your documents so every agent knows?")


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.mark.parametrize("said", [OWNER, "Please tell all the agents that Quay moves to 30-day terms.",
                                  "Let everyone know the Guji has landed."])
def test_asking_auto_to_tell_the_team_is_seen(said):
    assert asks_to_tell_the_team(said) is True


@pytest.mark.parametrize("said", ["Let me know how many boxes everyone ordered.", "Can you tell me what the team "
                                  "found?", "Which cafés are on 14-day terms?"])
def test_a_question_to_auto_is_not(said):
    assert asks_to_tell_the_team(said) is False


def test_the_turn_is_told_that_auto_s_memory_is_its_own():
    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": "passages from wholesale-terms-2026.md"})
        yield "searched"

    def run(chat):
        messages = [{"role": "user", "content": OWNER}]

        async def turn():
            return [f async for f in tells_the_team_honestly(retrieval_first)(chat, OWNER, messages, None, "c", [])]
        return asyncio.run(turn()), messages

    frames, messages = run(NS(widget_mode=False))
    assert frames == ["searched"] and messages[-1] == {"role": "system", "content": TELL_THE_TEAM_NOTE}
    assert "platform_upload_document" in TELL_THE_TEAM_NOTE and "only when they say yes" in TELL_THE_TEAM_NOTE
    assert len(run(NS(widget_mode=True))[1]) == 2                         # a widget visitor: no note


def test_saying_the_team_knows_after_only_a_memory_is_a_claim():
    assert claimed_action_not_done(SAID, {"platform_store_memory"}, promises=True) == TEAM_LABEL
    assert claimed_action_not_done(SAID, {"platform_store_memory", "platform_upload_document"}, promises=True) is None
    assert claimed_action_not_done(HONEST, {"platform_store_memory"}, promises=True) is None   # an offer
    assert claimed_action_not_done("Your agents don't know yet.", set(), promises=True) is None
    assert claimed_action_not_done(SAID, {"platform_store_memory"}, promises=False) is None    # an agent's draft


def test_the_reply_is_nudged_then_corrected_if_it_still_says_so():
    model = f187._Model(TOLD)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(TOLD), owner=OWNER)

    (sent,) = model.sent                                                  # the loop's one nudge
    assert "something was told to the whole team" in sent[-1]["content"]
    assert final["_f187"].claim == TEAM_LABEL
    assert final["_f187"].correction == not_done(TEAM_LABEL) == TEAM_LINE
    assert TEAM_LINE.startswith("Just to be clear: your agents haven't been told.")


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first
    for _ in range(5):                                                    # under F241, F307, F303, F316, F317
        inner = inner.__wrapped__
    assert inner.__code__ is tells_the_team_honestly(lambda: None).__code__
