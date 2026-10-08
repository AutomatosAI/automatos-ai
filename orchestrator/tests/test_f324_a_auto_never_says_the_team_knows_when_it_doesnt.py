"""F324 (night 9b): "Can you make sure the whole team knows that, so I don't have to tell each of
them?" got "I've stored the information … in my memory. This will ensure that all agents,
including the Support Agent, are aware of this change going forward" (5e404e17), with only
platform_store_memory behind it; Support #1962 and the Analyst #1963 then said nothing was
changing. The turn was told that Auto's memory is its own (consumers/chatbot/team_corrections.py),
and "the agents know" with no call that put it where they read it was a family's claim.

PRD-256 FX-007 (D10): the lane and the family are deleted. "I've stored it in my memory" is
read by the receipts' rule against the memory write (backed, or caught when none ran); "all
agents are aware" is no report of work done and is cleared. The receipts above the answer show
the owner what reached the team: a memory note, and no document.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from consumers.chatbot.receipts import build_receipts
from core.llm.usage_context import LANE_CHAT, usage_scope
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187
from tests.helpers_receipts_rule import line, nudged, tracker_of

CHATBOT = Path(__file__).resolve().parents[1] / "consumers" / "chatbot"
OWNER = ("My Support Agent just told me nothing is changing for Quay Coffee House's terms. I told you this "
         "afternoon they move to 30-day terms from November. Can you make sure the whole team knows that, so I "
         "don't have to tell each of them?")
SAID = ("You're right, Gerard. My apologies for the oversight. I've stored the information that Quay Coffee House "
        "moves to 30-day payment terms from November in my memory. This will ensure that all agents, including the "
        "Support Agent, are aware of this change going forward.")
TOLD = "This will ensure that all agents, including the Support Agent, are aware of this change going forward."
HONEST = ("I've kept this in my own memory, but your agents read your documents, not my memory. Shall I write it "
          "into a short note in your documents so every agent knows?")
MEMORY = ("platform_store_memory",)


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def test_the_memory_write_backs_stored_and_the_receipts_show_no_document():
    assert nudged(SAID, *MEMORY) is None and line(SAID, *MEMORY) is None
    assert [r["effect"] for r in build_receipts(tracker_of(MEMORY))] == ["memory saved"]
    assert nudged(SAID) == "stored"                                            # nothing ran: caught
    assert nudged("I've stored it in my memory so that all agents know.") == "stored"


@pytest.mark.parametrize("said", [TOLD, HONEST, "Your agents don't know yet.",
                                  "I'll write it into a note so every agent knows.",
                                  "The team will know once I post the note.",
                                  "If I write the note, the team will know."])
def test_what_the_team_will_know_is_no_report_of_work_done(said):
    assert nudged(said, *MEMORY) is None and line(said, *MEMORY) is None


def test_the_reply_is_not_nudged_by_a_team_family_any_more():
    model = f187._Model(TOLD)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(TOLD), owner=OWNER)

    assert model.sent == [] and final["_final_response"].content == TOLD
    assert final["_f187"].correction is None


def test_the_lane_is_deleted_and_the_chat_no_longer_runs_it():
    from consumers.chatbot.service import StreamingChatService

    assert not (CHATBOT / "team_corrections.py").exists()
    inner, names = StreamingChatService._retrieval_first, []
    while hasattr(inner, "__wrapped__"):
        names.append(inner.__code__.co_qualname)
        inner = inner.__wrapped__
    assert not any("tells_the_team_honestly" in name for name in names)
