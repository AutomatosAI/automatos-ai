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

P256-FIX-RVW-37: "all agents are aware" over a saved memory is a wrong statement the owner acts on
(F324's own bug, which FX-007 cleared). A claim that the team or the agents know needs a write other
than memory (``claims_by_kind``): over the memory alone it is nudged and the line says the agents
haven't been told; a plan's purpose, a condition, a denial or an offer stays no claim.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from consumers.chatbot.claims_by_kind import TEAM_LABEL
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


TEAM_NOT_TOLD = ("Just to be clear: your agents haven't been told: I only keep this in my own memory, and they read "
                 "your documents and their cards, not my memory. Ask me again if you want it done.")
NOTE = ("platform_store_memory", "platform_upload_document")


def test_the_memory_write_backs_stored_never_the_team_knowing():
    assert nudged(SAID, *MEMORY) == TEAM_LABEL and line(SAID, *MEMORY) == TEAM_NOT_TOLD
    assert [r["effect"] for r in build_receipts(tracker_of(MEMORY))] == ["memory saved"]
    assert nudged(SAID) == "stored"                                            # nothing ran: caught
    assert nudged("I've stored it in my memory so that all agents know.") == "stored"
    assert nudged("I've stored it in my memory so that all agents know.", *MEMORY) == TEAM_LABEL


def test_a_note_in_the_documents_backs_the_team_knowing():
    assert nudged(SAID, *NOTE) is None and line(SAID, *NOTE) is None
    assert nudged(TOLD, *NOTE) is None


def test_all_agents_aware_over_the_memory_alone_is_caught():
    assert nudged(TOLD, *MEMORY) == TEAM_LABEL and line(TOLD, *MEMORY) == TEAM_NOT_TOLD


@pytest.mark.parametrize("said", [HONEST, "Your agents don't know yet.",
                                  "I'll write it into a note so every agent knows.",
                                  "The team will know once I post the note.",
                                  "If I write the note, the team will know."])
def test_what_the_team_will_know_is_no_report_of_work_done(said):
    assert nudged(said, *MEMORY) is None and line(said, *MEMORY) is None


def test_the_reply_is_nudged_once_by_the_receipts_rule():
    """No team family: the first reply goes through the loop (``claims_work_done``) and its one nudge
    names the claim; the honest retry is the answer."""
    model = f187._Model(HONEST)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(TOLD), owner=OWNER)

    (sent,) = model.sent
    assert sent[-2] == {"role": "assistant", "content": TOLD}
    assert f"says something was {TEAM_LABEL}" in sent[-1]["content"]
    assert final["_final_response"].content == HONEST and final["_f187"].correction is None


def test_the_lane_is_deleted_and_the_chat_no_longer_runs_it():
    from consumers.chatbot.service import StreamingChatService

    assert not (CHATBOT / "team_corrections.py").exists()
    inner, names = StreamingChatService._retrieval_first, []
    while hasattr(inner, "__wrapped__"):
        names.append(inner.__code__.co_qualname)
        inner = inner.__wrapped__
    assert not any("tells_the_team_honestly" in name for name in names)
