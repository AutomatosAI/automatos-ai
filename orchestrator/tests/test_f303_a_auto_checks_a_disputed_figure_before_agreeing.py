"""F303 (night 9, L106): told "The Analyst counted 11 club cancellations April to
September on card 1886. You told me 4 a minute ago. Which is right, and why did you
say 4?", Auto said the Analyst was correct and that its own 4 "was based on an
incomplete query", with no tool call at all (chat ec665ae6). The turn was told to check
again before it agreed (consumers/chatbot/figure_disputes.py), and a reply that said a
figure was right or wrong with no count behind it was a "re-checked" claim (a family).

PRD-256 FX-007 (D10): the lane and the family are deleted. A figure said to be right, and
what an earlier figure "was based on", are no report of work done: the receipts' rule clears
them, and the receipts above the answer show the owner that nothing was counted this turn
(an empty receipts part: US-002's replay row "F303 the Analyst's count is correct").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from consumers.chatbot.receipts import build_receipts
from core.llm.usage_context import LANE_CHAT, usage_scope
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187
from tests.helpers_receipts_rule import line, nudged, tracker_of

CHATBOT = Path(__file__).resolve().parents[1] / "consumers" / "chatbot"
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


@pytest.mark.parametrize("calls", [(), ("platform_get_task",), ("platform_query_data",)])
def test_giving_way_reports_no_work_done_and_the_receipts_show_what_ran(calls):
    assert nudged(GAVE_WAY, *calls) is None and line(GAVE_WAY, *calls) is None
    assert [r["kind"] for r in build_receipts(tracker_of(calls))] == ["read"] * len(calls)


def test_an_offer_to_count_again_is_no_claim():
    assert nudged("Is 11 right? I can count it again.") is None


def test_the_reply_is_not_nudged_by_a_figure_family_any_more():
    model = f187._Model(GAVE_WAY)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(GAVE_WAY), owner=OWNER)

    assert model.sent == [] and final["_final_response"].content == GAVE_WAY
    assert final["_f187"].correction is None


def test_the_lane_is_deleted_and_the_chat_no_longer_runs_it():
    from consumers.chatbot.service import StreamingChatService

    assert not (CHATBOT / "figure_disputes.py").exists()
    inner, names = StreamingChatService._retrieval_first, []
    while hasattr(inner, "__wrapped__"):
        names.append(inner.__code__.co_qualname)
        inner = inner.__wrapped__
    assert not any("rechecks_disputed_figures" in name for name in names)
