"""F316 (night 9b): Auto's shop counts came from a document or from memory, and changed from
chat to chat: Q5 "doesn't specify the total number" (the October note), then 5,957, 87, 63;
Q8 2, 118, "12"; "where did that come from?" → "our previous conversation", "my memory",
"I seem to have misplaced the source". A shop-figure turn was told to count it from the shop
(consumers/chatbot/shop_figures.py), and a reply that gave a figure with no call to the shop was
a claim of a family (shop_and_team_claims.py), nudged once and then corrected.

PRD-256 FX-007 (D10): the lane and the family are deleted. A figure, "it isn't there" and a
source named from memory are no report of work done: the receipts' rule clears them, and the
receipts above the answer show the owner what the turn read (a document search, a memory
search, and no query of the shop).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from consumers.chatbot.receipts import build_receipts
from core.llm.usage_context import LANE_CHAT, usage_scope
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187
from tests.helpers_receipts_rule import call, line, nudged, tracker_of

CHATBOT = Path(__file__).resolve().parents[1] / "consumers" / "chatbot"
EXECUTION = Path(__file__).resolve().parents[1] / "modules" / "tools" / "execution"
Q5 = "How many Harvest Club boxes go out on Monday 5 October?"
FROM_THE_NOTE = ("I found that the Harvest Club boxes are posted on Monday, October 5th. However, the document doesn't "
                 "specify the total number of boxes going out that day.")
FROM_MEMORY = "I retrieved that information from our previous conversation."
UNCOUNTED = "There are 87 subscribers on the Harvest Club."
SEARCHED = (call("search_knowledge", {"query": Q5}), "platform_search_memory")


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.mark.parametrize("reply", [UNCOUNTED, FROM_THE_NOTE, FROM_MEMORY,
                                   "I seem to have misplaced the source of that information.",
                                   "The club box posts on Monday, October 5th, card #0101.",
                                   "Shall I count the 63 again from the shop?"])
def test_a_figure_or_its_source_is_no_report_of_work_done(reply):
    assert nudged(reply, *SEARCHED) is None and line(reply, *SEARCHED) is None


def test_the_receipts_show_the_shop_was_not_queried():
    shown = build_receipts(tracker_of(SEARCHED))
    assert [r["kind"] for r in shown] == ["read", "read"]
    assert not any("query" in r["action"] for r in shown)


def test_the_reply_is_not_nudged_by_a_shop_family_any_more():
    model = f187._Model(UNCOUNTED)
    with usage_scope(request_type=LANE_CHAT):
        _frames, final = f187._turn(model, f187._round(UNCOUNTED), owner=Q5)

    assert model.sent == [] and final["_final_response"].content == UNCOUNTED
    assert final["_f187"].correction is None


def test_the_lane_and_the_family_are_deleted_and_the_chat_no_longer_runs_them():
    from consumers.chatbot.service import StreamingChatService

    assert not (CHATBOT / "shop_figures.py").exists() and not (EXECUTION / "shop_and_team_claims.py").exists()
    inner, names = StreamingChatService._retrieval_first, []
    while hasattr(inner, "__wrapped__"):
        names.append(inner.__code__.co_qualname)
        inner = inner.__wrapped__
    assert not any("counts_from_the_shop" in name for name in names)
