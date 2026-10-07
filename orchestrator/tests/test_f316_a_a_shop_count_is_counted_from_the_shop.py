"""F316 (night 9b): Auto's shop counts came from a document or from memory, and changed from
chat to chat: Q5 "doesn't specify the total number" (the October note), then 5,957, 87, 63;
Q8 2, 118, "12"; "where did that come from?" → "our previous conversation", "my memory",
"I seem to have misplaced the source". Now a shop-figure turn is told to count it from the shop
in this reply, and a reply that gives a figure, says it isn't there, or names memory as its
source with no call to the shop is nudged once and then corrected.
"""
from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot import knowledge_prefetch as kp
from consumers.chatbot.claim_check import not_done
from consumers.chatbot.shop_figures import (
    SHOP_NOTE, WHERE_FROM_NOTE, asks_a_shop_figure, counts_from_the_shop, shop_note,
)
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.execution.action_claims import claimed_action_not_done
from modules.tools.execution.shop_and_team_claims import (
    SHOP_LABEL, SHOP_LINE, mark_shop_figure_turn, shop_figure_turn,
)
from tests import test_f187_a_claim_no_action_backs_is_corrected as f187

Q5 = "How many Harvest Club boxes go out on Monday 5 October?"
Q8 = "How many Harvest Club members cancelled between April and September, and what's the most common reason?"
FROM_THE_NOTE = ("I found that the Harvest Club boxes are posted on Monday, October 5th. However, the document doesn't "
                 "specify the total number of boxes going out that day.")
FROM_MEMORY = "I retrieved that information from our previous conversation."
UNCOUNTED = "There are 87 subscribers on the Harvest Club."


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.fixture
def shop(monkeypatch):
    """A workspace with its shop database connected."""
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: True)


def _in_a_turn(fn, *args, **kwargs):
    """Run ``fn`` in a context of its own, as a chat turn's task is: the turn's mark never leaks."""
    return contextvars.copy_context().run(fn, *args, **kwargs)


@pytest.mark.parametrize("said", [
    Q5, Q8, "So how many cancelled in total, all reasons?", "Were any Harvest Club boxes late in September?",
    "Do I need to reorder any Kirinyaga?", "What did the shop take in retail orders in September?",
    "Have we got enough Guji for October's club boxes?", "The members are in the shop system. Please count them.",
    "How many Harvest Club boxes ship Monday, and how many cards are still open?",
])
def test_a_figure_the_shop_holds_is_a_shop_question(said):
    assert asks_a_shop_figure(said) is True


@pytest.mark.parametrize("said", [
    "A café wants 10 kg of coffee next week. What do we charge them for delivery, and is it ever free?",
    "How many cards are in review?", "How many words should the Harbour Log intro be?",
    "How much margin do we make on a bag of Kirinyaga?",
    "Which importers are behind the coffees in the October club box?", "Is the intro short enough?",
])
def test_a_document_or_board_question_is_not(said):
    assert asks_a_shop_figure(said) is False


def _run(chat, said, messages):
    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": "passages from club-box-october-2026.md"})
        yield "searched"

    async def turn():
        frames = [f async for f in counts_from_the_shop(retrieval_first)(chat, said, messages, None, "c", [])]
        return frames, shop_figure_turn()
    return asyncio.run(turn())


def test_the_turn_is_told_to_count_it_from_the_shop_after_the_passages(shop):
    messages = [{"role": "user", "content": Q5}]
    frames, marked = _run(NS(widget_mode=False, db=None, workspace_id="ws"), Q5, messages)

    assert frames == ["searched"] and marked is True
    assert messages[-1] == {"role": "system", "content": SHOP_NOTE}       # last: after the October note's passages
    assert "platform_query_data" in SHOP_NOTE and "dated document" in SHOP_NOTE
    assert shop_figure_turn() is False                                    # the mark stays in the turn


def test_where_did_that_come_from_counts_again(shop):
    messages = [{"role": "user", "content": Q8}, {"role": "assistant", "content": "2 cancelled."},
                {"role": "user", "content": "Where did that come from?"}]
    _frames, marked = _run(NS(widget_mode=False, db=None, workspace_id="ws"), "Where did that come from?", messages)

    assert marked is True and messages[-1]["content"] == WHERE_FROM_NOTE
    assert "your memory or an earlier chat is no source" in WHERE_FROM_NOTE


def test_no_database_a_widget_or_another_question_gets_no_note(monkeypatch, shop):
    for chat, said in ((NS(widget_mode=True, db=None, workspace_id="ws"), Q5),
                       (NS(widget_mode=False, db=None, workspace_id="ws"), "Which importers supply the Guji?")):
        messages = [{"role": "user", "content": said}]
        assert _run(chat, said, messages)[1] is False and len(messages) == 2
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    assert _in_a_turn(shop_note, NS(widget_mode=False, db=None, workspace_id="ws"), Q5, []) is None


def _claims(reply, done):
    def check():
        mark_shop_figure_turn(True)
        return claimed_action_not_done(reply, done, promises=True)
    return _in_a_turn(check)


@pytest.mark.parametrize("reply", [UNCOUNTED, FROM_THE_NOTE, FROM_MEMORY,
                                   "I seem to have misplaced the source of that information."])
def test_a_figure_with_no_count_from_the_shop_is_a_claim(reply):
    assert _claims(reply, {"search_knowledge", "platform_search_memory"}) == SHOP_LABEL
    assert _claims(reply, {"platform_query_data"}) is None                # counted from the shop this turn


def test_a_date_a_question_or_another_turn_is_not():
    assert _claims("The club box posts on Monday, October 5th, card #0101.", set()) is None
    assert _claims("Shall I count the 63 again from the shop?", set()) is None
    assert claimed_action_not_done(UNCOUNTED, set(), promises=True) is None   # not a shop-figure turn


def test_the_reply_is_nudged_to_count_then_corrected_if_it_still_has_not():
    def turn():
        mark_shop_figure_turn(True)
        model = f187._Model(UNCOUNTED)
        with usage_scope(request_type=LANE_CHAT):
            return model, f187._turn(model, f187._round(UNCOUNTED), owner=Q5)[1]

    model, final = _in_a_turn(turn)
    (sent,) = model.sent                                                  # the loop's one nudge
    assert "something was counted from your shop system" in sent[-1]["content"]
    assert final["_f187"].claim == SHOP_LABEL
    assert final["_f187"].correction is None and not_done(SHOP_LABEL) == SHOP_LINE   # PRD-256: said from receipts
    assert SHOP_LINE.startswith("Just to be clear: I didn't count this from your shop system in this reply")


def test_the_turn_s_mark_reaches_the_tool_loop_s_check(shop):
    """The seam itself: the mark is set by the decorator on _retrieval_first and read by the claim
    check inside the tool loop, which runs its rounds in a task of its own (F196: contextvars)."""
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, f187.WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), None
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    model, first = f187._Model(UNCOUNTED), f187._round(UNCOUNTED)
    runtime = NS(llm_manager=model, agent_id=322, workspace_id=f187.WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": Q5}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        yield "searched"

    async def turn():
        async for _frame in counts_from_the_shop(retrieval_first)(svc, Q5, messages, runtime, "c", []):
            pass
        chunks = [c async for c in svc._stream_tool_loop(first, messages, runtime, {}, f187.TOOLS,
                                                         streamed_rounds=[first], reasoning_log=[])]
        return next(c for c in chunks if isinstance(c, dict) and c.get("_final_response"))

    with usage_scope(request_type=LANE_CHAT):
        final = asyncio.run(turn())
    assert final["_f187"].claim == SHOP_LABEL and final["_f187"].correction is None


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first.__wrapped__.__wrapped__.__wrapped__.__wrapped__  # PRD-256, F241, F307, F303
    assert inner.__code__ is counts_from_the_shop(lambda: None).__code__
