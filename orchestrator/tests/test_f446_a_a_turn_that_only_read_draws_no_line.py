"""P256-FIX-N17-5 (F446, nights 16–17): a turn that only read draws no "Just to be clear" line for what it found.

After N15-1/N15-2 the line fired 26 times on night 17, 19 of them on turns with no write receipt. With no done
write, every third-person shape the completed-action pattern reads was unbacked: "It has been unassigned.", "It is
now in the Review column.", "They're finished drafts waiting for your sign-off." each said what the turn's read
found. The one read-only exemption (``claims_not_reports.looked_something_up``) covered a stative passive after a
read of one thing; and the reads before the loop (retrieval first, the board) were recorded with record_execution,
never as outcomes, so the loop's check saw a turn that read nothing.

Now, in Auto's voice, a turn with no write receipt (done, refused or waiting) that read something (a done read, or
its reads before the loop: ``read_ahead``) is answered only when one of its claims is Auto's own report: the first
person, "Done.", a tick, "you'll now see it", a bare or ticked participle, a line opening on its verb
(``claims_backed._own``). When one is, every unbacked claim goes as before. A turn that read nothing keeps the full
check (the zero-read catch). The sentences are synthetic, shaped like the night's replies: night data stays local.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.receipts import the_loop_writes_receipts
from core.llm.clients.base import LLMResponse
from modules.tools.execution import claims_in_place
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.helpers_receipts_rule import call, tracker_of

OK = {"success": True}
TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]
NOTHING_DONE_LINE = "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done."
SEARCHED = ("search_knowledge", {"query": "autumn tier"}, OK)
QUERIED = ("query_database", {"sql": "select margin from sheets"}, OK)
F280 = "My apologies, Gerard. I understand. Task #0422 has been moved to 'cancelled'."


def _named(verb):
    return f"Just to be clear: I haven't {verb} anything in this reply. Ask me again if you want it done."


def _removed(answer, *calls, read_ahead=False):
    """The reply as the owner gets it in Auto's voice; None when nothing is taken out."""
    removed = without_unbacked_claims(answer, tracker_of(calls).outcomes, True, read_ahead=read_ahead)
    return None if removed is None else removed.text


# (the sentence, the reads of its turn): none of these turns wrote anything.
FOUND_BY_A_READ = [
    ("They're finished drafts waiting for your sign-off.", ["platform_board_summary"]),
    ("It is now in the Review column.", ["platform_list_tasks"]),
    ("It's saved in Deliverables.", ["platform_list_tasks", "platform_list_deliverables"]),
    ("It has been unassigned.", ["platform_list_tasks"]),
    ("#1755 was started tonight and is still in progress.", ["platform_get_task"]),
    ("It is done.", [SEARCHED, "platform_grep_documents"]),
    ("The margin sheet was built on 34p.", [QUERIED]),
    ("The 1 October issue was sent, so this is for the 15th.", ["platform_read_document"]),
    ("CELLAR's draft is set up for weekly 4 kg.", ["platform_list_tasks"]),
    ("- **Set up KERBSIDE CLERK** as a new agent for supplier chasing.", ["platform_list_agents"]),   # a menu bullet
]


@pytest.mark.parametrize("said, reads", FOUND_BY_A_READ, ids=[said[:40] for said, _ in FOUND_BY_A_READ])
def test_what_a_read_found_draws_no_line(said, reads):
    assert _removed(said, *reads) is None


@pytest.mark.parametrize("said", ["It has been unassigned.", "It is now in the Review column."])
def test_the_reads_before_the_loop_count(said):
    """The automatic reads are never an outcome: ``read_ahead`` says the turn made them."""
    assert _removed(said) is not None                                 # the turn read nothing: still caught
    assert _removed(said, read_ahead=True) is None


# ── recall kept: Auto's own reports after a read, and the whole removal set once one is ──

def test_an_own_claim_after_a_read_takes_what_goes_on_from_it_too():
    assert _removed("I've approved the mission. It's now running.", "platform_get_mission") == _named("approved")


def test_the_night_6_paragraph_loses_both_claims_and_keeps_the_pointer():
    said = ("I've just created a new task on your board: **\"Draft two-line thank-you note to Maya at Lamplight "
            "Café\"**. This task has been assigned to your **Shopify Support Agent**. You can find it on the board.")
    assert _removed(said, "platform_list_tasks") == (
        "Just to be clear: I haven't created or assigned anything in this reply. Ask me again if you want it done. "
        "You can find it on the board.")


@pytest.mark.parametrize("said, line", [
    ("Done.", NOTHING_DONE_LINE),
    ("Sent the email to Declan.", _named("sent")),
    ("Email sent.", _named("sent")),
    ("Mission launched ✅", _named("launched")),
    ("✅ Set up the report for Monday.", _named("set up")),           # ticked: not the menu bullet's "Set up"
    ("You should now see it in Deliverables.", NOTHING_DONE_LINE),
])
def test_autos_own_report_after_a_read_still_draws_the_line(said, line):
    assert _removed(said, "platform_list_tasks") == line


def test_the_zero_read_catch_stays_and_the_reads_ahead_trade_is_pinned():
    """Night 8's F280 with no call at all is still caught; after the turn's automatic reads it says what the
    board showed, and draws no line (the documented trade: Gerard's call before merge)."""
    assert _removed(F280) == ("My apologies, Gerard. I understand. Just to be clear: I haven't moved anything in "
                              "this reply. Ask me again if you want it done.")
    assert _removed(F280, read_ahead=True) is None


def test_a_refused_write_keeps_the_full_check():
    refused = call("platform_update_task_status", {"task_id": "#0422", "status": "cancelled"},
                   {"success": False, "error": "A running ticket can only be cancelled from its run."})
    assert _removed("It has been cancelled.", "platform_list_tasks", refused) == (
        "Just to be clear: I haven't cancelled anything in this reply; I tried to move the card #0422 and it didn't "
        "go through. Ask me again if you want it done.")


def test_an_agents_text_is_untouched_by_the_gate():
    """The gate is Auto's: an agent's run is held to its first-person claims, read or no read."""
    removed = without_unbacked_claims("I've cancelled your order. Thanks for waiting. It has been refunded.",
                                      tracker_of(["platform_list_tasks"]).outcomes, False, read_ahead=True)
    assert removed.text == "Thanks for waiting. It has been refunded."
    assert removed.taken_out == ("I've cancelled your order.",)


# ── the loop ────────────────────────────────────────────────────────────────

class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


async def _board(name, args, call_id, workspace_id):
    return {"success": True, "review": [{"number": "#1751"}, {"number": "#1752"}]}


def test_a_board_summary_round_returns_the_reply_unchanged(caplog):
    said = "They're finished drafts waiting for your sign-off."
    board = {"id": "call_board", "type": "function", "function": {
        "name": "platform_execute", "arguments": json.dumps({"action": "platform_board_summary", "params": {}})}}
    executor = ToolLoopExecutor(llm_callback=_Model(LLMResponse(content=said, tool_calls=None)),
                                tool_callback=_board, max_iterations=5, promises=True)
    with caplog.at_level(logging.WARNING, logger="modules.tools.execution.claims_in_place"):
        result = asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[board]),
                                          messages=[{"role": "user", "content": "What's in review?"}],
                                          tools=TOOLS, workspace_id="ws"))
    assert result.response.content == said
    assert not [r for r in caplog.records if "[tool-loop] unbacked claim" in r.getMessage()]


def _answered(said, read_ahead):
    executor = ToolLoopExecutor(llm_callback=_Model(), tool_callback=None, max_iterations=5, promises=True,
                                read_ahead=read_ahead)
    return asyncio.run(executor.run(initial_response=LLMResponse(content=said, tool_calls=None),
                                    messages=[{"role": "user", "content": "Where is #1755?"}], tools=TOOLS,
                                    workspace_id="ws")).response.content


def test_the_loop_passes_read_ahead_to_the_check():
    assert _answered("It has been unassigned.", read_ahead=True) == "It has been unassigned."
    assert _answered("It has been unassigned.", read_ahead=False) == _named("unassigned")
    source = inspect.getsource(claims_in_place.in_place)
    assert 'read_ahead=getattr(loop, "read_ahead", False))' in source


# ── the chat's loop is told it read ahead; an agent's run keeps the default ──

def _made_in_a_chat_loop(prefetched):
    """The executors made while ``the_loop_writes_receipts`` runs a loop, then one made after it."""
    made = []

    async def loop(chat, *args, **kwargs):
        made.append(ToolLoopExecutor(llm_callback=_Model(), tool_callback=None))      # the chat's own
        made.append(ToolLoopExecutor(llm_callback=_Model(), tool_callback=None))      # an agent's run within it
        yield "chunk"

    async def drain():
        return [chunk async for chunk in the_loop_writes_receipts(loop)(NS(streaming_handler=None),
                                                                        prefetched=prefetched)]

    assert asyncio.run(drain()) == ["chunk"]
    made.append(ToolLoopExecutor(llm_callback=_Model(), tool_callback=None))
    return [executor.read_ahead for executor in made]


def test_only_the_chats_own_executor_reads_ahead():
    assert _made_in_a_chat_loop([("search_knowledge", {"query": "autumn tier"})]) == [True, False, False]
    assert _made_in_a_chat_loop([]) == [False, False, False]


def test_the_chats_tool_loop_is_wrapped_by_the_seam_that_sets_it():
    """service.py's _stream_tool_loop is never edited (its body is over the shape gate's limit): the seam
    that already wraps it, ``the_loop_writes_receipts``, hands the loop its ``prefetched`` reads."""
    from consumers.chatbot.service import StreamingChatService

    seam = inspect.getsource(the_loop_writes_receipts)
    assert 'READ_AHEAD.set(bool(kwargs.get("prefetched")))' in seam
    chain, fn = [], StreamingChatService._stream_tool_loop
    while fn is not None:
        chain.append(fn.__code__.co_qualname)
        fn = getattr(fn, "__wrapped__", None)
    assert "the_loop_writes_receipts.<locals>.wrapped" in chain
    assert chain[-1] == "StreamingChatService._stream_tool_loop"
