"""P256-FIX-RVW-8 (FX-007 F187): a number the turn's automatic reads put in front of the model is
quoted, not invented.

Night 12, A439/A593: the owner asked what was waiting for them, the turn read the board's Needs you
by prefetch (no tool call), and its note listed "#0931 'Card #0930 Approval Request'". The reply named
#0930 from that title and got "task 0930 does not exist": FX-007 counted only the loop tracker's tool
results as quoted, so the first reply went through the loop on the id, was re-prompted, and the saved
answer kept the tier-2 line. ``receipts.its_reads_are_receipted`` now keeps what the reads' notes say
(``claim_check.reads_put_in_front``) and ``invented_ids`` reads it, in the loop and outside it.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot import claim_check, needs_you_turn
from consumers.chatbot.needs_you_turn import answers_what_needs_you
from consumers.chatbot.receipts import its_reads_are_receipted, writes_its_receipts

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
ASKED = "What is waiting for me?"
TOOLS = [{"type": "function", "function": {"name": "platform_list_tasks", "parameters": {"type": "object",
                                                                                         "properties": {}}}}]
WAITING = {"total": 1, "by_kind": {"review": 1},
           "cards": [{"number": "#0931", "title": "Card #0930 Approval Request", "kind": "review"}]}
NAMES_0930 = "One thing waits for you: #0931, the approval request for card #0930. It needs your OK on the board."
NAMES_0929 = "One thing waits for you: #0931, the approval request for card #0929. It needs your OK on the board."


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F187 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.fixture(autouse=True)
def board(monkeypatch):
    """#0931 is a card of the workspace; #0930 and #0929 are no card numbers. Needs you holds #0931."""
    monkeypatch.setattr(claim_check, "existing_ids", lambda ws, named: {p for p in named if p == ("task", "0931")})
    monkeypatch.setattr(needs_you_turn, "_read", lambda chat: copy.deepcopy(WAITING))


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None,
              finish_reason="tool_calls" if calls else "stop")


class _Model:
    def __init__(self, *texts):
        self.texts, self.sent = list(texts), []

    async def generate_response(self, messages, tools=None, on_delta=None):
        self.sent.append(copy.deepcopy(messages))
        text = self.texts.pop(0)
        if on_delta is not None:
            await on_delta("text", text)
        return _round(text)


def _service():
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), None
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


def _reads(said=ASKED, passage=None):
    """The turn's automatic reads, under the decorators service.py stacks: Needs you for a
    'what needs me' turn; ``passage`` stands for a read under it (retrieval first, the team's findings)."""
    async def inner(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        if passage:
            llm_messages.append({"role": "system", "content": passage})
        yield "retrieval"

    return its_reads_are_receipted(answers_what_needs_you(inner)), said


async def _read_first(svc, messages, prefetched, **reads):
    reader, said = _reads(**reads)
    return [frame async for frame in reader(svc, said, messages, None, "chat-1", prefetched)]


def _first_reply(reply, **reads):
    """The first reply's route: through the loop (True) or saved as it is (False), after the reads."""
    svc = _service()
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": ASKED}]

    async def run():
        prefetched = []
        await _read_first(svc, messages, prefetched, **reads)
        return await svc._first_reply_goes_through_the_loop(_round(reply), TOOLS, prefetched, ASKED)
    return asyncio.run(run())


def _loop(model, first, **reads):
    """The chat's tool loop over ``first``, after the turn's reads, in the same task."""
    svc = _service()
    runtime = NS(llm_manager=model, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": ASKED}]

    async def run():
        prefetched, final = [], None
        await _read_first(svc, messages, prefetched, **reads)
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, streamed_rounds=[first],
                                                 reasoning_log=[], prefetched=prefetched):
            if isinstance(chunk, dict) and chunk.get("_final_response"):
                final = chunk
        return final
    return asyncio.run(run())


# ── the first reply ─────────────────────────────────────────────────────────

def test_a_card_number_the_needs_you_note_quotes_keeps_the_first_reply_out_of_the_loop():
    """A439/A593: #0930 is in #0931's title, read by Needs you: no F187 route through the loop."""
    assert _first_reply(NAMES_0930) is False


def test_a_number_in_no_read_still_sends_the_first_reply_through_the_loop():
    assert _first_reply(NAMES_0929) is True


def test_a_number_a_read_under_needs_you_quotes_is_backed_too():
    """The retrieval-first passages and the team's findings go in under the same wrapper."""
    assert _first_reply(NAMES_0929, said="Is card #0931 approved yet?",
                        passage="Team findings: card #0931 answers it, 'Approve card #0929'.") is False


# ── the loop ─────────────────────────────────────────────────────────────────

def test_the_loop_neither_re_prompts_nor_corrects_a_number_the_read_quoted():
    model = _Model()
    final = _loop(model, _round(NAMES_0930))

    assert model.sent == []                                          # no F187 re-prompt
    verdict = final["_f187"]
    assert verdict.ids == [] and verdict.reprompted is False and verdict.correction is None


def test_the_loop_still_re_prompts_then_corrects_a_number_in_no_read():
    model = _Model(NAMES_0929)
    final = _loop(model, _round(NAMES_0929))

    (sent,) = model.sent
    assert sent[-1]["content"].startswith("Your previous reply names task 0929, which does not exist")
    assert final["_f187"].correction == ("Just to be clear: task 0929 does not exist — "
                                         "I named it without looking it up.")


# ── the seam ─────────────────────────────────────────────────────────────────

def test_the_reads_text_is_the_notes_they_added_and_each_turn_starts_with_none():
    svc = _service()
    messages = [{"role": "system", "content": "Ticket #0930 is mentioned in the persona, not by a read."},
                {"role": "user", "content": ASKED}]

    async def run():
        await _read_first(svc, messages, [])
        read = claim_check.READS_SAID.get()

        async def turn(chat):
            yield claim_check.READS_SAID.get()
        fresh = [said async for said in writes_its_receipts(turn)(NS(widget_mode=False, streaming_handler=None))]
        return read, fresh

    read, fresh = asyncio.run(run())
    assert "- #0931 'Card #0930 Approval Request'" in read and "persona" not in read
    assert fresh == [""]


def test_a_widget_visitor_reads_nothing_so_nothing_is_quoted():
    svc = _service()
    svc.widget_mode = True

    async def run():
        await _read_first(svc, [{"role": "user", "content": ASKED}], [])
        return claim_check.invented_ids(NAMES_0930, ASKED, WS)
    assert asyncio.run(run()) == [("task", "0930")]


def test_the_wrapper_is_outermost_on_retrieval_first():
    """Every automatic read sits under ``its_reads_are_receipted``, so each note it adds is kept."""
    from consumers.chatbot import service

    source = inspect.getsource(service.StreamingChatService)
    order = [source.index(mark) for mark in ("@its_reads_are_receipted", "@grounds_the_cards",
                                             "@answers_what_needs_you", "@reads_what_the_team_found",
                                             "@fills_the_named_template", "async def _retrieval_first(")]
    assert order == sorted(order)
