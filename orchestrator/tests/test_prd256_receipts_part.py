"""PRD-256 US-001 — the receipts part: what Auto did this turn, written by the platform.

Eleven customer nights found Auto claiming work it did not do, most often after a call that
FAILED on its arguments and was reported as done. Every reply now carries the platform's own
account of the turn: one receipt per call, built after the loop from the tool tracker
(``ToolExecutionTracker.outcomes``, never ``tool_execution_logs``), streamed as one
``receipts`` frame and saved as the message's ``receipts`` part. The model never writes it.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.receipts import (
    AUTOMATIC_READS, DONE, READ, REFUSED, SKIPPED, WRITE, build_receipts, receipt, turn_receipts,
)
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
MODEL = "google/gemini-2.5-flash"
BAD_JSON = "DATA_BAD_JSON: the document's data is not valid JSON (line 1, column 12)."
CARD_MOVE = {"action": "platform_update_task_status", "params": {"task_id": "#0422", "status": "done"}}
CARD_MOVED = {"success": True, "llm_context": "Ticket #0422 moved to done.",
              "raw_result": {"success": True, "task_id": 422, "number": "#0422", "status": "done"}}
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object",
                                                                                      "properties": {}}}}]


def _on_the_board(receipts):
    """What the block reads to say "No actions on the board." (frontend/lib/chat/receipts.ts):
    a done write that names a card by its number."""
    return [r for r in receipts if r["kind"] == WRITE and r["status"] == DONE and r["subject"].startswith("#")]


# ── the builder, from the tracker ───────────────────────────────────────────

def test_a_store_memory_only_turn_has_one_write_receipt_and_nothing_on_the_board():
    tracker = ToolExecutionTracker()
    tracker.record_outcome("store_memory", {"content": "Refunds over £50 need Gerard's OK."},
                           {"success": True, "llm_context": "Stored."})

    (only,) = build_receipts(tracker)
    assert only == {"action": "store_memory", "kind": WRITE, "status": DONE, "subject": "",
                    "effect": "memory saved", "link": None, "reason": None}
    assert _on_the_board([only]) == []          # the block says "No actions on the board."


def test_a_turn_with_no_calls_has_no_receipts():
    assert build_receipts(ToolExecutionTracker()) == []
    assert build_receipts(None) == []
    assert turn_receipts(None, ()) == []


def test_a_refused_generate_document_is_refused_with_its_reason():
    tracker = ToolExecutionTracker()
    tracker.record_outcome("generate_document", {"title": "Letter to Maya Osei", "data": "{\"body\": \"We\\'re\"}"},
                           {"success": False, "llm_context": f"Tool generate_document failed: {BAD_JSON}",
                            "raw_result": {"success": False, "status": "error", "error": f"{BAD_JSON}\nTrace: …"}})

    (refused,) = build_receipts(tracker)
    assert refused["status"] == REFUSED and refused["kind"] == WRITE
    assert refused["subject"] == "Letter to Maya Osei"
    assert refused["reason"] == BAD_JSON                     # the executor's reason, one line
    assert refused["link"] is None


def test_a_card_moved_to_done_names_the_card_and_where_it_went():
    tracker = ToolExecutionTracker()
    tracker.record_outcome("platform_execute", CARD_MOVE, CARD_MOVED)

    (moved,) = build_receipts(tracker)
    assert moved["action"] == "platform_update_task_status"     # the dispatcher's inner action
    assert (moved["kind"], moved["status"]) == (WRITE, DONE)
    assert moved["subject"] == "#0422" and moved["effect"] == "moved to Done"
    assert moved["link"] == "/command-center?tab=board&task_id=422"
    assert _on_the_board([moved]) == [moved]


@pytest.mark.parametrize("status, effect", [("cancelled", "moved to Cancelled"), ("send back", "sent back to its agent"),
                                            ("rejected", "sent back to its agent")])
def test_a_card_move_is_said_the_boards_way(status, effect):
    params = {"task_id": "#0451", "status": status}
    assert receipt("platform_update_task_status", params, {"success": True})["effect"] == effect


def test_a_refused_reason_is_trimmed_to_one_line():
    long = "x" * 500
    said = receipt("platform_update_agent", {"agent_name": "Social Media Director"},
                   {"success": False, "error": long})
    assert said["subject"] == "Social Media Director" and len(said["reason"]) <= 200


def test_a_skipped_repeat_is_a_skipped_receipt():
    tracker = ToolExecutionTracker()
    args = {"action": "platform_create_task", "params": {"title": "Weekly report"}}
    tracker.record_execution("platform_execute", args)
    tracker.record_outcome("platform_execute", args, {"success": True, "raw_result": {"number": "#0500"}})
    assert tracker.should_skip_execution("platform_execute", args)[0] is True

    made, again = build_receipts(tracker)
    assert (made["status"], made["subject"]) == (DONE, "#0500")
    assert (again["status"], again["subject"], again["reason"]) == (
        SKIPPED, "Weekly report", "the same request already ran in this reply")


def test_the_automatic_reads_fold_into_one_read():
    prefetched = [("search_knowledge", {"query": "refund policy"}), ("search_knowledge", {"query": "returns"}),
                  ("platform_board_summary", {"automatic": True, "needs_you": True})]
    tracker = ToolExecutionTracker()
    tracker.record_outcome("platform_execute", {"action": "platform_list_tasks", "params": {}}, {"success": True})

    folded, listed = build_receipts(tracker, prefetched)
    assert folded == {"action": AUTOMATIC_READS, "kind": READ, "status": DONE, "subject": "",
                      "effect": "read your documents and the board", "link": None, "reason": None}
    assert (listed["kind"], listed["status"]) == (READ, DONE)
    assert build_receipts(None, prefetched[:1])[0]["effect"] == "read your documents"


# ── the loop: one frame, from the tracker, the model that answered ──────────

@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F186 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None, model=MODEL,
              finish_reason="tool_calls" if calls else "stop")


def _call(arguments, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps(arguments)}}


class _Model:
    def __init__(self, *rounds):
        self.rounds = list(rounds)

    async def generate_response(self, messages, tools=None, on_delta=None):
        text, calls = self.rounds.pop(0)
        if on_delta is not None and text:
            await on_delta("text", text)
        return _round(text, calls)


class _Router:
    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        return CARD_MOVED


def _service():
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.widget_scopes, svc.widget_team, svc.widget_agent_lock = (), None, None
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router()
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


async def _loop(svc, agent_id=322):
    """The tool loop of a turn: the model moves #0422 to done, then answers."""
    runtime = NS(llm_manager=_Model(("Done, it's approved.", None)), agent_id=agent_id, workspace_id=WS,
                 metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": "Approve #0422"}]
    first = _round("", [_call(CARD_MOVE)])
    async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, prefetched=[]):
        yield chunk


def _frames_of(chunks, kind):
    return [json.loads(c[2:]) for c in chunks if isinstance(c, str) and c.startswith(f'd:{{"type": "{kind}"')]


def test_the_loop_sends_one_receipts_frame_before_its_answer():
    async def run():
        return [chunk async for chunk in _loop(_service())]
    chunks = asyncio.run(run())

    (frame,) = _frames_of(chunks, "receipts")
    (moved,) = frame["data"]["receipts"]
    assert (moved["subject"], moved["effect"], moved["status"]) == ("#0422", "moved to Done", DONE)
    assert frame["data"]["model"] == MODEL
    at = next(i for i, c in enumerate(chunks) if isinstance(c, str) and '"type": "receipts"' in c)
    final = next(i for i, c in enumerate(chunks) if isinstance(c, dict) and c.get("_final_response"))
    assert at < final                                 # before the answer and its additions


# ── the turn: every agent's, saved with the message ─────────────────────────

def _delegate_turn(monkeypatch, *, with_loop):
    """A DELEGATE turn: a specialist (agent 77) answers through the chat's own turn."""
    from consumers.chatbot.narration import reply_parts
    from consumers.chatbot.service import StreamingChatService

    monkeypatch.setattr(StreamingChatService, "_hidden_action_scope", lambda self, agent_id: nullcontext())
    svc, saved = _service(), []

    async def scoped(*args, **kwargs):
        if with_loop:
            async for chunk in _loop(svc, agent_id=77):
                if isinstance(chunk, str):
                    yield chunk
        yield svc.streaming_handler.format_aisdk_text("Here is what I found.")
        yield svc.streaming_handler.format_aisdk_finish()
        saved.append(reply_parts("", "", "Here is what I found."))

    svc._stream_response_with_agent_scoped = scoped

    async def run():
        return [c async for c in svc.stream_response_with_agent(
            chat_id="chat-1", messages=[{"role": "user", "content": "Approve #0422"}], agent_id=77, user_id=1)]
    return asyncio.run(run()), saved


def test_a_delegate_turn_with_calls_saves_their_receipts(monkeypatch):
    chunks, (parts,) = _delegate_turn(monkeypatch, with_loop=True)

    (frame,) = _frames_of(chunks, "receipts")                      # one frame: the loop's
    assert parts[0]["type"] == "receipts" and parts[0]["receipts"] == frame["data"]["receipts"]
    assert parts[0]["receipts"][0]["subject"] == "#0422"
    assert parts[-1] == {"type": "text", "text": "Here is what I found."}


def test_a_turn_that_ran_nothing_saves_empty_receipts_and_says_so_before_it_finishes(monkeypatch):
    chunks, (parts,) = _delegate_turn(monkeypatch, with_loop=False)

    (frame,) = _frames_of(chunks, "receipts")
    assert frame["data"]["receipts"] == []
    finish = next(i for i, c in enumerate(chunks) if '"type":"finish"' in c)
    assert chunks.index(next(c for c in chunks if '"type": "receipts"' in c)) == finish - 1
    assert parts[0] == {"type": "receipts", "receipts": []}


def test_the_turns_seams_are_wrapped_so_every_lane_has_receipts():
    """The turn entry every lane goes through (Auto's own, DELEGATE, the first reply, the
    loop and its forced synthesis), the loop, the automatic reads and the answer's model."""
    from consumers.chatbot.service import StreamingChatService

    for method, wrapper in (("stream_response_with_agent", "writes_its_receipts"),
                            ("_stream_tool_loop", "the_loop_writes_receipts"),
                            ("_retrieval_first", "its_reads_are_receipted"),
                            ("_answer_additions", "notes_the_answering_model")):
        assert getattr(StreamingChatService, method).__code__.co_qualname == f"{wrapper}.<locals>.wrapped", method


# ── the model never sees it ─────────────────────────────────────────────────

def test_the_receipts_part_is_never_read_as_the_reply():
    from api.chat import _parts_text
    from modules.context.sections.conversation import _parts_to_text

    parts = [{"type": "receipts", "receipts": [{"action": "store_memory", "kind": WRITE, "status": DONE,
                                                "subject": "", "effect": "memory saved", "link": None,
                                                "reason": None}]},
             {"type": "text", "text": "Noted."}]
    assert _parts_to_text(parts) == "Noted." and _parts_text(parts) == "Noted."


def test_outside_a_turn_the_saved_parts_are_as_before():
    from consumers.chatbot.narration import reply_parts

    assert reply_parts("", "", "Hi.") == [{"type": "text", "text": "Hi."}]
    assert reply_parts("", "", "Hi.", []) == [{"type": "receipts", "receipts": []}, {"type": "text", "text": "Hi."}]
