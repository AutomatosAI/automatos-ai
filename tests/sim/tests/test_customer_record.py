"""P256-FIX-N17-11 (F443): a night's chats.jsonl row says what the model thought and what each tool gave back.

Night 16's ask 318 (the November price letter) could not be explained from the record: a row held
each call's arguments, not its results, and nothing of the turn's reasoning or of the order the
call, result and end frames came in. A ``cmd_chat`` row now carries ``reasoning``, ``tool_results``
(the stream's 500-character excerpts) and ``frame_order``.
"""

import argparse
import json

from tests.sim import customer
from tests.sim.sse import parse_data_stream

SEARCH = "platform_search_documents"
EXCERPT = '{"results": [{"file_name": "wholesale-terms-2026-10.md", "score": 0.62}]}'
BODY = "\n".join([
    'd:{"type": "chat-id", "data": {"chatId": "fb259dec"}}',
    'd:{"type": "reasoning", "data": {"delta": "The owner wants November prices. "}}',
    'd:{"type": "reasoning", "data": {"delta": "Search the documents."}}',
    'd:{"type": "tool-start", "data": {"toolCallId": "c1", "toolName": "%s", "input": {"query": "November prices"}}}'
    % SEARCH,
    'd:{"type": "tool-result", "data": {"toolCallId": "c1", "toolName": "%s", "result": %s}}'
    % (SEARCH, json.dumps(EXCERPT)),
    'd:{"type": "tool-end", "data": {"toolCallId": "c1", "toolName": "%s", "success": true, "durationMs": 412}}'
    % SEARCH,
    '0:"Here is the November letter."',
])


class Response:
    status, ms, body = 200, 1840, BODY


class ChatApi:
    def stream_chat(self, text, *, chat_id=None, agent_id=None, timeout_s=None):
        return Response(), chat_id or "sent-id"


def test_the_stream_keeps_reasoning_and_the_tool_frames_in_order():
    turn = parse_data_stream(BODY)

    assert turn.reasoning == "The owner wants November prices. Search the documents."
    assert [frame["frame"] for frame in turn.tool_frames] == ["tool-start", "tool-result", "tool-end"]
    assert customer.turn_trace(turn) == {
        "reasoning": "The owner wants November prices. Search the documents.",
        "tool_results": [{"id": "c1", "name": SEARCH, "result": EXCERPT}],
        "frame_order": [f"tool-start:{SEARCH}#c1", f"tool-result:{SEARCH}#c1", f"tool-end:{SEARCH}#c1"],
    }


def test_a_cmd_chat_row_carries_reasoning_tool_results_and_frame_order(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(customer.NIGHT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(customer, "_api", lambda args: (ChatApi(), argparse.Namespace(chat_timeout_s=60)))

    args = argparse.Namespace(text="I need the November price letter to go out", chat_id=None, agent_id=None,
                              json=False)
    assert customer.cmd_chat(args) == 0

    row = json.loads((tmp_path / "chats.jsonl").read_text(encoding="utf-8").strip())
    assert row["chat_id"] == "fb259dec" and row["text"] == "Here is the November letter."
    assert row["reasoning"] == "The owner wants November prices. Search the documents."
    assert row["tool_results"] == [{"id": "c1", "name": SEARCH, "result": EXCERPT}]
    assert row["frame_order"] == [f"tool-start:{SEARCH}#c1", f"tool-result:{SEARCH}#c1", f"tool-end:{SEARCH}#c1"]
    assert row["tool_calls"] == [{"name": SEARCH, "args": {"query": "November prices"}}]
    assert "tools: platform_search_documents" in capsys.readouterr().out
