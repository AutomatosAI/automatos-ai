"""PRD-256 FX-002: the night harness shows what the owner sees (F394).

Night 12's persona graded 779 asks without a single receipt, saw 28 honesty lines where
the saved replies hold 158, and read 20 doubled replies the browser would have shown once.
These streams are shaped as the backend sends them: ``receipts_frames`` (a ``tool-data``
copy, then the typed ``receipts`` frame) and ``format_aisdk_narration``.
"""

import argparse
import json

from tests.sim import customer, sse
from tests.sim.api import Response
from tests.sim.customer_ops import receipt_line, render_turn
from tests.sim.driver_chat import turn_record

NARRATION = "Let me look at the board. "
DRAFT = "Done: card #0422 is in Done."
ANSWER = "I moved card #0422 to Done."
ABOVE = "I tried to move the card #0422, but it did not go through: the card is locked."
RECEIPTS = [
    {"action": "platform_list_tasks", "kind": "read", "status": "done", "subject": "tasks",
     "effect": "looked up", "link": None, "reason": None},
    {"action": "platform_update_task", "kind": "write", "status": "refused", "subject": "#0422",
     "effect": "moved to Done", "link": None, "reason": "the card is locked"},
]


def _d(kind, data):
    return "d:" + json.dumps({"type": kind, "data": data})


def _stream(retry=ANSWER):
    frame = {"receipts": RECEIPTS, "model": "gemini", "above": [ABOVE]}
    return "\n".join([
        _d("chat-id", {"chatId": "chat-1"}),
        "0:" + json.dumps(NARRATION),
        _d("narration", {"text": NARRATION}),
        _d("tool-start", {"toolCallId": "t1", "toolName": "platform_list_tasks", "input": {}}),
        _d("tool-end", {"toolCallId": "t1", "success": True}),
        "0:" + json.dumps(DRAFT),
        _d("narration", {"text": DRAFT, "retracted": True}),
        "0:" + json.dumps(retry),
        'd:{"type":"tool-data","data":' + json.dumps({"receipts": frame}) + "}",
        _d("receipts", frame),
        'd:{"type":"finish","finishReason":"stop"}',
    ])


def test_a_retracted_draft_leaves_the_reply_and_the_receipts_are_kept():
    turn = sse.parse_data_stream(_stream())

    assert turn.text == NARRATION + ANSWER                  # what the browser shows
    assert turn.text_raw == NARRATION + DRAFT + ANSWER      # every delta, as streamed
    assert turn.receipts == tuple(RECEIPTS)
    assert turn.above == (ABOVE,)
    assert turn.chat_id == "chat-1"


def test_a_retry_identical_to_its_draft_is_shown_once():
    """The retraction lands before the retry streams, so the retry is the one kept."""
    turn = sse.parse_data_stream(_stream(retry=DRAFT))

    assert turn.text == NARRATION + DRAFT
    assert turn.text_raw == NARRATION + DRAFT + DRAFT


def test_a_plain_narration_frame_and_a_turn_without_receipts_change_nothing():
    turn = sse.parse_data_stream("\n".join(["0:" + json.dumps(NARRATION), _d("narration", {"text": NARRATION}),
                                            "0:" + json.dumps(ANSWER)]))

    assert turn.text == turn.text_raw == NARRATION + ANSWER
    assert turn.receipts == () and turn.above == ()


def test_without_narration_matches_the_frontend():
    assert sse.without_narration("a b a", "a") == "a b "
    assert sse.without_narration("abc", "x") == "abc"
    assert sse.without_narration("abc", "") == "abc"


def test_the_turn_is_laid_out_receipts_then_lines_then_reply():
    shown = render_turn(RECEIPTS, [ABOVE], ANSWER)

    assert shown.splitlines() == [
        "receipts:",
        "  read · platform_list_tasks · tasks · done · -",
        "  write · platform_update_task · #0422 · refused · the card is locked",
        ABOVE,
        "",
        ANSWER,
    ]
    assert receipt_line({"kind": "write", "action": "x"}) == "write · x · - · - · -"
    assert render_turn([], [], "") == "(no reply text)"
    assert render_turn([], [ABOVE], "") == f"{ABOVE}\n\n(no reply text)"


def test_the_night_record_carries_the_owner_view():
    record = turn_record(1, "move #0422", sse.parse_data_stream(_stream()), 200, 900, 80)

    assert record["text"] == NARRATION + ANSWER
    assert record["text_raw"] == NARRATION + DRAFT + ANSWER
    assert record["receipts"] == RECEIPTS and record["above"] == [ABOVE]


def test_customer_chat_prints_the_owner_view_and_records_it(monkeypatch, tmp_path, capsys):
    class _Api:
        def stream_chat(self, text, **kw):
            return Response(200, _stream(), 900, {}), "sent-id"

    monkeypatch.setattr(customer, "_api", lambda args: (_Api(), type("S", (), {"chat_timeout_s": 5})()))
    monkeypatch.setenv(customer.NIGHT_DIR_ENV, str(tmp_path))

    args = argparse.Namespace(text="move #0422", chat_id=None, agent_id=None, json=False)
    assert customer.cmd_chat(args) == 0

    out = capsys.readouterr().out
    assert out.index("receipts:") < out.index(ABOVE) < out.index(ANSWER)
    assert DRAFT not in out
    record = json.loads((tmp_path / "chats.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["text"] == NARRATION + ANSWER and record["text_raw"] == NARRATION + DRAFT + ANSWER
    assert record["receipts"] == RECEIPTS and record["above"] == [ABOVE]
