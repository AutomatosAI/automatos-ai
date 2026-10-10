"""PRD-256 FX-001: the web chat takes its live receipts from a ``tool-data`` frame.

The web chat's stream reader (frontend lib/chat/hooks.ts ``useChat``) hands a ``tool-data`` frame
to its data callback and passes over a frame type it does not know; the receipts are taken there
(lib/chat/use-chat-with-receipts.ts), so ``useChat`` and ``Message``, both far over the
function-length limit, are not touched. The turn sends the same data twice: on a ``tool-data``
frame under ``receipts``, then as the one ``receipts`` frame. A public widget visitor gets neither
(F155).
"""
from __future__ import annotations

import contextvars
import json
from types import SimpleNamespace as NS

from consumers.chatbot import receipts as rc
from consumers.chatbot.streaming import get_streaming_handler

MOVED = {"action": "platform_update_task_status", "kind": "write", "status": "done", "subject": "#0422",
         "effect": "moved to Done", "link": "/command-center?tab=board&task_id=422", "reason": None}
ABOVE_LINE = "I tried to approve the mission and it didn't go through: Mission m-1 is not waiting for approval."


def _decoded(frame):
    assert frame.startswith("d:") and frame.endswith("\n")
    return json.loads(frame[2:])


def test_the_tool_data_frame_carries_the_receipts_frames_data_just_before_it():
    live, frame = rc.receipts_frames(get_streaming_handler(), [MOVED], "google/gemini-2.5-flash", [ABOVE_LINE])

    copy, receipts = _decoded(live), _decoded(frame)
    assert copy["type"] == "tool-data" and receipts["type"] == rc.FRAME
    assert copy["data"] == {rc.LIVE_KEY: receipts["data"]}
    assert receipts["data"] == {"receipts": [MOVED], "model": "google/gemini-2.5-flash", rc.ABOVE: [ABOVE_LINE]}


def test_a_turn_sends_them_once_and_a_widget_visitor_never():
    def frames(widget):
        chat = NS(streaming_handler=get_streaming_handler(), widget_mode=widget)
        rc._SENT.set(False)                                  # a fresh turn (writes_its_receipts sets it so)
        return rc._frames_once(chat, [MOVED], None), rc._frames_once(chat, [MOVED], None)

    first, again = contextvars.copy_context().run(frames, False)
    assert [_decoded(f)["type"] for f in first] == ["tool-data", rc.FRAME] and again == ()
    assert contextvars.copy_context().run(frames, True) == ((), ())
