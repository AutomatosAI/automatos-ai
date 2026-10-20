"""PRD-258 US-001 — each spoken turn records where its time went.

* the pure timing fields (``modules/voice/turn_timing.py``): all moments, missing
  moments, the ack never counted as Auto's frame, tool lines counted only before
  the first text (the prefixes ``StreamingHandler`` really emits);
* ``voice_turn_fields`` / ``record_voice_turn`` carry the new columns;
* a turn driven through the WS handler with the agent stream stubbed (300 ms,
  then two text units) writes ``first_auto_frame_ms`` ≥ 300 and
  ``audio_delivered``; a barge-in records ``first_frame_ms`` only if a frame went;
* the agent stream reports its facts in-band (where it landed, tools, prep_ms);
* D6: ``api/voice_retell.py`` never grows past its 1,005-line base.

No Retell, no model, no network, no database.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

# voice_retell.py on the PRD-258 base (D6: it ends every story no longer than this).
VOICE_RETELL_BASE_LINES = 1005
_WS_ID = str(uuid.uuid4())
_CHAT_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# The pure fields
# ---------------------------------------------------------------------------

def test_timing_fields_with_every_moment():
    from modules.voice.turn_timing import voice_timing_fields

    fields = voice_timing_fields(
        started=10.0, first_frame_at=10.25, first_auto_frame_at=10.8,
        tools_before_speech=2, prep_ms=812.6,
    )
    assert fields == {
        "first_frame_ms": 250,
        "first_auto_frame_ms": 800,
        "prep_ms": 813,
        "tools_before_speech": 2,
        "lead_in": False,
        "prefetch_hit": False,
    }


def test_timing_fields_never_invent_a_missing_moment():
    from modules.voice.turn_timing import voice_timing_fields

    fields = voice_timing_fields(
        started=10.0, first_frame_at=None, first_auto_frame_at=None,
        tools_before_speech=None, prep_ms=None,
    )
    assert fields["first_frame_ms"] is None
    assert fields["first_auto_frame_ms"] is None
    assert fields["prep_ms"] is None
    assert fields["tools_before_speech"] is None
    assert fields["lead_in"] is False and fields["prefetch_hit"] is False


def test_tool_lines_count_only_before_the_first_text():
    from consumers.chatbot.streaming import StreamingHandler
    from modules.voice.turn_timing import SpeechTally, is_tool_call_line, tally_chunk

    h = StreamingHandler()
    tool = h.format_aisdk_tool_start("t1", "search_knowledge", {"q": "x"})
    assert is_tool_call_line(tool)  # the constants match what the handler emits
    assert not is_tool_call_line(h.format_aisdk_tool_end("t1", "search_knowledge", True))
    assert not is_tool_call_line(h.format_aisdk_text("hello"))

    lines = [
        h.format_aisdk_data("progress", {"text": "looking"}),
        tool,
        h.format_aisdk_tool_end("t1", "search_knowledge", True),
        h.format_aisdk_tool_start("t2", "read_board", {}),
        h.format_aisdk_text(""),  # an empty text line is not speech
        h.format_aisdk_text("Here it is."),
        h.format_aisdk_tool_start("t3", "late_tool", {}),  # after speech: not counted
    ]
    tally = SpeechTally()
    for line in lines:
        tally = tally_chunk(tally, line)
    assert tally == SpeechTally(tools_before_speech=2, spoke=True)

    # one chunk carrying several lines counts them all, still up to the first text
    joined = tally_chunk(SpeechTally(), tool + tool + h.format_aisdk_text("Hi") + tool)
    assert joined == SpeechTally(tools_before_speech=2, spoke=True)


def test_speech_tap_passes_the_stream_through_unchanged():
    from consumers.chatbot.streaming import StreamingHandler
    from modules.voice.turn_timing import SpeechTap

    h = StreamingHandler()
    lines = [h.format_aisdk_tool_start("t1", "x", {}), h.format_aisdk_text("Hi."), "d:junk\n"]

    async def source():
        for line in lines:
            yield line

    async def run():
        tap = SpeechTap(source())
        seen = [line async for line in tap]
        return seen, tap.tally

    seen, tally = asyncio.run(run())
    assert seen == lines
    assert tally.tools_before_speech == 1 and tally.spoke is True


def test_facts_merge_keeps_what_each_item_set():
    from modules.voice.turn_timing import TurnFacts, facts_of, finished, landed, merge_facts

    facts = merge_facts(TurnFacts(), facts_of(landed("ws", "chat")))
    facts = merge_facts(facts, facts_of(finished(3, 640.0)))
    assert facts == TurnFacts(workspace_id="ws", conversation_id="chat", tools_before_speech=3, prep_ms=640.0)
    assert facts_of({"response_id": 1, "content": "hi", "content_complete": False}) is None


# ---------------------------------------------------------------------------
# Telemetry carries the new columns
# ---------------------------------------------------------------------------

_BASE_TURN = dict(
    stt_latency_ms=0, tts_latency_ms=0, total_ms=1200.4, transcript="hey",
    response_text="x" * 20, truncated=False, audio_delivered=True,
)


def test_voice_turn_fields_cover_the_new_keys():
    from modules.voice.telemetry import voice_turn_fields

    unmeasured = voice_turn_fields(**_BASE_TURN)
    for key in ("first_frame_ms", "first_auto_frame_ms", "prep_ms", "tools_before_speech"):
        assert key in unmeasured and unmeasured[key] is None
    assert unmeasured["lead_in"] is False and unmeasured["prefetch_hit"] is False

    measured = voice_turn_fields(
        **_BASE_TURN, first_frame_ms=250, first_auto_frame_ms=800, prep_ms=812.6, tools_before_speech=0,
    )
    assert (measured["first_frame_ms"], measured["first_auto_frame_ms"]) == (250, 800)
    assert measured["prep_ms"] == 813
    assert measured["tools_before_speech"] == 0  # zero tools is a measurement, not a gap
    assert measured["response_len"] == 20 and measured["transcript_len"] == 3  # lengths only


def test_record_voice_turn_writes_the_timing_columns(monkeypatch):
    import core.database.database as dbmod
    from modules.voice import telemetry as tel
    from modules.voice.turn_timing import voice_timing_fields

    added: list = []

    @contextmanager
    def fake_session():
        db = MagicMock()
        db.add.side_effect = added.append
        yield db

    monkeypatch.setattr(dbmod, "get_db_session", fake_session)
    timing = voice_timing_fields(
        started=1.0, first_frame_at=1.3, first_auto_frame_at=1.5, tools_before_speech=1, prep_ms=400.0
    )
    tel.record_voice_turn(
        workspace_id=_WS_ID, conversation_id="c1", message_id=None, call_id="call_t", timing=timing, **_BASE_TURN
    )
    assert len(added) == 1
    row = added[0]
    assert (row.first_frame_ms, row.first_auto_frame_ms, row.prep_ms, row.tools_before_speech) == (300, 500, 400, 1)
    assert row.lead_in is False and row.prefetch_hit is False
    assert row.call_id == "call_t"


def test_reply_prep_ms_is_read_from_the_context_trace():
    from modules.voice.call_binding import assistant_reply_prep_ms

    def db_with(row):
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = row
        return db

    since = datetime(2026, 10, 11, 9, 0)
    assert assistant_reply_prep_ms(db_with(({"prep_ms": 812.4},)), chat_id=_CHAT_ID, workspace_id=_WS_ID, turn_started_at=since) == 812.4
    assert assistant_reply_prep_ms(db_with(({"mode": "chat"},)), chat_id=_CHAT_ID, workspace_id=_WS_ID, turn_started_at=since) is None
    assert assistant_reply_prep_ms(db_with((None,)), chat_id=_CHAT_ID, workspace_id=_WS_ID, turn_started_at=since) is None
    assert assistant_reply_prep_ms(db_with(None), chat_id=_CHAT_ID, workspace_id=_WS_ID, turn_started_at=since) is None
    assert assistant_reply_prep_ms(db_with(({"prep_ms": True},)), chat_id=_CHAT_ID, workspace_id=_WS_ID, turn_started_at=since) is None


# ---------------------------------------------------------------------------
# Through the WS handler
# ---------------------------------------------------------------------------

class _ScriptedSocket:
    """A Retell socket double: each inbound message waits its delay; after the
    script, the socket stays open ``tail_wait`` seconds, then disconnects."""

    def __init__(self, script, tail_wait: float):
        self.script = list(script)
        self.tail_wait = tail_wait
        self.sent: list = []
        self.close_code = None

    async def accept(self):
        return None

    async def close(self, code=1000):
        self.close_code = code

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        if not self.script:
            await asyncio.sleep(self.tail_wait)
            raise WebSocketDisconnect(1000)
        delay, message = self.script.pop(0)
        await asyncio.sleep(delay)
        return message


def _turn(rid: int, text: str = "how is the board") -> dict:
    return {
        "interaction_type": "response_required",
        "response_id": rid,
        "transcript": [{"role": "user", "content": text}],
    }


_CALL_DETAILS = {
    "interaction_type": "call_details",
    "call": {"retell_llm_dynamic_variables": {"workspace_id": _WS_ID, "chat_id": _CHAT_ID}},
}


def _arm_socket(monkeypatch, stream, ack_seconds: float) -> list:
    """Patch the route around ``stream``; return the list record_voice_turn calls land in."""
    import api.voice_retell as vr
    from config import config as app_config
    from modules.voice import ai_notice, telemetry

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ("minted",)
    db.query.return_value.filter.return_value.scalar.return_value = "started"

    @contextmanager
    def fake_session():
        yield db

    recorded: list = []
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", fake_session)
    monkeypatch.setattr(vr, "_agent_retell_stream", stream)
    monkeypatch.setattr(telemetry, "record_voice_turn", lambda **kw: recorded.append(kw))
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", ack_seconds)
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_TEXT", "One moment.")
    # The call's first socket, not a reconnect: no PRD-264 AI reminder frame
    # comes before Auto's own (the timings below count frames).
    monkeypatch.setattr(ai_notice, "_mark_socket_open_sync", lambda call_id: False)
    return recorded


def _two_units_after(delay: float):
    from modules.voice.turn_timing import finished, landed

    async def stream(req):
        yield landed(_WS_ID, _CHAT_ID)
        await asyncio.sleep(delay)
        yield {"response_id": req.response_id, "content": "The board is clear. ", "content_complete": False}
        yield {"response_id": req.response_id, "content": "Nothing is blocked.", "content_complete": False}
        yield {"response_id": req.response_id, "content": "", "content_complete": True}
        yield finished(tools_before_speech=1, prep_ms=900.0)

    return stream


def test_a_stubbed_turn_writes_its_timings(monkeypatch):
    import api.voice_retell as vr
    from modules.voice.turn_timing import TURN_FACTS_KEY

    recorded = _arm_socket(monkeypatch, _two_units_after(0.3), ack_seconds=0)
    ws = _ScriptedSocket([(0, _CALL_DETAILS), (0, _turn(1))], tail_wait=0.6)
    asyncio.run(vr.retell_llm_websocket(ws, "call_timed"))

    assert len(recorded) == 1, "exactly one voice_turns row per live turn"
    row = recorded[0]
    timing = row["timing"]
    # no ack this turn: the first frame sent IS Auto's, sent once it arrived
    assert timing["first_frame_ms"] >= timing["first_auto_frame_ms"] >= 300
    assert timing["tools_before_speech"] == 1 and timing["prep_ms"] == 900
    assert timing["lead_in"] is False and timing["prefetch_hit"] is False
    assert row["audio_delivered"] is True
    assert row["workspace_id"] == _WS_ID and row["conversation_id"] == _CHAT_ID
    assert row["call_id"] == "call_timed"
    # total_ms is rounded only when the row is written; compare like with like
    assert int(round(row["total_ms"])) >= timing["first_auto_frame_ms"]
    # lengths-only: the reply's text never reaches telemetry, only its length
    assert set(row["response_text"]) == {"x"}
    assert len(row["response_text"]) == len("The board is clear. ") + len("Nothing is blocked.")
    # the in-band facts never go out on the socket
    assert not any(TURN_FACTS_KEY in m for m in ws.sent)


def test_the_ack_is_a_first_frame_but_not_autos(monkeypatch):
    import api.voice_retell as vr

    recorded = _arm_socket(monkeypatch, _two_units_after(0.3), ack_seconds=0.05)
    ws = _ScriptedSocket([(0, _CALL_DETAILS), (0, _turn(1))], tail_wait=0.6)
    asyncio.run(vr.retell_llm_websocket(ws, "call_acked"))

    assert any("One moment." in str(m.get("content")) for m in ws.sent)  # the ack went out
    timing = recorded[0]["timing"]
    assert timing["first_frame_ms"] is not None and timing["first_frame_ms"] < 300
    assert timing["first_auto_frame_ms"] >= 300  # Auto's own words, not the ack


def test_a_barge_in_records_a_first_frame_only_if_one_was_sent(monkeypatch):
    import api.voice_retell as vr
    from modules.voice import live_socket
    from modules.voice.turn_timing import landed

    async def stream(req):
        yield landed(_WS_ID, _CHAT_ID)
        if req.response_id == 3:  # the last turn ends on its own: nothing waits on teardown
            yield {"response_id": 3, "content": "", "content_complete": True}
            return
        if req.response_id == 1:  # speaks once, then is talked over
            yield {"response_id": 1, "content": "Let me check…", "content_complete": False}
        await asyncio.Event().wait()  # streams until superseded

    recorded = _arm_socket(monkeypatch, stream, ack_seconds=0)
    monkeypatch.setattr(live_socket, "TURN_EXIT_GRACE_SECONDS", 0.05)
    ws = _ScriptedSocket(
        [(0, _CALL_DETAILS), (0, _turn(1, "first")), (0.1, _turn(2, "second")), (0.1, _turn(3, "third"))],
        tail_wait=0.2,
    )
    asyncio.run(vr.retell_llm_websocket(ws, "call_barged"))

    rows = {row["transcript"]: row for row in recorded}
    assert set(rows) == {"first", "second", "third"}, "one row per turn, superseded or not"
    assert rows["first"]["timing"]["first_frame_ms"] is not None  # a frame went before the cancel
    assert rows["first"]["audio_delivered"] is True
    silent = rows["second"]["timing"]
    assert silent["first_frame_ms"] is None and silent["first_auto_frame_ms"] is None  # nothing invented
    assert rows["second"]["audio_delivered"] is False
    assert rows["third"]["timing"]["first_frame_ms"] is None  # an empty close is not a content frame


def test_a_turn_with_no_binding_writes_no_row(monkeypatch):
    """No proven workspace (the stream never said where it landed) → no row."""
    import api.voice_retell as vr

    async def refused(req):
        yield {"response_id": req.response_id, "content": "", "content_complete": True}

    recorded = _arm_socket(monkeypatch, refused, ack_seconds=0)
    ws = _ScriptedSocket([(0, _CALL_DETAILS), (0, _turn(1))], tail_wait=0.1)
    asyncio.run(vr.retell_llm_websocket(ws, "call_refused"))
    assert recorded == []


# ---------------------------------------------------------------------------
# The agent stream reports its facts in-band
# ---------------------------------------------------------------------------

def test_the_agent_stream_reports_where_it_landed_and_how_it_went(monkeypatch):
    import consumers.chatbot as chatbot_pkg

    import api.chat as chat_api
    import api.voice_retell as vr
    import modules.voice.call_binding as cb
    import services.board_events as be
    from consumers.chatbot.streaming import StreamingHandler
    from modules.voice.call_binding import CallBinding
    from modules.voice.providers.retell import RetellLLMRequest
    from modules.voice.turn_timing import TurnFacts, facts_of

    h = StreamingHandler()

    class FakeChatService:
        def __init__(self, db):
            pass

        def get_messages_by_chat_id(self, cid):
            return []

    class FakeStreaming:
        def __init__(self, db, workspace_id=None):
            pass

        def stream_response_with_agent(self, **kwargs):
            async def gen():
                yield h.format_aisdk_tool_start("t1", "read_board", {})
                yield h.format_aisdk_text("The board is clear.")

            return gen()

    @contextmanager
    def fake_session():
        yield MagicMock()

    monkeypatch.setattr(chatbot_pkg, "ChatService", FakeChatService)
    monkeypatch.setattr(chatbot_pkg, "StreamingChatService", FakeStreaming)
    monkeypatch.setattr(
        cb, "resolve_call_binding",
        lambda db, **k: CallBinding(chat_id=_CHAT_ID, user_id=7, workspace_id=_WS_ID, bound=True),
    )
    monkeypatch.setattr(cb, "upsert_voice_user_message", lambda db, **k: None)
    monkeypatch.setattr(cb, "stamp_assistant_voice_source", lambda db, **k: 0)
    monkeypatch.setattr(cb, "assistant_reply_prep_ms", lambda db, **k: 640.0)
    monkeypatch.setattr(chat_api, "get_default_agent_id", lambda db, w: 1)
    monkeypatch.setattr(be, "notify_chat_event", lambda db, **k: None)
    monkeypatch.setattr(vr, "get_db_session", fake_session)

    req = RetellLLMRequest(
        response_id=4, user_text="board?", interaction_type="response_required",
        workspace_id=_WS_ID, agent_id=None, call_id="call_inner",
    )

    async def run():
        return [item async for item in vr._agent_retell_stream_inner(req)]

    items = asyncio.run(run())
    facts = [facts_of(i) for i in items if facts_of(i) is not None]
    frames = [i for i in items if facts_of(i) is None]
    assert facts[0] == TurnFacts(workspace_id=_WS_ID, conversation_id=_CHAT_ID)  # first, before any frame
    assert items.index(next(i for i in items if facts_of(i) is not None)) == 0
    assert facts[-1] == TurnFacts(tools_before_speech=1, prep_ms=640.0)
    assert any(f.get("content") for f in frames)
    assert frames[-1]["content_complete"] is True


# ---------------------------------------------------------------------------
# D6 — the oversized route file never grows
# ---------------------------------------------------------------------------

def test_voice_retell_is_no_longer_than_its_base():
    path = _orchestrator_root / "api" / "voice_retell.py"
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= VOICE_RETELL_BASE_LINES, f"voice_retell.py grew to {lines} lines (D6)"


def test_the_moved_live_path_never_imports_the_pod_voice_client():
    """The PRD-207 §6 guard (test_prd207_voice_live.py) follows the code that left
    voice_retell.py: the moved live-path modules never import modules/voice/client."""
    import re

    pattern = re.compile(
        r"from\s+modules\.voice\.client\s+import|from\s+modules\.voice\s+import\s+([\w,\s]*\b)?client\b"
        r"|import\s+modules\.voice\.client\b"
    )
    moved = ("live_socket.py", "call_events.py", "caller.py", "turn_timing.py")
    offenders = [
        name for name in moved
        if pattern.search((_orchestrator_root / "modules" / "voice" / name).read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_helpers_left_the_route_file():
    source = (_orchestrator_root / "api" / "voice_retell.py").read_text(encoding="utf-8")
    for name in ("_apply_call_event", "_named_call_agent", "_resolve_caller_int_id"):
        assert f"def {name}(" not in source
