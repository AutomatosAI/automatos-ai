"""PRD-258 US-004 — the voice-latency runner (``scripts/eval/voice_latency/``).

* the frames it speaks: a 3-word ask is three growing ``update_only`` frames,
  then one ``response_required`` with the right id and transcript; the
  ``call_details`` frame carries what ``parse_llm_request`` and the WS handler
  (``modules/voice/live_socket.LiveCallSocket``) read;
* ``TurnWatch`` times one reply the way the server sends it (the one-time ack
  is a first frame but never Auto's);
* the report: nearest-rank p50/p90, the results line, a real call's join;
* the ask set: a bad row is refused with its index.

No Retell, no model, no network, no database.
"""
from __future__ import annotations

import asyncio
import io
import sys
import uuid
from pathlib import Path

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_WS_ID = str(uuid.uuid4())
_CHAT_ID = str(uuid.uuid4())
_USER_ID = 42
_CALL_ID = "sim_test"
_EVAL_DIR = _orchestrator_root / "scripts" / "eval" / "voice_latency"


# ---------------------------------------------------------------------------
# The frames
# ---------------------------------------------------------------------------

def test_a_three_word_ask_grows_word_by_word_then_requires_a_response():
    from scripts.eval.voice_latency import protocol

    history = protocol.after_turn((), "hello there", "Hi Gerard.")
    frames = protocol.ask_frames(history, "what's  on board", response_id=2)

    assert [f["interaction_type"] for f in frames] == ["update_only"] * 3 + ["response_required"]
    spoken = [f["transcript"][-1]["content"] for f in frames[:3]]
    assert spoken == ["what's", "what's on", "what's on board"]
    for frame in frames:
        # earlier turns ride along, as in Retell's running transcript
        assert frame["transcript"][:2] == [
            {"role": "user", "content": "hello there"},
            {"role": "agent", "content": "Hi Gerard."},
        ]
        assert frame["transcript"][-1]["role"] == "user"
    required = frames[-1]
    assert required["response_id"] == 2
    assert required["transcript"][-1] == {"role": "user", "content": "what's on board"}
    assert all("response_id" not in f for f in frames[:3])


def test_the_pace_is_150_words_a_minute():
    from scripts.eval.voice_latency import protocol

    assert protocol.WORDS_PER_MINUTE == 150
    assert protocol.WORD_INTERVAL_SECONDS == pytest.approx(0.4)


def test_the_dynamic_variables_parse_as_the_handler_reads_them():
    from modules.voice.providers.retell import parse_llm_request
    from scripts.eval.voice_latency import protocol

    details = protocol.call_details_frame(_CALL_ID, workspace_id=_WS_ID, user_id=_USER_ID, chat_id=_CHAT_ID)
    required = protocol.ask_frames((), "what is on my board", response_id=1)[-1]
    req = parse_llm_request({**required, "call": details["call"]})

    assert req.interaction_type == "response_required"
    assert req.response_id == 1
    assert req.user_text == "what is on my board"
    assert req.call_id == _CALL_ID
    assert req.workspace_id == _WS_ID
    assert req.user_id == str(_USER_ID)
    assert req.chat_id == _CHAT_ID


class _ScriptedSocket:
    """A Retell socket double fed the simulator's own frames, then a disconnect."""

    def __init__(self, script):
        self.script = list(script)
        self.sent: list = []

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        return None

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        if not self.script:
            await asyncio.sleep(0.05)
            raise WebSocketDisconnect(1000)
        return self.script.pop(0)


def test_the_ws_handler_reads_the_simulators_frames():
    from modules.policy import ai_disclosure
    from modules.voice import live_socket
    from scripts.eval.voice_latency import protocol

    seen: list = []

    async def stream(req):
        seen.append(req)
        yield {"response_id": req.response_id, "content": "", "content_complete": True}

    details = protocol.call_details_frame(_CALL_ID, workspace_id=_WS_ID, user_id=_USER_ID, chat_id=_CHAT_ID)
    script = [details, protocol.ping_frame(123)] + protocol.ask_frames((), "how is the board", response_id=1)
    ws = _ScriptedSocket(script)
    async def first_socket(call_id):  # not a reconnect: no Redis, no AI reminder frame
        return False

    ports = live_socket.SocketPorts(
        stream_turn=stream,
        call_status=lambda cid: "minted",
        ai_greeting=ai_disclosure.VOICE_GREETING,
        ai_reminder=ai_disclosure.VOICE_REMINDER,
        socket_reconnected=first_socket,
    )
    session = live_socket.LiveCallSocket(ws, _CALL_ID, ports)
    asyncio.run(session.run())

    assert ws.sent[0]["response_type"] == "config"
    assert protocol.wants_call_details(ws.sent[0]) and protocol.wants_pings(ws.sent[0])
    assert {"response_type": "ping_pong", "timestamp": 123} in ws.sent
    assert session.stats["updates"] == 4 and session.stats["turns"] == 1
    assert len(seen) == 1
    req = seen[0]
    assert (req.response_id, req.user_text) == (1, "how is the board")
    assert (req.workspace_id, req.user_id, req.chat_id) == (_WS_ID, str(_USER_ID), _CHAT_ID)


# ---------------------------------------------------------------------------
# One reply, timed on the client side
# ---------------------------------------------------------------------------

def _frame(rid, content, complete=False):
    from modules.voice.providers.retell import wrap_ws_response

    return wrap_ws_response({"response_id": rid, "content": content, "content_complete": complete})


def test_the_ack_is_a_first_frame_but_not_autos():
    from scripts.eval.voice_latency.protocol import TurnWatch

    watch = TurnWatch(response_id=3, started=10.0, ack_text="One moment.")
    watch.observe(_frame(2, "late words from the last turn"), 10.1)  # another turn's: ignored
    watch.observe({"response_type": "ping_pong", "timestamp": 1}, 10.2)
    watch.observe(_frame(3, "One moment. "), 12.5)
    watch.observe(_frame(3, "The board is clear. "), 13.0)
    watch.observe(_frame(3, "", complete=True), 14.25)

    assert watch.ack_heard and watch.done
    assert watch.elapsed_ms(watch.first_frame_at) == 2500
    assert watch.elapsed_ms(watch.first_auto_frame_at) == 3000
    assert watch.elapsed_ms(watch.complete_at) == 4250
    assert watch.reply == "The board is clear."


def test_a_reply_that_never_comes_is_none_not_a_number():
    from scripts.eval.voice_latency.protocol import TurnWatch

    watch = TurnWatch(response_id=1, started=0.0)
    watch.observe(_frame(1, "", complete=True), 1.0)
    assert watch.done
    assert watch.elapsed_ms(watch.first_frame_at) is None
    assert watch.elapsed_ms(watch.first_auto_frame_at) is None


def test_the_ai_reminder_is_a_first_frame_but_not_autos():
    """PRD-264 EUAI-04: the reminder that opens a response never times as Auto's reply."""
    from modules.policy.ai_disclosure import VOICE_REMINDER
    from scripts.eval.voice_latency.protocol import TurnWatch

    watch = TurnWatch(response_id=5, started=0.0)
    watch.observe(_frame(5, VOICE_REMINDER + " "), 0.1)
    watch.observe(_frame(5, "The board is clear."), 0.9)
    watch.observe(_frame(5, "", complete=True), 1.0)
    assert watch.elapsed_ms(watch.first_frame_at) == 100
    assert watch.elapsed_ms(watch.first_auto_frame_at) == 900
    assert watch.reply == "The board is clear."


def test_once_the_ack_is_spent_the_same_words_are_autos():
    from scripts.eval.voice_latency.protocol import TurnWatch

    watch = TurnWatch(response_id=4, started=0.0, ack_text="")
    watch.observe(_frame(4, "One moment. "), 0.5)
    assert watch.elapsed_ms(watch.first_auto_frame_at) == 500


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_nearest_rank_percentiles_on_known_lists():
    from scripts.eval.voice_latency.report import nearest_rank

    ten = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    assert nearest_rank(ten, 50) == 500
    assert nearest_rank(ten, 90) == 900
    assert nearest_rank(list(reversed(ten)), 50) == 500  # order does not matter
    assert nearest_rank([7], 50) == 7 and nearest_rank([7], 90) == 7
    assert nearest_rank([1, 2, 3, 4, 5], 50) == 3
    assert nearest_rank([1, 2, 3, 4, 5], 90) == 5
    assert nearest_rank([None, 40, None, 20], 50) == 20  # missing values are left out
    assert nearest_rank([], 50) is None and nearest_rank([None], 90) is None


def _result(kind, first, auto, full, prep, silent=False):
    from scripts.eval.voice_latency.report import TurnResult

    return TurnResult(kind, first, auto, full, prep, tools_before_speech=0, silent=silent)


def test_the_results_line_format():
    from scripts.eval.voice_latency import report

    results = [
        _result("small_talk", 800, 800, 2000, 1500),
        _result("small_talk", 1200, 1200, 3000, 2500),
        _result("board", 2500, None, 9000, None, silent=True),
    ]
    line = report.results_line("2026-10-11T18:00:00+00:00", "abc1234", report.summarize(results))

    assert line == (
        "2026-10-11T18:00:00+00:00 · abc1234 · "
        "overall: first_frame 1200/2500 · first_auto_frame 800/1200 · full 3000/9000 · prep 1500/2500 · silent 1"
        " | small_talk: first_frame 800/1200 · first_auto_frame 800/1200 · full 2000/3000 · prep 1500/2500 · silent 0"
        " | board: first_frame 2500/2500 · first_auto_frame -/- · full 9000/9000 · prep -/- · silent 1"
    )


def test_a_real_call_joins_its_turns_with_retells_latency():
    from scripts.eval.voice_latency import report

    rows = [
        {"first_frame_ms": 900, "first_auto_frame_ms": 900, "total_ms": 4000, "prep_ms": 2000,
         "tools_before_speech": 0, "audio_delivered": True},
        {"first_frame_ms": 2500, "first_auto_frame_ms": 6000, "total_ms": 9000, "prep_ms": None,
         "tools_before_speech": 2, "audio_delivered": False},
    ]
    vendor = {"e2e": {"p50": 1450.5, "p90": 2100, "num": 2}, "asr": {"p50": 210, "p90": 340, "num": 2},
              "tts": {"p50": 300, "num": 2}}
    line = report.call_line("T", "sha1", rows, vendor)

    assert line == (
        "T · sha1 · call: first_frame 900/2500 · first_auto_frame 900/6000 · full 4000/9000"
        " · prep 2000/2000 · silent 1 · vendor e2e 1450/2100 · asr 210/340 · llm -/- · tts 300/-"
    )
    assert report.call_line("T", "s", [], None).endswith("vendor e2e -/- · asr -/- · llm -/- · tts -/-")


def test_server_columns_join_only_when_every_ask_has_its_row():
    from scripts.eval.voice_latency import run
    from scripts.eval.voice_latency.protocol import Ask, TurnWatch

    asks = [Ask("tool", "check the weather"), Ask("knowledge", "what is our policy")]
    watches = [TurnWatch(response_id=1, started=0.0), TurnWatch(response_id=2, started=0.0)]
    rows = [{"prep_ms": 1100, "tools_before_speech": 1}, {"prep_ms": 900, "tools_before_speech": 0}]

    joined, matched = run.turn_results(asks, watches, rows)
    assert matched and [r.prep_ms for r in joined] == [1100, 900]
    assert [r.tools_before_speech for r in joined] == [1, 0]
    assert all(r.silent for r in joined)  # no Auto frame was seen

    short, matched = run.turn_results(asks, watches, rows[:1])
    assert not matched and [r.prep_ms for r in short] == [None, None]


# ---------------------------------------------------------------------------
# The ask set
# ---------------------------------------------------------------------------

def test_a_good_ask_set_parses():
    from scripts.eval.voice_latency.protocol import Ask, parse_asks

    asks = parse_asks([{"kind": "board", "text": " what's on   my board "}, {"kind": "tool", "text": "weather"}])
    assert asks == (Ask("board", "what's on my board"), Ask("tool", "weather"))


@pytest.mark.parametrize(
    "rows, index",
    [
        ([{"kind": "board", "text": "ok"}, {"kind": "chit_chat", "text": "hi"}], 1),
        ([{"kind": "board", "text": "ok"}, {"kind": "tool", "text": "ok"}, {"kind": "tool", "text": "   "}], 2),
        ([{"kind": "knowledge"}], 0),
        (["just a string"], 0),
    ],
)
def test_a_bad_ask_row_is_refused_with_its_index(rows, index):
    from scripts.eval.voice_latency.protocol import AskSetError, parse_asks

    with pytest.raises(AskSetError, match=rf"^ask {index}:"):
        parse_asks(rows)


@pytest.mark.parametrize("raw", ["{}", "[]", "not json"])
def test_an_ask_set_that_is_not_a_list_is_refused(raw):
    from scripts.eval.voice_latency import run
    from scripts.eval.voice_latency.protocol import AskSetError

    with pytest.raises(AskSetError):
        run.read_asks(io.StringIO(raw))


def test_the_owners_data_can_never_be_committed_here():
    ignored = (_EVAL_DIR / ".gitignore").read_text().split()
    assert "results.md" in ignored and "asks*.json" in ignored


def test_the_runner_never_writes_a_file():
    source = "\n".join(p.read_text() for p in _EVAL_DIR.glob("*.py"))
    for writer in ("open(", ".write_text(", ".write_bytes(", "os.environ", "os.getenv"):
        assert writer not in source, writer
