"""Voice latency A-4 — the quick-ask runner (``scripts/eval/voice_latency/quick_ask*.py``).

* the requests it sends: a spoken chat turn with a turn reference the server keeps, the
  agent only when asked for, the chat id only from the second ask on;
* ``QuickAskWatch`` times one reply over a recorded stream with a fake clock;
* the sentence rule is the phone's, and follows ``spoken_unit_ends``;
* a derived measure is None when a part is missing, never a guess;
* the report takes the measure list; Auto Live's line is unchanged with the default;
* the server's split is read from a trace, and a hosted run says ``server_marks=unavailable``;
* the token file is read again for each request and is in no output.

No model, no network, no database.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_WS_ID = str(uuid.uuid4())
_CHAT_ID = str(uuid.uuid4())
_TOKEN = "tok-not-a-real-bearer-0123456789"
_WALL_MS = 1762732800000
_ASK_TEXT = "what is on my board today"
_REPLY_ONE = "Sure."
_REPLY_TWO = " Two things are open."
_CHAT_ID_LINE = 'd:{"type":"chat-id","chatId":"%s","aiGenerated":true}' % _CHAT_ID
_FINISH_LINE = 'd:{"type":"finish","finishReason":"stop","aiGenerated":true}'
_REPLY_LINES = (_CHAT_ID_LINE, "0:" + json.dumps(_REPLY_ONE), "0:" + json.dumps(_REPLY_TWO), _FINISH_LINE)
_TRACE = {
    "prep_ms": 610,
    "timing": {
        "v": 1,
        "origin": "request",
        "spoken": True,
        "marks": {"authorised": 41, "routed": 95, "locked": 104, "agent": 310, "model_start": 1650,
                  "first_response": 4210, "reply_saved": 6020},
        "stages": {"agent_context": 12, "tools": 380, "context": 610, "knowledge": 230, "model": 2560},
        "counts": {"model_calls": 1},
    },
}


# ---------------------------------------------------------------------------
# The requests
# ---------------------------------------------------------------------------

def test_the_chat_body_is_a_spoken_turn_with_a_ref_the_server_keeps():
    from api.chat_models import ChatRequest
    from core.monitoring.turn_ref import REF_PATTERN
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    ref = wire.turn_ref(_WALL_MS, 3)
    body = wire.chat_body(_ASK_TEXT, ref)

    assert REF_PATTERN.fullmatch(ref)
    assert body["spokenMode"] is True and body["turnRef"] == ref
    assert body["message"] == {"role": "user", "parts": [{"type": "text", "text": _ASK_TEXT}]}
    assert "agentId" not in body and "id" not in body
    request = ChatRequest(**body)  # the route's own model takes it, and keeps the ref
    assert request.spokenMode is True and request.turnRef == ref and request.id is None


def test_the_agent_and_the_chat_id_are_sent_only_when_given():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    body = wire.chat_body(_ASK_TEXT, "vt-abc123-2", chat_id=_CHAT_ID, agent_id=7)
    assert body["id"] == _CHAT_ID and body["agentId"] == 7
    assert "id" not in wire.chat_body(_ASK_TEXT, "vt-abc123-2", chat_id=None, agent_id=7)
    assert "agentId" not in wire.chat_body(_ASK_TEXT, "vt-abc123-2", chat_id=_CHAT_ID)


def test_the_ref_is_the_send_time_and_the_ask_number_in_base_36():
    from core.monitoring.turn_ref import REF_PATTERN
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    assert [wire.base36(n) for n in (0, 9, 10, 35, 36, 1295, 1296)] == ["0", "9", "a", "z", "10", "zz", "100"]
    assert wire.turn_ref(36, 37) == "vt-10-11"
    assert wire.turn_ref(_WALL_MS, 1) == f"vt-{wire.base36(_WALL_MS)}-1"
    for number in (1, 20, 400):
        assert REF_PATTERN.fullmatch(wire.turn_ref(_WALL_MS, number))


def test_the_speech_body_is_the_sentence_and_the_same_ref():
    from modules.voice.speech import SpeechBody
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    body = wire.speech_body(_REPLY_ONE, "vt-abc123-1")
    assert body == {"text": _REPLY_ONE, "turnRef": "vt-abc123-1"}
    assert SpeechBody(**body).turnRef == "vt-abc123-1"


# ---------------------------------------------------------------------------
# The phone's sentence rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Sure. Two", "Sure."),
        ("3.5 is", None),
        ("Sure.", None),  # the next token may still change it
        ("Sure", None),
        ("", None),
        ("Talk to Dr. Lee today. Then", "Talk to Dr. Lee today."),
        ("Use e.g. the board", None),
        ("... Wait. Then", "Wait."),  # a run with nothing to say is not a sentence
        ('He said "Go." Then', 'He said "Go."'),
        ("Really?! Yes", "Really?!"),
    ],
)
def test_a_sentence_ends_at_a_terminator_then_whitespace(text, expected):
    from scripts.eval.voice_latency.quick_ask_protocol import first_sentence

    assert first_sentence(text) == expected
    assert first_sentence(text, at_text_end=False) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Sure.", "Sure."),  # a sentence only with the flag on
        ("Sure. Two", "Sure."),
        ("3.5 is", None),
        ("Talk to Dr.", None),
        ("It is 3.", "It is 3."),
        ("Sure", None),
    ],
)
def test_with_unit_ends_a_terminator_at_the_end_of_the_text_ends_a_sentence(text, expected):
    from scripts.eval.voice_latency.quick_ask_protocol import first_sentence

    assert first_sentence(text, at_text_end=True) == expected


def test_the_sentence_rule_follows_the_servers_capability_unless_forced():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    assert wire.unit_ends(["spoken_mode", "spoken_unit_ends"]) is True
    assert wire.unit_ends(["spoken_mode", "tts"]) is False
    assert wire.unit_ends(["spoken_mode", "spoken_unit_ends"], "off") is False
    assert wire.unit_ends(["spoken_mode"], "on") is True
    assert wire.unit_ends([], None) is False


# ---------------------------------------------------------------------------
# One reply, timed on the client side
# ---------------------------------------------------------------------------

def _watched(lines, *, at_text_end=False, times=None):
    """A watch fed ``lines`` at 1.0 s, 1.5 s, 2.0 s, …: headers at 0.2 s, the end 0.5 s after the last."""
    from scripts.eval.voice_latency.quick_ask_protocol import QuickAskWatch

    watch = QuickAskWatch(started=0.0, at_text_end=at_text_end)
    watch.headers(0.2)
    at = 0.5
    for index, line in enumerate(lines):
        at = times[index] if times else 1.0 + 0.5 * index
        watch.observe(line, at)
    watch.end(at + 0.5)
    return watch


def test_the_watch_times_a_recorded_stream():
    watch = _watched(_REPLY_LINES)  # chat-id 1.0, "Sure." 1.5, " Two things are open." 2.0, finish 2.5

    assert watch.chat_id == _CHAT_ID
    assert not watch.silent and not watch.failed
    assert watch.ms(watch.headers_at) == 200
    assert watch.ms(watch.first_text_at) == 1500
    assert watch.sentence == "Sure."
    assert watch.ms(watch.first_sentence_at) == 2000  # known only once the next unit came
    assert watch.ms(watch.ended_at) == 3000


def test_with_unit_ends_the_first_sentence_is_timed_at_its_own_frame():
    watch = _watched(_REPLY_LINES, at_text_end=True)

    assert watch.sentence == "Sure."
    assert watch.ms(watch.first_sentence_at) == watch.ms(watch.first_text_at) == 1500


def test_a_stream_with_no_text_is_silent():
    watch = _watched((_CHAT_ID_LINE, _FINISH_LINE))

    assert watch.silent
    assert watch.chat_id == _CHAT_ID
    assert watch.sentence is None
    assert watch.ms(watch.first_text_at) is None and watch.ms(watch.first_sentence_at) is None
    assert watch.ms(watch.ended_at) == 2000


def test_a_reply_with_no_sentence_end_is_one_sentence_when_the_stream_ends():
    watch = _watched((_CHAT_ID_LINE, '0:"It\'s five o\'clock"', _FINISH_LINE))

    assert watch.sentence == "It's five o'clock"
    assert watch.ms(watch.first_text_at) == 1500
    assert watch.ms(watch.first_sentence_at) == watch.ms(watch.ended_at) == 2500


def test_the_watch_ignores_what_is_no_frame_and_notes_an_error_frame():
    watch = _watched(("", "0:not json", '0:""', "0:17", 'd:["chat-id"]', "x:whatever", 'e:{"message":"no"}'))

    assert watch.silent and watch.failed
    assert watch.chat_id is None and watch.sentence is None


def test_the_first_chat_id_and_the_first_headers_win():
    from scripts.eval.voice_latency.quick_ask_protocol import QuickAskWatch

    watch = QuickAskWatch(started=0.0)
    watch.headers(0.1)
    watch.headers(0.9)
    watch.observe(_CHAT_ID_LINE, 0.2)
    watch.observe('d:{"type":"chat-id","chatId":"another"}', 0.3)
    watch.end(1.0)
    watch.end(5.0)

    assert watch.ms(watch.headers_at) == 100
    assert watch.chat_id == _CHAT_ID
    assert watch.ms(watch.ended_at) == 1000


# ---------------------------------------------------------------------------
# The measures of one ask
# ---------------------------------------------------------------------------

def _watch(headers=None, first_text=None, first_sentence=None, full=None):
    """A watch that has already seen these moments (seconds from its start)."""
    from scripts.eval.voice_latency.quick_ask_protocol import QuickAskWatch

    return QuickAskWatch(
        started=0.0, headers_at=headers, first_text_at=first_text, first_sentence_at=first_sentence, ended_at=full,
    )


def test_the_three_derived_measures_are_sums_of_measured_parts():
    from scripts.eval.voice_latency.quick_ask_protocol import SpeechTimes, quick_ask_result

    result = quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0), SpeechTimes(first_byte_ms=620, total_ms=910))

    assert (result.kind, result.silent) == ("board", False)
    assert (result.headers_ms, result.first_text_ms, result.first_sentence_ms, result.full_ms) == (100, 1000, 1200, 3000)
    assert (result.tts_first_byte_ms, result.tts_total_ms) == (620, 910)
    assert result.go_first_audio_ms == 1200 + 910
    assert result.streamed_first_audio_ms == 1200 + 620
    assert result.held_first_audio_ms == 3000 + 910


def test_a_reply_that_sent_an_error_frame_has_no_full_time():
    from scripts.eval.voice_latency.quick_ask_protocol import QuickAskWatch, SpeechTimes, quick_ask_result

    failed = QuickAskWatch(started=0.0, headers_at=0.1, first_text_at=1.0, first_sentence_at=1.2, ended_at=3.0,
                           failed=True)

    result = quick_ask_result("board", failed, SpeechTimes(first_byte_ms=620, total_ms=910))

    assert result.full_ms is None and result.held_first_audio_ms is None
    assert result.first_sentence_ms == 1200 and result.go_first_audio_ms == 2110  # what was measured is kept


def test_a_derived_measure_is_none_when_a_part_is_missing():
    from scripts.eval.voice_latency.quick_ask_protocol import NO_SPEECH, SpeechTimes, quick_ask_result

    spoken = SpeechTimes(first_byte_ms=620, total_ms=910)

    no_speech = quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0), NO_SPEECH)
    assert no_speech.first_sentence_ms == 1200 and no_speech.full_ms == 3000
    assert no_speech.go_first_audio_ms is None
    assert no_speech.streamed_first_audio_ms is None
    assert no_speech.held_first_audio_ms is None
    assert quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0)) == no_speech  # no speech is the default

    no_sentence = quick_ask_result("board", _watch(headers=0.1, full=3.0), spoken)
    assert no_sentence.silent
    assert no_sentence.go_first_audio_ms is None and no_sentence.streamed_first_audio_ms is None
    assert no_sentence.held_first_audio_ms == 3910

    never_ended = quick_ask_result("board", _watch(0.1, 1.0, 1.2, None), spoken)
    assert never_ended.full_ms is None and never_ended.held_first_audio_ms is None
    assert never_ended.go_first_audio_ms == 2110

    no_first_byte = quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0), SpeechTimes(None, 910))
    assert no_first_byte.streamed_first_audio_ms is None and no_first_byte.go_first_audio_ms == 2110


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _three_results():
    from scripts.eval.voice_latency.quick_ask_protocol import SpeechTimes, quick_ask_result

    return [
        quick_ask_result("small_talk", _watch(0.1, 1.0, 1.2, 3.0), SpeechTimes(500, 800)),
        quick_ask_result("small_talk", _watch(0.2, 2.0, 2.4, 5.0), SpeechTimes(700, 1000)),
        quick_ask_result("tool", _watch(headers=0.3, full=9.0)),
    ]


def test_the_quick_ask_summary_is_nearest_rank_per_kind_and_overall():
    from scripts.eval.voice_latency import report
    from scripts.eval.voice_latency.quick_ask_protocol import QUICK_ASK_MEASURES

    summary = report.summarize(_three_results(), QUICK_ASK_MEASURES)

    assert list(summary) == ["overall", "small_talk", "tool"]
    overall = summary["overall"]
    assert overall["headers"] == {"p50": 200, "p90": 300}
    assert overall["first_text"] == {"p50": 1000, "p90": 2000}
    assert overall["full"] == {"p50": 5000, "p90": 9000}
    assert overall["go_first_audio"] == {"p50": 2000, "p90": 3400}
    assert overall["streamed_first_audio"] == {"p50": 1700, "p90": 3100}
    assert overall["held_first_audio"] == {"p50": 3800, "p90": 6000}
    assert (overall["silent"], overall["turns"]) == (1, 3)
    assert summary["tool"]["first_text"] == {"p50": None, "p90": None}
    assert "first_frame" not in overall and "srv_authorised" not in overall


def test_the_quick_ask_results_line_shape():
    from scripts.eval.voice_latency import report
    from scripts.eval.voice_latency.quick_ask_protocol import QUICK_ASK_MEASURES

    summary = report.summarize(_three_results(), QUICK_ASK_MEASURES)
    line = report.results_line("2026-11-10T09:00:00+00:00", "abc1234", summary, QUICK_ASK_MEASURES)

    assert line == (
        "2026-11-10T09:00:00+00:00 · abc1234 · "
        "overall: headers 200/300 · first_text 1000/2000 · first_sentence 1200/2400"
        " · tts_first_byte 500/700 · tts_total 800/1000 · full 5000/9000 · go_first_audio 2000/3400"
        " · streamed_first_audio 1700/3100 · held_first_audio 3800/6000 · silent 1"
        " | small_talk: headers 100/200 · first_text 1000/2000 · first_sentence 1200/2400"
        " · tts_first_byte 500/700 · tts_total 800/1000 · full 3000/5000 · go_first_audio 2000/3400"
        " · streamed_first_audio 1700/3100 · held_first_audio 3800/6000 · silent 0"
        " | tool: headers 300/300 · first_text -/- · first_sentence -/-"
        " · tts_first_byte -/- · tts_total -/- · full 9000/9000 · go_first_audio -/-"
        " · streamed_first_audio -/- · held_first_audio -/- · silent 1"
    )


def test_auto_lives_results_line_is_unchanged_with_the_default_measures():
    """The string ``test_prd258_voice_latency_runner.py`` expects, byte for byte."""
    from scripts.eval.voice_latency import report
    from scripts.eval.voice_latency.report import TurnResult

    results = [
        TurnResult("small_talk", 800, 800, 2000, 1500, tools_before_speech=0, silent=False),
        TurnResult("small_talk", 1200, 1200, 3000, 2500, tools_before_speech=0, silent=False),
        TurnResult("board", 2500, None, 9000, None, tools_before_speech=0, silent=True),
    ]
    expected = (
        "2026-10-11T18:00:00+00:00 · abc1234 · "
        "overall: first_frame 1200/2500 · first_auto_frame 800/1200 · full 3000/9000 · prep 1500/2500 · silent 1"
        " | small_talk: first_frame 800/1200 · first_auto_frame 800/1200 · full 2000/3000 · prep 1500/2500 · silent 0"
        " | board: first_frame 2500/2500 · first_auto_frame -/- · full 9000/9000 · prep -/- · silent 1"
    )

    summary = report.summarize(results)
    assert report.results_line("2026-10-11T18:00:00+00:00", "abc1234", summary) == expected
    assert summary == report.summarize(results, report.MEASURES)
    assert report.results_line("2026-10-11T18:00:00+00:00", "abc1234", summary, report.MEASURES) == expected
    assert report.group_stats(results) == report.group_stats(results, report.MEASURES)
    assert report.group_segment("board", summary["board"]) == (
        "board: first_frame 2500/2500 · first_auto_frame -/- · full 9000/9000 · prep -/- · silent 1"
    )


# ---------------------------------------------------------------------------
# The server's split
# ---------------------------------------------------------------------------

def test_the_server_marks_are_read_from_a_trace_with_timing():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    marks = wire.server_marks(_TRACE)

    assert set(marks) == {name for name, _ in wire.SERVER_MEASURES}
    assert marks["srv_authorised"] == 41 and marks["srv_routed"] == 95 and marks["srv_locked"] == 104
    assert marks["srv_agent"] == 310 and marks["srv_model_start"] == 1650 and marks["srv_first_response"] == 4210
    assert marks["srv_first_text"] is None  # a snapshot at the save can lack it
    assert marks["srv_tools"] == 380 and marks["srv_model"] == 2560
    assert marks["srv_composio"] is None
    assert marks["srv_model_calls"] == 1


@pytest.mark.parametrize(
    "trace",
    [None, {}, {"prep_ms": 610}, {"timing": None}, {"timing": "soon"}, {"timing": {"marks": "x", "stages": None}},
     {"timing": {"marks": {"authorised": "41", "routed": True}}}, "not a trace"],
)
def test_every_server_mark_is_none_without_a_usable_timing(trace):
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    marks = wire.server_marks(trace)
    assert set(marks) == {name for name, _ in wire.SERVER_MEASURES}
    assert all(value is None for value in marks.values())


def test_a_result_built_with_no_trace_has_every_server_measure_none():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    result = wire.quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0), wire.SpeechTimes(620, 910))
    assert all(getattr(result, attr) is None for _, attr in wire.SERVER_MEASURES)

    with_trace = wire.quick_ask_result("board", _watch(0.1, 1.0, 1.2, 3.0), wire.SpeechTimes(620, 910), _TRACE)
    assert with_trace.srv_authorised == 41 and with_trace.srv_model_calls == 1
    assert with_trace.first_text_ms == 1000  # the runner's own first text is kept apart from the server's


def test_traces_are_keyed_by_the_ref_their_timing_carries():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    mine = {"timing": {**_TRACE["timing"], "ref": "vt-abc123-1"}}
    found = wire.timings_by_ref([None, {"prep_ms": 1}, _TRACE, mine, {"timing": {"ref": 7}}, "x"])

    assert found == {"vt-abc123-1": mine}


def test_the_measures_and_the_note_say_whether_the_server_was_read():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    assert wire.measures_for(False) == wire.QUICK_ASK_MEASURES
    assert wire.measures_for(True) == wire.QUICK_ASK_MEASURES + wire.SERVER_MEASURES
    assert wire.server_marks_note(False) == "server_marks=unavailable"
    assert wire.server_marks_note(True) == "server_marks=read"
    names = [name for name, _ in wire.measures_for(True)]
    assert len(names) == len(set(names))  # the server's first_text never shadows the runner's


# ---------------------------------------------------------------------------
# The run, against a server double
# ---------------------------------------------------------------------------

class _Response:
    """A streamed response double: a status, then lines (chat), chunks (speech) or a document."""

    def __init__(self, status=200, lines=(), chunks=(), document=None):
        self.status_code = status
        self._lines = tuple(lines)
        self._chunks = tuple(chunks)
        self._document = document

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def iter_bytes(self):
        yield from self._chunks

    def json(self):
        return self._document


class _Server:
    """The HTTP client double: discovery, then one scripted chat reply per ask, and speech."""

    def __init__(self, capabilities, replies, speech=None):
        self.capabilities = list(capabilities)
        self.replies = list(replies)
        self.speech = speech
        self.discovery_calls: list = []
        self.requests: list = []

    def get(self, url):
        self.discovery_calls.append(url)
        return _Response(document={"schema": 1, "capabilities": self.capabilities})

    def stream(self, method, url, json=None, headers=None):
        self.requests.append({"method": method, "url": url, "json": json, "headers": dict(headers or {})})
        if url.endswith("/api/voice/speech"):
            return self.speech or _Response(chunks=(b"ID3", b"", b"audio"))
        return self.replies.pop(0)

    def sent_to(self, path):
        return [request for request in self.requests if request["url"].endswith(path)]


def _chat(*lines):
    return _Response(lines=lines)


@pytest.fixture
def runner(monkeypatch):
    """The CLI module on a clock that ticks 100 ms per reading, with no pause and a fixed wall time."""
    from scripts.eval.voice_latency import quick_ask

    ticks = iter(range(10_000))
    monkeypatch.setattr(quick_ask, "now", lambda: next(ticks) / 10)
    monkeypatch.setattr(quick_ask, "pause", lambda: None)
    monkeypatch.setattr(quick_ask, "wall_ms", lambda: _WALL_MS)
    return quick_ask


def _asks(count=1):
    from scripts.eval.voice_latency.protocol import Ask

    return [Ask("board", _ASK_TEXT)] * count


def _args(runner, *extra):
    return runner.build_parser().parse_args(["--workspace", _WS_ID, "--sha", "abc1234", *extra])


def _no_database(runner, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a hosted run has no database to read")

    monkeypatch.setattr(runner, "read_timings", refuse)


def test_a_hosted_run_says_server_marks_unavailable(runner, monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(_TOKEN + "\n")
    _no_database(runner, monkeypatch)
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES)])

    outcome = runner.quick_ask(_args(runner, "--token-file", str(token_file)), _asks(), server)

    assert outcome["server_marks"] == "unavailable" and outcome["server_traces"] == 0
    assert outcome["results_line"].endswith(" · server_marks=unavailable")
    assert "srv_" not in outcome["results_line"]
    turn = outcome["turns"][0]
    assert all(turn[name] is None for name in turn if name.startswith("srv_"))
    assert turn["first_sentence_ms"] is not None and turn["go_first_audio_ms"] is not None
    assert outcome["chat_ids"] == [_CHAT_ID]


def test_the_token_file_is_read_for_each_request_and_is_in_no_output(runner, monkeypatch, tmp_path, caplog):
    _no_database(runner, monkeypatch)
    reads: list = []

    def reader(path):
        reads.append(path)
        return _TOKEN

    monkeypatch.setattr(runner, "read_token", reader)
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES), _chat(*_REPLY_LINES)])
    token_path = str(tmp_path / "token")

    with caplog.at_level(logging.DEBUG):
        outcome = runner.quick_ask(_args(runner, "--token-file", token_path), _asks(2), server)

    assert len(server.requests) == 4  # two chat turns, two speech requests
    assert reads == [token_path] * 4  # read again before each one
    assert all(request["headers"]["Authorization"] == f"Bearer {_TOKEN}" for request in server.requests)
    assert all(request["headers"]["X-Workspace-ID"] == _WS_ID for request in server.requests)
    printed = json.dumps(outcome, indent=2, default=str) + outcome["results_line"] + caplog.text
    assert _TOKEN not in printed
    # ... and neither is what was asked or answered
    assert _ASK_TEXT not in printed and "Two things are open" not in printed and "Sure." not in printed


def test_the_token_is_whatever_the_file_holds_now(tmp_path):
    from scripts.eval.voice_latency import quick_ask
    from scripts.eval.voice_latency.run import RunRefused

    token_file = tmp_path / "token"
    token_file.write_text("first-token\n")
    target = quick_ask.Target(base_url="http://localhost:8000", workspace_id=_WS_ID, token_file=str(token_file))
    assert target.headers()["Authorization"] == "Bearer first-token"

    token_file.write_text("  second-token  \n")
    assert target.headers()["Authorization"] == "Bearer second-token"

    token_file.write_text("\n")
    with pytest.raises(RunRefused) as empty:
        target.headers()
    with pytest.raises(RunRefused) as missing:
        quick_ask.read_token(str(tmp_path / "absent"))
    assert str(tmp_path) not in str(empty.value) and str(tmp_path) not in str(missing.value)

    local = quick_ask.Target(base_url="http://localhost:8000", workspace_id=_WS_ID)
    assert local.headers() == {"X-Workspace-ID": _WS_ID} and not local.hosted and target.hosted


def test_beside_a_database_the_servers_split_is_joined_by_workspace_chat_and_ref(runner, monkeypatch):
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    ref = wire.turn_ref(_WALL_MS, 1)
    read: list = []

    def timings(workspace_id, chat_ids):
        read.append((workspace_id, list(chat_ids)))
        return {ref: _TRACE, "vt-someone-else": {"timing": {"marks": {"authorised": 9999}}}}

    monkeypatch.setattr(runner, "read_timings", timings)
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES)])

    outcome = runner.quick_ask(_args(runner), _asks(), server)

    assert read == [(_WS_ID, [_CHAT_ID])]
    assert outcome["server_marks"] == "read" and outcome["server_traces"] == 1
    turn = outcome["turns"][0]
    assert turn["ref"] == ref
    assert turn["srv_authorised"] == 41 and turn["srv_model"] == 2560 and turn["srv_first_text"] is None
    assert "srv_authorised 41/41" in outcome["results_line"]
    assert "srv_first_text -/-" in outcome["results_line"]
    assert outcome["results_line"].endswith(" · server_marks=read")
    assert "Authorization" not in server.requests[0]["headers"]  # the local edition needs no sign-in


def test_one_chat_for_the_run_and_its_id_only_from_the_second_ask_on(runner, monkeypatch):
    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES) for _ in range(3)])

    outcome = runner.quick_ask(_args(runner, "--agent", "7"), _asks(3), server)

    bodies = [request["json"] for request in server.sent_to("/api/chat")]
    assert len(bodies) == 3
    assert "id" not in bodies[0]
    assert bodies[1]["id"] == _CHAT_ID and bodies[2]["id"] == _CHAT_ID
    assert all(body["spokenMode"] is True and body["agentId"] == 7 for body in bodies)
    refs = [body["turnRef"] for body in bodies]
    assert len(set(refs)) == 3 and refs == [turn["ref"] for turn in outcome["turns"]]
    speech = [request["json"] for request in server.sent_to("/api/voice/speech")]
    assert speech == [{"text": "Sure.", "turnRef": ref} for ref in refs]  # the first sentence, the same ref
    assert outcome["agent_id"] == 7


def test_new_chat_each_never_sends_a_chat_id_and_no_agent_means_no_agent_id(runner, monkeypatch):
    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES) for _ in range(2)])

    runner.quick_ask(_args(runner, "--new-chat-each"), _asks(2), server)

    bodies = [request["json"] for request in server.sent_to("/api/chat")]
    assert len(bodies) == 2
    assert all("id" not in body and "agentId" not in body for body in bodies)


@pytest.mark.parametrize(
    "capabilities, extra, at_its_own_frame",
    [
        (["spoken_mode", "tts"], (), False),
        (["spoken_mode", "tts", "spoken_unit_ends"], (), True),
        (["spoken_mode", "tts", "spoken_unit_ends"], ("--unit-ends", "off"), False),
        (["spoken_mode", "tts"], ("--unit-ends", "on"), True),
    ],
)
def test_a_run_cuts_the_first_sentence_as_a_phone_on_that_server_would(
    runner, monkeypatch, capabilities, extra, at_its_own_frame
):
    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    server = _Server(capabilities, [_chat(*_REPLY_LINES)])

    outcome = runner.quick_ask(_args(runner, *extra), _asks(), server)

    turn = outcome["turns"][0]
    assert outcome["unit_ends"] is at_its_own_frame
    # the clock ticks 100 ms a reading: start 0, headers 100, chat-id 200, "Sure." 300, the next unit 400
    assert turn["first_text_ms"] == 300
    assert turn["first_sentence_ms"] == (300 if at_its_own_frame else 400)
    assert server.sent_to("/api/voice/speech")[0]["json"]["text"] == "Sure."


def test_a_server_with_no_spoken_mode_is_refused_before_any_ask(runner, monkeypatch):
    from scripts.eval.voice_latency.run import RunRefused

    _no_database(runner, monkeypatch)
    server = _Server(["tts", "push"], [_chat(*_REPLY_LINES)])

    with pytest.raises(RunRefused, match="spoken_mode"):
        runner.quick_ask(_args(runner), _asks(), server)
    assert server.discovery_calls == ["http://localhost:8000/.well-known/automatos.json"]
    assert server.requests == []


def test_a_server_with_no_voice_is_refused_unless_no_speech_was_asked_for(runner, monkeypatch):
    from scripts.eval.voice_latency.run import RunRefused

    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    refused = _Server(["spoken_mode"], [_chat(*_REPLY_LINES)])
    with pytest.raises(RunRefused, match="--no-speech"):
        runner.quick_ask(_args(runner), _asks(), refused)
    assert refused.requests == []

    server = _Server(["spoken_mode"], [_chat(*_REPLY_LINES)])
    outcome = runner.quick_ask(_args(runner, "--no-speech"), _asks(), server)

    assert server.sent_to("/api/voice/speech") == [] and len(server.sent_to("/api/chat")) == 1
    turn = outcome["turns"][0]
    assert outcome["speech"] is False
    assert turn["first_sentence_ms"] == 400 and turn["full_ms"] is not None
    assert turn["tts_first_byte_ms"] is None and turn["tts_total_ms"] is None
    assert turn["go_first_audio_ms"] is None and turn["streamed_first_audio_ms"] is None
    assert turn["held_first_audio_ms"] is None


def test_no_speech_alone_does_not_excuse_a_missing_spoken_mode():
    from scripts.eval.voice_latency import quick_ask_protocol as wire

    assert wire.refusal(["tts"], no_speech=True) == wire.REFUSED_NO_SPOKEN_MODE
    assert wire.refusal(["spoken_mode"], no_speech=False) == wire.REFUSED_NO_TTS
    assert wire.refusal(["spoken_mode"], no_speech=True) is None
    assert wire.refusal(["spoken_mode", "tts"], no_speech=False) is None
    assert wire.capabilities_of({"capabilities": ["tts", 3, None, "spoken_mode"]}) == ("tts", "spoken_mode")
    assert wire.capabilities_of({"capabilities": "tts"}) == () and wire.capabilities_of(None) == ()


def test_a_refused_chat_turn_is_silent_and_asks_for_no_speech(runner, monkeypatch):
    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    server = _Server(["spoken_mode", "tts"], [_Response(status=500, lines=('{"detail":"no"}',))])

    outcome = runner.quick_ask(_args(runner), _asks(), server)

    turn = outcome["turns"][0]
    assert turn["silent"] is True and turn["headers_ms"] == 100
    assert turn["first_text_ms"] is None and turn["first_sentence_ms"] is None and turn["full_ms"] is None
    assert turn["go_first_audio_ms"] is None and turn["held_first_audio_ms"] is None
    assert server.sent_to("/api/voice/speech") == []
    assert outcome["chat_ids"] == []


def test_a_refused_speech_request_leaves_its_measures_empty(runner, monkeypatch):
    monkeypatch.setattr(runner, "read_timings", lambda workspace_id, chat_ids: {})
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES)], speech=_Response(status=429, chunks=(b"{}",)))

    outcome = runner.quick_ask(_args(runner), _asks(), server)

    turn = outcome["turns"][0]
    assert turn["first_sentence_ms"] == 400 and turn["full_ms"] is not None
    assert turn["tts_first_byte_ms"] is None and turn["tts_total_ms"] is None
    assert turn["go_first_audio_ms"] is None and turn["held_first_audio_ms"] is None


def test_speech_is_timed_from_its_own_send(runner):
    """Start 0, the first non-empty chunk 100 ms on, the end 200 ms on: the empty chunk reads no clock."""
    target = runner.Target(base_url="http://localhost:8000", workspace_id=_WS_ID)
    server = _Server([], [])

    times = runner.speak(server, target, "Sure.", "vt-abc123-1")

    assert (times.first_byte_ms, times.total_ms) == (100, 200)
    assert runner.speak(_Server([], [], speech=_Response(chunks=(b"",))), target, "Sure.", "vt-abc123-1") == (
        runner.NO_SPEECH
    )


def test_the_host_may_be_a_whole_base_url():
    from scripts.eval.voice_latency import quick_ask

    assert quick_ask.base_url("localhost", 8000) == "http://localhost:8000"
    assert quick_ask.base_url("https://api.example.test/", 8000) == "https://api.example.test"


@pytest.mark.parametrize(("base", "token_file", "in_the_clear"), [
    ("http://api.example.test:8000", "a-token-file", True),
    ("https://api.example.test", "a-token-file", False),
    ("http://localhost:8000", "a-token-file", False),
    ("http://127.0.0.1:8000", "a-token-file", False),
    ("http://[::1]:8000", "a-token-file", False),
    ("http://api.example.test:8000", None, False),
])
def test_a_token_crosses_plain_http_only_to_this_machine(base, token_file, in_the_clear):
    from scripts.eval.voice_latency import quick_ask

    target = quick_ask.Target(base_url=base, workspace_id=_WS_ID, token_file=token_file)

    assert quick_ask.sends_token_in_the_clear(target) is in_the_clear


def test_a_run_that_would_send_its_token_in_the_clear_is_refused_before_any_request(runner, tmp_path):
    from scripts.eval.voice_latency.run import RunRefused

    class NeverCalled:
        def __getattr__(self, name):
            raise AssertionError(f"the server was contacted ({name})")

    token_file = tmp_path / "token"
    token_file.write_text(_TOKEN + "\n")
    args = _args(runner, "--host", "api.example.test", "--token-file", str(token_file))

    with pytest.raises(RunRefused, match="https") as refused:
        runner.quick_ask(args, _asks(), NeverCalled())

    assert _TOKEN not in str(refused.value) and str(tmp_path) not in str(refused.value)


def test_the_workspace_must_be_an_id(runner):
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        runner._validate(parser, parser.parse_args(["--workspace", "not-a-uuid"]))
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --workspace is required
    runner._validate(parser, parser.parse_args(["--workspace", _WS_ID]))
