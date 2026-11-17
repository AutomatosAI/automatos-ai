"""
Voice latency A-4: what the quick-ask runner sends, and how it times one reply
==============================================================================

Pure (no I/O; standard library only). A quick-ask is what Automatos Go's voice screen does
over HTTP: ``POST /api/chat`` with ``spokenMode`` and a turn reference, then
``POST /api/voice/speech`` for the reply's first sentence with the same reference.

* ``chat_body`` / ``speech_body``: the two request bodies, as the phone builds them.
* ``turn_ref``: the reference, ``vt-<send time in ms, base 36>-<ask number, base 36>``.
* ``first_sentence``: the phone's sentence rule (``src/voice/speech/sentences.ts``), with the
  rule a phone uses on a server that lists ``spoken_unit_ends`` as ``at_text_end``.
* ``QuickAskWatch``: one reply's client-side clock, fed the stream line by line.
* ``QuickAskResult`` and ``quick_ask_result``: the measures of one ask. A derived measure is
  None when a part of it is missing, never a guess.
* ``server_marks``: the server's own split, from the reply's ``context_trace["timing"]``
  (``core.monitoring.turn_clock``); every name None when there is no trace.

Nothing here keeps or prints what was asked or answered beyond the one sentence the watch
hands to the speech request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CHAT_PATH = "/api/chat"
SPEECH_PATH = "/api/voice/speech"
DISCOVERY_PATH = "/.well-known/automatos.json"

# The request, as Automatos Go's buildChatRequest sends it (src/chat/request.ts).
ROLE_USER = "user"
PART_TEXT = "text"
VISIBILITY_PRIVATE = "private"
CHAT_PAGE = "chat"
CHAT_ROUTE = "/chat"

# The stream's frames (consumers/chatbot/streaming.py).
TEXT_FRAME_PREFIX = "0:"
DATA_FRAME_PREFIX = "d:"
ERROR_FRAME_PREFIX = "e:"
CHAT_ID_FRAME = "chat-id"

# Discovery capabilities (services/mobile/discovery.py).
CAPABILITY_SPOKEN_MODE = "spoken_mode"
CAPABILITY_TTS = "tts"
CAPABILITY_UNIT_ENDS = "spoken_unit_ends"
UNIT_ENDS_ON = "on"
UNIT_ENDS_OFF = "off"
UNIT_ENDS_CHOICES = (UNIT_ENDS_ON, UNIT_ENDS_OFF)

REF_PREFIX = "vt"
BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
MS_PER_SECOND = 1000.0

# A run of terminators, then any closing quote or bracket, followed by whitespace: the
# phone's rule. With ``at_text_end`` the end of the text so far counts as that whitespace.
_TERMINATOR = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
_TERMINATOR_OR_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|\Z)")
_RUN = re.compile(r"[.!?…]+")
_WORD_BEFORE = re.compile(r"[^\s(“\"'‘\[]+\Z")
_DOTTED = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]")  # e.g, i.e, a.m, U.S
_INITIAL = re.compile(r"[^\W\d_]")
ABBREVIATIONS = frozenset(
    ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "cf", "approx", "e.g", "i.e")
)

# Measure name → the QuickAskResult field it reads, in the results line's order.
QUICK_ASK_MEASURES = (
    ("headers", "headers_ms"),
    ("first_text", "first_text_ms"),
    ("first_sentence", "first_sentence_ms"),
    ("tts_first_byte", "tts_first_byte_ms"),
    ("tts_total", "tts_total_ms"),
    ("full", "full_ms"),
    ("go_first_audio", "go_first_audio_ms"),
    ("streamed_first_audio", "streamed_first_audio_ms"),
    ("held_first_audio", "held_first_audio_ms"),
)
# The server's split: (measure and field name, the trace's group, its name there). ``srv_``
# keeps the server's ``first_text`` apart from the runner's own.
SERVER_TRACE_NAMES = (
    ("srv_authorised", "marks", "authorised"),
    ("srv_routed", "marks", "routed"),
    ("srv_locked", "marks", "locked"),
    ("srv_agent", "marks", "agent"),
    ("srv_model_start", "marks", "model_start"),
    ("srv_first_response", "marks", "first_response"),
    ("srv_first_text", "marks", "first_text"),
    ("srv_agent_context", "stages", "agent_context"),
    ("srv_tools", "stages", "tools"),
    ("srv_context", "stages", "context"),
    ("srv_composio", "stages", "composio"),
    ("srv_knowledge", "stages", "knowledge"),
    ("srv_model", "stages", "model"),
    ("srv_model_calls", "counts", "model_calls"),
)
SERVER_MEASURES = tuple((name, name) for name, _, _ in SERVER_TRACE_NAMES)
TIMING_KEY = "timing"
REF_KEY = "ref"
SERVER_MARKS_READ = "read"
SERVER_MARKS_UNAVAILABLE = "unavailable"
SERVER_MARKS_LABEL = "server_marks"

REFUSED_NO_SPOKEN_MODE = "the server's discovery does not list spoken_mode: it has no quick-ask to time"
REFUSED_NO_TTS = "the server's discovery does not list tts (no server voice): pass --no-speech to time the chat alone"


# ── the requests ────────────────────────────────────────────────────────────


def base36(number: int) -> str:
    """A whole number, zero or more, in base 36 (JavaScript's ``toString(36)``)."""
    value = max(0, int(number))
    digits = BASE36_DIGITS[value % len(BASE36_DIGITS)]
    while value >= len(BASE36_DIGITS):
        value //= len(BASE36_DIGITS)
        digits = BASE36_DIGITS[value % len(BASE36_DIGITS)] + digits
    return digits


def turn_ref(wall_ms: int, ask_number: int) -> str:
    """``vt-<the send time, ms since 1970, base 36>-<the ask's number in this run, base 36>``."""
    return f"{REF_PREFIX}-{base36(wall_ms)}-{base36(ask_number)}"


def chat_body(
    text: str, ref: str, *, chat_id: Optional[str] = None, agent_id: Optional[int] = None
) -> Dict[str, Any]:
    """``POST /api/chat``'s body for one spoken ask. ``id`` only for a chat that exists (from
    the second ask on); ``agentId`` only when the run names an agent."""
    body: Dict[str, Any] = {
        "message": {"role": ROLE_USER, "parts": [{"type": PART_TEXT, "text": text}]},
        "selectedVisibilityType": VISIBILITY_PRIVATE,
        "context": {"page": CHAT_PAGE, "route": CHAT_ROUTE},
        "spokenMode": True,
        "turnRef": ref,
    }
    if chat_id:
        body = {"id": chat_id, **body}
    if agent_id is not None:
        body = {**body, "agentId": int(agent_id)}
    return body


def speech_body(sentence: str, ref: str) -> Dict[str, Any]:
    """``POST /api/voice/speech``'s body for the reply's first sentence, in the default voice."""
    return {"text": sentence, "turnRef": ref}


# ── discovery ───────────────────────────────────────────────────────────────


def capabilities_of(document: Any) -> Tuple[str, ...]:
    """The capability names a discovery document lists; anything else is no capability."""
    listed = document.get("capabilities") if isinstance(document, Mapping) else None
    if not isinstance(listed, (list, tuple)):
        return ()
    return tuple(name for name in listed if isinstance(name, str))


def refusal(capabilities: Iterable[str], *, no_speech: bool) -> Optional[str]:
    """Why this server cannot be timed, for the owner; None when it can."""
    listed = set(capabilities)
    if CAPABILITY_SPOKEN_MODE not in listed:
        return REFUSED_NO_SPOKEN_MODE
    if CAPABILITY_TTS not in listed and not no_speech:
        return REFUSED_NO_TTS
    return None


def unit_ends(capabilities: Iterable[str], forced: Optional[str] = None) -> bool:
    """Whether a sentence may end at the end of the text so far: what a phone does on a
    server that lists ``spoken_unit_ends``, unless the run forces it ``on`` or ``off``."""
    if forced in UNIT_ENDS_CHOICES:
        return forced == UNIT_ENDS_ON
    return CAPABILITY_UNIT_ENDS in set(capabilities)


# ── the phone's sentence rule ───────────────────────────────────────────────


def _ends_sentence(text: str, index: int, terminator: str) -> bool:
    """A lone '.' after an abbreviation or an initial does not end the sentence."""
    run = _RUN.match(terminator)
    if run is None or run.group(0) != ".":
        return True
    before = _WORD_BEFORE.search(text[:index])
    word = before.group(0).lower() if before else ""
    return not (word in ABBREVIATIONS or _DOTTED.fullmatch(word) or _INITIAL.fullmatch(word))


def _sayable(sentence: str) -> bool:
    return any(character.isalnum() for character in sentence)


def first_sentence(text_so_far: str, at_text_end: bool = False) -> Optional[str]:
    """The first complete sentence of the text so far, or None while there is none yet.

    A sentence ends at ``. ! ? …`` followed by whitespace. ``at_text_end`` is the rule a phone
    uses when the server lists ``spoken_unit_ends``: a terminator at the end of the text so
    far ends a sentence too, because every frame of a spoken turn is a whole speech unit.
    """
    pattern = _TERMINATOR_OR_END if at_text_end else _TERMINATOR
    start = 0
    for match in pattern.finditer(text_so_far):
        if not _ends_sentence(text_so_far, match.start(), match.group(0)):
            continue
        sentence = text_so_far[start:match.end()].strip()
        if _sayable(sentence):
            return sentence
        start = match.end()
    return None


# ── one reply, timed on the client side ─────────────────────────────────────


def elapsed_ms(started: float, moment: Optional[float]) -> Optional[int]:
    """Whole milliseconds from a start to a moment; None if the moment never came."""
    return None if moment is None else int(round((moment - started) * MS_PER_SECOND))


def _frame_json(line: str, prefix: str) -> Any:
    try:
        return json.loads(line[len(prefix):])
    except ValueError:
        return None


@dataclass
class QuickAskWatch:
    """One reply's client-side clock: fed the response's headers, every line, then the end.

    A per-ask accumulator owned by the runner's loop (never shared). ``started`` is the
    moment the chat request was sent; ``at_text_end`` is ``first_sentence``'s flag.
    """

    started: float
    at_text_end: bool = False
    headers_at: Optional[float] = None
    first_text_at: Optional[float] = None
    first_sentence_at: Optional[float] = None
    ended_at: Optional[float] = None
    chat_id: Optional[str] = None
    sentence: Optional[str] = None
    failed: bool = False
    text_parts: List[str] = field(default_factory=list)

    def headers(self, at: float) -> None:
        """The response's status line and headers are in."""
        if self.headers_at is None:
            self.headers_at = at

    def observe(self, line: str, at: float) -> None:
        """Note one line of the stream; a line that is no known frame is ignored."""
        if line.startswith(TEXT_FRAME_PREFIX):
            self._text(_frame_json(line, TEXT_FRAME_PREFIX), at)
        elif line.startswith(DATA_FRAME_PREFIX):
            self._data(_frame_json(line, DATA_FRAME_PREFIX))
        elif line.startswith(ERROR_FRAME_PREFIX):
            self.failed = True

    def _text(self, text: Any, at: float) -> None:
        if not isinstance(text, str) or text == "":
            return
        self.text_parts.append(text)
        if self.first_text_at is None:
            self.first_text_at = at
        if self.sentence is None:
            self._sentence(first_sentence("".join(self.text_parts), self.at_text_end), at)

    def _sentence(self, sentence: Optional[str], at: float) -> None:
        if sentence is not None:
            self.sentence, self.first_sentence_at = sentence, at

    def _data(self, frame: Any) -> None:
        if not isinstance(frame, Mapping) or frame.get("type") != CHAT_ID_FRAME:
            return
        chat_id = frame.get("chatId")
        if self.chat_id is None and isinstance(chat_id, str) and chat_id:
            self.chat_id = chat_id

    def end(self, at: float) -> None:
        """The stream ended. A reply with no sentence end is one sentence now, as the phone's
        flush hands over the rest when the turn finishes."""
        if self.ended_at is not None:
            return
        self.ended_at = at
        rest = "".join(self.text_parts).strip()
        if self.sentence is None and _sayable(rest):
            self._sentence(rest, at)

    @property
    def silent(self) -> bool:
        """No word of the reply arrived."""
        return self.first_text_at is None

    def ms(self, moment: Optional[float]) -> Optional[int]:
        return elapsed_ms(self.started, moment)


@dataclass(frozen=True)
class SpeechTimes:
    """The first sentence's speech request, from its send: the first body byte and the last."""

    first_byte_ms: Optional[int] = None
    total_ms: Optional[int] = None


NO_SPEECH = SpeechTimes()


# ── the result of one ask ───────────────────────────────────────────────────


def server_marks(trace: Optional[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    """The server's split of one turn from its reply's trace: every name of
    ``SERVER_TRACE_NAMES``, None when the trace, its ``timing`` or the value is missing."""
    timing = trace.get(TIMING_KEY) if isinstance(trace, Mapping) else None
    marks: Dict[str, Optional[int]] = {}
    for name, group, key in SERVER_TRACE_NAMES:
        values = timing.get(group) if isinstance(timing, Mapping) else None
        value = values.get(key) if isinstance(values, Mapping) else None
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        marks[name] = int(round(value)) if is_number else None
    return marks


def timings_by_ref(traces: Iterable[Any]) -> Dict[str, Mapping[str, Any]]:
    """Replies' traces keyed by the turn reference their ``timing`` carries; a trace with no
    timing or no reference is left out. The caller has already read by workspace and chat."""
    found: Dict[str, Mapping[str, Any]] = {}
    for trace in traces:
        timing = trace.get(TIMING_KEY) if isinstance(trace, Mapping) else None
        ref = timing.get(REF_KEY) if isinstance(timing, Mapping) else None
        if isinstance(ref, str) and ref:
            found[ref] = trace
    return found


@dataclass(frozen=True)
class QuickAskResult:
    """One measured ask, whole milliseconds. ``silent``: no word of the reply arrived."""

    kind: str
    silent: bool
    headers_ms: Optional[int]
    first_text_ms: Optional[int]
    first_sentence_ms: Optional[int]
    tts_first_byte_ms: Optional[int]
    tts_total_ms: Optional[int]
    full_ms: Optional[int]
    # first_sentence + tts_total: today's phone, which plays a sentence once its whole file is in.
    go_first_audio_ms: Optional[int]
    # first_sentence + tts_first_byte: what playing while downloading could reach.
    streamed_first_audio_ms: Optional[int]
    # full + tts_total: a client that holds the whole reply before it speaks.
    held_first_audio_ms: Optional[int]
    srv_authorised: Optional[int] = None
    srv_routed: Optional[int] = None
    srv_locked: Optional[int] = None
    srv_agent: Optional[int] = None
    srv_model_start: Optional[int] = None
    srv_first_response: Optional[int] = None
    srv_first_text: Optional[int] = None
    srv_agent_context: Optional[int] = None
    srv_tools: Optional[int] = None
    srv_context: Optional[int] = None
    srv_composio: Optional[int] = None
    srv_knowledge: Optional[int] = None
    srv_model: Optional[int] = None
    srv_model_calls: Optional[int] = None


def _sum(first: Optional[int], second: Optional[int]) -> Optional[int]:
    return None if first is None or second is None else first + second


def quick_ask_result(
    kind: str,
    watch: QuickAskWatch,
    speech: SpeechTimes = NO_SPEECH,
    trace: Optional[Mapping[str, Any]] = None,
) -> QuickAskResult:
    """One ask's measures from its watch, its speech request and (when read) its trace. A reply
    that sent an error frame has no ``full``: its stream ended, the reply did not finish."""
    first_sentence_ms = watch.ms(watch.first_sentence_at)
    full_ms = None if watch.failed else watch.ms(watch.ended_at)
    return QuickAskResult(
        kind=kind,
        silent=watch.silent,
        headers_ms=watch.ms(watch.headers_at),
        first_text_ms=watch.ms(watch.first_text_at),
        first_sentence_ms=first_sentence_ms,
        tts_first_byte_ms=speech.first_byte_ms,
        tts_total_ms=speech.total_ms,
        full_ms=full_ms,
        go_first_audio_ms=_sum(first_sentence_ms, speech.total_ms),
        streamed_first_audio_ms=_sum(first_sentence_ms, speech.first_byte_ms),
        held_first_audio_ms=_sum(full_ms, speech.total_ms),
        **server_marks(trace),
    )


def measures_for(server_marks_read: bool) -> Sequence[Tuple[str, str]]:
    """The measures a run reports: the server's split only when its traces were read."""
    return QUICK_ASK_MEASURES + SERVER_MEASURES if server_marks_read else QUICK_ASK_MEASURES


def server_marks_note(server_marks_read: bool) -> str:
    """``server_marks=read`` or ``server_marks=unavailable``: the results line's last field."""
    state = SERVER_MARKS_READ if server_marks_read else SERVER_MARKS_UNAVAILABLE
    return f"{SERVER_MARKS_LABEL}={state}"
