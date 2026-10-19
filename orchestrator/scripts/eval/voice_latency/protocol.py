"""
PRD-258 US-004: the Retell custom-LLM frames the simulator speaks
==================================================================

Pure (no I/O). These are the frames Retell sends to Auto's socket
(``modules/voice/live_socket.LiveCallSocket``, behind
``WS /api/voice/retell/llm-websocket/{call_id}``), built the way that
handler reads them:

* ``call_details`` carries ``call.retell_llm_dynamic_variables``
  {workspace_id, user_id, chat_id}, sent once the server's ``config`` frame
  asks for it;
* ``ping_pong`` carries a timestamp the server echoes back;
* one ask is one ``update_only`` per word as the utterance grows (the running
  ``transcript`` of {role, content}, earlier turns included), then
  ``response_required`` with the next ``response_id``.

``TurnWatch`` reads the server's ``response`` frames for one turn: the first
content frame, Auto's first frame (the one-time 'One moment.' ack and the
PRD-264 AI reminder never count) and ``content_complete``. The ask set the
owner keeps is validated here too, so a bad row is refused with its index
before anything is minted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.policy.ai_disclosure import VOICE_REMINDER
from modules.voice.providers.retell import (
    INTERACTION_CALL_DETAILS,
    INTERACTION_PING_PONG,
    INTERACTION_RESPONSE_REQUIRED,
    INTERACTION_UPDATE_ONLY,
)

# Speaking pace: about 150 words a minute, so one word every 0.4 s.
WORDS_PER_MINUTE = 150
SECONDS_PER_MINUTE = 60.0
WORD_INTERVAL_SECONDS = SECONDS_PER_MINUTE / WORDS_PER_MINUTE

# Timeouts (seconds).
CONNECT_TIMEOUT_SECONDS = 10.0  # the WebSocket handshake
HANDSHAKE_TIMEOUT_SECONDS = 10.0  # the server's config frame, then its begin frame
TURN_TIMEOUT_SECONDS = 120.0  # response_required → content_complete
PING_INTERVAL_SECONDS = 2.0  # Retell pings every 2 s while auto_reconnect is on
BETWEEN_ASKS_SECONDS = 1.0  # a breath between one reply and the next ask

# The response id Retell's first response_required carries (the server's begin frame is 0).
FIRST_RESPONSE_ID = 1

MS_PER_SECOND = 1000.0

# The transcript roles Retell uses.
ROLE_USER = "user"
ROLE_AGENT = "agent"

# The server's frame types (``response_type``).
RESPONSE_TYPE_CONFIG = "config"
RESPONSE_TYPE_RESPONSE = "response"

# The ask set: one row is {"kind": <one of ASK_KINDS>, "text": "..."}.
ASK_KINDS = ("small_talk", "knowledge", "board", "tool")

Transcript = Tuple[Dict[str, str], ...]


class AskSetError(ValueError):
    """The ask set on stdin is not a list of valid asks; the message names the row."""


@dataclass(frozen=True)
class Ask:
    """One spoken ask: its kind (for the per-kind percentiles) and its words."""

    kind: str
    text: str


def _parse_ask(index: int, row: Any) -> Ask:
    if not isinstance(row, dict):
        raise AskSetError(f"ask {index}: expected an object with 'kind' and 'text'")
    kind, text = row.get("kind"), row.get("text")
    if kind not in ASK_KINDS:
        raise AskSetError(f"ask {index}: kind {kind!r} is not one of {', '.join(ASK_KINDS)}")
    if not isinstance(text, str) or not text.split():
        raise AskSetError(f"ask {index}: 'text' must be a non-empty string")
    return Ask(kind=kind, text=" ".join(text.split()))


def parse_asks(raw: Any) -> Tuple[Ask, ...]:
    """Validate the ask set; the first bad row is refused with its (0-based) index."""
    if not isinstance(raw, list) or not raw:
        raise AskSetError("the ask set must be a non-empty JSON list of {kind, text} objects")
    return tuple(_parse_ask(index, row) for index, row in enumerate(raw))


def call_details_frame(call_id: str, *, workspace_id: str, user_id: int, chat_id: str) -> Dict[str, Any]:
    """The frame that delivers the call's dynamic variables, as Retell sends it."""
    return {
        "interaction_type": INTERACTION_CALL_DETAILS,
        "call": {
            "call_id": call_id,
            "retell_llm_dynamic_variables": {
                "workspace_id": str(workspace_id),
                "user_id": user_id,
                "chat_id": str(chat_id),
            },
        },
    }


def ping_frame(timestamp_ms: int) -> Dict[str, Any]:
    """Retell's keep-alive; the server echoes the timestamp."""
    return {"interaction_type": INTERACTION_PING_PONG, "timestamp": int(timestamp_ms)}


def _with_user_words(history: Transcript, words: Sequence[str]) -> List[Dict[str, str]]:
    return [dict(entry) for entry in history] + [{"role": ROLE_USER, "content": " ".join(words)}]


def ask_frames(history: Transcript, text: str, response_id: int) -> List[Dict[str, Any]]:
    """One ``update_only`` per word as the utterance grows, then ``response_required``."""
    words = text.split()
    updates = [
        {"interaction_type": INTERACTION_UPDATE_ONLY, "transcript": _with_user_words(history, words[:count])}
        for count in range(1, len(words) + 1)
    ]
    required = {
        "interaction_type": INTERACTION_RESPONSE_REQUIRED,
        "response_id": int(response_id),
        "transcript": _with_user_words(history, words),
    }
    return updates + [required]


def after_turn(history: Transcript, user_text: str, agent_text: str) -> Transcript:
    """The transcript the next ask carries: this turn's words and Auto's reply appended."""
    return tuple(history) + (
        {"role": ROLE_USER, "content": user_text},
        {"role": ROLE_AGENT, "content": agent_text},
    )


def is_config(message: Dict[str, Any]) -> bool:
    return message.get("response_type") == RESPONSE_TYPE_CONFIG


def wants_call_details(message: Dict[str, Any]) -> bool:
    return bool((message.get("config") or {}).get("call_details"))


def wants_pings(message: Dict[str, Any]) -> bool:
    return bool((message.get("config") or {}).get("auto_reconnect"))


def is_response_to(message: Dict[str, Any], response_id: int) -> bool:
    """True for a ``response`` frame answering this response id."""
    return message.get("response_type") == RESPONSE_TYPE_RESPONSE and message.get("response_id") == response_id


@dataclass
class TurnWatch:
    """One turn's client-side clock: fed every server frame with its arrival time.

    A per-turn accumulator owned by the runner's turn loop (never shared).
    ``ack_text`` is the one-time slow-turn filler; empty once the call has
    heard it, so a later identical sentence from Auto counts as Auto's.
    """

    response_id: int
    started: float
    ack_text: str = ""
    first_frame_at: Optional[float] = None
    first_auto_frame_at: Optional[float] = None
    complete_at: Optional[float] = None
    ack_heard: bool = False
    reply_parts: List[str] = field(default_factory=list)

    def observe(self, message: Dict[str, Any], at: float) -> None:
        """Note one server frame; frames for other turns are ignored."""
        if not is_response_to(message, self.response_id):
            return
        content = str(message.get("content") or "")
        if content:
            self._content(content, at)
        if message.get("content_complete") and self.complete_at is None:
            self.complete_at = at

    def _content(self, content: str, at: float) -> None:
        if self.first_frame_at is None:
            self.first_frame_at = at
        is_ack = bool(self.ack_text) and not self.ack_heard and content.strip() == self.ack_text.strip()
        if is_ack:
            self.ack_heard = True
            return
        if content.strip() == VOICE_REMINDER:
            return  # the AI reminder (PRD-264 EUAI-04) opens a response; it is not Auto's reply
        self.reply_parts.append(content)
        if self.first_auto_frame_at is None:
            self.first_auto_frame_at = at

    @property
    def done(self) -> bool:
        return self.complete_at is not None

    @property
    def reply(self) -> str:
        return "".join(self.reply_parts).strip()

    def elapsed_ms(self, moment: Optional[float]) -> Optional[int]:
        """Whole milliseconds from the turn's start to a moment; None if it never came."""
        return None if moment is None else int(round((moment - self.started) * MS_PER_SECOND))
