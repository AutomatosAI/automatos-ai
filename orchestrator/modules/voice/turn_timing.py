"""
PRD-258 US-001: where a spoken turn's time went
================================================

Pure timing logic for one Auto Live turn (no I/O). The socket marks three
moments on a monotonic clock — the turn's start (``response_required``), the
first content frame SENT (the one-time 'One moment.' ack counts) and the
first frame Auto itself produced (the ack never counts) — and the agent
stream reports how many tools Auto called before it first spoke and the
``prep_ms`` its context assembly stored on the reply. ``voice_timing_fields``
turns those into the ``voice_turns`` columns in whole milliseconds; a missing
moment is None, never an invented number.

The agent stream reports its facts to the socket in-band, as items that are
not Retell frames (``landed`` once the call is bound, ``finished`` after the
reply is saved); the socket keeps them and never sends them on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Dict, Mapping, Optional

from modules.voice.providers.retell import extract_agent_text

# The AI-SDK stream lines ``StreamingHandler`` emits (consumers/chatbot/streaming.py):
# ``format_aisdk_text`` writes ``0:"…"`` and ``format_aisdk_tool_start`` writes
# ``d:{"type": "tool-start", …}`` — the one line per tool call Auto makes.
AISDK_DATA_PREFIX = "d:"
AISDK_TOOL_CALL_EVENT = "tool-start"

# The key an in-band facts item carries; a Retell frame never has it.
TURN_FACTS_KEY = "_voice_turn_facts"

_MS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class TurnFacts:
    """What the agent stream knows about its turn: where it landed, and how it went."""

    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    tools_before_speech: Optional[int] = None
    prep_ms: Optional[float] = None


@dataclass(frozen=True)
class SpeechTally:
    """Tool calls counted in Auto's stream until its first text."""

    tools_before_speech: int = 0
    spoke: bool = False


def landed(workspace_id: str, conversation_id: str) -> Dict[str, TurnFacts]:
    """The in-band item that says which proven workspace and thread the turn writes into."""
    return {TURN_FACTS_KEY: TurnFacts(workspace_id=workspace_id, conversation_id=conversation_id)}


def finished(tools_before_speech: int, prep_ms: Optional[float]) -> Dict[str, TurnFacts]:
    """The in-band item sent once the reply is saved: the tool count and the stored prep time."""
    return {TURN_FACTS_KEY: TurnFacts(tools_before_speech=tools_before_speech, prep_ms=prep_ms)}


def facts_of(item: Mapping[str, Any]) -> Optional[TurnFacts]:
    """The facts an in-band item carries, or None for a Retell frame."""
    facts = item.get(TURN_FACTS_KEY) if isinstance(item, Mapping) else None
    return facts if isinstance(facts, TurnFacts) else None


def merge_facts(known: TurnFacts, new: TurnFacts) -> TurnFacts:
    """A new copy of ``known`` with every field ``new`` sets."""
    updates = {name: value for name, value in vars(new).items() if value is not None}
    return replace(known, **updates)


def is_tool_call_line(line: str) -> bool:
    """True for the AI-SDK data line that announces a tool call."""
    if not line.startswith(AISDK_DATA_PREFIX):
        return False
    try:
        payload = json.loads(line[len(AISDK_DATA_PREFIX):])
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("type") == AISDK_TOOL_CALL_EVENT


def tally_chunk(tally: SpeechTally, chunk: str) -> SpeechTally:
    """The tally after one stream chunk: tool lines count only before the first text."""
    for line in (chunk or "").splitlines():
        if tally.spoke:
            return tally
        if extract_agent_text(line).strip():
            tally = replace(tally, spoke=True)
        elif is_tool_call_line(line):
            tally = replace(tally, tools_before_speech=tally.tools_before_speech + 1)
    return tally


class SpeechTap:
    """Passes Auto's AI-SDK stream through unchanged, keeping its ``SpeechTally``.

    The one stateful piece here: ``tally`` is replaced (never mutated) per
    chunk, and read once the stream has ended.
    """

    def __init__(self, chunks: AsyncIterator[str]) -> None:
        self._chunks = chunks
        self.tally = SpeechTally()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._passthrough()

    async def _passthrough(self) -> AsyncIterator[str]:
        async for chunk in self._chunks:
            self.tally = tally_chunk(self.tally, chunk)
            yield chunk


def _elapsed_ms(start: float, at: Optional[float]) -> Optional[int]:
    if at is None:
        return None
    return int(round((at - start) * _MS_PER_SECOND))


def _whole_ms(value: Optional[float]) -> Optional[int]:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(round(value))


def _count(value: Optional[int]) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def voice_timing_fields(
    *,
    started: float,
    first_frame_at: Optional[float],
    first_auto_frame_at: Optional[float],
    tools_before_speech: Optional[int],
    prep_ms: Optional[float],
) -> Dict[str, Any]:
    """The PRD-258 ``voice_turns`` timing columns for one turn, in whole milliseconds.

    ``started`` and the two frame moments are readings of the same monotonic
    clock, in seconds. A moment that never happened (no frame sent, a turn
    cancelled first) is None. ``lead_in`` and ``prefetch_hit`` are written
    false until PRD-258 US-008 and US-009 ship.
    """
    return {
        "first_frame_ms": _elapsed_ms(started, first_frame_at),
        "first_auto_frame_ms": _elapsed_ms(started, first_auto_frame_at),
        "prep_ms": _whole_ms(prep_ms),
        "tools_before_speech": _count(tools_before_speech),
        "lead_in": False,
        "prefetch_hit": False,
    }
