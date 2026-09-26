"""F186 (night 6): a turn's reply is its answer. What the model said before
calling its tools is narration.

41 of night 6's 159 saved replies were two replies in one. The chat joined
every round's streamed text into the saved message and into memory (PRD-238
S2: "the saved message is exactly what the screen showed"). #1110 at 03:05: the
first round said the ticket was "waiting for the Shopify Support Agent", the
tools then showed it was done, and the second round answered "This task was
completed". Both halves were saved as one reply, and the pre-tool guess sat in
the next turn's history as if it had been checked.

A round that ended in tool calls is narration: shown as progress, and stored
as its own part, never as the answer. The final response is the answer: the
saved text, and what memory and the next turn read. A reply the loop nudged
and retried (F108: it claimed work no action backed) is neither, and is left
out. When the answer is empty, the narration stands in for it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def called_tools(response: Any) -> bool:
    return bool(getattr(response, "tool_calls", None))


def split_reply(rounds: Sequence[Any], final: Optional[Any], final_text: str) -> Tuple[List[str], str]:
    """``(narration, answer)`` for a turn: ``rounds`` are its streamed model
    responses in order, ``final`` the one that answered (None when the answer
    was made outside them), ``final_text`` the answer's text."""
    narration = [r.content for r in rounds if r is not final and called_tools(r) and (r.content or "").strip()]
    if (final_text or "").strip():
        return narration, final_text
    return [], "\n\n".join(narration)


def reply_parts(reasoning: str, narration: str, answer: str) -> List[Dict[str, Any]]:
    """The saved message's parts: its reasoning (PRD-238 S1), its narration and
    its answer, each on its own. Only the answer is ``text``; anything that
    reads a message's text (the next turn, previews, search) sees the answer."""
    parts: List[Dict[str, Any]] = []
    if reasoning:
        parts.append({"type": "reasoning", "reasoning": reasoning})
    if narration:
        parts.append({"type": "narration", "narration": narration})
    parts.append({"type": "text", "text": answer})
    return parts
