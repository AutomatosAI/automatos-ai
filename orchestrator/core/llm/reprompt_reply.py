"""The reply to a re-prompt opens with the work, not an apology (F295, night 9b).

Night 9b (6586c8bf, 8578eeaf): told by the platform's claim check that a reply said
something was done when no call did it, Auto answered "You are absolutely right to call
me out on that, Gerard. My apologies…" before the real answer, as if the owner had
spoken, and the apology streamed live into the chat. FIXER's ``nudges.without_the_apology``
takes those opening sentences off a nudged agent reply; this does the same for Auto's
chat, on the reply that is saved and on the text that streams.

``ApologyGate`` sits between the model's stream and the chat's ``on_delta``: it holds
the opening text until its first sentence is complete (or ``HOLD_CHARS`` arrive), drops
the opening sentences that are an apology, then lets everything through as it comes.
Reasoning deltas are never held.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

Delta = Callable[[str, str], Awaitable[None]]

TEXT = "text"
# Enough for an opening sentence or two; past this the opening is passed on as it stands.
HOLD_CHARS = 320
_SENTENCE_END = (". ", "! ", "? ", ".\n", "!\n", "?\n")


@dataclass(frozen=True)
class _Said:
    """A bit of text in the shape ``without_the_apology`` reads (a dataclass with content)."""
    content: str


def without_an_opening_apology(text: str) -> str:
    """``text`` without the apology sentences it opens with (FIXER's rule, the one place it lives)."""
    from modules.tools.execution.nudges import without_the_apology

    return without_the_apology(_Said(text)).content


class ApologyGate:
    """Holds a re-prompted reply's opening text until it can tell whether it is an apology."""

    def __init__(self, on_delta: Delta) -> None:
        self._on_delta = on_delta
        self._held = ""
        self._open = False

    async def __call__(self, kind: str, text: str) -> None:
        if kind != TEXT or self._open:
            await self._on_delta(kind, text)
            return
        self._held += text
        if self._ready():
            await self._release()

    def _ready(self) -> bool:
        """The opening can be judged: a sentence after any apology has started, or enough has come."""
        if len(self._held) >= HOLD_CHARS:
            return True
        kept = without_an_opening_apology(self._held)
        if kept != self._held.lstrip():
            return bool(kept.strip())          # an apology came off, and the work has started
        return any(end in self._held for end in _SENTENCE_END)

    async def _release(self) -> None:
        self._open = True
        kept, self._held = without_an_opening_apology(self._held), ""
        if kept:
            await self._on_delta(TEXT, kept)

    async def close(self) -> None:
        """Pass on whatever is still held when the stream ends."""
        if not self._open and self._held:
            await self._release()


def gated(on_delta: Optional[Delta]) -> Optional[ApologyGate]:
    return ApologyGate(on_delta) if on_delta is not None else None


__all__ = ["ApologyGate", "gated", "without_an_opening_apology"]
