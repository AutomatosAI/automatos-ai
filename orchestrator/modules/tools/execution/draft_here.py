"""F419 (night 14): a reply to a here-ask holds the draft, never a pointer to one.

Night 14 (12–13 Oct, Auto on Haiku 5.5): the owner asked for a draft here (P256-FIX-T5 routes
the ask to Auto's answer, ``consumers.chatbot.asked_here.HERE_DIRECTIVE``), and the whole
reply was "The draft is above, in chat only.", twice. There was no draft above.

Now a reply to a here-ask that points to a draft ("above", "below", "in chat") and holds no
body besides the pointing sentences (``bare_pointer``) is retried once with the directive:
the FX-017 retry seam (``nudge_about``, on the loop's final-reply check, after F413's), so the
pointer is retracted when the retry replaces it. A retry that still holds no draft (blank, the
account of the turn's calls in its place, or a pointer again) is replaced by a plain line that says no draft was written, never sent as a
bare pointer. A first reply that runs no tool is routed through the loop for this check
(``a_bare_pointer_goes_through_the_loop``, on the chat's first-reply check).

Stdlib only at import, like the loop (the here-ask rule is read when a reply ends).
"""
from __future__ import annotations

import functools
import logging
import re
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .fetched_work import owners_words, shows_fetched_work
from .nudges import nudge_about
from .turn_account import account_of, auto_speaks

logger = logging.getLogger(__name__)

# Where a reply says the draft is, instead of giving it: "above" or "below" as a place ("is above,",
# "see below:", not "above target"), "in (the) chat", "earlier in the chat", "in my last message".
_POINTER = re.compile(
    r"\b(?:above|below)\b(?=\s*(?:$|[,.!?;:)]|in\s+(?:the\s+)?chat\b))|\bin\s+(?:the\s+|this\s+|our\s+)?chat\b"
    r"|\b(?:earlier|previously)\s+in\s+(?:the|this|our)\s+(?:chat|conversation)\b"
    r"|\bin\s+(?:my|the)\s+(?:last|previous|earlier)\s+(?:message|reply)\b",
    re.IGNORECASE,
)
# A reply's sentences and lines: a draft's body is the pieces that point nowhere.
_PIECES = re.compile(r"(?<=[.!?])\s+|\n+")
# Fewer words than this besides the pointing sentences is no draft ("Let me know if you'd like changes.").
DRAFT_MIN_WORDS = 20
DRAFT_NOT_THERE_MSG = (
    "Your previous reply points to a draft ({where}), but there is no draft there: the owner sees only "
    "this reply. {directive}"
)
POINTER_SHOWN_CHARS = 160
# What the owner reads when the retry still gave no draft: never a pointer to one.
NO_DRAFT_SAID = "I haven't written the draft into this reply. Ask me again and I'll write it here in full."

Recover = Callable[..., Awaitable[Any]]
FirstReplyCheck = Callable[..., Awaitable[bool]]


def _pieces(reply: str) -> List[str]:
    return [piece.strip() for piece in _PIECES.split(reply or "") if piece.strip()]


def bare_pointer(reply: str) -> Optional[str]:
    """The sentence of ``reply`` that points to a draft elsewhere ("The draft is above, in chat only."),
    when the reply holds no draft besides its pointing sentences; else None. Pure."""
    pieces = _pieces(reply)
    pointing = [piece for piece in pieces if _POINTER.search(piece)]
    if not pointing:
        return None
    body = sum(len(piece.split()) for piece in pieces if not _POINTER.search(piece))
    return pointing[0][:POINTER_SHOWN_CHARS] if body < DRAFT_MIN_WORDS else None


def asked_here(said: str) -> bool:
    """Whether the owner's words ask for the work in the reply (P256-FIX-T5's one rule)."""
    from consumers.chatbot.asked_here import asks_for_it_here  # read when a reply ends

    return asks_for_it_here(said)


def draft_nudge(pointer: str) -> str:
    """The retry's words: the reply pointed to a draft that is not there, then the here directive."""
    from consumers.chatbot.asked_here import HERE_DIRECTIVE

    directive = HERE_DIRECTIVE.split("\n", 3)[-1].strip()   # its words, without the directive's heading
    return DRAFT_NOT_THERE_MSG.format(where=pointer, directive=directive)


def no_draft(response: Any) -> Any:
    """A copy of ``response`` (never mutated) whose content is the plain line that says no draft was
    written; never streamed, so the chat shows it as the answer's tail (``turn_account.with_account``'s shape)."""
    kept = vars(response) if response is not None and hasattr(response, "__dict__") else {}
    return SimpleNamespace(**{"usage": None, "finish_reason": "stop", **kept, "content": NO_DRAFT_SAID,
                              "tool_calls": None, "streamed": False})


def _holds_no_draft(loop: Any, current: Any, retry: Any) -> bool:
    """The retry gave no draft: it came back blank (the pointer kept, or F264's account of the turn's
    calls in its place) or points to a draft again."""
    said = getattr(retry, "content", "") or ""
    return retry is current or said == account_of(loop.tracker.outcomes) or bool(bare_pointer(said))


async def _retried(loop: Any, current: Any, messages: List[Dict[str, Any]],
                   tools: Optional[List[Dict[str, Any]]], pointer: str) -> Any:
    """The one retry for a bare pointer; a retry with no draft in it either becomes the plain line."""
    logger.warning("[tool-loop] a reply to a here-ask points to a draft that is not there — nudging once")
    retry = await nudge_about(loop._llm, current, messages, tools, draft_nudge(pointer))
    if getattr(retry, "tool_calls", None):
        return retry
    if _holds_no_draft(loop, current, retry):
        logger.warning("[tool-loop] the retry still holds no draft — saying so plainly")
        return no_draft(retry)
    return retry


def has_the_draft_asked_here(recover: Recover) -> Recover:
    """Wraps the loop's final-reply check (``ToolLoopExecutor._recover_claimed_action``): when it has
    nothing to recover, a reply in the owner's chat to a here-ask that only points to a draft is
    retried once, through ``nudge_about`` like the check's own nudges."""
    @functools.wraps(recover)
    async def check(loop: Any, current: Any, messages: List[Dict[str, Any]],
                    tools: Optional[List[Dict[str, Any]]]) -> Any:
        retry = await recover(loop, current, messages, tools)
        if retry is not None or not tools or getattr(current, "tool_calls", None) or not auto_speaks(loop.promises):
            return retry
        pointer = bare_pointer(getattr(current, "content", "") or "")
        if not pointer or not asked_here(owners_words(messages)):
            return None
        return await _retried(loop, current, messages, tools, pointer)
    return check


def shows_the_work(recover: Recover) -> Recover:
    """The loop's final-reply check, then F413's (the work the owner asked to see is shown), then
    F419's (a here-ask's reply holds its draft): each nudges only when those before it did not."""
    return has_the_draft_asked_here(shows_fetched_work(recover))


def a_bare_pointer_goes_through_the_loop(check: FirstReplyCheck) -> FirstReplyCheck:
    """Wraps the chat's first-reply check (``StreamingChatService._first_reply_goes_through_the_loop``):
    a first reply to a here-ask that ran no tool and only points to a draft goes through the loop too,
    whose final-reply check retries it once."""
    @functools.wraps(check)
    async def routed(chat: Any, response: Any, use_tools: Optional[List[Dict[str, Any]]],
                     prefetched: Any, latest_text: str) -> bool:
        if await check(chat, response, use_tools, prefetched, latest_text):
            return True
        if not use_tools or getattr(response, "tool_calls", None):
            return False
        return bool(bare_pointer(getattr(response, "content", "") or "")) and asked_here(latest_text)
    return routed


__all__ = ["DRAFT_MIN_WORDS", "DRAFT_NOT_THERE_MSG", "NO_DRAFT_SAID", "a_bare_pointer_goes_through_the_loop",
           "bare_pointer", "draft_nudge", "has_the_draft_asked_here", "shows_the_work"]
