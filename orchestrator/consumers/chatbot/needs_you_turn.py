"""F307 (night 9): "What needs me?" is answered from the board's Needs you, read first.

"What needs me right now?" got "It looks like all agents are currently idle, and
there are no anomalies or urgent tasks that require your immediate attention. You're
all clear!" with 16 cards in review (chat 2223df4d). Its one call was
platform_fleet_status, the agents' state, not the owner's. Twice more, in fresh chats,
the same sentence came back in about 2 s with no call at all (5950e0e1 with #1849,
#1856 and #1858 in review; 834c2e25 with mission step #1888 waiting). F263 put the
board's Needs you on platform_board_summary, but nothing made the turn read it: the
model chose the call, or answered from nothing. The workspace also sat at onboarding
stage not_started all night, so every Auto turn was pinned to AutoBrain's Tier 0 and
the board fast path never ran (F314, services/onboarding_content.py).

Now a message asking what needs the owner, or what waits for them, reads Needs you
(services.needs_you through ``board_waiting.whats_waiting``: cards in review,
questions, approvals, stuck work, the steps of a mission that ended, failures; the
same count the board shows) before the turn's first model call. The total and each
row by number go in last, with the rule that the answer is that list and is never
"all clear" unless the total is 0; the activity trail shows the read. A read that
fails says so, so the answer can't be "all clear" either. An answer that still says
nothing needs them while Needs you holds something gets one plain line with the count
(``says_the_board_is_clear``, on the saved answer's additions).

A widget visitor's turn reads nothing: the board is the owner's (F155).
"""
from __future__ import annotations

import contextvars
import functools
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

READ_TOOL = "platform_board_summary"
READ_ARGS = {"automatic": True, "needs_you": True}
SYSTEM_ROLE = "system"
LISTED = 40
TITLE_CHARS = 80
KIND_WORDS = {"review": "in review, waiting for your OK", "question": "a question for you",
              "approval": "waiting for your approval", "stuck": "stuck until you act", "failed": "failed"}
NOTE_HEAD = ("The owner is asking what needs them. This is the board's Needs you, read just now (the same "
             "count the board shows): {total} in all{by_kind}.")
NOTE_RULE = ("Answer from this list: give the total and name each item by its number and title. Never say "
             "nothing needs them, or that they are all clear, unless the total is 0.")
NOTE_UNREAD = ("The owner is asking what needs them, and the board's Needs you could not be read just now. "
               "Don't say nothing needs them: say you couldn't read the board and that Needs you on the board "
               "lists what waits for them.")
MORE = "- and {more} more on the board"
SUMMARY = "{total} waiting for you, read from your board's Needs you"
CLEAR_LINE = ("Just to be clear: your board's Needs you has {total} waiting for you right now ({by_kind}). "
              "Open Needs you on the board to see them.")

_ASKS = re.compile(
    r"\bneeds? (?:me|my (?:attention|ok|okay|approval|input|sign-?off|decision|review))\b"
    r"|\bneeds you\b|\b(?:waiting|waits?) (?:for|on) me\b|\banything (?:else )?for me\b|\bon my plate\b"
    r"|\bwhat (?:do|should) i (?:need to )?(?:do|look at|deal with|approve|review)"
    r"(?: (?:next|now|today|first|right now))?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_ALL_CLEAR = re.compile(
    r"\ball clear\b|\bnothing (?:needs|requires|is waiting|waits)\b"
    r"|\bno\b[^.!?\n]{0,40}\b(?:needs?|requires?|waiting)\b[^.!?\n]{0,25}\byour?\b",
    re.IGNORECASE,
)
# The Needs-you total read for this turn (None: not asked, or not read).
_WAITING: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("f307_waiting", default=None)


def asks_what_needs_me(text: object) -> bool:
    """Whether the owner's message asks what needs them, or what waits for them."""
    return bool(_ASKS.search(str(text or "").strip()))


def _by_kind(counts: Dict[str, int]) -> str:
    return ", ".join(f"{n} {KIND_WORDS.get(kind, kind)}" for kind, n in counts.items() if n)


def needs_you_note(waiting: Dict[str, Any]) -> str:
    """What the model reads: the total, by kind, and each row by number and title."""
    by_kind = _by_kind(waiting.get("by_kind") or {})
    head = NOTE_HEAD.format(total=waiting["total"], by_kind=f" ({by_kind})" if by_kind else "")
    cards = waiting.get("cards") or []
    rows = [f"- {c.get('number') or 'no number'} '{(c.get('title') or '')[:TITLE_CHARS]}': "
            f"{KIND_WORDS.get(c.get('kind'), c.get('kind'))}" for c in cards[:LISTED]]
    if len(cards) > LISTED:
        rows.append(MORE.format(more=len(cards) - LISTED))
    return "\n".join([head, *rows, NOTE_RULE])


def _read(chat: Any) -> Optional[Dict[str, Any]]:
    """Needs you for this chat's workspace, read in a savepoint; None when it can't be read."""
    from modules.tools.discovery.board_waiting import whats_waiting

    try:
        with chat.db.begin_nested():
            return whats_waiting(chat.db, chat.workspace_id)
    except Exception:
        logger.exception("[F307] Needs you could not be read for 'what needs me'")
        return None


def _frames(chat: Any, waiting: Dict[str, Any], elapsed_ms: int) -> List[Any]:
    """The activity trail's frames for the read."""
    call_id = f"needs-you-{uuid.uuid4().hex[:12]}"
    handler = chat.streaming_handler
    return [handler.format_aisdk_tool_start(call_id, READ_TOOL, dict(READ_ARGS)),
            handler.format_aisdk_tool_end(call_id, READ_TOOL, True, duration_ms=elapsed_ms,
                                          summary=SUMMARY.format(total=waiting["total"]))]


def _reads_needs_you(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]],
                     prefetched: List[Any]) -> List[Any]:
    """For a 'what needs me' turn: the note in last, the read in ``prefetched``; the frames to yield."""
    _WAITING.set(None)
    if getattr(chat, "widget_mode", False) or not asks_what_needs_me(latest_text):
        return []
    started = time.monotonic()
    waiting = _read(chat)
    if waiting is None:
        llm_messages.append({"role": SYSTEM_ROLE, "content": NOTE_UNREAD})
        return []
    _WAITING.set(waiting)
    llm_messages.append({"role": SYSTEM_ROLE, "content": needs_you_note(waiting)})
    prefetched.append((READ_TOOL, dict(READ_ARGS)))
    logger.info("[F307] 'what needs me' read Needs you first: %d waiting", waiting["total"])
    return _frames(chat, waiting, int((time.monotonic() - started) * 1000))


Turn = Callable[..., AsyncGenerator[Any, None]]


def answers_what_needs_you(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a 'what needs me' turn reads the
    board's Needs you before the first model call (see the module)."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], agent_runtime: Any,
                      chat_id: str, prefetched: List[Any]) -> AsyncGenerator[Any, None]:
        for frame in _reads_needs_you(chat, latest_text, llm_messages, prefetched):
            yield frame
        async for frame in retrieval_first(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
            yield frame
    return wrapped


def says_the_board_is_clear(answer: object, waiting: Optional[Dict[str, Any]]) -> bool:
    """An answer that says nothing needs the owner while Needs you holds something,
    and never gives the count."""
    total = (waiting or {}).get("total") or 0
    said = str(answer or "")
    return total > 0 and bool(_ALL_CLEAR.search(said)) and not re.search(rf"\b{total}\b", said)


Additions = Callable[..., List[str]]


def never_all_clear_unread(answer_additions: Additions) -> Additions:
    """Wrap ``StreamingChatService._answer_additions``: an answer that says the owner is
    all clear while this turn's Needs you holds something gains the plain count."""
    @functools.wraps(answer_additions)
    def wrapped(f187_verdict: Any, final_round: Any) -> List[str]:
        additions = answer_additions(f187_verdict, final_round)
        waiting = _WAITING.get()
        if not says_the_board_is_clear(getattr(final_round, "content", ""), waiting):
            return additions
        line = CLEAR_LINE.format(total=waiting["total"], by_kind=_by_kind(waiting.get("by_kind") or {}))
        return [*additions, f"\n\n{line}"]
    return wrapped


__all__ = ["answers_what_needs_you", "asks_what_needs_me", "needs_you_note", "never_all_clear_unread",
           "says_the_board_is_clear"]
