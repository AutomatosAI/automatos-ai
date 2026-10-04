"""F317 (night 9b, build 15): Auto didn't use what the team had already found.

"The club box count is on the board — the Analyst and the Business Analyst both did cards on
it. What number did they give?" got "neither the Analyst nor the Shopify Business Analyst's
completed tasks directly address the total number of club boxes" (b8d81ba9) while the Q5
cards said 63. In the same chat, "Check what the team has already found on the board this week
first" got a count of cards by status: the message names the board, so the turn went to the
board's fast path, and the one board read Auto chose was platform_board_summary. "Do I need to
reorder any Kirinyaga?" was answered from the 1 September paper while the Watchdog's card
from ten minutes earlier sat on the board (378c1e41); Auto read it only when told to. Nothing
on Auto's path read the cards' answers: F305's past-work scope is there for an agent that asks
for it by name, and Auto never did.

Now a question to Auto (or a message asking what the team found) reads the board first
(``services/team_findings.py``: the cards in review or done whose title or answer share the
question's words, best match first, at most four) and, after the document passages, gives the
model each card by number, title, agent, state and day with the start of its answer, labelled
an agent's answer and not the owner's facts. The rule: say what a card found, by its number
and who found it, before or beside its own answer, say plainly if a figure differs, and never
say the team hasn't covered something a card answers. A message asking what the team found
also gets "answer from what the cards say, not a count of cards by status", or, when no card
matches, how to look for them. The read shows in the activity trail and counts as this turn's
board read. A widget visitor's turn reads nothing: the board is the owner's (F155).
"""
from __future__ import annotations

import functools
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

READ_TOOL = "platform_list_tasks"
READ_ARGS = {"automatic": True, "answers": True, "status": "review,done"}
SYSTEM_ROLE = "system"
NOTE_HEAD = ("What your team already found on the board, read just now for this question. Each is an agent's "
             "answer on a card, not the owner's facts; a card in review isn't approved yet:")
ROW = "- {number} '{title}', {agent}, {state} {day}: {excerpt}"
STATES = {"done": "done", "review": "in review since"}
NOTE_RULE = ("Where a card here answers the owner's question, say what it found, by its number and who found it, "
             "before or beside your own answer, and say plainly if your figure differs from it. Never say the "
             "team hasn't covered something a card here answers. A card's figure is from the day it was written: "
             "for today's figure from the shop, count it again too.")
ASKED_RULE = "The owner asked what the team found: answer from what these cards say, not a count of cards by status."
NONE_FOUND = ("The owner asked what the team already found, and no card in review or done shares the words of their "
              "question. Look for the cards on it with platform_list_tasks, read their answers with "
              "platform_get_task, and answer from what they say, never from a count of cards by status.")
SUMMARY = "{count} card(s) on the board already answer this"
TITLE_CHARS = 100

_ASKS_THE_TEAM = re.compile(
    r"\b(?:the|my|our) team\b|\b(?:check|look at|read|go through) (?:the|my|our) (?:board|cards)\b"
    r"|\bon (?:the|my|our) board\b|\b(?:already|have) (?:found|counted|worked out|looked at|done)\b"
    r"|\bwhat did (?:the|my|our) (?:\w+ ){0,2}(?:find|say|get|give|count|work out)\b",
    re.IGNORECASE)


def asks_what_the_team_found(text: object) -> bool:
    """Whether the owner points Auto at the team's work: the team, the board, what's been found."""
    return bool(_ASKS_THE_TEAM.search(str(text or "")))


def findings_note(findings: List[Dict[str, Any]], asked: bool) -> str:
    """What the model reads: each card that answers, then the rule."""
    rows = [ROW.format(number=f["number"], title=str(f["title"] or "")[:TITLE_CHARS], agent=f["agent"],
                       state=STATES.get(f["status"], f["status"]), day=f["day"], excerpt=f["excerpt"])
            for f in findings]
    return "\n".join([NOTE_HEAD, *rows, NOTE_RULE, *([ASKED_RULE] if asked else [])])


def _read(chat: Any, latest_text: str) -> Optional[List[Dict[str, Any]]]:
    """The cards that answer, read in a savepoint; None when the board can't be read."""
    from services.team_findings import team_findings

    try:
        with chat.db.begin_nested():
            return team_findings(chat.db, chat.workspace_id, latest_text)
    except Exception:
        logger.exception("[F317] the team's findings could not be read for this turn")
        return None


def _frames(chat: Any, count: int, elapsed_ms: int) -> List[Any]:
    call_id = f"team-findings-{uuid.uuid4().hex[:12]}"
    handler = chat.streaming_handler
    return [handler.format_aisdk_tool_start(call_id, READ_TOOL, dict(READ_ARGS)),
            handler.format_aisdk_tool_end(call_id, READ_TOOL, True, duration_ms=elapsed_ms,
                                          summary=SUMMARY.format(count=count))]


def _wanted(chat: Any, latest_text: str) -> bool:
    from consumers.chatbot.knowledge_prefetch import is_question
    from consumers.chatbot.needs_you_turn import asks_what_needs_me

    if getattr(chat, "widget_mode", False) or asks_what_needs_me(latest_text):
        return False
    return is_question(latest_text) or asks_what_the_team_found(latest_text)


def reads_the_team(chat: Any, latest_text: str, prefetched: List[Any]) -> tuple:
    """For a question: (the note, or None; the frames to yield). The read goes in ``prefetched``."""
    if not _wanted(chat, latest_text):
        return None, []
    started = time.monotonic()
    findings = _read(chat, latest_text)
    asked = asks_what_the_team_found(latest_text)
    if findings is None or not (findings or asked):
        return None, []
    prefetched.append((READ_TOOL, dict(READ_ARGS)))
    logger.info("[F317] %d card(s) on the board answer this question", len(findings))
    note = findings_note(findings, asked) if findings else NONE_FOUND
    return note, _frames(chat, len(findings), int((time.monotonic() - started) * 1000))


Turn = Callable[..., AsyncGenerator[Any, None]]


def reads_what_the_team_found(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a question reads the cards that already
    answer it, and their answers go in after the document passages (see the module)."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], agent_runtime: Any,
                      chat_id: str, prefetched: List[Any]) -> AsyncGenerator[Any, None]:
        note, frames = reads_the_team(chat, latest_text, prefetched)
        for frame in frames:
            yield frame
        async for frame in retrieval_first(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
            yield frame
        if note:
            llm_messages.append({"role": SYSTEM_ROLE, "content": note})
    return wrapped


__all__ = ["ASKED_RULE", "NONE_FOUND", "NOTE_RULE", "asks_what_the_team_found", "findings_note",
           "reads_the_team", "reads_what_the_team_found"]
