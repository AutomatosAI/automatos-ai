"""P256-FIX-N13-5 (F412, night 13): a reply's opening says only what this conversation holds.

Two faults opened Auto's replies on night 13:

* Six brand-new chats opened with "My last reply…" or "As I said…" when the chat had no turn
  before it: the recalled thread checkpoints read to the model as its own earlier turns. They
  are now labelled as earlier conversations (``modules/context/earlier_conversations.py``), and
  the opening of a reply in a conversation with no earlier reply is checked by one rule
  (``without_a_back_reference``): "As I said, …" loses its lead-in, and "My last reply said …"
  or "In my last reply, …" names an earlier conversation instead ("In an earlier conversation,
  I said …").
* The not-done honesty line showed on replies that claimed nothing (a greeting, a fact). The
  receipts add it only for a claim they do not back (``consumers/chatbot/receipts.honesty_lines``),
  and every saved answer opens with the lines above it, so the conversation the model reads
  holds the line at the top of its own earlier replies, and the model wrote it again itself.
  A reply in such a conversation (or in a fresh one) loses a copy of the line it opens with
  (``without_the_platforms_line``): the line is the receipts' alone, and said once. Since
  P256-FIX-N15-2 the platform says it only in place of a claim it took out, never above a reply.

``opening_for`` gives the checks a call's reply opening goes through; ``core/llm/turn_order.py``
runs them on the text that streams (``reprompt_reply.ApologyGate``) and on the reply that is
saved, with the re-prompt's apology check (F295, night 9b). Only a call that streams (Auto's
chat) is checked for these two; an agent run or a distiller is not.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Messages = List[Dict[str, Any]]
Cut = Callable[[str], str]
ASSISTANT, USER = "assistant", "user"
EARLIER_CONVERSATION = "In an earlier conversation"

_REPLY = r"(?:reply|message|answer|response)"
_MY_LAST = rf"my\s+(?:last|previous|earlier|prior)\s+{_REPLY}\b"
_SAID = r"(?:said|mentioned|noted|explained|promised|suggested|discussed|covered|agreed|flagged|confirmed)"
# A short greeting the opening may start with ("Evening Gerard! "): kept, and the rule reads after it.
_GREETING = re.compile(r"^\s*(?:(?:hi|hello|hey|evening|morning|afternoon|good (?:morning|afternoon|evening))"
                       r"(?:\s+(?-i:[A-Z])[\w'’-]*)?\s*[,!.—–-]+\s*)?", re.I)
# A lead-in that refers back to an earlier reply of this conversation: dropped ("As I said, …").
_LEAD_IN = re.compile(
    rf"^(?:(?:as|like)\s+(?:I|we)(?:'ve|’ve| have)?\s+{_SAID}"
    rf"(?:\s+(?:earlier|before|above|previously|last time|in\s+{_MY_LAST}))?"
    rf"|as\s+discussed(?:\s+(?:earlier|before|previously|last time))?"
    rf"|(?:following up on|further to|picking up (?:from|on)|to pick up (?:from|on)|continuing (?:from|on))\s+"
    rf"(?:{_MY_LAST}|what I said)"
    rf"|(?:picking up|to pick up|continuing|carrying on)(?:\s+from)?\s+where we left off"
    rf")\s*[,:;—–-]\s*", re.I)
# "In my last reply, …" names the earlier conversation; "My last reply said …" says who said it there.
_SENTENCE_ENDS = (". ", "! ", "? ", ".\n", "!\n", "?\n")
_IN_MY_LAST = re.compile(rf"^in\s+{_MY_LAST}", re.I)
_MY_LAST_SAID = re.compile(rf"^{_MY_LAST}\s+(?P<verb>{_SAID})\b", re.I)


def _greeting_then(text: str) -> Tuple[str, str]:
    found = _GREETING.match(text)
    return text[: found.end()], text[found.end():]


def without_a_back_reference(text: str) -> str:
    """``text`` with the back-reference its opening makes to an earlier reply taken off ("As I said, "),
    or named as an earlier conversation ("My last reply said" → "In an earlier conversation, I said");
    the rest, and a text that opens with none, as it is."""
    greeting, rest = _greeting_then(text or "")
    dropped = _LEAD_IN.match(rest)
    if dropped:
        after = rest[dropped.end():]
        return greeting + after[:1].upper() + after[1:]
    said = _MY_LAST_SAID.match(rest)
    if said:
        return f"{greeting}{EARLIER_CONVERSATION}, I {said.group('verb').lower()}{rest[said.end():]}"
    named = _IN_MY_LAST.match(rest)
    return f"{greeting}{EARLIER_CONVERSATION}{rest[named.end():]}" if named else text


@functools.lru_cache(maxsize=1)
def _the_line() -> Tuple[str, str]:
    """The not-done line's opening and its close, around what it says was not done."""
    from consumers.chatbot.claim_check import NOT_DONE

    head, tail = NOT_DONE.split("{said}")
    return head, tail


def _a_copy_of_the_line(text: str) -> Optional[re.Match]:
    head, tail = _the_line()
    return re.match(rf"\s*{re.escape(head)}[^\n]*?{re.escape(tail)}\s*", text)


def _after_the_copies(text: str) -> str:
    """``text`` past the copies of the not-done line it opens with."""
    found = _a_copy_of_the_line(text)
    while found:
        text = text[found.end():]
        found = _a_copy_of_the_line(text)
    return text


def without_the_platforms_line(text: str) -> str:
    """``text`` without the copies of the not-done line it opens with; a text that is only the line stays."""
    rest = _after_the_copies(text or "")
    return rest if rest.strip() and rest != text else text


def could_be_the_platforms_line(held: str) -> bool:
    """Whether the opening held so far is only copies of the not-done line, or may still turn out to be one."""
    rest = _after_the_copies(held)
    if rest != held and not rest.strip():
        return True
    head, tail = _the_line()
    said = rest.lstrip()
    if len(said) < len(head):
        return head.startswith(said)
    return said.startswith(head) and tail not in said and "\n" not in said


def _back_reference_pending(held: str) -> bool:
    """Whether the opening held so far (past any copy of the line and a greeting) has not yet shown
    whether it refers back: no rule matched and its first sentence is not complete."""
    _greeting, rest = _greeting_then(_after_the_copies(held))
    if without_a_back_reference(rest) != rest:
        return False
    return not any(end in rest for end in _SENTENCE_ENDS)


def _never(_held: str) -> bool:
    return False


def _any_waits(waits: Tuple[Callable[[str], bool], ...], held: str) -> bool:
    return any(wait(held) for wait in waits)


def _content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def _a_check(message: Any) -> bool:
    """A user turn the platform wrote (a re-prompt or the loop's nudge), not the person."""
    from core.llm.turn_order import REPROMPT_FRAME
    from modules.tools.execution.nudges import PLATFORM_CHECK

    return _content(message).startswith((PLATFORM_CHECK, REPROMPT_FRAME))


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else None


def opens_the_conversation(messages: Optional[Messages]) -> bool:
    """Whether no reply of the model comes before the person's latest turn: a chat's first reply
    (its tool rounds and a re-prompt within it included)."""
    turns = [at for at, message in enumerate(messages or []) if _role(message) == USER and not _a_check(message)]
    return bool(turns) and not any(_role(message) == ASSISTANT for message in (messages or [])[: turns[-1]])


def holds_the_line(messages: Optional[Messages]) -> bool:
    """Whether the conversation the model reads holds the not-done line (an earlier saved answer's)."""
    _head, tail = _the_line()
    return any(tail in _content(message) for message in messages or [] if _role(message) != USER)


@dataclass(frozen=True)
class Opening:
    """The checks a reply's opening goes through, in order, and whether a held opening must wait."""

    cuts: Tuple[Cut, ...]
    pending: Callable[[str], bool] = _never

    def trim(self, text: str) -> str:
        for cut in self.cuts:
            text = cut(text)
        return text

    def checked(self, response: Any) -> Any:
        """``response`` with its opening checked (a copy; the response itself is never changed)."""
        text = getattr(response, "content", None)
        if not isinstance(text, str) or not text.strip():
            return response
        kept = self.trim(text)
        if kept == text or not dataclasses.is_dataclass(response):
            return response
        logger.info("[F412] a reply's opening was checked: %d characters became %d", len(text), len(kept))
        return dataclasses.replace(response, content=kept)


def opening_for(messages: Optional[Messages], *, reprompt: bool, streams: bool) -> Optional[Opening]:
    """The checks for the reply to ``messages``: a re-prompt's apology (F295); for a call that streams,
    a copy of the not-done line (in a fresh conversation or one that holds it) and, in a fresh one, a
    back-reference to an earlier reply. None when there is nothing to check."""
    from core.llm.reprompt_reply import without_an_opening_apology

    fresh = streams and opens_the_conversation(messages)
    the_line = streams and (fresh or holds_the_line(messages))
    cuts = ((without_the_platforms_line,) if the_line else ()) + ((without_an_opening_apology,) if reprompt else ()) \
        + ((without_a_back_reference,) if fresh else ())
    waits = ((could_be_the_platforms_line,) if the_line else ()) + ((_back_reference_pending,) if fresh else ())
    if not cuts:
        return None
    return Opening(cuts, functools.partial(_any_waits, waits) if waits else _never)


__all__ = ["EARLIER_CONVERSATION", "Opening", "could_be_the_platforms_line", "holds_the_line", "opening_for",
           "opens_the_conversation", "without_a_back_reference", "without_the_platforms_line"]
