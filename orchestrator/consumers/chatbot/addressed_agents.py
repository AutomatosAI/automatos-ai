"""PRD-256 FX-014 (E5): who a message hands work to, read from the roster whatever the tiers said.

Night 12: ``handoffs.the_lane`` turned a named agent into a ticket only on a DELEGATE verdict, which
the rubric no longer offers, so "Get OPS to…" and "Ask RESEARCHER…" stayed with Auto. Two shapes the
owner used that night are read here, beside the names ``handoffs`` already reads:

- an agent's id given as the answer to "which one?": "267, the operations one", "267 (ops)", "agent 284".
  The number must be an active teammate's id, and the words after it must describe that agent (a word
  of its name or job title), so "30, the big bags" is not agent 30. A bare number ("2") is an answer to
  any numbered question, so it hands nothing over unless it says "agent";
- a name several active agents carry ("Get OPS to…" with two called OPS): nobody is picked and no
  copy is made; Auto asks which, listing each as FX-012's refusal does ("267 · OPS · Operations
  Manager"), and the answer (the id) is the ticket's agent.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

# "agent 267", "267, the operations one", "267 — the shop floor one", "267 (ops)".
ID_REPLY = re.compile(r"^\s*(?P<agent>agent\s+)?(?P<id>\d{1,9})\s*(?:[.!]?\s*$|[,:;(\-–—]\s*(?P<rest>.*)$)", re.IGNORECASE)
# The answer to "which one?" is short: a longer message that opens with a number is about something else.
MAX_REPLY_WORDS = 8
MIN_WORD_CHARS = 3
# Two words name the same thing when one starts the other this far ("operation", "operations").
MIN_SHARED_PREFIX = 4
NOT_DESCRIPTIVE = frozenset({
    "the", "one", "please", "agent", "that", "this", "yes", "its", "it's", "okay", "thanks", "thank", "you",
    "use", "pick", "with", "and", "for", "him", "her", "them",
})
_WORD = re.compile(r"[a-z0-9']+")
DESCRIBED_BY = ("name", "job_title", "role", "team")
# Two teammates handed one piece of work together: "RESEARCHER and WRITER", "OPS, CLUB DESK".
JOINED = r"(?<![a-z0-9]){first}\s*(?:,|&|\band\b)\s*(?:the\s+|my\s+|our\s+)?{second}(?![a-z0-9])"

CLASH_DIRECTIVE = (
    "\n\n## Manager directive — ask which agent\n"
    "The user is handing {what} to '{name}', and {count} active agents are called that: {candidates}. "
    "Do NOT pick one yourself, do NOT create a new agent and do NOT file anything yet. Ask the user in ONE "
    "line which one they mean, listing each as above (id · name · job title). When they answer (the id, or "
    "which one), {then}, and confirm in one line with the card number and who has it.\n"
)
THEN_TICKET = ("file the ticket with platform_create_task and that agent's agent_id, written as the dispatch "
               "contract below, and start it unless they deferred")
THEN_CARD = "call platform_assign_task with task_id \"{card}\" and that agent's agent_id (do NOT create a new card)"


def _words(text: object) -> set:
    found = _WORD.findall(str(text or "").lower())
    return {word for word in found if len(word) >= MIN_WORD_CHARS and word not in NOT_DESCRIPTIVE}


def _same_word(said: str, known: str) -> bool:
    if said == known:
        return True
    shorter = min(len(said), len(known))
    return shorter >= MIN_SHARED_PREFIX and (said.startswith(known) or known.startswith(said))


def describes(rest: Optional[str], agent: Any) -> bool:
    """Whether the words after an id describe ``agent``: one of them is a word of its name or job."""
    said = _words(rest)
    known = set().union(*(_words(getattr(agent, attr, "")) for attr in DESCRIBED_BY))
    return any(_same_word(word, other) for word in said for other in known)


def id_reply(message: Optional[str], roster: Sequence[Any]) -> Optional[Any]:
    """The active teammate a short reply names by its id ("267, the operations one"), else None."""
    text = str(message or "")
    found = ID_REPLY.match(text)
    if not found or len(text.split()) > MAX_REPLY_WORDS:
        return None
    agent = next((agent for agent in roster if str(agent.id) == found.group("id").lstrip("0")), None)
    if agent is None or not (found.group("agent") or describes(found.group("rest"), agent)):
        return None
    return agent


def joined_with_another(message: Optional[str], name: str, roster: Sequence[Any]) -> bool:
    """Whether another teammate is named right beside ``name`` ("have RESEARCHER and WRITER plan …"):
    work for several, which a mission verdict keeps."""
    said, first = str(message or "").lower(), re.escape(name.strip().lower())
    others = {str(getattr(agent, "name", "") or "").strip().lower() for agent in roster} - {name.strip().lower(), ""}
    return any(re.search(JOINED.format(first=first, second=re.escape(other)), said) for other in others)


def shared_name(agents: Sequence[Any]) -> Optional[str]:
    """The name two or more ``agents`` all carry (any case), else None."""
    names = {str(getattr(agent, "name", "") or "").strip().casefold() for agent in agents}
    return str(agents[0].name).strip() if len(agents) >= 2 and len(names) == 1 else None


def clash_directive(agents: Sequence[Any], card: Optional[str] = None) -> str:
    """The ASSIGN directive for a name several active agents carry: ask which, never pick or copy."""
    from modules.tools.discovery.agent_refs import candidates

    directive = CLASH_DIRECTIVE.format(
        what=f"card {card}" if card else "this work", name=shared_name(agents), count=len(agents),
        candidates=candidates(list(agents)), then=THEN_CARD.format(card=card) if card else THEN_TICKET,
    )
    if card:
        return directive
    from consumers.chatbot.auto import BOARD_MOVES_THE_CARD
    from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT

    return f"{directive}\n{DISPATCH_CONTRACT_FRAGMENT}\n{BOARD_MOVES_THE_CARD}\n"


__all__ = ["CLASH_DIRECTIVE", "ID_REPLY", "MAX_REPLY_WORDS", "clash_directive", "describes", "id_reply",
           "joined_with_another", "shared_name"]
