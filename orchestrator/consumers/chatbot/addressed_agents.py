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

P256-FIX-RVW-12: the names ``handoffs`` reads are read here too. A name is handed work only when a task
follows it ("Get OPS to…", "Have OPS check…", "Ask RESEARCHER what…", "Ask Sales: …"), so "Have
support tickets been answered today?", "Get sales figures for Q3" and "Get OPS's stock report" stay
the tiers' with an agent called Support, Sales or OPS; work handed to two teammates together ("Have
RESEARCHER and WRITER plan the launch") stays the tiers' whatever they said.
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

# Work handed to a teammate by name (P256-FIX-RVW-12): the verb, the name, then the task ("ask Jim to …",
# "have the Researcher check …"). A name with no task after it is only a word of the sentence.
ADDRESSED_BY = r"\b(?P<verb>ask|have|get|tell|let)\s+(?:the\s+|my\s+|our\s+)?{name}(?![a-z0-9])"
_COLON = re.compile(r"^\s*:")
_POSSESSIVE = re.compile(r"^['’]")                       # "OPS's stock report", "the Sales' figures"
_NEXT_TWO = re.compile(r"^\s+(?P<first>[a-z][a-z'’-]*)(?:\s+(?P<second>[a-z0-9][a-z0-9'’-]*))?")
# "Get X to …" is the only way to get someone to do something: "get sales figures" is a fetch.
ONLY_WITH_TO = "get"
ASK = "ask"
# What an ask hands over after the name: "Ask RESEARCHER what…", "… whether…", "… about…", "… the price".
ASKED_WHAT = frozenset({
    "what", "how", "why", "when", "where", "who", "whom", "whose", "which", "whether", "if", "about", "for",
    "the", "a", "an",
})
AUXILIARIES = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "shall", "may", "might", "must",
})
# A word after the head noun that makes the name a noun's modifier: "Have support ticket been …".
NOUN_PHRASE_AUXILIARIES = frozenset({"been", "being"})
PARTICIPLES = frozenset({
    "done", "gone", "seen", "sent", "made", "taken", "given", "written", "got", "gotten", "heard", "told", "said",
    "paid", "sold", "bought", "brought", "thought", "found", "kept", "left", "met", "spoken", "chosen", "known",
    "shown", "drawn", "begun", "broken", "eaten", "fallen", "forgotten", "hidden", "grown", "thrown", "flown",
})
# Bare verbs that end in "ed" ("Have OPS feed …" ends in "eed" and is read as a verb already).
BARE_ED = frozenset({"embed", "shed", "shred", "wed"})
PARTICIPLE_ENDING, BARE_EED = "ed", "eed"
# A plural noun after the name makes the name its modifier ("support tickets", "sales figures"); a verb's
# bare form ends in a single s only after s, u or a ("process", "focus", "canvas").
_PLURAL = re.compile(r"[^sua'’]s$")

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


def _participle(word: str) -> bool:
    if word in PARTICIPLES:
        return True
    return word.endswith(PARTICIPLE_ENDING) and not word.endswith(BARE_EED) and word not in BARE_ED


def _bare_verb(first: str, second: str) -> bool:
    """Whether ``first`` can open the task after a name ("Have OPS check …", "Tell CLUB DESK the box …"):
    not an auxiliary or a past participle ("Have Support answered …?", "Have sales been …?"), not a plural
    noun the name modifies ("support tickets"), and not a noun with a verb of its own ("support ticket been")."""
    if first in AUXILIARIES or _participle(first) or _PLURAL.search(first):
        return False
    return second not in NOUN_PHRASE_AUXILIARIES


def _task_follows(verb: str, rest: str) -> bool:
    """Whether the words after an addressed name (``rest``) are a task for it: a colon, "to <verb>", an
    ask's question ("what", "whether", "about"), or, after have, let or tell, a bare verb."""
    if _COLON.match(rest):
        return True
    words = None if _POSSESSIVE.match(rest) else _NEXT_TWO.match(rest)
    if words is None:
        return False
    first, second = words.group("first"), words.group("second") or ""
    if first == "to":
        return bool(second)
    if verb == ONLY_WITH_TO:
        return False
    return first in ASKED_WHAT if verb == ASK else _bare_verb(first, second)


def addressed_by_name(message: Optional[str], name: str) -> bool:
    """Whether ``message`` hands ``name`` work: ask, have, get, tell or let, the name, then a task
    ("Get OPS to …", "Have OPS check …", "Ask RESEARCHER what …", "Ask Sales: …"). A possessive ("Get
    OPS's stock report"), a name that modifies a noun ("Have support tickets been answered?", "Get sales
    figures for Q3") or one followed by a past participle or an auxiliary hands nothing over."""
    said, wanted = str(message or "").lower(), re.escape(name.strip().lower())
    return any(_task_follows(found.group("verb"), said[found.end():])
               for found in re.finditer(ADDRESSED_BY.format(name=wanted), said))


def joined_with_another(message: Optional[str], name: str, roster: Sequence[Any]) -> bool:
    """Whether another teammate is named right beside ``name`` ("have RESEARCHER and WRITER plan …"):
    work for several, which stays the tiers' lane whatever they said (P256-FIX-RVW-12)."""
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


__all__ = ["ADDRESSED_BY", "CLASH_DIRECTIVE", "ID_REPLY", "MAX_REPLY_WORDS", "addressed_by_name", "clash_directive",
           "describes", "id_reply", "joined_with_another", "shared_name"]
