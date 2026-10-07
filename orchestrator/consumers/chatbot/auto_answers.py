"""PRD-256 US-010 (Decision D2): Auto always answers; work goes to agents on tickets.

72% of the Tier-3 verdicts were DELEGATE, and api/chat.py handed each one to the
Universal Router, so a specialist answered the owner's chat in its own persona with its
own tools. That lane is gone. What the classifier returns is read here, after its cache
and its tiers (and after the Jev shadow has recorded the classifier's own verdict):

- a DELEGATE verdict that hands work to an agent by name ("ask Jim to …") becomes that
  agent's ticket (ASSIGN); a name only mentioned ("what did Jim say?") hands nothing over;
- any other DELEGATE verdict is Auto's to answer (RESPOND) with its platform tools;
- a card handed to a named agent ("Give #0192 to the Support Agent") is the ASSIGN lane
  on that card: the ticket is the card itself, never a copy (F241);
- Auto itself (the system agent) is never the agent a ticket goes to;
- everything else is returned as the tiers decided it.

Only the owner's own choice of agent in the UI (``request.agentId``) puts another agent
on the chat.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import re
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple

from consumers.chatbot.board_questions import CARD_NUMBER

logger = logging.getLogger(__name__)

PLATFORM_HINT = "platform"
# A name shorter than this never counts as named: a stray "AI" or "Bo" is not an agent.
MIN_NAME_CHARS = 3
REASONING_ANSWERS = "PRD-256 D2: Auto answers; no specialist takes over the chat"
REASONING_NAMED = "PRD-256 D2: the named agent gets a ticket"
REASONING_CARD = "PRD-256 D2 / F241: the named card goes to the named agent"
# "give #0192 to …", "assign ticket 12 to …", "hand #0192 over to …": the card sits between the
# verb and the first "to" after it. "Move #0192 to review" is a status, so "move" is no verb.
HANDS_ON = re.compile(r"\b(?:give|assign|reassign|hand)\b(?P<what>[^.?!\n]*?)\bto\b", re.IGNORECASE)
# The reference platform_assign_task takes: "#0192" as written, "ticket 12" as "12".
CARD_REF = re.compile(r"#?\d+(?:\.\d+)?")
# Who the card goes to: the words after "to", without the article, and ending where a
# purpose, a reason or a courtesy starts ("to Jim to handle by Friday, please" is Jim).
SENTENCE_END = re.compile(r"[.?!,;\n]")
LEADING_ARTICLE = re.compile(r"^(?:the|my|our)\s+", re.IGNORECASE)
RECEIVER_END = re.compile(r"\s+(?:to|for|by|so|and|because|please|now|thanks|today|asap)\b.*$", re.IGNORECASE)
NOT_A_RECEIVER = frozenset({"him", "her", "them", "you", "myself", "yourself", "review", "done"})
# Work handed to an agent by name: "ask Jim to …", "have the Researcher …", "get Jim to …".
ADDRESSED = r"\b(?:ask|have|get|tell|let)\s+(?:the\s+|my\s+|our\s+)?{name}(?![a-z0-9])"

Assess = Callable[..., Awaitable[Any]]


def agents_named_in(message: Optional[str], agents: Sequence[Any]) -> List[Any]:
    """The active agents whose name is in ``message`` as a whole word, case-insensitive,
    one entry per agent. A name inside another word ("automatically" for an agent called
    Auto) does not count."""
    text = (message or "").lower()
    named = {}
    for agent in agents if text else ():
        name = (getattr(agent, "name", "") or "").strip().lower()
        if len(name) >= MIN_NAME_CHARS and re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text):
            named.setdefault(agent.id, agent)
    return list(named.values())


def _handoff(message: Optional[str]) -> Optional[Tuple[str, str]]:
    """(card, the receiver's words) when the message hands a card on, else None."""
    said = str(message or "")
    for hands_on in HANDS_ON.finditer(said):
        card = CARD_NUMBER.search(hands_on.group("what"))
        if card:
            who = SENTENCE_END.split(said[hands_on.end():], maxsplit=1)[0].strip()
            who = RECEIVER_END.sub("", LEADING_ARTICLE.sub("", who)).strip()
            return CARD_REF.search(card.group(0)).group(0), who
    return None


def handed_card(message: Optional[str]) -> Optional[str]:
    """The card number the message hands on to someone ("give #0192 to …"), else None."""
    handoff = _handoff(message)
    return handoff[0] if handoff else None


def handed_to(message: Optional[str]) -> Optional[str]:
    """Who the message hands a card to ("give #0192 to the Support Agent please" → "Support Agent")."""
    handoff = _handoff(message)
    return (handoff[1] or None) if handoff else None


def needs_no_apps(assessment: Any) -> bool:
    """Whether the turn may skip the workspace's connected apps (Composio).

    Chitchat, Auto's own platform work and a memory recall need none. Every other ask
    Auto now answers itself (the old DELEGATE work: "send an email to John") keeps them,
    so answering in Auto's chat never costs the owner an app."""
    from consumers.chatbot.auto import Action, Complexity

    if assessment is None or assessment.action != Action.RESPOND:
        return False
    hints = set(assessment.tool_hints or [])
    if assessment.complexity == Complexity.ATOM:
        return True
    return hints <= {PLATFORM_HINT} if hints else bool(assessment.needs_memory)


def answered_by_auto(assessment: Any) -> Any:
    """A DELEGATE verdict as Auto's own answer: RESPOND, with the platform tools kept
    beside any narrowing hint (no hint narrows nothing, so they are there already)."""
    from consumers.chatbot.auto import Action

    hints = list(assessment.tool_hints or [])
    if hints and PLATFORM_HINT not in hints:
        hints.append(PLATFORM_HINT)
    return dataclasses.replace(
        assessment, action=Action.RESPOND, tool_hints=hints,
        reasoning=f"{assessment.reasoning} ({REASONING_ANSWERS})",
    )


def _ticket_for(assessment: Any, target: Tuple[int, str], reasoning: str) -> Any:
    from consumers.chatbot.auto import Action

    agent_id, agent_name = target
    return dataclasses.replace(
        assessment, action=Action.ASSIGN, target_agent_id=agent_id, target_agent_name=agent_name,
        reasoning=f"{assessment.reasoning} ({reasoning})",
    )


def _teammates(brain: Any) -> List[Any]:
    """The active roster without the system agent: Auto never hands work to itself."""
    return [agent for agent in brain._active_agents() if not getattr(agent, "is_system_agent", False)]


def _card_receiver(brain: Any, message: str, who: str) -> Optional[Tuple[int, str]]:
    """The one teammate a card is handed to, as (id, name). Named in part ("the Support
    Agent"), AutoBrain's roster match resolves it; a pronoun or an unknown name is nobody."""
    if len(who) < MIN_NAME_CHARS or who.lower() in NOT_A_RECEIVER:
        return None
    roster = _teammates(brain)
    named = agents_named_in(who, roster)
    if len(named) == 1:
        return named[0].id, named[0].name
    agent_id, agent_name = brain._match_roster_agent(who, roster, message=message)
    return (agent_id, agent_name) if agent_id is not None else None


def _addressed_agent(brain: Any, message: str) -> Optional[Tuple[int, str]]:
    """The one teammate the message hands work to by name ("ask Jim to …"), as (id, name).
    A name that is only mentioned ("what did Jim say?") hands nothing over."""
    said = message.lower()
    addressed = [agent for agent in agents_named_in(message, _teammates(brain))
                 if re.search(ADDRESSED.format(name=re.escape(agent.name.strip().lower())), said)]
    return (addressed[0].id, addressed[0].name) if len(addressed) == 1 else None


def the_lane(brain: Any, message: str, assessment: Any) -> Any:
    """The verdict Auto acts on, from the one the tiers returned (see the module doc)."""
    from consumers.chatbot.auto import Action

    if assessment.action not in (Action.RESPOND, Action.DELEGATE):
        return assessment
    handoff = _handoff(message)
    if handoff:
        target, reasoning = _card_receiver(brain, message, handoff[1]), REASONING_CARD
    elif assessment.action == Action.DELEGATE:
        target, reasoning = _addressed_agent(brain, message), REASONING_NAMED
    else:
        return assessment
    if target is not None:
        logger.info("[PRD-256 D2] %s: ASSIGN to agent %s", reasoning, target[0])
        return _ticket_for(assessment, target, reasoning)
    return answered_by_auto(assessment) if assessment.action == Action.DELEGATE else assessment


def card_directive(card: str, agent_name: str, *, deferred: bool) -> str:
    """The ASSIGN directive for a card already on the board: assign it, never copy it."""
    start = ("Leave it where it is in the queue: the user asked to defer it." if deferred
             else f"Start it: platform_update_task_status \"{card}\" to 'in_progress'.")
    return (
        "\n\n## Manager directive — hand the card on\n"
        f"The user is giving card {card} to the agent '{agent_name}'. The card is already on "
        "the board: do NOT create a new card for it. Do this now:\n"
        f"1. platform_assign_task with task_id \"{card}\" and agent_name \"{agent_name}\".\n"
        f"2. {start}\n"
        f"3. Confirm in ONE line with the card number {card} and who has it now.\n"
    )


def with_card_directive(assessment: Any, message: Optional[str], *, deferred: bool) -> Any:
    """An ASSIGN turn that hands a card on to a resolved agent carries the card's
    directive in place of the new-ticket one; any other turn is returned as it is."""
    card = handed_card(message)
    if not card or assessment.target_agent_id is None or not assessment.target_agent_name:
        return assessment
    return dataclasses.replace(
        assessment, context_directive=card_directive(card, assessment.target_agent_name, deferred=deferred),
    )


def auto_always_answers(assess: Assess) -> Assess:
    """Wrap ``AutoBrain.assess`` (below its cache, its tiers and the shadow): the verdict
    it returns is the one Auto acts on, per ``the_lane``."""
    @functools.wraps(assess)
    async def wrapped(brain: Any, message: str, conversation_length: int = 0) -> Any:
        verdict = await assess(brain, message, conversation_length)
        return the_lane(brain, message, verdict)
    return wrapped


__all__ = [
    "agents_named_in", "answered_by_auto", "auto_always_answers", "card_directive", "handed_card", "handed_to",
    "needs_no_apps", "the_lane", "with_card_directive",
]
