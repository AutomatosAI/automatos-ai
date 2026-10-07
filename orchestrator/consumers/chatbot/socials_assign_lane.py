"""F379 (night 11, 7 Oct): social media work is the Social Media Director's ticket, decided before any model reads it.

Night 11, the owner asked Auto for "my social media person" (ticket 2155) and "my social media
director" (2157), and both went to NEWSROOM, a newsroom agent that wrote a markdown caption and a
storyboard and made no post. The agent called Social Media Director, the one that makes a real
Socials post, was never picked by Auto (friction 17); the owner had to file 2156 on the board
himself. "Social media person" is no agent's name, the classifier's name match needs one, and the
roster ranker scored NEWSROOM higher.

So AutoBrain pins social media work to the ASSIGN lane for the workspace's Social Media Director,
as the brand lane pins brand work to the Brand designer (``brand_assign_lane``), before its cache
and its tiers: a message that names the owner's social media person, director, manager or team,
or asks for an Instagram, LinkedIn, X, TikTok or Facebook post, a carousel, a reel or a social
video. The Director is the Socials package's own agent in this workspace (its clone of the
marketplace row), or the active agent called Social Media Director; a workspace without one keeps
the tiers. The ticket's description says what done means: a Socials post that exists and has
rendered (``SOCIALS_TICKET``): the turn is marked (``director_turn``), and its last note is that,
never the note that tells Auto how to make the post itself. A question about the work, an ask Auto keeps for itself
("do it yourself"), an ask that names another of the workspace's agents ("get Jim to make a
carousel": the owner chose who, ``names_another_agent``), and any message mid-onboarding are
assessed as before.

The role, said in the classifier's words ("my social media person"), resolves to the Director too
(``social_media_role``), so the ASSIGN lane never asks which agent, nor picks another.
"""
from __future__ import annotations

import functools
import logging
import re
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple
from uuid import UUID

from consumers.chatbot.paperwork_to_the_team import keeps_it_with_auto

logger = logging.getLogger(__name__)

ACTIVE = "active"
DIRECTOR_NAME = "social media director"
REASONING = "F379: social media work is the Social Media Director's ticket"
PLATFORM_HINT = "platform"
# A name shorter than this ("Al", "X") would match ordinary words, so it never counts as named.
MIN_NAMED_CHARS = 3
AUTO_NAME = "auto"  # the owner talks to Auto: its name never hands the work to another agent

_ROLE = (r"social(?:\s+media)?\s+(?:person|director|manager|team|lead|guy|girl|lady|agent|people|expert|"
         r"specialist|marketer|whizz|wiz|wizard|bod|folks?)")
SOCIAL_ROLE = re.compile(r"\b" + _ROLE + r"\b", re.I)
_ADDRESSED = re.compile(r"\b(?:my|our|the)\s+" + _ROLE + r"\b", re.I)
_MAKE = (r"\b(?:make|create|draft|write|do|design|prepare|put together|build|produce|turn|plan|need|want|get|"
         r"mock up|knock up)\b")
_SOCIAL_THING = (r"(?:(?:instagram|insta|ig|linkedin|x|twitter|tiktok|facebook|social(?:\s+media)?)\s+"
                 r"(?:posts?|carousels?|reels?|stor(?:y|ies)|videos?|cards?|captions?|updates?)"
                 r"|carousels?|reels?|tweets?)")
_ASKS_FOR_IT = re.compile(_MAKE + r"[^.?!\n]{0,60}?\b" + _SOCIAL_THING + r"\b", re.I)
# "Have you…?" asks; "Have our social media team…" hands the work over.
_QUESTION = re.compile(r"^\s*(?:what|which|who|whose|when|where|why|how|did|does|is|are|was|were|"
                       r"(?:has|have)\s+(?:you|we|i|they)\b)", re.I)
# The posts already made: listing, reading, approving or removing them is Auto's, with its own tools.
_ABOUT_MADE_ONES = re.compile(r"\b(?:list|show|which|how many|waiting|approv\w*|status|look at|read|check|delete|"
                              r"remove|publish\w*|schedul\w*|already)\b", re.I)
_ADDRESS_TO_AUTO = re.compile(r"^\s*auto\s*[:,\-–—]\s*", re.I)

SOCIALS_TICKET = (
    "The ticket is a Socials post. This is social media work, the {name}'s job. In the ticket's "
    "description, after the owner's words and facts exactly as they gave them, say: make it as a Socials "
    "post with platform_create_social_post, on one of this workspace's social templates (platform_list_"
    "templates with format social_image or social_video), its fields filled from these facts only, render "
    "true; done means the post exists in the Socials tab and has rendered, with its post id in the report; "
    "if a fact or a photo is missing, ask the owner on this card instead of making something up. Don't make "
    "the post yourself in this reply. Tell the owner in one line that the {name} has it, and the card's number."
)

_director: ContextVar[Optional[str]] = ContextVar("socials_director_turn", default=None)


def asks_for_social_work(text: object) -> bool:
    """Whether ``text`` hands social media work over: it names the owner's social media person (or
    director, manager, team), or asks for a channel's post, a carousel, a reel or a social video.
    Not a question about it, not about posts already made (a list, an approval), and not an ask the
    owner keeps with Auto."""
    said = _ADDRESS_TO_AUTO.sub("", str(text or ""))
    if not said.strip() or keeps_it_with_auto(said) or _QUESTION.search(said) or _ABOUT_MADE_ONES.search(said):
        return False
    return bool(_ADDRESSED.search(said) or _ASKS_FOR_IT.search(said))


def find_social_media_director(db: Any, workspace_id: Any) -> Optional[Any]:
    """The workspace's active Social Media Director: the Socials package's clone of the marketplace
    agent, else the active agent of that name."""
    from sqlalchemy import func

    from core.models.core import Agent
    from core.seeds.seed_socials_package import DIRECTOR, MARKETPLACE

    workspace = UUID(str(workspace_id))
    mine = db.query(Agent).filter(Agent.workspace_id == workspace, Agent.status == ACTIVE)
    marketplace = db.query(Agent).filter(Agent.slug == DIRECTOR, Agent.owner_type == MARKETPLACE).first()
    clone = mine.filter(Agent.cloned_from_id == marketplace.id).first() if marketplace is not None else None
    return clone or mine.filter(func.lower(Agent.name) == DIRECTOR_NAME).first()


def active_director(db: Any, workspace_id: Any) -> Optional[Any]:
    """The Director, or None (none, paused, or unreadable: the tiers decide)."""
    try:
        return find_social_media_director(db, workspace_id)
    except Exception:
        logger.exception("[F379] the Social Media Director could not be read for workspace %s; the tiers decide",
                         workspace_id)
        return None


def names_another_agent(message: str, names: Sequence[str], director_name: str) -> bool:
    """Whether ``message`` names one of ``names`` (the workspace's active agents) other than the
    Director, as a whole word: then the owner chose who does the work, and the tiers route it. Pure."""
    said = _ADDRESS_TO_AUTO.sub("", str(message or ""))
    others = {str(name).strip() for name in names if name} - {str(director_name or "").strip()}
    others = {name for name in others if name.casefold() != AUTO_NAME}
    return any(
        len(name) >= MIN_NAMED_CHARS and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", said, re.I)
        for name in others
    )


def agent_names(db: Any, workspace_id: Any) -> Sequence[str]:
    """The names of the workspace's active agents; none when they can't be read (the lane then
    goes by the words alone, as before)."""
    from core.models.core import Agent

    try:
        rows = db.query(Agent.name).filter(Agent.workspace_id == UUID(str(workspace_id)), Agent.status == ACTIVE).all()
    except Exception:
        logger.exception("[F379] the agents of workspace %s could not be read; no other agent counts as named",
                         workspace_id)
        return []
    return [row[0] for row in rows]


def director_assignment(db: Any, workspace_id: Any, message: str) -> Optional[Any]:
    """The ASSIGN-lane assessment for social media work, the workspace's Director resolved; None otherwise
    (also when the message names another agent: the owner chose who)."""
    if not asks_for_social_work(message):
        return None
    director = active_director(db, workspace_id)
    if director is None or names_another_agent(message, agent_names(db, workspace_id), director.name):
        return None
    from consumers.chatbot.auto import Action, Complexity, ComplexityAssessment

    logger.info("[F379] social media work: ASSIGN to the Social Media Director (agent %s)", director.id)
    return ComplexityAssessment(
        complexity=Complexity.MOLECULE, action=Action.ASSIGN, reasoning=REASONING, confidence=1.0,
        target_agent_id=director.id, target_agent_name=director.name, tool_hints=[PLATFORM_HINT],
    )


def director_turn() -> Optional[str]:
    """The Director this turn was given to, or None when the turn is not the Director's ticket."""
    return _director.get()


def ticket_note() -> Optional[str]:
    """What the Director's ticket must say, for the turn's last note (``socials_turn_note``); None
    when the turn is not the Director's."""
    name = director_turn()
    return SOCIALS_TICKET.format(name=name) if name else None


Assess = Callable[..., Awaitable[Any]]


def social_work_goes_to_the_director(assess: Assess) -> Assess:
    """Wrap ``AutoBrain.assess``: social media work is the ASSIGN lane for the Social Media Director,
    ahead of the cache and the tiers; every other message, and any message mid-onboarding, as before."""
    @functools.wraps(assess)
    async def wrapped(brain: Any, message: str, conversation_length: int = 0) -> Any:
        pinned = director_assignment(brain._db, brain._workspace_id, message)
        social = pinned is not None and not brain._onboarding_active()
        _director.set(pinned.target_agent_name if social else None)
        return pinned if social else await assess(brain, message, conversation_length)
    return wrapped


def social_media_role(target: str, agents: Sequence[Any]) -> Tuple[Optional[int], Optional[str]]:
    """``(id, name)`` of the one roster agent a social media role names ("my social media person" is
    the Social Media Director), or ``(None, None)``: the classifier's name match, when no name matched."""
    if not SOCIAL_ROLE.search(target or ""):
        return None, None
    found = {agent.id: agent.name for agent in agents if "social media" in str(getattr(agent, "name", "") or "").lower()}
    if len(found) != 1:
        return None, None
    (agent_id, name), = found.items()
    return agent_id, name


__all__ = ["SOCIALS_TICKET", "SOCIAL_ROLE", "active_director", "agent_names", "asks_for_social_work", "director_assignment",
           "director_turn", "find_social_media_director", "names_another_agent", "ticket_note", "social_media_role",
           "social_work_goes_to_the_director"]
