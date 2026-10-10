"""PRD-256 US-011: one hand-off table: who owns each kind of ask, and what its ticket carries.

Four lanes grew one night at a time, each its own module: brand work went to the Brand designer
(PRD-255 US-014, F362), customer paperwork with no template named to the team (F337(c)), social
media work to the Social Media Director (F379), and each lane had its own words, its own pin on
AutoBrain.assess or its own note. They are one table now (``HANDOFFS``): the kind of ask, the
words that say it, the agent role that owns it, and the note the ticket carries. The next kind
of work is one row here, not a new module.

The table is read once per turn, by the classifier (``hands_off`` on ``AutoBrain.assess``,
ahead of its cache and its tiers): the first row whose role resolves to an active agent pins
the turn to the ASSIGN lane for that agent; mid-onboarding, Auto owns every turn. The reading
is the turn's (a context variable): the turn's last note (``named_template_note.read_note``)
reads it back instead of reading the words again. A turn no classifier read (an agent the owner
chose in the UI) is read there, once.

The owner keeps any of it with Auto by saying so ("do it yourself", "don't bother the team");
a question about the work hands nothing over. Paperwork's role is the agent whose job it is,
found when the ticket is filed (``platform_recommend_agent``), so it pins nothing: its note
files the ticket, and only when the turn names no template (a named template Auto fills, F351).

US-010 (Decision D2): Auto always answers. A classifier DELEGATE verdict used to hand the owner's
chat to the Universal Router, so a specialist answered in its own persona; that lane is gone.
``auto_always_answers`` wraps ``AutoBrain.assess`` below its cache, its tiers and the Jev shadow:
a DELEGATE verdict that hands work to an agent by name ("ask Jim to …") becomes that agent's
ticket (ASSIGN); a name only mentioned ("what did Jim say?") hands nothing over; any other
DELEGATE verdict is Auto's to answer (RESPOND) with its platform tools; a card handed to a named
agent ("Give #0192 to the Support Agent") is the ASSIGN lane on that card, never a copy (F241);
Auto itself is never the agent a ticket goes to. Only the owner's own choice of agent in the UI
(``request.agentId``) puts another agent on the chat.

FX-014 (night 12): the hand-over is read from the roster whatever the tiers said, not only on a
DELEGATE verdict the rubric no longer offers: "Get OPS to …", "Give #1057 to CHRISTMAS BOX" and the
id given after a clash ("267, the operations one") are that agent's ticket; a name several active
agents carry asks which (``addressed_agents``); work naming several teammates stays the tiers' lane.
"""
from __future__ import annotations

import functools
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple
from uuid import UUID

from consumers.chatbot.addressed_agents import (
    ID_REPLY, addressed_by_name, clash_directive, id_reply, joined_with_another, shared_name,
)
from consumers.chatbot.board_questions import CARD_NUMBER

logger = logging.getLogger(__name__)

ACTIVE = "active"
# AutoBrain counts the conversation with the latest message in it: 1 is a message that opens it.
OPENING_LENGTH = 1
PLATFORM_HINT = "platform"
BRAND, SOCIALS, PAPERWORK = "brand work", "social media work", "customer paperwork"

# ── Every row: the owner keeps the work with Auto ("do it yourself", "don't bother the team").
_AUTO_ITSELF = re.compile(r"\byourself\b|\b(?:don['’]?t|do not|no need to)\s+(?:bother|involve|use|ask)\s+the\s+team\b"
                          r"|\bwithout the team\b", re.I)
_QUESTION = re.compile(r"^\s*(?:what|which|who|whose|when|where|why|how|did|does|is|are|was|were|has|have)\b", re.I)

# ── Customer paperwork (F337(c)): what it is called, and the asking that makes one.
PAPER_KINDS = (r"letters?|invoices?|quotes?|quotations?|estimates?|proposals?|agreements?|contracts?|"
               r"flyers?|leaflets?|posters?|brochures?|price ?lists?|welcome sheets?|receipts?|newsletters?|"
               r"one-pagers?|menus?")
_MAKE_PAPER = (r"\b(?:make|write|draft|create|prepare|produce|generate|design|put together|draw up|knock up|"
               r"mock up|do (?:me |us )?(?:a|an|the))\b")
_ASKS_FOR_PAPER = re.compile(_MAKE_PAPER + r"[^.!?\n]{0,80}?\b" + rf"(?P<kind>{PAPER_KINDS})" + r"\b", re.I)

# ── Brand work (PRD-255 US-014, F362): the kit, the look of the documents, the templates.
_MAKE_BRAND = (r"\b(?:design|redesign|create|build|make|improve|refresh|revamp|rework|polish|moderni[sz]e|tidy up|"
               r"sort out|work on|freshen up|upgrade|fix|develop|put together|come up with|"
               r"help (?:me |us )?(?:with|design|build|create|make|improve|sort out))\b")
# Up to three words between the asking and the brand, none of them customer paperwork or a preposition:
# "design a flyer in our brand colours" asks for a flyer, "make something with my template" fills one.
_GAP = rf"(?:(?!(?:{PAPER_KINDS}|with|on|using|in|into|for|from|to|at|about)\b)[\w'’-]+\s+){{0,3}}?"
# The everyday words (kit, colours, fonts, palette) count only as the owner's own ("our colours").
_BRAND = (r"(?:brand(?:ing)?(?:\s+(?:kit|board|identity|guidelines?|look|style|colou?rs|palette|fonts?))?|"
          r"visual identity|look and feel|house style|type ?scale|typography|"
          r"(?:our|my|the)\s+kit|(?:our|my)\s+(?:colou?r\s+)?(?:colou?rs|fonts?|palette)|"
          rf"(?:(?:{PAPER_KINDS})\s+)?templates?)\b")
_ASKS_FOR_BRAND = re.compile(_MAKE_BRAND + r"\s+" + _GAP + _BRAND, re.I)
_COLOURS = r"orange|red|blue|green|yellow|purple|pink|teal|navy|black|grey|gray|gold|brown|beige|colou?rs?|colou?rful"
# "less orange, please": a change to the colours said on its own, not "less red tape".
_COLOUR_TWEAK = re.compile(rf"\b(?:less|more)\s+(?:{_COLOURS}|accent)\b(?=\s*(?:[,.;:!?]|please\b|$))", re.I)
# "more space", "warmer": a change to the look when the message is about the look.
_LOOK_TWEAK = re.compile(rf"\b(?:(?:less|more)\s+(?:space|spacing|white ?space|breathing room|contrast|{_COLOURS}|accent)|"
                         r"warmer|cooler|bolder|brighter|darker|lighter|softer|calmer|friendlier|cleaner|"
                         r"more (?:modern|playful|premium|professional|elegant|minimal))\b", re.I)
_LOOK_WORD = re.compile(r"\b(?:brand(?:ing)?|kit|palette|colou?rs?|fonts?|look|style|templates?|board|layout|"
                        r"pages?|margins?|header|headings?|design|sections?|paragraphs?)\b", re.I)
# F362: a colour given a role in the look ("make the orange an accent only").
_COLOUR_ROLE = re.compile(rf"\b(?:make|use|keep|turn|set|have)\s+(?:the|our|my)\s+(?:{_COLOURS})\s+"
                          r"(?:(?:as|into|to)\s+)?(?:an?\s+|the\s+|our\s+)?(?:accent|highlight|main colou?r|"
                          r"primary|secondary|background)\b", re.I)
# F362: a style word on its own ("Warmer, please"), brand work only when it opens the conversation.
_BARE_TWEAK = re.compile(r"^\W*(?:(?:a\s+)?(?:bit|little|touch|lot)\s+)?(?:" + _LOOK_TWEAK.pattern + r"|"
                         + _COLOUR_TWEAK.pattern + r")[\s,]*(?:please|thanks|thank you)?\W*$", re.I)

# ── Social media work (F379): the owner's social media person, or a channel's post, a carousel, a reel.
_ROLE = (r"social(?:\s+media)?\s+(?:person|director|manager|team|lead|guy|girl|lady|agent|people|expert|"
         r"specialist|marketer|whizz|wiz|wizard|bod|folks?)")
SOCIAL_ROLE = re.compile(r"\b" + _ROLE + r"\b", re.I)
_ADDRESSED = re.compile(r"\b(?:my|our|the)\s+" + _ROLE + r"\b", re.I)
_MAKE_SOCIAL = (r"\b(?:make|create|draft|write|do|design|prepare|put together|build|produce|turn|plan|need|want|get|"
                r"mock up|knock up)\b")
_SOCIAL_THING = (r"(?:(?:instagram|insta|ig|linkedin|x|twitter|tiktok|facebook|social(?:\s+media)?)\s+"
                 r"(?:posts?|carousels?|reels?|stor(?:y|ies)|videos?|cards?|captions?|updates?)"
                 r"|carousels?|reels?|tweets?)")
_ASKS_FOR_SOCIAL = re.compile(_MAKE_SOCIAL + r"[^.?!\n]{0,60}?\b" + _SOCIAL_THING + r"\b", re.I)
# "Have you…?" asks; "Have our social media team…" hands the work over.
_SOCIAL_QUESTION = re.compile(r"^\s*(?:what|which|who|whose|when|where|why|how|did|does|is|are|was|were|"
                              r"(?:has|have)\s+(?:you|we|i|they)\b)", re.I)
# The posts already made: listing, approving or removing them is Auto's, with its own tools.
_ABOUT_MADE_ONES = re.compile(r"\b(?:list|show|which|how many|waiting|approv\w*|status|look at|read|check|delete|"
                              r"remove|publish\w*|schedul\w*|already)\b", re.I)
_ADDRESS_TO_AUTO = re.compile(r"^\s*auto\s*[:,\-–—]\s*", re.I)
DIRECTOR_NAME = "social media director"
# A name shorter than this ("Al", "X") would match ordinary words, so it never counts as named.
MIN_NAMED_CHARS = 3
AUTO_NAME = "auto"  # the owner talks to Auto: its name never hands the work to another agent


def keeps_it_with_auto(text: object) -> bool:
    """The owner keeps the work with Auto: "do it yourself", "don't bother the team"."""
    return bool(_AUTO_ITSELF.search(str(text or "")))


def asks_for_paperwork(text: object) -> Optional[str]:
    """The kind of customer paperwork ``text`` asks to be made ("price list"), or None: a question
    about one, a message that keeps it with Auto, or nothing of the kind."""
    said = str(text or "")
    if keeps_it_with_auto(said):
        return None
    found = _ASKS_FOR_PAPER.search(said)
    return " ".join(found.group("kind").lower().split()) if found else None


def asks_for_brand_work(text: object, opening: bool = False) -> bool:
    """Whether ``text`` asks for the brand, the kit or the templates to be designed or changed in look:
    not a question about them, and not one the owner keeps with Auto. ``opening``: ``text`` opens the
    conversation, so a style word on its own ("Warmer, please") can only mean the look (F362)."""
    said = str(text or "")
    if not said.strip() or keeps_it_with_auto(said) or _QUESTION.search(said):
        return False
    if _ASKS_FOR_BRAND.search(said) or _COLOUR_TWEAK.search(said) or _COLOUR_ROLE.search(said):
        return True
    if opening and _BARE_TWEAK.search(said):
        return True
    return bool(_LOOK_TWEAK.search(said) and _LOOK_WORD.search(said))


def asks_for_social_work(text: object) -> bool:
    """Whether ``text`` hands social media work over: it names the owner's social media person (or
    director, manager, team), or asks for a channel's post, a carousel, a reel or a social video. Not a
    question about it, not about posts already made, and not an ask the owner keeps with Auto."""
    said = _ADDRESS_TO_AUTO.sub("", str(text or ""))
    if not said.strip() or keeps_it_with_auto(said) or _SOCIAL_QUESTION.search(said) or _ABOUT_MADE_ONES.search(said):
        return False
    return bool(_ADDRESSED.search(said) or _ASKS_FOR_SOCIAL.search(said))


# ── The notes the tickets carry ───────────────────────────────────────────────

DESIGNER_NOTE = (
    "The owner asked for brand work: their brand kit, the look of their documents, or their document templates. "
    "That is the {name}'s job (the owner's choice: Auto delegates, agents do the work), so don't design it yourself "
    "in this reply, and don't call platform_update_brand_kit, platform_propose_brand_kit or the template tools. File "
    "it for the {name}: platform_create_task with assigned_agent_name \"{name}\" and a self-contained description, "
    "written as the dispatch contract below, that carries the owner's words exactly (every colour, name and change "
    "they asked for) and invents none. The card's flow, for the description: read the logo; propose the kit on a "
    "card the owner approves, with the Brand Board drawn from the proposal and not yet saved; save it only after "
    "the owner approves; make the sample set (an invoice, a letter, a proposal and three social cards) as "
    "Deliverables; report back with the board and the set. Start it (platform_update_task_status to "
    "'in_progress'). Tell the owner in one line that the {name} has it, its card number, and that its proposal "
    "will come to them as a card to approve or send back. A style ask (warmer, more space, less orange) is that "
    "one ticket: never a mission (platform_create_mission) and never a platform setting "
    "(platform_update_system_setting)."
)
REMOVED_NOTE = (
    "The owner asked for brand work, and this workspace has no Brand designer: the owner removed it. Don't change "
    "the brand kit or the templates yourself. Tell the owner in one line that brand work is the Brand designer's, "
    "and that installing the Socials package from the marketplace adds it back."
)
UNAVAILABLE_NOTE = (
    "The owner asked for brand work, and the Brand designer could not be added to their team just now. Don't "
    "change the brand kit or the templates yourself: tell the owner it could not be filed and to ask again shortly."
)
SOCIALS_TICKET = (
    "The ticket is a Socials post. This is social media work, the {name}'s job. In the ticket's "
    "description, after the owner's words and facts exactly as they gave them, say: make it as a Socials "
    "post with platform_create_social_post, on one of this workspace's social templates (platform_list_"
    "templates with format social_image or social_video), its fields filled from these facts only, render "
    "true; done means the post exists in the Socials tab and has rendered, with its post id in the report; "
    "if a fact or a photo is missing, ask the owner on this card instead of making something up. Don't make "
    "the post yourself in this reply. Tell the owner in one line that the {name} has it, and the card's number."
)
TEAM_NOTE = (
    "The owner asked for {article} {kind} for their customers and named none of their templates. Customer paperwork "
    "goes to the team (the owner's choice): don't write it yourself in this reply. Call platform_recommend_agent "
    "with the request (platform_list_agents is the plain roster), then file it for the agent whose job it is: "
    "platform_create_task with assigned_agent_name and a self-contained description, written as the dispatch "
    "contract below, that carries every name, address, figure, date and term exactly as the owner gave them and "
    "invents none. Start it (platform_update_task_status to 'in_progress'). Tell the owner in one line who has it "
    "and its card number, and offer to make it yourself now on one of their templates if they'd rather."
)
VOWELS = "aeiou"


def _with_the_contract(note: str) -> str:
    """A note that files a ticket, with the ASSIGN lane's dispatch contract and board rule after it."""
    from consumers.chatbot.auto import BOARD_MOVES_THE_CARD
    from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT

    return f"{note}\n\n{DISPATCH_CONTRACT_FRAGMENT}\n{BOARD_MOVES_THE_CARD}"


# ── The roles: the active agent that owns each kind, for the pin ──────────────

def active_designer(db: Any, workspace_id: Any, message: str = "") -> Optional[Any]:
    """The workspace's Brand designer while it is active, else None (none, paused, or unreadable)."""
    from core.seeds.seed_brand_designer import find_brand_designer

    try:
        designer = find_brand_designer(db, UUID(str(workspace_id)))
    except Exception:
        logger.exception("[F362] the Brand designer could not be read for workspace %s; the tiers decide", workspace_id)
        return None
    return designer if designer is not None and getattr(designer, "status", None) == ACTIVE else None


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


def agent_names(db: Any, workspace_id: Any) -> Sequence[str]:
    """The names of the workspace's active agents; none when they can't be read (the words alone decide)."""
    from core.models.core import Agent

    try:
        rows = db.query(Agent.name).filter(Agent.workspace_id == UUID(str(workspace_id)), Agent.status == ACTIVE).all()
    except Exception:
        logger.exception("[F379] the agents of workspace %s could not be read; no other agent counts as named",
                         workspace_id)
        return []
    return [row[0] for row in rows]


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


def active_director(db: Any, workspace_id: Any, message: str = "") -> Optional[Any]:
    """The Social Media Director, or None: none, unreadable, or ``message`` names another agent (the owner chose)."""
    try:
        director = find_social_media_director(db, workspace_id)
    except Exception:
        logger.exception("[F379] the Social Media Director could not be read for workspace %s; the tiers decide",
                         workspace_id)
        return None
    if director is None or names_another_agent(message, agent_names(db, workspace_id), director.name):
        return None
    return director


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


# ── The table ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HandOff:
    """One row: a kind of ask, the words that say it, the role that owns it, the note its ticket carries.

    ``owner`` finds the role's active agent, for the ASSIGN pin; None pins nothing (the note files the
    ticket). ``unless_a_template_is_named``: the note is given only when the turn names no template.
    ``no_setting_or_mission``: a pinned turn changes no platform setting and starts no mission."""

    kind: str
    role: str
    asked: Callable[[str, bool], Optional[str]]
    owner: Optional[Callable[[Any, Any, str], Optional[Any]]]
    note: Callable[[Any, UUID, "Turn"], Optional[str]]
    unless_a_template_is_named: bool = False
    no_setting_or_mission: bool = False


@dataclass(frozen=True)
class Turn:
    """The table read for one turn: every row the latest message matches (in the table's order) with
    what it asked for, and the row and agent the turn was pinned to (ASSIGN), if any."""

    matches: Tuple[Tuple[HandOff, str], ...] = ()
    pinned: Optional[HandOff] = None
    agent: Optional[str] = None

    def agent_for(self, kind: str) -> Optional[str]:
        """The agent this turn was pinned to when the pin was ``kind``'s row, else None."""
        return self.agent if self.pinned is not None and self.pinned.kind == kind else None


def _designer_note(db: Any, workspace_id: UUID, turn: Turn) -> str:
    """The designer's ticket: named by the pin, else found or seeded now; or why it can't be filed."""
    from consumers.chatbot import brand_to_the_designer

    try:
        name = turn.agent_for(BRAND) or brand_to_the_designer.designer_name(db, workspace_id)
    except Exception:
        logger.exception("[PRD-255] the Brand designer could not be found or seeded for workspace %s", workspace_id)
        return UNAVAILABLE_NOTE
    return _with_the_contract(DESIGNER_NOTE.format(name=name)) if name else REMOVED_NOTE


def _director_ticket(db: Any, workspace_id: UUID, turn: Turn) -> Optional[str]:
    """What the Director's ticket must say, on a turn pinned to the Director only."""
    name = turn.agent_for(SOCIALS)
    return SOCIALS_TICKET.format(name=name) if name else None


def _team_note(db: Any, workspace_id: UUID, turn: Turn) -> Optional[str]:
    """The hand-to-the-team note for the paperwork the turn asked for."""
    kind = next((asked for row, asked in turn.matches if row.kind == PAPERWORK), None)
    if not kind:
        return None
    article = "an" if kind[:1] in VOWELS else "a"
    return _with_the_contract(TEAM_NOTE.format(article=article, kind=kind))


HANDOFFS: Tuple[HandOff, ...] = (
    HandOff(BRAND, "the Brand designer", lambda text, opening: BRAND if asks_for_brand_work(text, opening) else None,
            active_designer, _designer_note, no_setting_or_mission=True),
    HandOff(SOCIALS, "the Social Media Director", lambda text, opening: SOCIALS if asks_for_social_work(text) else None,
            active_director, _director_ticket),
    HandOff(PAPERWORK, "the agent whose job it is (platform_recommend_agent, when the ticket is filed)",
            lambda text, opening: asks_for_paperwork(text), None, _team_note, unless_a_template_is_named=True),
)

_turn: ContextVar[Optional[Turn]] = ContextVar("handoffs_turn", default=None)


def read_the_table(message: object, opening: bool = False) -> Tuple[Tuple[HandOff, str], ...]:
    """Every row whose words ``message`` matches, in the table's order, with what it asked for. Pure.
    ``opening``: the message opens the conversation."""
    said = str(message or "")
    matches = ((row, row.asked(said, opening)) for row in HANDOFFS)
    return tuple((row, asked) for row, asked in matches if asked)


def the_turn(texts: Sequence[str]) -> Turn:
    """This turn's reading: the classifier's, or, when no classifier read the turn (an agent the owner
    chose), the table read now from the owner's latest message (``texts[0]``)."""
    turn = _turn.get()
    if turn is not None:
        return turn
    return Turn(read_the_table(texts[0] if texts else "", opening=len(texts) == 1))


def turn_note(db: Any, workspace_id: UUID, turn: Turn, *, no_template_named: bool) -> Optional[str]:
    """The first note the turn's rows give, in the table's order: the rows that hold whatever the turn
    names, or (``no_template_named``) the rows given only when it names no template."""
    for row, _asked in turn.matches:
        if row.unless_a_template_is_named == no_template_named:
            note = row.note(db, workspace_id, turn)
            if note:
                return note
    return None


# ── The pin on AutoBrain.assess ───────────────────────────────────────────────

Assess = Callable[..., Awaitable[Any]]


def _pin(db: Any, workspace_id: Any, message: str,
         matches: Sequence[Tuple[HandOff, str]]) -> Tuple[Optional[HandOff], Optional[Any]]:
    """The first matching row whose role resolves to an active agent, with the agent; else (None, None)."""
    for row, _asked in matches:
        agent = row.owner(db, workspace_id, message) if row.owner is not None else None
        if agent is not None:
            return row, agent
    return None, None


def _assignment(row: HandOff, agent: Any) -> Any:
    """The ASSIGN-lane assessment for the row's agent."""
    from consumers.chatbot.auto import Action, Complexity, ComplexityAssessment

    logger.info("[PRD-256 US-011] %s: ASSIGN to %s (agent %s)", row.kind, row.role, agent.id)
    return ComplexityAssessment(
        complexity=Complexity.MOLECULE, action=Action.ASSIGN, confidence=1.0,
        reasoning=f"PRD-256 hand-off table: {row.kind} is {row.role}'s ticket",
        target_agent_id=agent.id, target_agent_name=agent.name, tool_hints=[PLATFORM_HINT],
    )


def hands_off(assess: Assess) -> Assess:
    """Wrap ``AutoBrain.assess``: read the table once for the turn; the first matching row whose role
    resolves pins the ASSIGN lane for that agent, ahead of the cache and the tiers. Mid-onboarding,
    and for every other message, the tiers decide."""
    @functools.wraps(assess)
    async def wrapped(brain: Any, message: str, conversation_length: int = 0) -> Any:
        from modules.tools.discovery.brand_turns import mark_brand_turn

        matches = read_the_table(message, opening=conversation_length <= OPENING_LENGTH)
        row, agent = _pin(brain._db, brain._workspace_id, message, matches)
        if row is not None and brain._onboarding_active():
            row, agent = None, None
        _turn.set(Turn(matches, row, agent.name if agent is not None else None))
        guarded = row is not None and row.no_setting_or_mission
        # Gerard, 7 Oct: for the rest of a brand turn no setting is changed and no mission started.
        mark_brand_turn(agent.name if guarded else None, brand=guarded)
        return _assignment(row, agent) if row is not None else await assess(brain, message, conversation_length)
    return wrapped


# ── Auto answers; a named agent gets a ticket (US-010, Decision D2) ──────────────────

# A name shorter than this never counts as named: a stray "AI" or "Bo" is not an agent.
MIN_NAME_CHARS = 3
REASONING_ANSWERS = "PRD-256 D2: Auto answers; no specialist takes over the chat"
REASONING_NAMED = "PRD-256 D2: the named agent gets a ticket"
REASONING_CARD = "PRD-256 D2 / F241: the named card goes to the named agent"
REASONING_ID = "PRD-256 FX-014: the agent the owner picked by its id gets the ticket"
REASONING_ASK = "PRD-256 FX-014: several active agents carry the name, so Auto asks which"
# "give #0192 to …", "assign ticket 12 to …", "hand #0192 over to …": the card sits between the
# verb and the first "to" after it. "Move #0192 to review" is a status, so "move" is no verb.
HANDS_ON = re.compile(r"\b(?:give|assign|reassign|hand)\b(?P<what>[^.?!\n]*?)\bto\b", re.IGNORECASE)
# A card handed on by its number past #0999 too ("give #1057 to …"); an order's number never is one.
HANDED_CARD = re.compile(CARD_NUMBER.pattern + r"|(?<![\w&#])(?<!order )#\d{3,6}(?:\.\d{1,3})?\b", re.IGNORECASE)
# Work handed to someone: a cheap check before the roster is read.
ADDRESS_VERB = re.compile(r"\b(?:ask|have|get|tell|let)\s", re.IGNORECASE)
# The reference platform_assign_task takes: "#0192" as written, "ticket 12" as "12".
CARD_REF = re.compile(r"#?\d+(?:\.\d+)?")
CARD_SIGN = "#"
# Who the card goes to: the words after "to", without the article, and ending where a
# purpose, a reason or a courtesy starts ("to Jim to handle by Friday, please" is Jim).
SENTENCE_END = re.compile(r"[.?!,;\n]")
LEADING_ARTICLE = re.compile(r"^(?:the|my|our)\s+", re.IGNORECASE)
RECEIVER_END = re.compile(r"\s+(?:to|for|by|so|and|because|please|now|thanks|today|asap)\b.*$", re.IGNORECASE)
NOT_A_RECEIVER = frozenset({"him", "her", "them", "you", "myself", "yourself", "review", "done"})


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
        card = HANDED_CARD.search(hands_on.group("what"))
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
    return replace(
        assessment, action=Action.RESPOND, tool_hints=hints,
        reasoning=f"{assessment.reasoning} ({REASONING_ANSWERS})",
    )


def _ticket_for(assessment: Any, target: Tuple[int, str], reasoning: str) -> Any:
    from consumers.chatbot.auto import Action

    agent_id, agent_name = target
    return replace(
        assessment, action=Action.ASSIGN, target_agent_id=agent_id, target_agent_name=agent_name,
        reasoning=f"{assessment.reasoning} ({reasoning})",
    )


def _teammates(brain: Any) -> List[Any]:
    """The active roster without the system agent: Auto never hands work to itself."""
    return [agent for agent in brain._active_agents() if not getattr(agent, "is_system_agent", False)]


def _card_receiver(brain: Any, message: str, who: str, roster: List[Any]) -> Optional[Any]:
    """The one teammate a card is handed to, as (id, name); the agents a shared name could be (FX-014);
    or None. Named in part ("the Support Agent"), AutoBrain's roster match resolves it; an id the
    owner gives ("267") is that agent; a pronoun or an unknown name is nobody."""
    if who.lower() in NOT_A_RECEIVER:
        return None
    by_id = id_reply(who, roster)
    if by_id is not None:
        return by_id.id, by_id.name
    if len(who) < MIN_NAME_CHARS:
        return None
    named = agents_named_in(who, roster)
    if len(named) == 1:
        return named[0].id, named[0].name
    if shared_name(named):
        return named
    agent_id, agent_name = brain._match_roster_agent(who, roster, message=message)
    return (agent_id, agent_name) if agent_id is not None else None


def _addressed_agent(message: str, roster: List[Any]) -> Optional[Any]:
    """The one teammate the message hands work to by name ("ask Jim to …"), as (id, name); the
    agents a shared name could be ("Get OPS to …" with two OPS); or None. A name that is only
    mentioned ("what did Jim say?", "Get sales figures") hands nothing over, and neither does work
    handed to it with another teammate ("Have RESEARCHER and WRITER plan the launch")."""
    addressed = [agent for agent in agents_named_in(message, roster) if addressed_by_name(message, agent.name)]
    if not addressed or joined_with_another(message, addressed[0].name, roster):
        return None
    if len(addressed) == 1:
        return addressed[0].id, addressed[0].name
    return addressed if shared_name(addressed) else None


def _on_the_board(brain: Any, card: str) -> bool:
    """Whether a card handed on by its '#' number is one of this workspace's cards (P256-FIX-RVW-42): "Hand
    #1043 over to Support - the customer wants a refund" names an order, not a card. "ticket 12" says so."""
    from services.ticket_numbers import resolve_ticket_ref

    return not card.startswith(CARD_SIGN) or resolve_ticket_ref(brain._db, brain._workspace_id, card) is not None


def _handed_over(brain: Any, message: str) -> Tuple[Optional[Any], str]:
    """(who the message hands work to, why): a card handed on, an id given as the answer to "which
    one?", or a teammate asked by name. A message about a card that hands it to no one (approving it,
    moving it) stays Auto's, and so does a '#' number that is no card of the workspace's (an order's),
    one with nobody in it, or one handing work to several teammates together ("Have RESEARCHER and
    WRITER plan the launch"): the tiers' lane, whatever they said."""
    handoff = _handoff(message)
    if handoff and _on_the_board(brain, handoff[0]):
        return _card_receiver(brain, message, handoff[1], _teammates(brain)), REASONING_CARD
    if HANDED_CARD.search(message) or not (ADDRESS_VERB.search(message) or ID_REPLY.match(message)):
        return None, REASONING_NAMED
    roster = _teammates(brain)
    by_id = id_reply(message, roster)
    if by_id is not None:
        return (by_id.id, by_id.name), REASONING_ID
    return _addressed_agent(message, roster), REASONING_NAMED


def _ask_which(assessment: Any, agents: List[Any], message: str) -> Any:
    """ASSIGN with no agent picked: the directive lists the namesakes and asks which (FX-012's list)."""
    from consumers.chatbot.auto import Action

    logger.info("[PRD-256 FX-014] %d active agents are called %r: Auto asks which", len(agents), agents[0].name)
    return replace(
        assessment, action=Action.ASSIGN, target_agent_id=None, target_agent_name=shared_name(agents),
        context_directive=clash_directive(agents, handed_card(message)),
        reasoning=f"{assessment.reasoning} ({REASONING_ASK})",
    )


def the_lane(brain: Any, message: str, assessment: Any) -> Any:
    """The verdict Auto acts on, from the one the tiers returned (see the module doc). FX-014: work
    handed to a teammate by name or id is that teammate's ticket whatever the tiers said."""
    from consumers.chatbot.auto import Action

    target, reasoning = _handed_over(brain, str(message or ""))
    if isinstance(target, list):
        return _ask_which(assessment, target, message)
    if target is not None:
        logger.info("[PRD-256 D2] %s: ASSIGN to agent %s", reasoning, target[0])
        return _ticket_for(assessment, target, reasoning)
    return answered_by_auto(assessment) if assessment.action == Action.DELEGATE else assessment


def card_directive(card: str, agent_name: str, *, deferred: bool, agent_id: Optional[int] = None) -> str:
    """The ASSIGN directive for a card already on the board: assign it, never copy it."""
    start = ("Leave it where it is in the queue: the user asked to defer it." if deferred
             else f"Start it: platform_update_task_status \"{card}\" to 'in_progress'.")
    return (
        "\n\n## Manager directive — hand the card on\n"
        f"The user is giving card {card} to the agent '{agent_name}'. The card is already on "
        "the board: do NOT create a new card for it. Do this now:\n"
        f"1. platform_assign_task with task_id \"{card}\" and agent_name \"{agent_name}\""
        f"{f' and agent_id {agent_id}' if agent_id else ''}.\n"
        f"2. {start}\n"
        f"3. Confirm in ONE line with the card number {card} and who has it now.\n"
    )


def with_card_directive(assessment: Any, message: Optional[str], *, deferred: bool, card: Optional[str] = None) -> Any:
    """An ASSIGN turn that hands a card on to a resolved agent carries the card's
    directive in place of the new-ticket one; any other turn is returned as it is.
    ``card``: the card an earlier message handed on, for an answer that names none (RVW-33)."""
    card = handed_card(message) or card
    if not card or assessment.target_agent_id is None or not assessment.target_agent_name:
        return assessment
    return replace(
        assessment, context_directive=card_directive(card, assessment.target_agent_name, deferred=deferred,
                                                     agent_id=assessment.target_agent_id),
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
    "BRAND", "DESIGNER_NOTE", "HANDOFFS", "HandOff", "PAPERWORK", "PAPER_KINDS", "REMOVED_NOTE", "SOCIALS",
    "SOCIALS_TICKET", "SOCIAL_ROLE", "TEAM_NOTE", "Turn", "UNAVAILABLE_NOTE", "active_designer",
    "active_director", "agent_names", "agents_named_in", "answered_by_auto", "asks_for_brand_work",
    "asks_for_paperwork", "asks_for_social_work", "auto_always_answers", "card_directive",
    "find_social_media_director", "handed_card", "handed_to", "hands_off", "keeps_it_with_auto",
    "names_another_agent", "needs_no_apps", "read_the_table", "social_media_role", "the_lane", "the_turn",
    "turn_note", "with_card_directive",
]
