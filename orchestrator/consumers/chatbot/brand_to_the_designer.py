"""PRD-255 US-014: a brand ask goes to the Brand designer (thesis: Auto delegates).

The owner, 4 Oct: "Automatos becomes your brand; Auto is the one voice and delegates,
agents do the work." So when the owner asks Auto to design or improve the brand, the kit
or the document templates ("help me design my brand", "improve the kit", "make our
templates", "less orange"), the turn gets a note, as F337(c)'s paperwork note does:
file a ticket for the workspace's Brand designer with the owner's words and start it.
Auto never designs it itself and never changes the kit on such an ask: the designer
proposes on a card the owner approves (FR-11).

Only the owner's latest message is read. A question about the brand ("what colour is
our accent?") asks nothing to be made, and "do it yourself" keeps it with Auto. A
workspace with no designer yet gets its one (``seed_brand_designer``, find-or-seed; an
existing hosted workspace whose Auto was seeded before PRD-255 gets it here). One the
owner removed is not brought back: the note says so instead.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional
from uuid import UUID

from consumers.chatbot.paperwork_to_the_team import PAPER_KINDS, keeps_it_with_auto

logger = logging.getLogger(__name__)

_MAKE = (r"\b(?:design|redesign|create|build|make|improve|refresh|revamp|rework|polish|moderni[sz]e|tidy up|"
         r"sort out|work on|freshen up|upgrade|fix|develop|put together|come up with|"
         r"help (?:me |us )?(?:with|design|build|create|make|improve|sort out))\b")
# Up to three words between the asking and the brand, none of them customer paperwork or a
# preposition: "design a flyer in our brand colours" asks for a flyer (F337(c)), and "make
# something with my template" fills one (F351); neither asks for the brand.
_GAP = rf"(?:(?!(?:{PAPER_KINDS}|with|on|using|in|into|for|from|to|at|about)\b)[\w'’-]+\s+){{0,3}}?"
# What the brand is. The everyday words (kit, colours, fonts, palette) count only as the
# owner's own ("our colours", "the kit"): "fix the font size" and "a colour palette for my
# garden" are not brand work.
_BRAND = (r"(?:brand(?:ing)?(?:\s+(?:kit|board|identity|guidelines?|look|style|colou?rs|palette|fonts?))?|"
          r"visual identity|look and feel|house style|type ?scale|typography|"
          r"(?:our|my|the)\s+kit|(?:our|my)\s+(?:colou?r\s+)?(?:colou?rs|fonts?|palette)|"
          rf"(?:(?:{PAPER_KINDS})\s+)?templates?)\b")
_ASKS_FOR_BRAND = re.compile(_MAKE + r"\s+" + _GAP + _BRAND, re.I)
_COLOURS = r"orange|red|blue|green|yellow|purple|pink|teal|navy|black|grey|gray|gold|brown|beige|colou?rs?|colou?rful"
# "less orange": a change to the colours, said on its own ("less orange, please"), not
# "less red tape" or "more gold stock".
_COLOUR_TWEAK = re.compile(rf"\b(?:less|more)\s+(?:{_COLOURS}|accent)\b(?=\s*(?:[,.;:!?]|please\b|$))", re.I)
# "more space", "warmer", "more orange in the header": a change to the look when the
# message is about the look ("more space in the calendar" is not).
_LOOK_TWEAK = re.compile(rf"\b(?:(?:less|more)\s+(?:space|spacing|white ?space|breathing room|contrast|{_COLOURS}|accent)|"
                         r"warmer|cooler|bolder|brighter|darker|lighter|softer|calmer|friendlier|cleaner|"
                         r"more (?:modern|playful|premium|professional|elegant|minimal))\b", re.I)
_LOOK_WORD = re.compile(r"\b(?:brand(?:ing)?|kit|palette|colou?rs?|fonts?|look|style|templates?|board|layout|"
                        r"pages?|margins?|header|headings?|design)\b", re.I)
_QUESTION = re.compile(r"^\s*(?:what|which|who|whose|when|where|why|how|did|does|is|are|was|were|has|have)\b", re.I)

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
    "will come to them as a card to approve or send back."
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


def asks_for_brand_work(text: object) -> bool:
    """Whether ``text`` asks for the brand, the kit or the templates to be designed or changed in look:
    not a question about them, and not one the owner keeps with Auto."""
    said = str(text or "")
    if not said.strip() or keeps_it_with_auto(said) or _QUESTION.search(said):
        return False
    if _ASKS_FOR_BRAND.search(said) or _COLOUR_TWEAK.search(said):
        return True
    return bool(_LOOK_TWEAK.search(said) and _LOOK_WORD.search(said))


def _seed_in_own_session(workspace_id: UUID) -> Optional[str]:
    """Seed the workspace's designer in a session of its own, committed before the turn files the
    ticket; its name, or None when the owner removed it."""
    from core.database.database import SessionLocal
    from core.seeds.seed_brand_designer import seed_brand_designer

    db = SessionLocal()
    try:
        agent = seed_brand_designer(db, workspace_id)
        db.commit()
        return getattr(agent, "name", None)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


Seeder = Callable[[UUID], Optional[str]]


def designer_name(db: Any, workspace_id: UUID, seed: Seeder = _seed_in_own_session) -> Optional[str]:
    """The workspace's Brand designer's name: found, or seeded now. None when the owner removed it."""
    from core.seeds.seed_brand_designer import find_brand_designer

    found = find_brand_designer(db, workspace_id)
    return found.name if found is not None else seed(workspace_id)


def designer_note(db: Any, workspace_id: UUID, text: object, seed: Seeder = _seed_in_own_session) -> Optional[str]:
    """The hand-to-the-designer note for the owner's latest message, or None when it asks for no brand work.
    The ASSIGN lane's dispatch contract and board rule go with it."""
    if not asks_for_brand_work(text):
        return None
    try:
        name = designer_name(db, workspace_id, seed)
    except Exception:
        logger.exception("[PRD-255] the Brand designer could not be found or seeded for workspace %s", workspace_id)
        return UNAVAILABLE_NOTE
    if name is None:
        return REMOVED_NOTE
    from consumers.chatbot.auto import BOARD_MOVES_THE_CARD
    from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT

    return f"{DESIGNER_NOTE.format(name=name)}\n\n{DISPATCH_CONTRACT_FRAGMENT}\n{BOARD_MOVES_THE_CARD}"


__all__ = ["DESIGNER_NOTE", "REMOVED_NOTE", "UNAVAILABLE_NOTE", "asks_for_brand_work", "designer_name",
           "designer_note"]
