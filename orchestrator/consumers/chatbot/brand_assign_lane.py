"""F362 (night 10c): a brand ask is the Brand designer's ticket, decided before any model reads it.

PRD-255 US-014 gave a brand ask a note: file a ticket for the workspace's Brand designer.
Night 10c (6 Oct), four brand asks, and one went to the designer first time. The note is a
prompt, and the turn's lane was the classifier's to choose: "Make the orange an accent only"
was read as chat (the ATOM lane, a narrowed dispatcher), and Auto called the kit's proposal
tool itself three times; "More space between sections" reached a platform setting; "Warmer,
please" became a three-task mission with a researcher.

So AutoBrain pins a brand ask (``brand_to_the_designer.asks_for_brand_work``) to the ASSIGN
lane with the workspace's designer resolved, before its cache and its tiers: the turn takes
the full path, its manager directive files the ticket for the designer by name and starts it,
and the designer note says what goes in it. The designer is read, never seeded here: a
workspace without one keeps the tiers, and the note seeds it or says it was removed. The
onboarding pin still comes first: mid-onboarding, Auto owns every turn.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from consumers.chatbot.brand_to_the_designer import asks_for_brand_work

logger = logging.getLogger(__name__)

ACTIVE = "active"
# AutoBrain counts the conversation with the latest message in it: 1 is a message that opens it.
OPENING_LENGTH = 1
REASONING = "PRD-255 US-014 / F362: brand work is the Brand designer's ticket"
PLATFORM_HINT = "platform"

Assess = Callable[..., Awaitable[Any]]


def active_designer(db: Any, workspace_id: Any) -> Optional[Any]:
    """The workspace's Brand designer while it is active, else None (none, paused, or unreadable)."""
    from core.seeds.seed_brand_designer import find_brand_designer

    try:
        designer = find_brand_designer(db, UUID(str(workspace_id)))
    except Exception:
        logger.exception("[F362] the Brand designer could not be read for workspace %s; the tiers decide", workspace_id)
        return None
    return designer if designer is not None and getattr(designer, "status", None) == ACTIVE else None


def designer_assignment(db: Any, workspace_id: Any, message: str, conversation_length: int) -> Optional[Any]:
    """The ASSIGN-lane assessment for a brand ask, the workspace's designer resolved; None otherwise."""
    if not asks_for_brand_work(message, opening=conversation_length <= OPENING_LENGTH):
        return None
    designer = active_designer(db, workspace_id)
    if designer is None:
        return None
    from consumers.chatbot.auto import Action, Complexity, ComplexityAssessment

    logger.info("[F362] brand ask: ASSIGN to the Brand designer (agent %s)", designer.id)
    return ComplexityAssessment(
        complexity=Complexity.MOLECULE, action=Action.ASSIGN, reasoning=REASONING, confidence=1.0,
        target_agent_id=designer.id, target_agent_name=designer.name, tool_hints=[PLATFORM_HINT],
    )


def brand_work_goes_to_the_designer(assess: Assess) -> Assess:
    """Wrap ``AutoBrain.assess``: a brand ask is the ASSIGN lane for the Brand designer, ahead of the
    cache and the tiers; every other message, and any message mid-onboarding, is assessed as before."""
    @functools.wraps(assess)
    async def wrapped(brain: Any, message: str, conversation_length: int = 0) -> Any:
        from modules.tools.discovery.brand_turns import mark_brand_turn

        pinned = designer_assignment(brain._db, brain._workspace_id, message, conversation_length)
        brand = pinned is not None and not brain._onboarding_active()
        # Gerard, 7 Oct: for the rest of a brand turn no setting is changed and no mission started.
        mark_brand_turn(pinned.target_agent_name if brand else None, brand=brand)
        return pinned if brand else await assess(brain, message, conversation_length)
    return wrapped


__all__ = ["active_designer", "brand_work_goes_to_the_designer", "designer_assignment"]
