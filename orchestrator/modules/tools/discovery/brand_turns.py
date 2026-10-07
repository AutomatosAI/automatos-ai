"""A turn routed to the Brand designer changes no platform setting and starts no mission (Gerard, 7 Oct).

F362 (night 10c) pins a brand ask to the ASSIGN lane for the workspace's Brand designer
(``consumers.chatbot.brand_assign_lane``), and its directive files the ticket. A directive is
still a prompt: on night 10c "More space between sections" reached
platform_update_system_setting and "Warmer, please" became a three-task mission. So the turn
carries a flag, and for the rest of that turn the executor refuses the setting and every action
that starts a mission, with the call that does the work: a ticket for the designer.

The flag is a context variable, as the turn's usage scope is: the lane sets it in the chat
request's context when it pins the turn (and clears it when it does not), and the turn's tool
calls run in that context. Everything else in the turn runs as before.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

# The actions a brand turn may not run: the setting, and every action that starts a mission
# (platform_create_blog_post plans its post as a mission, handlers_blog.create_blog_post_from_topic).
REFUSED_ON_A_BRAND_TURN = frozenset({
    "platform_update_system_setting",
    "platform_create_mission",
    "platform_create_blog_post",
})
GOES_TO_THE_DESIGNER = ("This is a brand change: it goes to the Brand Designer on a ticket "
                        "(platform_create_task assigned to {designer}).")
THE_DESIGNER = "the Brand Designer"

_designer: ContextVar[Optional[str]] = ContextVar("brand_turn_designer", default=None)


def mark_brand_turn(designer_name: Optional[str], *, brand: bool) -> None:
    """Say whether this turn went to the Brand designer (``designer_name``); every assessment sets it."""
    _designer.set((designer_name or THE_DESIGNER) if brand else None)


def brand_turn_designer() -> Optional[str]:
    """The designer this turn went to, or None when it is not a brand turn."""
    return _designer.get()


def refusal_on_a_brand_turn(action_name: str) -> Optional[str]:
    """Why ``action_name`` is refused on this turn, or None. ``follows_the_owner``, the executor's
    check before any gate or handler, asks it first: a brand ask is the owner's words too."""
    designer = brand_turn_designer()
    if designer is None or action_name not in REFUSED_ON_A_BRAND_TURN:
        return None
    return GOES_TO_THE_DESIGNER.format(designer=designer)


__all__ = ["GOES_TO_THE_DESIGNER", "REFUSED_ON_A_BRAND_TURN", "brand_turn_designer", "mark_brand_turn",
           "refusal_on_a_brand_turn"]
