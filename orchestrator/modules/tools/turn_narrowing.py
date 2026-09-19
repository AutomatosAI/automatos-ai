"""Per-turn action ranking, carried from the tool build to the prompt (F025).

The problem this solves is a prompt-cache one. Tools are serialised into the
request BEFORE the system prompt, so anything in the tool block that changes
between turns invalidates the entire cached prefix behind it. The
``platform_execute`` action enum was re-narrowed per query (PRD-138 US-009), so
on night 1 (2026-09-18) the first call of every one of the 77 Auto turns read
back only 2,432-3,456 tokens of a ~34k prefix, while calls 2+ within the same
turn read ~33k. That miss was 55% of Auto's spend on gpt-5.4, and prod runs
Auto on gpt-5.5, where the same miss costs about twice as much.

Narrowing is still worth doing — it is the steer, and the tool-selection-graph
work measures it. It just must not live in the cached prefix. So the enum ships
as the full eligible set (byte-stable between turns) and the ranking is
delivered as a LATE system line, after the stable blocks.

The ranking is computed deep inside the tool build and consumed by the chat
service, which does not call that code directly. A ContextVar carries it across
that gap for the life of one turn — the same pattern
``core/llm/usage_context.py`` uses to carry the spending lane.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# How many ranked actions the late line names. Past a dozen it stops being a
# steer and becomes the enum again, in prose.
MAX_NARROWED_IN_PROMPT = 12

_narrowed_actions: ContextVar[Optional[List[str]]] = ContextVar(
    "automatos_turn_narrowed_actions", default=None,
)


def enum_is_cache_stable() -> bool:
    """Whether the dispatcher enum must stay byte-identical between turns."""
    try:
        from config import config

        return bool(getattr(config, "TOOL_ENUM_CACHE_STABLE", True))
    except Exception:  # noqa: BLE001 — a config fault must not change the surface
        return True


def publish_narrowed_actions(actions: Sequence[str]) -> None:
    """Record what this turn's ranking chose. Never raises."""
    try:
        _narrowed_actions.set([str(a) for a in actions])
    except Exception:  # noqa: BLE001
        logger.debug("turn narrowing: could not publish the ranked actions", exc_info=True)


def clear_narrowed_actions() -> None:
    """Forget the previous turn's ranking, so a turn that does not narrow does
    not inherit the last one's line."""
    try:
        _narrowed_actions.set(None)
    except Exception:  # noqa: BLE001
        pass


def narrowed_actions() -> List[str]:
    """This turn's ranked actions, or ``[]`` when nothing narrowed."""
    return list(_narrowed_actions.get() or [])


def narrowed_actions_prompt_line() -> str:
    """The late system line, or ``""`` when there is nothing to say.

    Goes AFTER the stable blocks: it is the one part of the prompt that is
    meant to change every turn, so it must sit where changing it costs only
    itself.
    """
    if not enum_is_cache_stable():
        return ""      # the enum itself is already narrowed; saying it twice is noise
    actions = narrowed_actions()
    if not actions:
        return ""
    shown = actions[:MAX_NARROWED_IN_PROMPT]
    more = len(actions) - len(shown)
    tail = f" (and {more} more)" if more > 0 else ""
    return (
        "## Most relevant actions for this request\n"
        "Ranked against what was just asked. Any other platform action is still "
        "available if this list does not fit:\n"
        + ", ".join(f"`{name}`" for name in shown)
        + tail
    )
