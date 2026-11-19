"""The actions a prompt names stay reachable, whatever the ranking says (19 Nov 2026).

The ``platform_execute`` dispatcher offers a turn only its top-ranked actions (PRD-138
US-009). A Playbook step's prompt that says "Read the plan with platform_get_social_plan"
still lost that action to the ranking: the Content bank research run of 18 Nov had none of
``platform_get_social_plan``, ``platform_add_social_topics`` and ``platform_list_deliverables``,
added no topics and ended "completed", and a week of posts was made from titles with no facts.

:func:`named_in` reads the action names a text spells out; :func:`with_named_first` puts the
ones that clear the caller's role gate at the head of a narrowed allow-list, so the enum's cap
never cuts them. A full enum (``None``) already offers every action and is left as it is.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

Narrowing = Tuple[Optional[List[str]], Optional[str], bool]

NAMED_REASON = "named_in_prompt"
# A platform action's name as a prompt spells it: platform_ and lowercase words joined by underscores.
_ACTION_NAME = re.compile(r"\bplatform_[a-z0-9]+(?:_[a-z0-9]+)*\b")
# The most names one prompt may pin: a prompt is not a way round the enum's cap.
MAX_NAMED = 12


def named_in(text: Optional[str]) -> List[str]:
    """The distinct ``platform_*`` names ``text`` spells out, in the order they first appear."""
    if not text:
        return []
    seen: List[str] = []
    for name in _ACTION_NAME.findall(text):
        if name not in seen:
            seen.append(name)
    return seen[:MAX_NAMED]


def with_named_first(
    narrowing: Narrowing, query: Optional[str], passes_gate: Callable[[str], bool], cap: int,
) -> Narrowing:
    """``narrowing`` with the actions ``query`` names, that ``passes_gate`` clears (a registered
    action the principal may call), first, then the ranked ones, capped at ``cap``. Unchanged
    for a full enum or a query that names none."""
    allowed, reason, from_pins = narrowing
    if allowed is None:
        return narrowing
    named = [name for name in named_in(query) if passes_gate(name)]
    if not named:
        return narrowing
    merged = _ordered([*named, *allowed])
    if cap > 0:
        merged = merged[: max(cap, len(named))]
    return merged, (reason or NAMED_REASON), from_pins


def _ordered(names: Sequence[str]) -> List[str]:
    """``names`` once each, in their first order."""
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


__all__ = ["MAX_NAMED", "NAMED_REASON", "named_in", "with_named_first"]
