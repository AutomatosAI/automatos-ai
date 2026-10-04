"""Which generation toolkit makes a slot (PRD-251 S1.8, D12; PRD-251B US-B304; PRD-251C US-C302).

The render's footage plan (``footage.plan_for``) asks here for each slot's route:

* :func:`route_for`: the first toolkit, in ``SOCIALS_FOOTAGE_TOOLKITS`` order with the
  workspace's default for that kind first, that is connected and whose recipe makes the kind
  with every action it calls on offer in the media capability registry; else why none does.
* :func:`route_named`: the one toolkit a plan row names for its AI media (a slot's ``via``),
  never another; else why it cannot make the kind here. A plan save checks the same
  (``modules/socials/plan_row_visuals.py``).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

from config import config
from modules.socials.capabilities import MediaCapabilities
from modules.socials.recipes.footage_toolkits import KIND_CAPABILITY, KIND_WORDS, RECIPES, Route


def preferred_toolkits() -> Tuple[str, ...]:
    """The generation toolkits a render tries, in order (``SOCIALS_FOOTAGE_TOOLKITS``)."""
    names = (name.strip().lower() for name in (config.SOCIALS_FOOTAGE_TOOLKITS or "").split(","))
    return tuple(dict.fromkeys(name for name in names if name in RECIPES))


def route_for(kind: str, caps: MediaCapabilities, prefer: Optional[str] = None) -> Union[Route, str]:
    """The first toolkit route that makes ``kind`` here, the workspace's default for it
    first (``prefer``, PRD-251B US-B304); else why none does."""
    if caps.problem:
        return caps.problem
    reasons = []
    order = preferred_toolkits()
    for toolkit in dict.fromkeys((prefer, *order) if prefer in order else order):
        if toolkit not in caps.connected:
            continue
        route, why = RECIPES[toolkit].route(kind, caps)
        if route is not None:
            return route
        reasons.append(why)
    if reasons:
        return "; ".join(reasons)
    connectable = [RECIPES[t].label for t in preferred_toolkits() if t in caps.connectable(KIND_CAPABILITY[kind])]
    hint = f": connect {' or '.join(connectable)} in Composio" if connectable else ""
    return f"no generation toolkit that makes {KIND_WORDS[kind]} is connected{hint}"


def route_named(kind: str, caps: MediaCapabilities, toolkit: str) -> Union[Route, str]:
    """The route of the one toolkit a plan row names for its AI media (PRD-251C US-C302), else
    why it cannot make ``kind`` here: never another toolkit."""
    if caps.problem:
        return caps.problem
    recipe = RECIPES.get(toolkit) if toolkit in preferred_toolkits() else None
    if recipe is None:
        return f"{toolkit} is not a toolkit Socials makes images or footage with"
    if toolkit not in caps.connected:
        return f"{recipe.label} is not connected: connect it in Composio"
    route, why = recipe.route(kind, caps)
    return route if route is not None else str(why)


def route_of(kind: str, caps: MediaCapabilities, via: Any, prefer: Optional[Mapping[str, str]]) -> Union[Route, str]:
    """A slot's route: the toolkit its request names (``via``), else the first that makes ``kind``."""
    return route_named(kind, caps, via) if isinstance(via, str) and via else route_for(kind, caps, (prefer or {}).get(kind))
