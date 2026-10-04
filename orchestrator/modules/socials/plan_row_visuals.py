"""PRD-251C (C6, US-C302): a cadence row's own visual, checked against what the workspace has.

A row may set ``visual`` (``plans._row_visual``): the source its posts' visuals come from,
over the plan's mix, and for AI images or footage the toolkit that makes them. That toolkit
must make the row's media here now, as a render would route it
(``recipes.footage_routes.route_named``): connected in Composio, with its actions on the Socials
media allowlist and not deny-listed. A plan save naming one it cannot use is refused (422)
and nothing is saved. The spend stays as for every AI shot (D13): the render prices, caps
and books it; a row's choice changes which toolkit makes the shot, never the caps.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Tuple

from core.social_templates import IMAGE_SLOT, VIDEO_SLOT
from modules.socials import plans
from modules.socials.capabilities import media_capabilities
from modules.socials.recipes import footage_routes

SLOT_KINDS = {plans.AI_IMAGES: IMAGE_SLOT, plans.AI_FOOTAGE: VIDEO_SLOT}


def named_toolkits(cadence: Iterable[Any]) -> List[Tuple[int, str, str]]:
    """``(row index, slot kind, toolkit)`` for each row that names the toolkit of its AI media."""
    found = []
    for index, row in enumerate(cadence or ()):
        visual = row.get("visual") if isinstance(row, Mapping) else None
        kind = SLOT_KINDS.get(visual.get("source")) if isinstance(visual, Mapping) else None
        if kind is not None and visual.get("toolkit"):
            found.append((index, kind, str(visual["toolkit"])))
    return found


def check_toolkits(db: Any, workspace_id: Any, cadence: Iterable[Any]) -> None:
    """:class:`plans.InvalidPlan` unless every toolkit a row names makes its media here now.
    Reads the capability registry only when a row names one (it may commit the session:
    call it before the save stages anything)."""
    wanted = named_toolkits(cadence)
    if not wanted:
        return
    caps = media_capabilities(db, workspace_id)
    for index, kind, toolkit in wanted:
        route = footage_routes.route_named(kind, caps, toolkit)
        if isinstance(route, str):
            raise plans.InvalidPlan(f"cadence[{index}].visual.toolkit: {route}")
