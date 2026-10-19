"""Auto rows that never chose a model follow the platform default (Gerard, 19 Oct 2026).

The live default for Auto is data: the ``orchestrator_llm`` provider and model in
system settings (``core/llm/new_agent_model.default_route``). A new Auto row takes
it and is marked as inheriting (``inheriting_model_config``, used by
``core/seeds/seed_auto_agent``). When an admin changes that setting
(``api/system_settings.py``), ``restamp_inheriting_auto_agents`` moves the Auto rows
still on the platform default to the new one. A model a workspace chose is never
moved, and no other agent is touched.

A row inherits when either:

* it carries the marker AND still holds the route it was stamped with. Settings >
  Orchestrator's save (``api/workspaces.py``) copies the stored dict and writes only
  the provider and model over it, so a row a workspace re-pointed keeps the marker
  but no longer matches its recorded route: that is a choice, and it stays;
* it carries no marker at all and holds, exactly, a pair a shipped seed wrote
  before the marker existed (``KNOWN_SEED_ROUTES``). A row with any other marker
  is never read as a seed's.

A known gap, accepted: a row from before the marker that a workspace re-pointed, in
Settings, to exactly one of those two pairs cannot be told from a seed row, and a
restamp moves it. Settings > Orchestrator writes no marker of its own; until it does
(``api/workspaces.py``), the stored dict is all there is to go on.

The restamp locks the rows it reads (``SELECT ... FOR UPDATE``) until the caller
commits, so a workspace's save that lands meanwhile is either seen (and kept as a
choice) or waits and is written after it: never overwritten by a stale plan.

Nothing runs this at boot: deploying it moves no row. It runs on an admin change of
the setting, or by hand through ``scripts/restamp_auto_models.py --apply``.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)

Route = Tuple[str, str]  # (provider, model_id)

SOURCE_KEY = "source"
PLATFORM_DEFAULT_SOURCE = "platform_default"
# The route the row was stamped with, so a later change made in Settings shows.
STAMPED_ROUTE_KEY = "platform_route"
AUTO_SLUG_PATTERN = "auto-%"

# The (provider, model_id) pairs shipped seeds wrote into Auto rows before the marker
# existed. A row holding one exactly was never chosen in Settings.
KNOWN_SEED_ROUTES = frozenset({
    # core/llm/defaults.py DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL until 19 Oct 2026,
    # written by core/seeds/seed_auto_agent._get_default_model_config.
    ("openrouter", "google/gemini-2.5-flash"),
    # alembic/versions/seed_auto_agents_existing_workspaces.py:73, the migration that
    # gave the workspaces of its day their Auto row.
    ("openrouter", "openai/gpt-4o"),
})


def _route_of(model_config: Mapping[str, Any]) -> Route:
    return model_config.get("provider"), model_config.get("model_id")


def _stamped_route(model_config: Mapping[str, Any]) -> Optional[Route]:
    stamped = model_config.get(STAMPED_ROUTE_KEY)
    if not isinstance(stamped, Mapping):
        return None
    return _route_of(stamped)


def inheriting_model_config(model_config: Mapping[str, Any], route: Optional[Route] = None) -> Dict[str, Any]:
    """A NEW dict: ``model_config`` on ``route`` (default: its own pair), marked as
    following the platform default. Every other key is kept."""
    provider, model_id = route or _route_of(model_config)
    return {
        **model_config,
        "provider": provider,
        "model_id": model_id,
        SOURCE_KEY: PLATFORM_DEFAULT_SOURCE,
        STAMPED_ROUTE_KEY: {"provider": provider, "model_id": model_id},
    }


def is_inheriting(model_config: Optional[Mapping[str, Any]]) -> bool:
    """Is this Auto row still on the platform default (see the module docstring)?"""
    if not model_config:
        return False
    route = _route_of(model_config)
    if SOURCE_KEY in model_config:
        return (model_config[SOURCE_KEY] == PLATFORM_DEFAULT_SOURCE
                and _stamped_route(model_config) == route)
    return route in KNOWN_SEED_ROUTES


def auto_agent_rows(db: Session, *, lock: bool = False) -> List[Any]:
    """Every workspace's Auto row, in id order. Platform-wide on purpose:
    ``orchestrator_llm`` is one platform setting, changed by a platform admin.
    ``lock`` holds the rows (``FOR UPDATE``) until the transaction ends: for a
    restamp that writes them, never on the dry run's read-only connection."""
    from core.models.core import Agent

    query = (
        db.query(Agent)
        .filter(
            Agent.is_system_agent.is_(True),
            Agent.slug.like(AUTO_SLUG_PATTERN),
            Agent.workspace_id.isnot(None),
        )
        .order_by(Agent.id)
    )
    return (query.with_for_update() if lock else query).all()


def _label(route: Route) -> str:
    return f"{route[0]}/{route[1]}"


@dataclass(frozen=True)
class RestampPlan:
    """What a restamp to ``target`` would do: the rows to move (each with its new
    model_config), and how many stay because they are current or were chosen."""

    target: Route
    moves: List[Tuple[Any, Dict[str, Any]]] = field(default_factory=list)
    current: int = 0
    chosen: int = 0

    def by_route(self) -> Dict[str, int]:
        """``"from -> to"`` → rows that would move along it."""
        counts = Counter(f"{_label(_route_of(agent.model_config))} -> {_label(self.target)}"
                         for agent, _ in self.moves)
        return dict(counts)

    def counts(self) -> Dict[str, Any]:
        return {"target": _label(self.target), "moved": len(self.moves), "current": self.current,
                "chosen": self.chosen, "by_route": self.by_route()}


def plan_restamp(rows: List[Any], target: Route) -> RestampPlan:
    """Sort ``rows`` (Auto agents) into moves, current and chosen. Reads only."""
    moves: List[Tuple[Any, Dict[str, Any]]] = []
    current = chosen = 0
    for agent in rows:
        model_config = agent.model_config or {}
        if not is_inheriting(model_config):
            chosen += 1
            continue
        new_config = inheriting_model_config(model_config, target)
        if new_config == model_config:
            current += 1
        else:
            moves.append((agent, new_config))
    return RestampPlan(target=target, moves=moves, current=current, chosen=chosen)


def _assign_model_config(agent: Any, model_config: Dict[str, Any]) -> None:
    agent.model_config = model_config
    if sa_inspect(agent, raiseerr=False) is not None:
        flag_modified(agent, "model_config")


def restamp_inheriting_auto_agents(db: Session) -> Dict[str, Any]:
    """Move every inheriting Auto row to the platform default as it reads now
    (``default_route``); ``{"target", "moved", "current", "chosen", "by_route"}``.
    The rows stay locked until the caller commits; it flushes, the caller commits."""
    from core.llm.new_agent_model import default_route

    plan = plan_restamp(auto_agent_rows(db, lock=True), default_route())
    counts = plan.counts()  # before the moves: by_route reads each row's old pair
    for agent, new_config in plan.moves:
        _assign_model_config(agent, new_config)
    db.flush()
    logger.info("Auto model restamp to %s: %d moved, %d already current, %d chosen in Settings kept (%s)",
                counts["target"], counts["moved"], counts["current"], counts["chosen"], counts["by_route"])
    return counts


__all__ = [
    "KNOWN_SEED_ROUTES", "PLATFORM_DEFAULT_SOURCE", "RestampPlan", "auto_agent_rows",
    "inheriting_model_config", "is_inheriting", "plan_restamp", "restamp_inheriting_auto_agents",
]
