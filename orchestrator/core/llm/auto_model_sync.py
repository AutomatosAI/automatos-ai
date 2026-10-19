"""Auto rows that never chose a model follow the platform default (Gerard, 19 Oct 2026).

The live default for Auto is data: the ``orchestrator_llm`` provider and model in
system settings (``core/llm/new_agent_model.default_route``), which a platform admin
sets in System Settings > LLM tiers. A new Auto row takes it and is marked as
inheriting (``inheriting_model_config``, used by ``core/seeds/seed_auto_agent``).
When an admin changes that setting (``api/system_settings.py``, the
``platform_update_system_setting`` tool), ``restamp_inheriting_auto_agents`` moves
the Auto rows still on the platform default to the new one. A model a workspace
chose is never moved, and no other agent is touched.

A row inherits when either:

* it carries the platform-default marker AND still holds the route it was stamped
  with (a row re-pointed by any writer that copies the dict and writes only the
  provider and model no longer matches its recorded route: that is a choice);
* it carries no marker at all and holds, exactly, a pair a shipped seed wrote
  before the marker existed (``KNOWN_SEED_ROUTES``). A row with any other marker,
  ``WORKSPACE_CHOICE_SOURCE`` included, is never read as a seed's.

Settings > Orchestrator (``services/orchestrator_auto_agent.sync_auto_llm``) writes
only that workspace's Auto row and marks it (``workspace_saved_model_config``): a save
of the platform default's own pair, or one that leaves an inheriting row's model as it
was, keeps the row inheriting; any other pair marks it ``workspace_choice``. A known
gap, accepted: a row from before the markers that a workspace re-pointed, in
Settings, to exactly one of the two seed pairs and never saved again cannot be told
from a seed row. Another: a page loaded before an admin's change that saves the old
pair afterwards records it as the workspace's choice.

The restamp locks the rows it reads (``SELECT ... FOR UPDATE``) until the caller
commits, so a workspace's save that lands meanwhile is either seen (and kept as a
choice) or waits and is written after it: never overwritten by a stale plan.

Nothing runs this at boot. The one deploy-time move is the Alembic data migration
``auto_haiku_default_20261019``, which carries its own copy of these rules.
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
# A workspace saved its own model in Settings > Orchestrator: never restamped.
WORKSPACE_CHOICE_SOURCE = "workspace_choice"
# The route the row was stamped with, so a later change made in Settings shows.
STAMPED_ROUTE_KEY = "platform_route"
AUTO_SLUG_PATTERN = "auto-%"
# The orchestrator_llm keys default_route reads for the platform default (the
# legacy names too): a change of one of them moves the inheriting Auto rows.
ORCHESTRATOR_CATEGORY = "orchestrator_llm"
ORCHESTRATOR_ROUTE_KEYS = frozenset({"provider", "model", "llm_provider", "llm_model"})
RESTAMP_UNDONE = ("The Auto agents on the platform default could not be moved to the new "
                  "model, so nothing was saved. Try again.")
RESTAMP_FAILED = ("The setting was saved, but the Auto agents on the platform default were "
                  "not moved to it. Save another model, then this one again, to move them.")

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


def workspace_saved_model_config(stored: Mapping[str, Any], update: Mapping[str, Any],
                                 platform_route: Route) -> Dict[str, Any]:
    """A NEW dict: the Auto row's ``stored`` model_config with a Settings > Orchestrator
    save's ``update`` over it, marked. The row follows the platform default
    (``inheriting_model_config``) when the save names the platform default's own
    pair, or leaves the model of a row that was following it as it was (a Soul or
    sampling save is no choice of model). Any other pair is the workspace's choice,
    marked ``WORKSPACE_CHOICE_SOURCE``, which no restamp moves."""
    saved = {**stored, **update}
    route = _route_of(saved)
    unchanged = route == _route_of(stored) and is_inheriting(stored)
    if route == tuple(platform_route) or unchanged:
        return inheriting_model_config(saved)
    kept = {key: value for key, value in saved.items() if key != STAMPED_ROUTE_KEY}
    return {**kept, SOURCE_KEY: WORKSPACE_CHOICE_SOURCE}


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


Changed = List[Tuple[Any, Optional[str]]]  # (system setting, its value before)


class AutoRestampError(RuntimeError):
    """The restamp after a settings change failed; ``str(exc)`` is the admin's message."""


def changes_platform_default(category: str, key: str, old_value: Optional[str], new_value: Optional[str]) -> bool:
    """Does this settings write change the provider or model of the platform default?"""
    return (category == ORCHESTRATOR_CATEGORY and key in ORCHESTRATOR_ROUTE_KEYS
            and (old_value or "") != (new_value or ""))


def _undo_settings(db: Session, changed: Changed) -> str:
    """Put ``changed`` back as it was, so the same save retries the move (the
    restamp fires on a changed value only); the message for the admin."""
    try:
        for setting, old_value in changed:
            setting.value = old_value
        db.commit()
        return RESTAMP_UNDONE
    except Exception:
        db.rollback()
        logger.exception("Could not undo the settings saved before a failed Auto model restamp")
        return RESTAMP_FAILED


def restamp_after_settings_change(db: Session, changed: Changed) -> Dict[str, Any]:
    """Move the Auto rows still on the platform default to the one just saved, and
    commit. The settings in ``changed`` are committed first: ``default_route`` reads
    them on its own connection. If the move fails they are put back, so the setting
    and the rows never part, and ``AutoRestampError`` says what happened."""
    try:
        counts = restamp_inheriting_auto_agents(db)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Auto model restamp failed after an orchestrator_llm change")
        raise AutoRestampError(_undo_settings(db, changed)) from exc
    logger.info("orchestrator_llm changed: Auto model restamp %s", counts)
    return counts


__all__ = [
    "AutoRestampError", "KNOWN_SEED_ROUTES", "ORCHESTRATOR_ROUTE_KEYS", "PLATFORM_DEFAULT_SOURCE",
    "RESTAMP_FAILED", "RESTAMP_UNDONE", "RestampPlan", "WORKSPACE_CHOICE_SOURCE", "auto_agent_rows",
    "changes_platform_default", "inheriting_model_config", "is_inheriting", "plan_restamp",
    "restamp_after_settings_change", "restamp_inheriting_auto_agents", "workspace_saved_model_config",
]
