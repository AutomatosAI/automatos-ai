"""The public Marketplace's Featured and Popular items (PRD-270 §5.7, one page for everyone).

A signed-out visitor's Marketplace draws the member page's Featured hero and rail and, in place of the
personal "Recommended for You", a Popular strip. ``read_featured`` answers both as
``{featured, popular}``:

* ``featured``: the approved ``owner_type='marketplace'`` agents marked featured, most installed first
  then newest, with the id as the last tie, at most ``FEATURED_LIMIT``. Agents only, as the member
  ``GET /api/marketplace/featured`` lists (``api/marketplace.py`` ``get_featured``);
* ``popular``: what the member page's "Recommended" strip computes (``marketplace-homepage.tsx``): the
  ``POPULAR_POOL`` most installed approved Marketplace agents and Playbooks together, by install count
  then newest, less the featured ones, the first ``POPULAR_LIMIT``.

Each item maps through ``FEATURED_FIELDS`` (``fields.py``): the display fields, its ``type`` (``agents``
or ``playbooks``, the public read it comes from), ``version`` and ``installCount``. The queries name
their columns, so no configuration, prompt, step, owner, clone or workspace id is ever read; the
publisher is ``publishers.py``'s display name, never an email.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sqlalchemy.orm import Session

from core.models.core import Agent, WorkflowTemplate as WorkflowRecipe
from services.marketplace_public.agents import apps_by_agent
from services.marketplace_public.fields import (
    FEATURED_FIELDS, MARKETPLACE_OWNER, count_of, display_item, strings, text_or_none,
)
from services.marketplace_public.publishers import publisher_names, publisher_of

AGENTS_TYPE, PLAYBOOKS_TYPE = "agents", "playbooks"
FEATURED_LIMIT = 5  # the member homepage asks /featured for five: the hero and a rail of four
POPULAR_POOL = 20  # the member homepage's "Recommended" reads the 20 most installed items
POPULAR_LIMIT = 6  # ... and shows six of them
NO_DATE = datetime.min  # a row with no created_at sorts last among equal install counts

Ranked = Tuple[str, Any]  # (the item's type, its row)

_AGENT_COLUMNS = (
    Agent.id, Agent.name, Agent.description, Agent.marketplace_category, Agent.marketplace_icon, Agent.tags,
    Agent.original_creator_id, Agent.version, Agent.install_count, Agent.created_at,
)
_PLAYBOOK_COLUMNS = (
    WorkflowRecipe.id, WorkflowRecipe.name, WorkflowRecipe.description, WorkflowRecipe.marketplace_category,
    WorkflowRecipe.marketplace_icon, WorkflowRecipe.tags, WorkflowRecipe.required_tools,
    WorkflowRecipe.original_creator_id, WorkflowRecipe.version, WorkflowRecipe.install_count,
    WorkflowRecipe.created_at,
)


def _agent_rows(db: Session, limit: int, featured_only: bool) -> List[Any]:
    """Approved Marketplace agents, most installed then newest, the featured ones only when asked."""
    query = db.query(*_AGENT_COLUMNS).filter(Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True))
    if featured_only:
        query = query.filter(Agent.is_featured.is_(True))
    order = (Agent.install_count.desc().nullslast(), Agent.created_at.desc().nullslast(), Agent.id.desc())
    return query.order_by(*order).limit(limit).all()


def _playbook_rows(db: Session, limit: int) -> List[Any]:
    """Approved Marketplace Playbooks, most installed then newest."""
    order = (
        WorkflowRecipe.install_count.desc().nullslast(), WorkflowRecipe.created_at.desc().nullslast(),
        WorkflowRecipe.id.desc(),
    )
    return (
        db.query(*_PLAYBOOK_COLUMNS)
        .filter(WorkflowRecipe.owner_type == MARKETPLACE_OWNER, WorkflowRecipe.is_approved.is_(True))
        .order_by(*order)
        .limit(limit)
        .all()
    )


def _rank(entry: Ranked) -> Tuple[int, datetime]:
    _kind, row = entry
    return count_of(row.install_count), row.created_at or NO_DATE


def popular_from(agents: Sequence[Any], playbooks: Sequence[Any], featured_ids: Iterable[int]) -> List[Ranked]:
    """The member page's "Recommended" over these rows: the pool by installs then newest, less the featured."""
    pool = [(AGENTS_TYPE, row) for row in agents] + [(PLAYBOOKS_TYPE, row) for row in playbooks]
    ranked = sorted(pool, key=_rank, reverse=True)[:POPULAR_POOL]
    shown = {(AGENTS_TYPE, agent_id) for agent_id in featured_ids}
    return [entry for entry in ranked if (entry[0], entry[1].id) not in shown][:POPULAR_LIMIT]


def _item(entry: Ranked, names: Mapping[int, str], apps: Mapping[int, List[str]]) -> Dict[str, Any]:
    kind, row = entry
    required = apps.get(row.id, ()) if kind == AGENTS_TYPE else strings(row.required_tools)
    return display_item(
        FEATURED_FIELDS, key=row.id, name=row.name, description=row.description,
        category=row.marketplace_category, icon=row.marketplace_icon, tags=row.tags,
        publisher=publisher_of(names, row.original_creator_id), required_apps=required,
        type=kind, version=text_or_none(row.version), installCount=count_of(row.install_count),
    )


def read_featured(db: Session) -> Dict[str, List[Dict[str, Any]]]:
    """``{featured, popular}``: the featured agents, then the most installed agents and Playbooks besides."""
    featured = [(AGENTS_TYPE, row) for row in _agent_rows(db, FEATURED_LIMIT, featured_only=True)]
    agents = _agent_rows(db, POPULAR_POOL, featured_only=False)
    popular = popular_from(agents, _playbook_rows(db, POPULAR_POOL), (row.id for _kind, row in featured))
    shown = featured + popular
    names = publisher_names(db, (row.original_creator_id for _kind, row in shown))
    agent_ids = [row.id for kind, row in shown if kind == AGENTS_TYPE]
    apps = apps_by_agent(db, agent_ids) if agent_ids else {}
    return {
        "featured": [_item(entry, names, apps) for entry in featured],
        "popular": [_item(entry, names, apps) for entry in popular],
    }


__all__ = [
    "AGENTS_TYPE", "FEATURED_LIMIT", "PLAYBOOKS_TYPE", "POPULAR_LIMIT", "POPULAR_POOL", "popular_from",
    "read_featured",
]
