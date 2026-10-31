"""The public Marketplace's counts (PRD-270 §5.7, one page for everyone): "100 items · 113 installs".

The member page's header counts the items its ``/api/marketplace/items`` read lists (approved
Marketplace agents and Playbooks) and adds up their install counts. ``read_counts`` answers the same two
numbers for a signed-out visitor as ``{items, installs}``, over every approved ``owner_type='marketplace'``
agent and Playbook: one ``count`` and one ``sum`` per table, an install count that is NULL read as 0.
Nothing here reads a workspace, a person or an install record: only the catalogue rows' own totals.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.core import Agent, WorkflowTemplate as WorkflowRecipe
from services.marketplace_public.fields import COUNTS_FIELDS, MARKETPLACE_OWNER, allowed_item

NO_INSTALLS = 0


def _totals(db: Session, model: Any) -> Tuple[int, int]:
    """The number of ``model``'s approved Marketplace rows and the sum of their install counts."""
    count, installs = (
        db.query(func.count(model.id), func.coalesce(func.sum(model.install_count), NO_INSTALLS))
        .filter(model.owner_type == MARKETPLACE_OWNER, model.is_approved.is_(True))
        .one()
    )
    return int(count), int(installs)


def read_counts(db: Session) -> Dict[str, int]:
    """``{items, installs}`` over the approved Marketplace agents and Playbooks."""
    agents, playbooks = _totals(db, Agent), _totals(db, WorkflowRecipe)
    return allowed_item(COUNTS_FIELDS, items=agents[0] + playbooks[0], installs=agents[1] + playbooks[1])


__all__ = ["read_counts"]
