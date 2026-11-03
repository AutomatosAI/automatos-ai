"""PRD-270 US-270-07a (K10) — the public read of the Marketplace's Playbooks.

Approved ``owner_type='marketplace'`` Playbook rows (``workflow_recipes``; the member routes' wire type is
``recipe``), most installed first then newest, with the id as the last tie. The query names its
columns, so a Playbook's steps, inputs, outputs, execution and schedule configs, template definition,
learning data, owner, clone and workspace ids are never read. Its required apps are its
``required_tools``, as app slugs; its publisher is ``publishers.py``'s, never the creator's email the
member route answers.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from core.models.core import WorkflowTemplate as WorkflowRecipe
from services.marketplace_public.fields import MARKETPLACE_OWNER, PLAYBOOK_FIELDS, display_item, strings
from services.marketplace_public.publishers import publisher_names, publisher_of


def read_playbooks(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` approved Marketplace Playbooks from ``offset``, each as its public item."""
    rows = (
        db.query(
            WorkflowRecipe.id, WorkflowRecipe.name, WorkflowRecipe.description, WorkflowRecipe.marketplace_category,
            WorkflowRecipe.marketplace_icon, WorkflowRecipe.tags, WorkflowRecipe.required_tools,
            WorkflowRecipe.original_creator_id,
        )
        .filter(WorkflowRecipe.owner_type == MARKETPLACE_OWNER, WorkflowRecipe.is_approved.is_(True))
        .order_by(
            WorkflowRecipe.install_count.desc().nullslast(), WorkflowRecipe.created_at.desc(),
            WorkflowRecipe.id.desc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )
    names = publisher_names(db, (row.original_creator_id for row in rows))
    return [
        display_item(
            PLAYBOOK_FIELDS, key=row.id, name=row.name, description=row.description,
            category=row.marketplace_category, icon=row.marketplace_icon, tags=row.tags,
            publisher=publisher_of(names, row.original_creator_id), required_apps=strings(row.required_tools),
        )
        for row in rows
    ]


__all__ = ["read_playbooks"]
