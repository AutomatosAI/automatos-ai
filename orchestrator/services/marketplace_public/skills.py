"""PRD-270 US-270-07b (K10) — the public read of the Marketplace's skills.

Active global skills (``workspace_id`` NULL), the rows the member Skills tab offers to enable
(``api/workspace_skills.py`` ``list_available_skills``), by name with the id as the last tie. The tab
also marks the ones a workspace has enabled: that is a workspace's, so ``workspace_enabled_skills`` is
never read here. The query names its columns, so a skill's prompt template, implementation,
parameters, tool schema, metadata, Git and file-system source, package and creator are never read.

The key is the skill's name, which is its slug (``uq_skills_marketplace_name`` keeps it unique among
global skills). A skill has no creator a visitor can be shown, so Automatos publishes it, and it lists
no required apps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from core.models.core import Skill
from services.marketplace_public.fields import SKILL_FIELDS, display_item, text_or_none


def _item(row: Any) -> Dict[str, Any]:
    return display_item(
        SKILL_FIELDS, key=row.name, name=row.name, description=row.description, category=row.category,
        tags=row.tags, version=text_or_none(row.skill_version),
    )


def read_skills(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` active global skills from ``offset``, each as its public item."""
    rows = (
        db.query(Skill.name, Skill.description, Skill.category, Skill.tags, Skill.skill_version)
        .filter(Skill.workspace_id.is_(None), Skill.is_active.is_(True))
        .order_by(Skill.name, Skill.id)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_item(row) for row in rows]


__all__ = ["read_skills"]
