"""F383 (night 11, 7 Oct): an installed Socials Playbook its owner never edited takes the seed's new prompt.

The plan's research runs the workspace's own copy of Content bank research
(``services/socials_plan_research``), cloned from the marketplace row when the plan was
first saved. The seed brought the marketplace row's prompt up to date at every boot
(``SEEDED_BEFORE``, PRD-251C US-C105) and never the copies, so night 11's research ran
the prompt it was installed with.

Every boot, each workspace copy of a refreshed Socials Playbook (``cloned_from_id`` the
marketplace row, owned by its workspace) whose step still holds a prompt an earlier seed
wrote takes the seed's prompt for that step, the rest of the step (its agent, its order)
as the copy has it. A step whose prompt the owner edited holds no earlier seed's prompt,
so it keeps its own: an owner-edited Playbook is never overwritten. The caller commits.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from sqlalchemy.orm import Session

from core.models.core import WorkflowTemplate

logger = logging.getLogger(__name__)

WORKSPACE_OWNER = "workspace"
Refresher = Callable[[Any], Optional[List[Any]]]


def installed_copies(db: Session, row: WorkflowTemplate) -> List[WorkflowTemplate]:
    """The workspaces' copies of the marketplace Playbook ``row``."""
    return (db.query(WorkflowTemplate)
            .filter(WorkflowTemplate.cloned_from_id == row.id, WorkflowTemplate.owner_type == WORKSPACE_OWNER)
            .all())


def refresh_installed_copies(db: Session, row: WorkflowTemplate, refreshed: Refresher) -> int:
    """Give each unedited copy of ``row`` the seed's prompts: ``refreshed(steps)`` is the
    steps brought up to date as new dicts, or None when nothing in them is an earlier
    seed's. Returns how many copies changed."""
    changed = 0
    for copy in installed_copies(db, row):
        steps = refreshed(copy.steps)
        if steps is None:
            continue
        copy.steps = steps
        changed += 1
        logger.info("Socials package: workspace %s's Playbook %s takes the seed's new prompt",
                    copy.workspace_id, copy.id)
    return changed


__all__ = ["installed_copies", "refresh_installed_copies"]
