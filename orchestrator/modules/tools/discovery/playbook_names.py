"""Which names a new playbook may not take (F185, F222).

- F185 (night 6): asked to run the playbook it had just made, Auto made another of
  the same name. A namesake of one of this workspace's playbooks is refused,
  naming the one that exists.
- F222 (night 6, 2 Oct): asked to add the marketplace's ready-made "Weekly social
  posts", Auto made an empty playbook of that name and said it was done. A
  marketplace playbook's name is refused too: the result says what the
  marketplace holds, and that the owner installs it there. No action installs
  a single marketplace playbook for Auto, so it says it can't, never pretends.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_playbooks import _playbook_namesakes_refusal, _playbooks_called

MARKETPLACE_OWNER = "marketplace"
MARKETPLACE_REFUSAL = (
    "'{name}' is a ready-made playbook in the Marketplace. A new playbook with its name would be an "
    "empty copy, not that one, so none was made. The owner installs it themselves, from the Playbooks "
    "tab of the Marketplace page: you can't install a single marketplace playbook for them. Tell them so."
)


def marketplace_playbook_called(db: Session, name: Any) -> Optional[Any]:
    """The approved marketplace playbook carrying ``name`` (trimmed, any case), if any."""
    from core.models.core import WorkflowTemplate

    return (
        db.query(WorkflowTemplate.id, WorkflowTemplate.name)
        .filter(
            WorkflowTemplate.owner_type == MARKETPLACE_OWNER,
            WorkflowTemplate.is_approved.is_(True),
            func.lower(func.trim(WorkflowTemplate.name)) == str(name).strip().lower(),
        )
        .order_by(WorkflowTemplate.id)
        .first()
    )


def name_refusal(db: Session, workspace_id: UUID, name: Any) -> Optional[Dict[str, Any]]:
    """Why a new playbook may not be called ``name``, as the tool's refusal; None when it may."""
    namesakes = _playbooks_called(db, workspace_id, name)
    if namesakes:
        return {
            "success": False,
            "existing_playbook_id": namesakes[0].id,
            "existing_playbook_ids": [playbook.id for playbook in namesakes],
            "error": _playbook_namesakes_refusal(namesakes),
        }
    listed = marketplace_playbook_called(db, name)
    if listed is not None:
        return {"success": False, "marketplace_playbook_id": listed.id,
                "error": MARKETPLACE_REFUSAL.format(name=listed.name)}
    return None


__all__ = ["MARKETPLACE_REFUSAL", "marketplace_playbook_called", "name_refusal"]
