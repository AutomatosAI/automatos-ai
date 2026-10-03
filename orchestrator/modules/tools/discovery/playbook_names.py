"""Which names a new playbook may not take (F185, F222).

- F185 (night 6): asked to run the playbook it had just made, Auto made another of
  the same name. A namesake of one of this workspace's playbooks is refused,
  naming the one that exists.
- F222 (night 6, 2 Oct): asked to add the marketplace's ready-made "Weekly social
  posts", Auto made an empty playbook of that name and said it was done. A
  marketplace playbook's name is refused too: the result says what the
  marketplace holds, and that the owner installs it there. No action installs
  a single marketplace playbook for Auto, so it says it can't, never pretends.
- F231 (night 6, B119): asked to run the café playbook for Quayside Pantry, Auto
  made "New Cafe Onboarding - Quayside Pantry", a copy with the café written in,
  and ran that: each new café would leave another playbook behind. A name that is
  one of the workspace's playbooks with a qualifier added is refused; the run's
  details go to platform_execute_playbook as input_data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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


# How a copy names what it was copied for: "<playbook> - Quayside Pantry", "(…)", ": …", "for …".
QUALIFIERS = (" - ", " – ", " — ", " | ", ": ", " (", " for ")
QUALIFIED_REFUSAL = (
    "'{name}' is playbook {id} '{base}' with details added, so none was made: a copy per run leaves "
    "another playbook behind each time. To run it for {detail}, call platform_execute_playbook with "
    "playbook_id {id} and those details in input_data. If the owner asked for a separate playbook, "
    "give it a name of its own."
)


def _cuts(name: str) -> List[Tuple[int, str]]:
    """Every place a qualifier starts in ``name`` (lower case), latest first."""
    cuts = []
    for qualifier in QUALIFIERS:
        at = name.find(qualifier)
        while at > 0:
            cuts.append((at, qualifier))
            at = name.find(qualifier, at + 1)
    return sorted(cuts, reverse=True)


def qualified_namesake(db: Session, workspace_id: UUID, name: Any) -> Optional[Tuple[Any, str]]:
    """The workspace playbook that ``name`` is with a qualifier added, and the
    qualifier's detail ("Quayside Pantry"); None when it is no such name. The
    longest playbook name wins, so a name with a qualifier of its own still matches."""
    from core.models.core import WorkflowTemplate

    text = str(name).strip()
    cuts = _cuts(text.lower())
    if not cuts:
        return None
    playbooks = {str(p.name).strip().lower(): p for p in db.query(WorkflowTemplate.id, WorkflowTemplate.name)
                 .filter(WorkflowTemplate.workspace_id == workspace_id).order_by(WorkflowTemplate.id.desc())}
    for at, qualifier in cuts:
        playbook = playbooks.get(text[:at].strip().lower())
        if playbook is not None:
            detail = text[at + len(qualifier):].strip().rstrip(")").strip()
            return playbook, detail or "this run"
    return None


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
    qualified = qualified_namesake(db, workspace_id, name)
    if qualified is not None:
        playbook, detail = qualified
        return {"success": False, "existing_playbook_id": playbook.id,
                "error": QUALIFIED_REFUSAL.format(name=str(name).strip(), id=playbook.id, base=playbook.name,
                                                  detail=detail)}
    return None


__all__ = ["MARKETPLACE_REFUSAL", "QUALIFIED_REFUSAL", "marketplace_playbook_called", "name_refusal",
           "qualified_namesake"]
