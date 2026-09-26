"""F091 (night 3): a confirmation card names what it acts on, and never asks
about something that is not there.

Night 3's delete card named no document, and #503 was an id Auto had made up:
the owner was asked to approve deleting a document that did not exist. Before
a gated call stages its card, the ids in its parameters are looked up in the
workspace. A document, agent or ticket that is not there fails the call back
to the model — nothing is asked, nothing is done — and one that is there is
named on the card.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

# param → (table, the column that names a row, what the owner calls it). Fixed
# here, never taken from the call, so the table and column are safe to format.
_TARGETS: Dict[str, Tuple[str, str, str]] = {
    "document_id": ("documents", "filename", "document"),
    "agent_id": ("agents", "name", "agent"),
    "task_id": ("board_tasks", "title", "ticket"),
    "playbook_id": ("workflow_recipes", "name", "playbook"),  # F185: the delete card named none
}
# Where an action's id points when it is not the default table above.
_ACTION_TARGETS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "platform_cancel_scheduled_task": {"task_id": ("agent_scheduled_tasks", "description", "scheduled task")},
}
NAME_CHARS = 80


@dataclass(frozen=True)
class Target:
    param: str
    ident: Any
    noun: str
    name: Optional[str] = None


def resolve_targets(db: Any, workspace_id: Any, params: Any,
                    action: Optional[str] = None) -> Tuple[List[Target], List[Target]]:
    """``(found, missing)`` for the id parameters of ``action``'s call, in this
    workspace. A lookup that errors counts as found-without-a-name: it is not a
    verdict."""
    found: List[Target] = []
    missing: List[Target] = []
    if not isinstance(params, dict):
        return found, missing
    targets = {**_TARGETS, **_ACTION_TARGETS.get(action or "", {})}
    for param, (table, label, noun) in targets.items():
        raw = params.get(param)
        if raw in (None, ""):
            continue
        try:
            ident = int(raw)
        except (TypeError, ValueError):
            missing.append(Target(param, raw, noun))
            continue
        try:
            row = db.execute(
                text(f"SELECT {label} FROM {table} WHERE id = :id AND workspace_id = CAST(:ws AS uuid)"),
                {"id": ident, "ws": str(workspace_id)},
            ).first()
        except Exception:  # noqa: BLE001 — cannot tell: stage the card as before
            logger.warning("[subject-targets] could not look up %s %s", noun, ident, exc_info=True)
            found.append(Target(param, ident, noun))
            continue
        if row is None:
            missing.append(Target(param, ident, noun))
        else:
            found.append(Target(param, ident, noun, str(row[0])[:NAME_CHARS] if row[0] is not None else None))
    return found, missing


def named_subject(found: List[Target]) -> str:
    """ " on 'christmas-box-2026.csv' (document #716)", or "" when nothing is named."""
    parts = [f"'{t.name}' ({t.noun} #{t.ident})" if t.name else f"{t.noun} #{t.ident}" for t in found]
    return f" on {', '.join(parts)}" if parts else ""


def missing_targets_error(action: str, missing: List[Target]) -> Dict[str, Any]:
    """The call's result when it names something that is not in the workspace."""
    what = ", ".join(f"no {t.noun} #{t.ident}" for t in missing)
    return {
        "success": False,
        "error": (
            f"{what[0].upper()}{what[1:]} in this workspace — nothing was asked or done. "
            f"Look it up first (list or search), then call {action} with a real id."
        ),
        "missing_targets": [{"param": t.param, "id": t.ident, "kind": t.noun} for t in missing],
    }
