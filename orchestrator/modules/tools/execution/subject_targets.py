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
SCHEDULED_TASKS = "agent_scheduled_tasks"
# Where an action's id points when it is not the default table above.
_ACTION_TARGETS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "platform_cancel_scheduled_task": {"task_id": (SCHEDULED_TASKS, "description", "scheduled task")},
}
NAME_CHARS = 80
TICKETS = "board_tasks"
TICKET = "ticket"
# PRD-268 K8: a person's follow-up is theirs alone. No card names its reason and no generic
# call acts on it: to every other caller its id names no scheduled task.
_A_FOLLOW_UP = text(
    "SELECT 1 FROM agent_scheduled_tasks "
    "WHERE id = :id AND workspace_id = CAST(:ws AS uuid) AND deliver_as = :follow_up"
)


@dataclass(frozen=True)
class Target:
    param: str
    ident: Any
    noun: str
    name: Optional[str] = None
    # PRD-252 R4: a ticket as messages name it, "ticket #0042".
    label: Optional[str] = None
    # PRD-256 FX-009: why it names nothing the call can act on, in the tool's own words.
    why: Optional[str] = None

    @property
    def called(self) -> str:
        """"ticket #0042"; "ticket 612" for a ticket known only by its id (a '#'
        means a ticket's number); "document #716" for anything else."""
        if self.label:
            return self.label
        return f"{self.noun} {self.ident}" if self.noun == TICKET else f"{self.noun} #{self.ident}"


def resolve_targets(db: Any, workspace_id: Any, params: Any,
                    action: Optional[str] = None) -> Tuple[List[Target], List[Target]]:
    """``(found, missing)`` for the id parameters of ``action``'s call, in this
    workspace. A lookup that errors counts as found-without-a-name: it is not a
    verdict."""
    if not isinstance(params, dict):
        return [], []
    from modules.tools.discovery.mission_targets import mission_targets

    found, missing = mission_targets(db, workspace_id, params, action)  # FX-009: a mission, as its tool reads it
    targets = {**_TARGETS, **_ACTION_TARGETS.get(action or "", {})}
    for param, (table, label, noun) in targets.items():
        raw = params.get(param)
        if raw in (None, ""):
            continue
        ident = _row_id(db, workspace_id, table, raw)
        if ident is None:
            missing.append(Target(param, raw, noun))
            continue
        try:
            row = db.execute(
                text(f"SELECT {label} FROM {table} WHERE id = :id AND workspace_id = CAST(:ws AS uuid)"),
                {"id": ident, "ws": str(workspace_id)},
            ).first()
            hidden = row is not None and _a_follow_up(db, workspace_id, table, ident)
        except Exception:  # noqa: BLE001 — cannot tell: stage the card as before
            logger.warning("[subject-targets] could not look up %s %s", noun, ident, exc_info=True)
            found.append(Target(param, ident, noun))
            continue
        if row is None or hidden:
            missing.append(Target(param, ident, noun))
        else:
            name = str(row[0])[:NAME_CHARS] if row[0] is not None else None
            found.append(Target(param, ident, noun, name, _ticket_called(db, workspace_id, table, ident)))
    return found, missing


def names_a_ticket(action: Optional[str], param: str) -> bool:
    """Whether ``param`` of ``action``'s call names a board ticket (a scheduled task's
    ``task_id`` does not, P256-FIX-N13-7)."""
    entry = {**_TARGETS, **_ACTION_TARGETS.get(action or "", {})}.get(param)
    return entry is not None and entry[0] == TICKETS


def _row_id(db: Any, workspace_id: Any, table: str, raw: Any) -> Optional[int]:
    """The row a parameter names: its id, or for a ticket the ticket the ticket tools
    will act on (#0042, "0042" and 42 are its number, PRD-252 R4 and Gerard, 7 Oct),
    so the card names the ticket the call will change. None when it names no row."""
    from services.ticket_refs import ticket_id_named

    if table == TICKETS:
        return ticket_id_named(db, workspace_id, raw)[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _a_follow_up(db: Any, workspace_id: Any, table: str, ident: int) -> bool:
    """Whether the scheduled task ``ident`` is a person's follow-up (PRD-268 K8)."""
    if table != SCHEDULED_TASKS:
        return False
    from services.scheduled_task_steps import DELIVER_FOLLOW_UP

    found = db.execute(_A_FOLLOW_UP, {"id": ident, "ws": str(workspace_id), "follow_up": DELIVER_FOLLOW_UP})
    return found.first() is not None


def _ticket_called(db: Any, workspace_id: Any, table: str, ident: int) -> Optional[str]:
    if table != TICKETS:
        return None
    from services.ticket_numbers import ticket_label_for

    return ticket_label_for(db, workspace_id, ident)


# Params that identify WHAT an action will act on, in the order a person reads them. An approval
# card that says only "cancel a scheduled task" gives the operator nothing to decide on (night 1,
# 2026-09-18: the card named neither the id nor the title of the schedule it would cancel).
SUBJECT_PARAMS: Tuple[str, ...] = (
    "title", "name", "task_id", "agent_id", "document_id", "mission_id",
    "report_id", "id", "app_name", "query", "playbook_name",
)


def subject_line(params: Dict[str, Any]) -> str:
    """" on <what this will act on>", or "" when nothing identifies it: the card's subject
    when no looked-up target names one."""
    if not isinstance(params, dict):
        return ""
    named = [
        f"{key}={params[key]!r}"
        for key in SUBJECT_PARAMS
        if params.get(key) not in (None, "", [], {})
    ]
    return f" on {', '.join(named[:3])}" if named else ""


def named_subject(found: List[Target]) -> str:
    """ " on 'christmas-box-2026.csv' (document #716)", or "" when nothing is named."""
    parts = [f"'{t.name}' ({t.called})" if t.name else t.called for t in found]
    return f" on {', '.join(parts)}" if parts else ""


def missing_targets_error(action: str, missing: List[Target]) -> Dict[str, Any]:
    """The call's result when it names something that is not in the workspace (or, FX-009,
    names something its tool would refuse: the tool's own refusal)."""
    what = ", ".join(f"no {t.called}" for t in missing)
    told = " ".join(t.why for t in missing if t.why)
    return {
        "success": False,
        "error": told or (
            f"{what[0].upper()}{what[1:]} in this workspace — nothing was asked or done. "
            f"Look it up first (list or search), then call {action} with a real id."
        ),
        "missing_targets": [{"param": t.param, "id": t.ident, "kind": t.noun} for t in missing],
    }
