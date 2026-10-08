"""PRD-256 FX-010: what an approval card says for a playbook it creates, times or deletes.

Night 12 created playbooks with no card. Creating, scheduling and deleting a playbook are
owner-only now (Decision D1, amended 8 Oct), and the card says what the owner approves: a
new playbook's name and what it is for; a timer's cron, zone and switch 'from → to' (read
from the playbook's own row); or which playbook is deleted for good. A playbook is looked
up in the caller's workspace only, by its id or by the one playbook with that name.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_text import change_line, said_line, value_line

PLAYBOOK_LINE = "playbook: '{name}' (playbook #{id})"
DELETED_FOR_GOOD = "playbook: '{name}' (playbook #{id}) is deleted for good, with its timer and triggers"
# platform_schedule_playbook: param (its schedule_config key too) → what the owner calls it.
TIMER_FIELDS = (("cron_expression", "runs at (cron)"), ("timezone", "time zone"), ("enabled", "timer on"))
WAIT_FOR_ME = ("wait_for_me", "each run waits for your check")


def create_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """A new playbook's name and what it is for."""
    lines = [said_line("name", params.get("name")), said_line("what it is for", params.get("description"))]
    if params.get(WAIT_FOR_ME[0]) is not None:
        lines.append(said_line(WAIT_FOR_ME[1], params[WAIT_FOR_ME[0]]))
    return lines


def schedule_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The playbook, then each part of its timer the call changes, 'from → to'."""
    playbook = _playbook(db, workspace_id, params)
    if playbook is None:
        return []
    timer = playbook.schedule_config if isinstance(playbook.schedule_config, dict) else {}
    changes = [change_line(label, timer.get(param), params[param])
               for param, label in TIMER_FIELDS if params.get(param) is not None]
    if params.get(WAIT_FOR_ME[0]) is not None:
        changes.append(said_line(WAIT_FOR_ME[1], params[WAIT_FOR_ME[0]]))
    return [value_line(PLAYBOOK_LINE.format(name=playbook.name, id=playbook.id)), *changes]


def delete_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """Which playbook goes, and that it cannot be undone."""
    playbook = _playbook(db, workspace_id, params)
    return [value_line(DELETED_FOR_GOOD.format(name=playbook.name, id=playbook.id))] if playbook is not None else []


def _playbook(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[Any]:
    """This workspace's playbook by id, or the one playbook carrying the name said (two
    namesakes name none: the handler asks which)."""
    from core.models.core import WorkflowTemplate
    from modules.tools.discovery.handlers_playbooks import _playbooks_called

    query = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == workspace_id)
    if params.get("playbook_id") not in (None, ""):
        try:
            return query.filter(WorkflowTemplate.id == int(params["playbook_id"])).first()
        except (TypeError, ValueError):
            return None
    named = _playbooks_called(db, workspace_id, params["playbook_name"]) if params.get("playbook_name") else []
    return query.filter(WorkflowTemplate.id == named[0].id).first() if len(named) == 1 else None


__all__ = ["create_lines", "delete_lines", "schedule_lines"]
