"""PRD-256 FX-010: what an approval card says for a playbook it creates, times or deletes.

Night 12 created playbooks with no card. Creating, scheduling and deleting a playbook are
owner-only now (Decision D1, amended 8 Oct), and the card says what the owner approves: a
new playbook's name and what it is for; a timer's cron, zone and switch 'from → to' (read
from the playbook's own row); or which playbook is deleted for good. P256-FIX-RVW-14: an
update that carries a timer (``schedule_config``) shows the same timer lines, and any new
name or purpose it gives with them. A playbook is looked
up in the caller's workspace only, by its id: a name is bound to one before the card
(``playbook_binding``, P256-FIX-RVW-45).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_text import change_line, said_line, value_line

PLAYBOOK_LINE = "playbook: '{name}' (playbook #{id})"
DELETED_FOR_GOOD = "playbook: '{name}' (playbook #{id}) is deleted for good, with its timer and triggers"
# platform_schedule_playbook: param (its schedule_config key too) → what the owner calls it.
TIMER_FIELDS = (("cron_expression", "runs at (cron)"), ("timezone", "time zone"), ("enabled", "timer on"))
WAIT_FOR_ME = ("wait_for_me", "each run waits for your check")
SCHEDULE_CONFIG = "schedule_config"
# schedule_config's other keys: how it runs ('manual' | 'cron' | 'trigger') and its trigger.
TIMER_KINDS = (("type", "runs"), ("trigger_config", "trigger"))
# platform_update_playbook: every other param its handler applies → what the owner calls it.
# The click runs the whole call, so the card shows all of it beside the timer.
BESIDE_THE_TIMER = (("name", "name"), ("description", "what it is for"), ("tags", "tags"),
                    ("execution_config", "how its steps run"), ("inputs", "what each run needs"))


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


def update_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """platform_update_playbook with a timer: the schedule card's lines from its
    ``schedule_config``, how it runs and its trigger 'from → to', then every other field
    the call changes (its name, purpose, tags, step config or inputs)."""
    playbook = _playbook(db, workspace_id, params)
    timer = params.get(SCHEDULE_CONFIG) if isinstance(params.get(SCHEDULE_CONFIG), dict) else {}
    if playbook is None:
        return []
    # Only the timer's own fields: a schedule_config never names the playbook the card reads.
    said = {param: timer[param] for param, _label in TIMER_FIELDS if param in timer}
    lines = schedule_lines(db, workspace_id, action, {**params, **said})
    now = playbook.schedule_config if isinstance(playbook.schedule_config, dict) else {}
    lines.extend(change_line(label, now.get(key), timer[key]) for key, label in TIMER_KINDS if timer.get(key) is not None)
    lines.extend(change_line(label, getattr(playbook, field, None), params[field])
                 for field, label in BESIDE_THE_TIMER if params.get(field) is not None)
    return lines


def delete_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """Which playbook goes, and that it cannot be undone."""
    playbook = _playbook(db, workspace_id, params)
    return [value_line(DELETED_FOR_GOOD.format(name=playbook.name, id=playbook.id))] if playbook is not None else []


def _playbook(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[Any]:
    """This workspace's playbook by its id: a name is bound to one before the card is
    asked (``playbook_binding``, P256-FIX-RVW-45), and the click runs on that id."""
    from core.models.core import WorkflowTemplate

    try:
        playbook_id = int(params.get("playbook_id"))
    except (TypeError, ValueError):
        return None
    return (db.query(WorkflowTemplate)
            .filter(WorkflowTemplate.workspace_id == workspace_id, WorkflowTemplate.id == playbook_id).first())


__all__ = ["create_lines", "delete_lines", "schedule_lines", "update_lines"]
