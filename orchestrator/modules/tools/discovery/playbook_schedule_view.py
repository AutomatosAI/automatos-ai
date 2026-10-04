"""F290 (night 8): a playbook's own answer names its timer.

Auto told the owner Tom's Monday Dispatch Checklist (#101) "does not currently have
a schedule configured at all" while its Monday 07:00 timer was on: "The
platform_get_playbook call did not return any schedule_config". Asked to switch
off "the one with a timer" of two namesakes, it read platform_list_playbooks, which
named neither the timers nor the steps, and picked the empty one.

Each playbook in platform_get_playbook's and platform_list_playbooks's answers now
carries "timer": {"on": true, "when": "weekdays at 09:00 (Europe/London)", "cron":
"0 9 * * 1-5", "timezone": "Europe/London"}, or null when it has none; the list
also carries each one's step_count. Nothing is added on a public widget turn
(F155): a timer is as much the owner's business as the steps that turn leaves out.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from modules.tools.discovery.cron_when import plain_cron
from modules.tools.discovery.timer_target import has_timer, timer_on

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def adds_each_timer(handler: Handler) -> Handler:
    """platform_get_playbook / platform_list_playbooks: each playbook in the answer
    carries its own "timer" (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        result = await handler(db, workspace_id, params)
        if widget_turn() or not (isinstance(result, dict) and result.get("success")):
            return result
        return _with_timers(db, workspace_id, result)
    return wrapped


def _with_timers(db: Session, workspace_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    one = result.get("playbook") if isinstance(result.get("playbook"), dict) else None
    listed = result.get("playbooks") if isinstance(result.get("playbooks"), list) else []
    rows = _rows(db, workspace_id, [p.get("id") for p in [one, *listed] if isinstance(p, dict)])
    out = dict(result)
    if one is not None:
        out["playbook"] = _with_timer(one, rows)
    if listed:
        out["playbooks"] = [_with_timer(p, rows) if isinstance(p, dict) else p for p in listed]
    return out


def _rows(db: Session, workspace_id: Any, ids: List[Any]) -> Dict[Any, Any]:
    """This workspace's playbooks with these ids, by id: one query for the whole answer."""
    from core.models.core import WorkflowTemplate

    wanted = [i for i in ids if i is not None]
    if not wanted:
        return {}
    found = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id, WorkflowTemplate.id.in_(wanted)).all()
    return {getattr(row, "id", None): row for row in found}


def _with_timer(playbook: Dict[str, Any], rows: Dict[Any, Any]) -> Dict[str, Any]:
    row = rows.get(playbook.get("id"))
    added: Dict[str, Any] = {"timer": timer_view(row)}
    if "step_count" not in playbook and row is not None:
        added["step_count"] = len(getattr(row, "steps", None) or [])
    return {**playbook, **added}


def timer_view(playbook: Any) -> Optional[Dict[str, Any]]:
    """{"on", "when", "cron", "timezone"}, or None when the playbook has no timer saved."""
    if playbook is None or not has_timer(playbook):
        return None
    sc = playbook.schedule_config
    return {"on": timer_on(playbook), "when": plain_cron(sc["cron_expression"], sc.get("timezone")),
            "cron": sc["cron_expression"], "timezone": sc.get("timezone")}


__all__ = ["adds_each_timer", "timer_view"]
