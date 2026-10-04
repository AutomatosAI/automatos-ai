"""platform_update_playbook switches a timer off without throwing its time away (F290, night 8).

Night 8: {schedule_config: {enabled: false}} was refused for having no "type" (#917's
check), and Auto gave up ("Since I don't have the task_id, I cannot fulfill your
request"). Elsewhere {type: "manual", enabled: false} was saved and the owner's
weekday-9 setting was gone, and {type: "cron", cron_expression, enabled: false} was
written to the wrong one of two namesakes (#110 for #111).

A schedule_config that only switches the timer ({enabled: …}, {type: "manual"},
{type: "cron"} with no cron_expression, or {}) is now read against the saved
timer: its cron and zone are kept and only "enabled" changes ("manual" and {} mean
off, as they did before, but the time is no longer lost). A playbook with no
timer says so and nothing changes. A complete new cron replaces the timer as
before, but never on the wrong namesake (timer_target.py). A type the scheduler
doesn't know ('none') still meets #917's refusal; null still changes nothing.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

from modules.tools.discovery.timer_off import is_off
from modules.tools.discovery.timer_target import NO_TIMER, by_id, failed, has_timer, timer_said, wrong_namesake

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

SWITCH_TYPES = (None, "manual", "cron")
KEPT_OFF = (" The timer is off: it no longer runs by itself. Its time, {when}, is kept; send schedule_config "
            "{{\"enabled\": true}} to switch it back on.")
BACK_ON = " The timer is back on at its own time: {when}."


def merges_the_schedule(handler: Handler) -> Handler:
    """platform_update_playbook: see the module. Outermost, so #917's check sees
    the merged schedule."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        given = (params or {}).get("schedule_config")
        playbook = by_id(db, workspace_id, params.get("playbook_id")) if isinstance(given, dict) else None
        if playbook is None:
            return await handler(db, workspace_id, params)
        off = _switches_off(given)
        refusal = wrong_namesake(db, workspace_id, playbook, off) if _names_a_timer(given) else None
        if refusal is not None:
            return refusal
        if not _only_switches(given):
            return await handler(db, workspace_id, params)
        if not has_timer(playbook):
            return failed(NO_TIMER.format(id=playbook.id, name=playbook.name, way="off" if off else "on"))
        saved = playbook.schedule_config
        merged = {"type": "cron", "cron_expression": saved["cron_expression"],
                  "timezone": given.get("timezone") or saved.get("timezone"), "enabled": not off}
        out = await handler(db, workspace_id, {**params, "schedule_config": merged})
        return _said(out, playbook, off)
    return wrapped


def _switches_off(given: Dict[str, Any]) -> bool:
    """enabled false, type "manual", or no word on it at all ({}): the timer goes off."""
    if given.get("type") == "manual":
        return True
    return is_off(given.get("enabled")) if "enabled" in given else given.get("type") is None


def _only_switches(given: Dict[str, Any]) -> bool:
    """A schedule_config that carries no time of its own: it only switches the timer."""
    return given.get("type") in SWITCH_TYPES and not given.get("cron_expression") \
        and not given.get("trigger_config")


def _names_a_timer(given: Dict[str, Any]) -> bool:
    """A schedule_config about a cron timer: one that switches it, or a new cron."""
    return _only_switches(given) or given.get("type") == "cron"


def _said(out: Any, playbook: Any, off: bool) -> Any:
    """The answer says the time was kept (off) or is running again (on)."""
    if not (isinstance(out, dict) and out.get("success")):
        return out
    note = (KEPT_OFF if off else BACK_ON).format(when=timer_said(playbook))
    return {**out, "message": f"{out.get('message', '')}{note}"}


__all__ = ["merges_the_schedule"]
