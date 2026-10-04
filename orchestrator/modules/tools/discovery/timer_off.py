"""Switching a playbook's timer off, and back on, through platform_schedule_playbook (F290, night 8).

Night 8: "turn the timer off" worked 3 times of 8. Auto sent {enabled: false} with no
cron and was refused for a missing cron_expression (the schema required one, and so
did the tool), then asked the owner "Could you please provide the cron expression?",
sent "None", null and "" as the cron, searched the web for cron syntax and once
proposed a "February 31st" timer. The owner doesn't know what a cron is; the timer
already had one.

enabled: false needs no cron now. The playbook's saved time and zone are kept,
switched off, and the answer says so in plain words. enabled: true with no cron
switches a saved timer back on at its own time. A call with no time for a playbook
that has no timer is refused saying so, and to write the cron from the owner's
words, never to ask them for one. Which playbook is meant is timer_target.py's.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Tuple

from sqlalchemy.orm import Session

from modules.tools.discovery.timer_target import (
    NO_TIMER, failed, has_timer, target, timer_said, wrong_namesake,
)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

OFF_WORDS = frozenset({"false", "off", "no", "0"})
BLANK_CRONS = frozenset({"", "none", "null"})
NEEDS_A_TIME = ("Playbook #{id} '{name}' has no timer yet, so it needs a cron_expression. Write it from the "
                "owner's words ('every weekday at 9' is '0 9 * * 1-5') and call this again; never ask the owner "
                "for a cron expression.")
NOW_OFF = ("The timer on playbook #{id} '{name}' is off: it no longer runs by itself. Its time, {when}, is kept: "
           "call this again with enabled: true and no cron_expression to switch it back on.")
BACK_ON = " It is switched back on at its own time: {when}."


def is_off(value: Any) -> bool:
    """enabled: false, as a model sends it (a boolean, or "false" as text)."""
    return value is False or (isinstance(value, str) and value.strip().lower() in OFF_WORDS)


def blank_cron(value: Any) -> bool:
    """No usable cron was sent: missing, "", "None" or null (all three reached the tool on night 8)."""
    return value is None or str(value).strip().lower() in BLANK_CRONS


def switches_the_timer(handler: Handler) -> Handler:
    """platform_schedule_playbook: the playbook the owner means, and a timer
    switched off or on without a cron (see the module). Outermost: the
    decorators under it get the playbook's id and its saved time."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params or {})
        off = is_off(params.get("enabled"))
        playbook, refusal, picked = target(db, workspace_id, params, off)
        if refusal is not None or playbook is None:
            return refusal or await handler(db, workspace_id, params)
        refusal = wrong_namesake(db, workspace_id, playbook, off)
        call, refusal = (None, refusal) if refusal else _with_its_time(playbook, params, off)
        if refusal is not None:
            return refusal
        back_on = not off and blank_cron(params.get("cron_expression"))
        out = await handler(db, workspace_id, call)
        return _said(out, playbook, off=off, back_on=back_on, picked=picked)
    return wrapped


def _with_its_time(playbook: Any, params: Dict[str, Any], off: bool) -> Tuple[Any, Any]:
    """(the call to make, None), or (None, the refusal): a call with no cron takes
    the playbook's saved one, and its zone unless the call names another."""
    call = {**params, "playbook_id": playbook.id, **({"enabled": False} if off else {})}
    if not blank_cron(params.get("cron_expression")):
        return call, None
    if not has_timer(playbook):
        refusal = NO_TIMER if off else NEEDS_A_TIME
        return None, failed(refusal.format(id=playbook.id, name=playbook.name, way="off"))
    saved = playbook.schedule_config
    return {**call, "cron_expression": saved["cron_expression"],
            "timezone": params.get("timezone") or saved.get("timezone")}, None


def _said(out: Any, playbook: Any, *, off: bool, back_on: bool, picked: Any) -> Any:
    """The answer in the owner's words: off with its time kept, or back on at its time."""
    if not (isinstance(out, dict) and out.get("success")):
        return out
    said = {**out, "chosen_because": picked} if picked else dict(out)
    when = timer_said(playbook)
    if off:
        return {**said, "message": NOW_OFF.format(id=playbook.id, name=playbook.name, when=when)}
    if back_on:
        return {**said, "message": f"{out.get('message', '')}{BACK_ON.format(when=when)}"}
    return said


__all__ = ["blank_cron", "is_off", "switches_the_timer"]
