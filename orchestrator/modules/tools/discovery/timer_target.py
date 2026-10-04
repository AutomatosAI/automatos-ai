"""Which playbook a timer call acts on (F290, night 8).

Two playbooks were named alike: "Weekly Social Posts" (#110, no steps) and
"Weekly social posts" (#111, steps and the weekday-9 timer). Asked to switch "the
one with a timer" off, Auto read the list, which named no timer, picked 110 and
switched its timer "off" twice while 111 kept running. Asked by name, the timer
tool refused because two playbooks share it, and Auto asked the owner for an id.

Both timer tools (platform_schedule_playbook, platform_update_playbook) now read
the playbook the same way:
- by name, with namesakes: switching off takes the only one whose timer is on,
  and a new timer goes on the only one with steps; the answer says which and why;
- by id: switching off a playbook whose timer is not on, while a namesake's is,
  is refused naming that one; a timer on an empty playbook, while a namesake has
  steps, is refused naming that one. Nothing is written to the wrong playbook.

Night 9 (F310): a name only one playbook has came back as that playbook alone, not
(playbook, None, None). platform_schedule_playbook raised unpacking it, the executor
answered "Action 'platform_schedule_playbook' failed" and rolled back, so "put my
Monday Stock Check on a timer" failed three times in two chats, and switching it off
by name failed too. Only an id worked (platform_update_playbook). Every answer is
the three now.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from modules.tools.discovery.cron_when import plain_cron
from modules.tools.discovery.playbook_lookup import playbooks_called

NOT_FOUND = "No playbook {said} in this workspace, so nothing changed. List them (platform_list_playbooks) for its id."
NO_TIMER = ("Playbook #{id} '{name}' has no timer, so there is nothing to switch {way}. Nothing changed. "
            "To give it one, call platform_schedule_playbook with a cron_expression written from the owner's words.")
NONE_ON = "No playbook called '{name}' has a timer on, so there is nothing to switch off. Nothing changed."
NAMESAKE_TIMER = ("Playbook #{id} '{name}' has no timer on, so nothing changed. {others} Switch that one off: "
                  "call this again with its playbook_id.")
NAMESAKE_STEPS = ("Playbook #{id} '{name}' has no steps, so a timer on it would run nothing. Nothing changed. "
                  "{others} Call this again with its playbook_id.")
PICKED_ON = "{count} playbooks are called '{name}'. #{id} is the one whose timer was on, so it was switched off."
PICKED_STEPS = "{count} playbooks are called '{name}'. #{id} got the timer: it is the only one with steps."
Target = Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]


def has_timer(playbook: Any) -> bool:
    """A cron saved on the playbook, switched on or off."""
    sc = getattr(playbook, "schedule_config", None) or {}
    return isinstance(sc, dict) and sc.get("type") == "cron" and bool(sc.get("cron_expression"))


def timer_on(playbook: Any) -> bool:
    """A cron saved on the playbook and switched on (what the scheduler fires)."""
    from services.playbook_scheduler import is_live_cron

    return is_live_cron(getattr(playbook, "schedule_config", None))


def timer_said(playbook: Any) -> str:
    """The playbook's saved time in plain words: "weekdays at 09:00 (Europe/London)"."""
    sc = playbook.schedule_config or {}
    return plain_cron(sc.get("cron_expression"), sc.get("timezone"))


def failed(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def by_id(db: Session, workspace_id: Any, playbook_id: Any) -> Optional[Any]:
    """This workspace's playbook with this id, or None."""
    from core.models.core import WorkflowTemplate

    return db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == playbook_id, WorkflowTemplate.workspace_id == workspace_id).first()


def namesakes(db: Session, workspace_id: Any, playbook: Any) -> List[Any]:
    """The other playbooks with this one's name (trimmed, any case)."""
    return [p for p in playbooks_called(db, workspace_id, playbook.name)
            if p.id != playbook.id and str(p.name).strip().lower() == str(playbook.name).strip().lower()]


def target(db: Session, workspace_id: Any, params: Dict[str, Any], off: bool) -> Target:
    """(playbook, refusal, why it was picked). (None, None, None) leaves the call to
    the tool as it was: no playbook named, or several equally likely ones."""
    said = params.get("playbook_id")
    if isinstance(said, str) and said.strip() and not said.strip().isdigit():
        params = {**params, "playbook_id": None, "playbook_name": params.get("playbook_name") or said.strip()}
        said = None
    if said not in (None, ""):
        playbook = by_id(db, workspace_id, said)
        return (playbook, None, None) if playbook is not None else (None, failed(NOT_FOUND.format(said=f"#{said}")), None)
    name = params.get("playbook_name")
    if not name:
        return None, None, None
    found = playbooks_called(db, workspace_id, name)
    if not found:
        return None, failed(NOT_FOUND.format(said=f"called '{name}'")), None
    return (found[0], None, None) if len(found) == 1 else _of_several(found, str(name).strip(), off)


def _of_several(found: List[Any], name: str, off: bool) -> Target:
    """A name several playbooks match: off takes the only one whose timer is on, on
    the only one with steps; otherwise the tool's own refusal lists them. Names
    that only contain the one given are never chosen between."""
    if any(str(p.name).strip().lower() != name.lower() for p in found):
        return None, None, None
    if off:
        live = [p for p in found if timer_on(p)]
        if not live:
            return None, failed(NONE_ON.format(name=name)), None
        picked, why = (live[0], PICKED_ON) if len(live) == 1 else (None, None)
    else:
        staffed = [p for p in found if p.steps]
        picked, why = (staffed[0], PICKED_STEPS) if len(staffed) == 1 else (None, None)
    if picked is None:
        return None, None, None
    return picked, None, why.format(count=len(found), name=name, id=picked.id)


def wrong_namesake(db: Session, workspace_id: Any, playbook: Any, off: bool) -> Optional[Dict[str, Any]]:
    """The refusal when another playbook of the same name is plainly the one meant."""
    if off and not timer_on(playbook):
        live = [p for p in namesakes(db, workspace_id, playbook) if timer_on(p)]
        if live:
            others = " ".join(f"#{p.id} '{p.name}' has its timer on ({timer_said(p)})." for p in live)
            return failed(NAMESAKE_TIMER.format(id=playbook.id, name=playbook.name, others=others))
    if not off and not playbook.steps:
        staffed = [p for p in namesakes(db, workspace_id, playbook) if p.steps]
        if staffed:
            others = " ".join(f"#{p.id} '{p.name}' has {len(p.steps)} step(s)." for p in staffed)
            return failed(NAMESAKE_STEPS.format(id=playbook.id, name=playbook.name, others=others))
    return None


__all__ = ["NO_TIMER", "by_id", "failed", "has_timer", "target", "timer_on", "timer_said", "wrong_namesake"]
