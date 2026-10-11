"""PRD-256 P256-FIX-RVW-45 (FX-010, FX-008): a playbook timed or deleted by name is bound to
one row before its card, so the card and the click name the same playbook.

The fourth fix-wave review: the card read the playbook by its exact name only (two namesakes
named none), while the click found it through ``timer_target.target``: the one playbook whose
name contains the one said, or, of exact namesakes, the only one with steps. {playbook_name:
'Stock Check', cron_expression: '0 8 * * 1'} showed "Set a playbook's timer
playbook_name='Stock Check'." with no lines, and the click timed 'Monday Stock Check'. A
playbook made or renamed before the click could change what it ran on.

Now the ask binds the name the click's own way (``assigned_subjects``' pattern): a timer by
``timer_target.target`` and its namesake check, a delete by ``delete_playbook``'s rule (the
one playbook with that whole name, never a system one). The call the card asks about names
the playbook by its id alone, the card reads that row, and the click runs on it. None, or
several, is refused before any grant, listing them (F091).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

PLAYBOOK_ID, PLAYBOOK_NAME = "playbook_id", "playbook_name"
SCHEDULE, DELETE = "platform_schedule_playbook", "platform_delete_playbook"
BINDS = frozenset({SCHEDULE, DELETE})
MAX_LISTED = 12
NAME_IT = "Provide playbook_id or playbook_name: nothing was asked or done."
NONE_CALLED = ("No playbook {said} in this workspace, so nothing was asked or done.{near} "
               "Call {action} again with the playbook_id of the one meant (platform_list_playbooks lists them).")
NEAR = " Playbooks whose name contains it: {listed}."
SEVERAL = ("{count} playbooks match '{name}': {listed}. Nothing was asked or done. Call {action} again "
           "with the playbook_id of the one the owner means.")
SYSTEM = "Playbook #{id} '{name}' is a system playbook and cannot be deleted. Nothing was asked or done."
Bound = Tuple[Optional[Any], Optional[Dict[str, Any]]]


def bound_to_the_playbook(db: Any, workspace_id: Any, action: str,
                          params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call naming its playbook by the row's own id alone, the refusal when it names
    none the click would act on, or several). Any other call is as it is."""
    from modules.tools.discovery.playbook_lookup import _name_in_id

    if action not in BINDS:
        return params, None
    said = _name_in_id(params)
    if said.get(PLAYBOOK_ID) in (None, "") and not str(said.get(PLAYBOOK_NAME) or "").strip():
        return params, {"success": False, "error": NAME_IT}
    playbook, refusal = _timed(db, workspace_id, said) if action == SCHEDULE else _deleted(db, workspace_id, said)
    if playbook is None:
        return params, refusal
    rest = {key: value for key, value in said.items() if key != PLAYBOOK_NAME}
    return {**rest, PLAYBOOK_ID: playbook.id}, None


def _timed(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Bound:
    """The playbook ``timer_target.target`` times, unless its namesake check refuses it."""
    from modules.tools.discovery.timer_off import is_off
    from modules.tools.discovery.timer_target import target, wrong_namesake

    off = is_off(params.get("enabled"))
    playbook, refusal, _why = target(db, workspace_id, params, off)
    if refusal is not None:
        return None, refusal
    if playbook is None:
        return None, _several(db, workspace_id, SCHEDULE, params)
    refusal = wrong_namesake(db, workspace_id, playbook, off)
    return (None, refusal) if refusal is not None else (playbook, None)


def _deleted(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Bound:
    """The playbook ``delete_playbook`` deletes: by id, else the one with that whole name."""
    from modules.tools.discovery.handlers_playbooks import _playbooks_called
    from modules.tools.discovery.timer_target import by_id

    said_id = params.get(PLAYBOOK_ID)
    if said_id not in (None, ""):
        playbook = by_id(db, workspace_id, said_id)
        if playbook is None:
            return None, _refused(NONE_CALLED.format(said=f"#{said_id}", near="", action=DELETE))
    else:
        named = _playbooks_called(db, workspace_id, params[PLAYBOOK_NAME])
        if len(named) != 1:
            return None, _several(db, workspace_id, DELETE, params, named)
        playbook = by_id(db, workspace_id, named[0].id)
    if getattr(playbook, "is_system", False):
        return None, _refused(SYSTEM.format(id=playbook.id, name=playbook.name))
    return playbook, None


def _several(db: Any, workspace_id: Any, action: str, params: Dict[str, Any],
             found: Optional[List[Any]] = None) -> Dict[str, Any]:
    """The refusal listing what the name matches: several, or none (with any whose name contains it)."""
    from modules.tools.discovery.playbook_lookup import playbooks_called

    name = str(params[PLAYBOOK_NAME]).strip()
    found = playbooks_called(db, workspace_id, name) if found is None else found
    if len(found) > 1:
        return _refused(SEVERAL.format(count=len(found), name=name, listed=_listed(found), action=action))
    near = playbooks_called(db, workspace_id, name)
    return _refused(NONE_CALLED.format(said=f"called '{name}'", near=NEAR.format(listed=_listed(near)) if near else "",
                                       action=action))


def _listed(playbooks: List[Any]) -> str:
    shown = ", ".join(f"#{playbook.id} '{playbook.name}'" for playbook in playbooks[:MAX_LISTED])
    extra = len(playbooks) - MAX_LISTED
    return f"{shown}, and {extra} more" if extra > 0 else shown


def _refused(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


__all__ = ["BINDS", "bound_to_the_playbook"]
