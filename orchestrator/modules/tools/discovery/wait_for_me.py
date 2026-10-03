"""F242: the owner's "wait for me" on Auto's playbook tools.

``wait_for_me`` on platform_create_playbook or platform_update_playbook is the
playbook's setting: every run's card waits for the owner in Review. On
platform_execute_playbook it is that run's own request. The handlers do their
own work first; these decorators then keep the setting where
services/playbook_wait.py reads it.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict

from services.playbook_wait import CLOSES_ITSELF, OWNER_REVIEWS, WAIT_FOR_ME

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _without(params: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in (params or {}).items() if key != WAIT_FOR_ME}


def keeps_wait_for_me(handler: Handler) -> Handler:
    """platform_create_playbook / platform_execute_playbook: once the playbook
    (or the run) exists, it keeps the owner's ``wait_for_me``."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        wants = (params or {}).get(WAIT_FOR_ME)
        out = await handler(db, workspace_id, _without(params))
        if not isinstance(wants, bool) or not isinstance(out, dict) or out.get("success") is not True:
            return out
        if out.get("execution_id"):
            return {**out, **_on_the_run(db, workspace_id, out["execution_id"], wants)}
        playbook_id = (out.get("playbook") or {}).get("id")
        return {**out, **_on_the_playbook(db, workspace_id, playbook_id, wants)} if playbook_id else out
    return wrapped


def updates_wait_for_me(handler: Handler) -> Handler:
    """platform_update_playbook: the setting is a change of its own, so a call that
    changes nothing else is no longer "nothing changed"."""
    from .action_registry import nothing_changed

    unchanged = nothing_changed("platform_update_playbook", "playbook_id")

    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        wants = (params or {}).get(WAIT_FOR_ME)
        out = await handler(db, workspace_id, _without(params))
        if not isinstance(wants, bool) or not isinstance(out, dict):
            return out
        only_this = out.get("error") == unchanged
        if out.get("success") is not True and not only_this:
            return out  # the update was refused: the setting is not changed either
        kept = _on_the_playbook(db, workspace_id, params.get("playbook_id"), wants)
        if kept and only_this:
            out = {"success": True, "playbook_id": out.get("playbook_id"),
                   "message": "Every run of this playbook now " + ("waits for the owner's check."
                                                                   if wants else "closes itself.")}
        return {**out, **kept}
    return wrapped


def _on_the_playbook(db: Any, workspace_id: Any, playbook_id: Any, wants: bool) -> Dict[str, Any]:
    """The playbook's setting, in its ``execution_config`` (its caller commits it with
    the playbook's other changes). Nothing when the playbook is not this workspace's."""
    from core.models.core import WorkflowTemplate

    playbook = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == playbook_id, WorkflowTemplate.workspace_id == workspace_id).first()
    if playbook is None:
        return {}
    playbook.execution_config = {**(playbook.execution_config or {}), WAIT_FOR_ME: wants}
    db.flush()
    return {WAIT_FOR_ME: wants}


def _on_the_run(db: Any, workspace_id: Any, execution_id: str, wants: bool) -> Dict[str, Any]:
    """One run's request, and its card's when it is made already. Committed: the
    run works on a session of its own, and reads this when it makes its card."""
    from core.models.core import BoardTask, RecipeExecution

    run = db.query(RecipeExecution).filter(
        RecipeExecution.execution_id == execution_id, RecipeExecution.workspace_id == workspace_id).first()
    if run is None:
        return {}
    run.execution_metadata = {**(run.execution_metadata or {}), WAIT_FOR_ME: wants}
    card = db.query(BoardTask).filter(BoardTask.source_type == "recipe", BoardTask.source_id == execution_id,
                                      BoardTask.workspace_id == workspace_id).first()
    if card is not None:
        card.review_mode = OWNER_REVIEWS if wants else CLOSES_ITSELF
    db.commit()
    return {WAIT_FOR_ME: wants}
