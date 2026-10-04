"""F291 (night 8): the mission's own Approve takes the owner's note too.

The board's Approve on a mission's card keeps its note for every step
(api/board_mission_card). The mission's Approve (``POST /api/missions/{id}/approve``),
which the board's verdict buttons and the mission page call, took no note: #0214's
"Plan is fine. Use our real Thursday delivery day and keep the email short." had
nowhere to go. Its body now takes ``note``, kept on the mission
(``modules.coordination.owner_note``) in the approval's own transaction: the
endpoint's commit keeps it with the approval, and its rollback drops it with one.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

Endpoint = Callable[..., Awaitable[Any]]


def keeps_the_owners_note(endpoint: Endpoint) -> Endpoint:
    """Wrap the mission approve endpoint: a note in its body goes on the mission first."""
    @functools.wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        _keep_note(kwargs.get("db"), kwargs.get("mission_id"), kwargs.get("ctx"), kwargs.get("body"))
        return await endpoint(*args, **kwargs)
    return wrapped


def _keep_note(db: Session, mission_id: Any, ctx: Any, body: Any) -> None:
    """The note on the caller's mission while its plan waits. Anything else (no note,
    no such mission, a plan already decided) is left for the endpoint to answer."""
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import RunState
    from modules.coordination.owner_note import keep_owner_note

    note = getattr(body, "note", None)
    if db is None or ctx is None or mission_id is None or not note:
        return
    run = db.query(OrchestrationRun).filter(OrchestrationRun.id == mission_id,
                                            OrchestrationRun.workspace_id == ctx.workspace_id).first()
    if run is not None and run.state == RunState.AWAITING_APPROVAL.value:
        keep_owner_note(run, note)


__all__ = ["keeps_the_owners_note"]
