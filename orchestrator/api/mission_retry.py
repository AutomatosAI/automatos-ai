"""F247 (night 7): the mission page's Resume retries a failed mission.

``POST /missions/{id}/resume`` answers only for a paused mission. For a failed one,
in the caller's workspace, ``modules.coordination.mission_retry.retry_failed``
first makes it a paused mission whose failed steps wait to run again. The
endpoint then resumes it as it resumes any paused mission, in the same
transaction.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable

from fastapi import HTTPException
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Endpoint = Callable[..., Awaitable[Any]]


def resume_retries_a_failed_mission(endpoint: Endpoint) -> Endpoint:
    """Wrap the resume endpoint: a failed mission is made ready to run again, then resumed."""
    @functools.wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        _retry_if_failed(kwargs.get("db"), kwargs.get("mission_id"), kwargs.get("ctx"))
        return await endpoint(*args, **kwargs)
    return wrapped


def _retry_if_failed(db: Session, mission_id: Any, ctx: Any) -> None:
    """Retry the caller's mission when it failed. Anything else, a missing mission
    included, is left for the endpoint to answer."""
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import RunState
    from modules.coordination.mission_retry import retry_failed
    from services.orchestration_state import ConflictError, InvalidTransitionError

    if db is None or ctx is None or mission_id is None:
        return
    run = db.query(OrchestrationRun).filter(OrchestrationRun.id == mission_id,
                                            OrchestrationRun.workspace_id == ctx.workspace_id).first()
    if run is None or RunState(run.state) != RunState.FAILED:
        return
    try:
        retry_failed(db, run, ctx.user.id or "unknown")
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to retry mission %s", mission_id)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error") from exc


__all__ = ["resume_retries_a_failed_mission"]
