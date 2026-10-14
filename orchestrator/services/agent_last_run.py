"""PRD-256 N15-6 (F427): whether an agent can run is read from what it did, not its flag alone.

Night 15, A371: "RESEARCHER and WRITER can't run right now… no connected machine serves
either", while both had finished work for the owner that night; WRITER's record still said
``inactive``. A CLI host claims a session agent's ticket whatever its stored ``status``, and
the host's presence (a heartbeat in the last 90 seconds) can lapse between beats, so neither
the flag nor the presence alone says whether the agent can run.

- An agent with a run in the last ``AGENT_RECENT_RUN_WINDOW_SECONDS`` (a ticket in progress,
  or one it finished into Review or Done) has shown it can run: the no-host reasons are
  dropped for it, its listing carries ``last_run``, and a CLI agent's stale ``inactive`` is
  listed as ``active``.
- The flag is refreshed when the agent runs: a CLI host's claim sets a claimed agent whose
  record says ``inactive`` to ``active``. The API lane refuses an inactive agent (the agent
  factory's active guard), so its claims leave the flag alone, and so does the listing.
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# A ticket in these statuses is a run: going now, or finished for the owner to see.
RAN_STATUSES = ("in_progress", "review", "done")
ACTIVE, INACTIVE = "active", "inactive"
LAST_RUN_KEY = "last_run"


def _cutoff() -> datetime:
    from config import config

    return datetime.now(timezone.utc) - timedelta(seconds=config.AGENT_RECENT_RUN_WINDOW_SECONDS)


def last_runs(db: Session, workspace_id: Any, agent_ids: Iterable[Any]) -> Dict[int, str]:
    """Each of ``agent_ids`` with a run inside the window, in this workspace: its last run's
    time (ISO). An agent with none is left out."""
    from core.models.core import BoardTask

    ids = sorted({int(i) for i in agent_ids if i is not None})
    if not ids:
        return {}
    ran_at = func.coalesce(BoardTask.completed_at, BoardTask.started_at)
    rows = (db.query(BoardTask.assigned_agent_id, func.max(ran_at))
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.assigned_agent_id.in_(ids),
                    BoardTask.status.in_(RAN_STATUSES), ran_at >= _cutoff())
            .group_by(BoardTask.assigned_agent_id).all())
    return {agent_id: at.isoformat() for agent_id, at in rows if at is not None}


def read_last_runs(db: Session, workspace_id: Any, agent_ids: Iterable[Any]) -> Dict[int, str]:
    """``last_runs``, or nothing when it can't be read: what is said stands on the rest."""
    try:
        return last_runs(db, workspace_id, agent_ids)
    except Exception:
        logger.exception("[agents] could not read the last runs of agents in workspace %s", workspace_id)
        return {}


def with_last_run(agent: Dict[str, Any], ran: Dict[int, str]) -> Dict[str, Any]:
    """``agent`` (a listing's dict) with its last run, and a CLI agent's stale ``inactive``
    shown as ``active``; unchanged when it has no run in the window. A new dict."""
    from core.cli_runtime import RUNTIME_CLI

    last = ran.get(agent.get("id"))
    if last is None:
        return agent
    stale = agent.get("runtime") == RUNTIME_CLI and agent.get("status") == INACTIVE
    return {**agent, LAST_RUN_KEY: last, **({"status": ACTIVE} if stale else {})}


def _refresh_flags(db: Session, claimed: List[Any]) -> None:
    """Set the claimed tickets' agents whose record says ``inactive`` to ``active``, each in its
    ticket's workspace."""
    from core.models.core import Agent

    ids = sorted({t.assigned_agent_id for t in claimed if t.assigned_agent_id is not None})
    workspaces = list({t.workspace_id for t in claimed})
    if not ids:
        return
    try:
        refreshed = (db.query(Agent)
                     .filter(Agent.id.in_(ids), Agent.workspace_id.in_(workspaces), Agent.status == INACTIVE)
                     .update({Agent.status: ACTIVE}, synchronize_session=False))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("[agents] could not refresh the active flag of claimed agents %s", ids)
        return
    if refreshed:
        logger.info("[agents] %d claimed agent(s) of %s were flagged inactive; now active", refreshed, ids)


def refreshes_the_flag(claim: Callable[..., List[Any]]) -> Callable[..., List[Any]]:
    """Wrap ``board_dispatcher.claim_tasks``: a CLI host's claim is its agent running, so the
    agent's stale ``inactive`` flag is refreshed. The claim stands either way."""
    from core.cli_runtime import RUNTIME_CLI

    @functools.wraps(claim)
    def wrapped(db: Session, *args: Any, **kwargs: Any) -> List[Any]:
        claimed = claim(db, *args, **kwargs)
        if claimed and kwargs.get("runtime") == RUNTIME_CLI:
            _refresh_flags(db, claimed)
        return claimed
    return wrapped


__all__ = ["LAST_RUN_KEY", "RAN_STATUSES", "last_runs", "read_last_runs", "refreshes_the_flag", "with_last_run"]
