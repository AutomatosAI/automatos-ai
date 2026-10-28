"""PRD-268 US-C04 (§6.2.2, K13) — the heartbeat bell after a live check-in.

A heartbeat that runs as a check-in (``services.proactive.heartbeat_bridge``, live mode)
rings ``heartbeat_complete`` only when its run made a suggestion or failed for the first
time in a streak: an error of a class the check-in's run before it (skips passed over,
``modules.proactive.streaks``) did not fail with. A silent run, a skipped run and a
streak's later failures ring nothing, where every tick used to ring.

The row goes through ``NotificationDispatcher``, as the tick's did: the workspace's
preferences choose where (``heartbeat_complete`` is not a push type), and it links to the
Command Center feed (``link_type`` heartbeat) with the run's id. Its copy is the server's,
from ``STRINGS``, built from structured fields only (the agent's name, the count, the run's
error class); no word the model wrote reaches it. Background copy is English (K13).

The streak's own suggestion, its count and the doubling interval are US-C05's.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.proactive import RUN_ERROR, RUN_SUGGESTED
from modules.proactive.streaks import consecutive_failures
from services.proactive.runs import KIND_AGENT, KIND_WORKSPACE, previous_finished

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO_NAME",
    "BACKGROUND_LOCALE",
    "BELL_EVENT",
    "BELL_LINK",
    "STRINGS",
    "bell_copy",
    "ring_if_new",
    "rings",
    "starts_a_streak",
]

BELL_EVENT = "heartbeat_complete"
# The bell links a heartbeat's notice to the Command Center feed (notification-bell.tsx linkFor).
BELL_LINK = "heartbeat"
# Auto's workspace check-in rings as Auto (the orchestrator heartbeat it replaces).
AUTO_NAME = "Auto"
BACKGROUND_LOCALE = "en"
STATUS_OK = "ok"
STATUS_ERROR = "error"
# Today's name for an agent whose row is gone (_dispatch_heartbeat_notification).
UNNAMED_AGENT = "agent-{agent_id}"

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        "title": "{name} Heartbeat",
        "suggested_one": "{name}'s check-in suggested a next step.",
        "suggested_many": "{name}'s check-in suggested {count} next steps.",
        "failing": "{name}'s check-in failed ({error}).",
    }),
    "pt-BR": MappingProxyType({
        "title": "Heartbeat de {name}",
        "suggested_one": "O check-in de {name} sugeriu um próximo passo.",
        "suggested_many": "O check-in de {name} sugeriu {count} próximos passos.",
        "failing": "O check-in de {name} falhou ({error}).",
    }),
})


@contextmanager
def _session() -> Iterator[Session]:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _kind(agent_id: Optional[int]) -> str:
    return KIND_WORKSPACE if agent_id is None else KIND_AGENT


def starts_a_streak(db: Session, workspace_id: UUID, agent_id: Optional[int], run: Any) -> bool:
    """Whether ``run`` (``outcome``, ``error_class``, ``run_id``) failed, and is the first
    failure of its streak."""
    if run.outcome != RUN_ERROR:
        return False
    before = previous_finished(db, workspace_id, _kind(agent_id), agent_id, run.run_id)
    runs = (run,) if before is None else (run, before)
    return consecutive_failures(runs, run.error_class) == 1


def rings(db: Session, workspace_id: UUID, agent_id: Optional[int], run: Any) -> bool:
    """Whether a live check-in's run rings the bell: it made a suggestion or started a streak."""
    return run.outcome == RUN_SUGGESTED or starts_a_streak(db, workspace_id, agent_id, run)


def bell_copy(run: Any, name: str, locale: str = BACKGROUND_LOCALE) -> Tuple[str, str, str]:
    """The bell row's title, message and status, from the strings table."""
    table = STRINGS.get(locale, STRINGS[BACKGROUND_LOCALE])
    title = table["title"].format(name=name)
    if run.outcome == RUN_SUGGESTED:
        line = table["suggested_one" if run.suggestions == 1 else "suggested_many"]
        return title, line.format(name=name, count=run.suggestions), STATUS_OK
    return title, table["failing"].format(name=name, error=run.error_class), STATUS_ERROR


def _name(db: Session, workspace_id: UUID, agent_id: Optional[int]) -> str:
    """Who rings: Auto for the workspace check-in, else the agent, read in its workspace."""
    from core.models.core import Agent

    if agent_id is None:
        return AUTO_NAME
    row = db.query(Agent.name).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    return row[0] if row is not None and row[0] else UNNAMED_AGENT.format(agent_id=agent_id)


def _planned(db: Session, workspace_id: UUID, agent_id: Optional[int], run: Any) -> Optional[Tuple[str, ...]]:
    """The bell's name, title, message and status, or None when the run rings nothing."""
    if not rings(db, workspace_id, agent_id, run):
        return None
    name = _name(db, workspace_id, agent_id)
    return (name, *bell_copy(run, name))


async def ring_if_new(run: Any, workspace_id: Any, agent_id: Optional[int]) -> bool:
    """Ring ``heartbeat_complete`` for a live check-in's run (a runner ``CheckInResult``)
    when it made a suggestion or started a streak. True when it rang; a failure is logged
    and rings nothing, and never reaches the heartbeat."""
    from core.services.notification_dispatcher import NotificationDispatcher

    workspace = UUID(str(workspace_id))
    with _session() as db:
        try:
            planned = await asyncio.to_thread(_planned, db, workspace, agent_id, run)
            if planned is None:
                return False
            name, title, message, status = planned
            await NotificationDispatcher(db, workspace).dispatch(
                event_type=BELL_EVENT, title=title, message=message, link_type=BELL_LINK,
                link_id=str(run.run_id), agent_id=agent_id, agent_name=name, status=status,
            )
            db.commit()
        except Exception:  # noqa: BLE001 — logged; a bell that fails never fails the heartbeat
            db.rollback()
            logger.exception("[proactive] the bell for run %s in %s failed", run.run_id, workspace)
            return False
    logger.info("[proactive] run %s in %s rang %s (%s)", run.run_id, workspace, BELL_EVENT, run.outcome)
    return True
