"""PRD-268 §6.2.1, FR-4, K7, K10 — the heartbeat bridge: check-ins beside or instead of today's ticks.

The heartbeat scheduler runs the bridge's jobs (``agent_job``, ``orchestrator_job``), not the
ticks. At every fire a job reads the workspace's proactive mode
(``services.proactive.switches.effective_mode``) and whether this heartbeat converts:

* an agent's, when its mode is ``check_in`` (``modules.proactive.heartbeats``, K10) and
  it is not a Claude Code agent, whose heartbeat stays its board ticket in W1;
* the orchestrator's, at ``proactive_level`` silent or notify; act_notify and autonomous
  keep today's tick.

Then:

* **off** (or a heartbeat that does not convert): today's tick only;
* **shadow**: today's tick, then the check-in, which records its run and suggestions and
  rings nothing;
* **live**: the check-in only. No tick runs, so no ticket is filed, ``report_to`` routes
  nothing and no per-tick report is written; a ``heartbeat_complete`` bell row is written
  only when the run made a suggestion or started an error streak
  (``services.proactive.heartbeat_bell``), and never for the orchestrator at silent.

A check-in runs within the heartbeat's active hours, as its tick does; the runner checks
the switches, credits, spend and caps at its start. A mode that cannot be read is off, so
today's tick runs. The ticks are unchanged, and the Run-now routes still run them (K10).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from functools import partial
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.proactive.heartbeats import (
    CHECK_IN_LEVELS,
    DEFAULT_LEVEL,
    HEARTBEAT_CHECK_IN,
    SILENT_LEVEL,
    heartbeat_mode,
)
from services.proactive.switches import MODE_LIVE, MODE_OFF, MODE_SHADOW, effective_mode

logger = logging.getLogger(__name__)

__all__ = [
    "CHECK_IN_FAILED",
    "HeartbeatJobs",
    "OUTSIDE_ACTIVE_HOURS",
    "agent_job",
    "orchestrator_job",
]

# The skip result a tick returns outside active hours (HeartbeatService._agent_tick).
OUTSIDE_ACTIVE_HOURS = "outside_active_hours"
CHECK_IN_FAILED = "check_in_failed"
# The key that names a check-in's result beside a shadow tick's.
CHECK_IN_KEY = "check_in"

Result = Dict[str, Any]
CheckIn = Callable[..., Awaitable[Result]]


@contextmanager
def _session() -> Iterator[Session]:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _runs_on_the_cli(db: Session, agent_id: int, workspace_id: Any) -> bool:
    """Whether the agent is a Claude Code agent, whose heartbeat stays its ticket (K10)."""
    from core.cli_runtime import RUNTIME_CLI, runtime_kind_of
    from core.models.core import Agent

    row = db.query(Agent.configuration).filter(
        Agent.id == agent_id, Agent.workspace_id == UUID(str(workspace_id))).first()
    return runtime_kind_of(row[0] if row is not None else None) == RUNTIME_CLI


def _agent_mode(agent_id: int, workspace_id: Any, hb_config: Dict[str, Any]) -> str:
    """The mode an agent's heartbeat runs in: the workspace's for a ``check_in`` heartbeat
    of an agent that is not on the CLI, else off. A read that fails is off."""
    if heartbeat_mode(hb_config) != HEARTBEAT_CHECK_IN:
        return MODE_OFF
    try:
        with _session() as db:
            mode = effective_mode(db, UUID(str(workspace_id)))
            if mode == MODE_OFF or _runs_on_the_cli(db, agent_id, workspace_id):
                return MODE_OFF
            return mode
    except Exception:  # noqa: BLE001 — a mode that cannot be read is off: today's tick runs
        logger.exception("[proactive] the mode of agent %s's heartbeat could not be read: off", agent_id)
        return MODE_OFF


def _proactive_level(workspace_id: Any, hb_config: Dict[str, Any]) -> str:
    """The orchestrator's proactive level, read as its tick reads it: the heartbeat's own,
    else the workspace's orchestrator settings, else notify."""
    from consumers.chatbot.personality import load_orchestrator_settings

    stated = hb_config.get("proactive_level")
    if stated:
        return str(stated)
    return str(load_orchestrator_settings(str(workspace_id)).get("proactive_level") or DEFAULT_LEVEL)


def _orchestrator_mode(workspace_id: Any, hb_config: Dict[str, Any]) -> Tuple[str, str]:
    """The mode the orchestrator heartbeat runs in, and its proactive level: the workspace's
    mode at silent or notify, else off. A read that fails is off."""
    try:
        level = _proactive_level(workspace_id, hb_config)
        if level not in CHECK_IN_LEVELS:
            return MODE_OFF, level
        with _session() as db:
            return effective_mode(db, UUID(str(workspace_id))), level
    except Exception:  # noqa: BLE001 — a mode that cannot be read is off: today's tick runs
        logger.exception("[proactive] the mode of %s's orchestrator heartbeat could not be read: off", workspace_id)
        return MODE_OFF, DEFAULT_LEVEL


def _summary(run: Any, rang: bool) -> Result:
    """A check-in's result, as a job returns it."""
    return {"source": CHECK_IN_KEY, "run_id": str(run.run_id), "outcome": run.outcome,
            "suggestions": run.suggestions, "error_class": run.error_class, "rang": rang}


async def _agent_check_in(agent_id: int, workspace_id: Any, hb_config: Dict[str, Any], *, ring: bool) -> Result:
    from services.proactive.heartbeat_bell import ring_if_new
    from services.proactive.runner import run_agent_check_in

    run = await run_agent_check_in(agent_id, workspace_id, hb_config)
    return _summary(run, await ring_if_new(run, workspace_id, int(agent_id)) if ring else False)


async def _workspace_check_in(workspace_id: Any, *, ring: bool) -> Result:
    from services.proactive.heartbeat_bell import ring_if_new
    from services.proactive.runner import run_workspace_check_in

    run = await run_workspace_check_in(workspace_id)
    return _summary(run, await ring_if_new(run, workspace_id, None) if ring else False)


class HeartbeatJobs:
    """The scheduler's jobs for one heartbeat service. The scheduler holds their bound
    methods, as it held the ticks' (``service._agent_tick``), so a job store keeps them no
    differently."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def agent(self, agent_id: int, workspace_id: str, hb_config: Dict[str, Any]) -> Optional[Result]:
        """One fire of an agent's heartbeat: today's tick, its check-in, or both, by mode."""
        mode = await asyncio.to_thread(_agent_mode, agent_id, workspace_id, hb_config)
        tick = partial(self._service._agent_tick, agent_id, workspace_id, hb_config)
        check_in = partial(_agent_check_in, agent_id, workspace_id, hb_config)
        return await self._bridged(mode, workspace_id, hb_config, tick, check_in, ring=True)

    async def orchestrator(self, workspace_id: str, hb_config: Dict[str, Any]) -> Optional[Result]:
        """One fire of the orchestrator heartbeat: today's tick, Auto's workspace check-in,
        or both, by mode; at silent the check-in rings nothing, as the tick did not."""
        mode, level = await asyncio.to_thread(_orchestrator_mode, workspace_id, hb_config)
        tick = partial(self._service._orchestrator_tick, workspace_id, hb_config)
        check_in = partial(_workspace_check_in, workspace_id)
        return await self._bridged(mode, workspace_id, hb_config, tick, check_in, ring=level != SILENT_LEVEL)

    async def _bridged(self, mode: str, workspace_id: str, hb_config: Dict[str, Any],
                       tick: Callable[[], Awaitable[Optional[Result]]], check_in: CheckIn,
                       *, ring: bool) -> Optional[Result]:
        if mode == MODE_LIVE:
            return await self._check_in(workspace_id, hb_config, partial(check_in, ring=ring))
        result = await tick()
        if mode != MODE_SHADOW:
            return result
        shadow = await self._check_in(workspace_id, hb_config, partial(check_in, ring=False))
        return {**(result or {}), CHECK_IN_KEY: shadow}

    async def _check_in(self, workspace_id: str, hb_config: Dict[str, Any], check_in: CheckIn) -> Result:
        """Run a check-in within the heartbeat's active hours. A failure is logged and is the
        job's result; it never reaches the scheduler or a shadow's tick."""
        if not await self._service._is_within_active_hours(hb_config, workspace_id):
            return {"status": "skipped", "reason": OUTSIDE_ACTIVE_HOURS}
        try:
            return await check_in()
        except Exception:  # noqa: BLE001 — logged; the job's result says the check-in failed
            logger.exception("[proactive] a heartbeat's check-in in %s failed", workspace_id)
            return {"status": "error", "reason": CHECK_IN_FAILED}


def agent_job(service: Any) -> Callable[[int, str, Dict[str, Any]], Awaitable[Optional[Result]]]:
    """What the scheduler runs for an agent's heartbeat, in place of ``service._agent_tick``."""
    return HeartbeatJobs(service).agent


def orchestrator_job(service: Any) -> Callable[[str, Dict[str, Any]], Awaitable[Optional[Result]]]:
    """What the scheduler runs for the orchestrator heartbeat, in place of
    ``service._orchestrator_tick``."""
    return HeartbeatJobs(service).orchestrator
