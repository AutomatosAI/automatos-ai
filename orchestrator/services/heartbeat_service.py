"""
Heartbeat Service (PRD-55: Autonomous Assistant Platform).

Schedules and executes periodic heartbeat checks for both the
orchestrator (workspace-level health) and individual agents.
Uses APScheduler with an optional Redis job store for persistence.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limiting semaphores (per-workspace and per-agent)
# ---------------------------------------------------------------------------
_workspace_semaphores: Dict[str, asyncio.Semaphore] = {}
_agent_locks: Dict[str, asyncio.Lock] = {}

MAX_CONCURRENT_PER_WORKSPACE = 5


def _get_workspace_semaphore(workspace_id: str) -> asyncio.Semaphore:
    if workspace_id not in _workspace_semaphores:
        _workspace_semaphores[workspace_id] = asyncio.Semaphore(MAX_CONCURRENT_PER_WORKSPACE)
    return _workspace_semaphores[workspace_id]


def _get_agent_lock(agent_id: str) -> asyncio.Lock:
    if agent_id not in _agent_locks:
        _agent_locks[agent_id] = asyncio.Lock()
    return _agent_locks[agent_id]


class HeartbeatService:
    """Manages scheduled heartbeat jobs for orchestrator and agents."""

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_scheduler(self) -> AsyncIOScheduler:
        """Lazy-init the scheduler, optionally backed by Redis."""
        if self._scheduler is not None:
            return self._scheduler

        jobstores: Dict[str, Any] = {}
        redis_url = config.REDIS_URL
        if redis_url:
            try:
                from apscheduler.jobstores.redis import RedisJobStore

                jobstores["default"] = RedisJobStore.from_url(redis_url)
                logger.info("Heartbeat scheduler using Redis job store")
            except Exception as exc:
                logger.warning(
                    "Could not initialise Redis job store, falling back to memory: %s",
                    exc,
                )

        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores or None,
            job_defaults={"coalesce": True, "max_instances": 1},
        )
        return self._scheduler

    async def start(self) -> None:
        """Load heartbeat configs and start the scheduler."""
        if self._started:
            return

        scheduler = self._init_scheduler()

        # Load workspace-level heartbeat configs
        try:
            from core.database.database import SessionLocal
            from core.models.system_settings import SystemSetting

            db = SessionLocal()
            try:
                # Look for heartbeat settings per workspace
                settings = (
                    db.query(SystemSetting)
                    .filter(SystemSetting.category == "heartbeat")
                    .all()
                )
                for setting in settings:
                    try:
                        workspace_id = setting.key  # key stores workspace_id
                        cfg = setting.value if isinstance(setting.value, dict) else {}
                        if cfg.get("enabled"):
                            self.schedule_orchestrator_heartbeat(workspace_id, cfg)
                    except Exception as exc:
                        logger.warning(
                            "Failed to schedule orchestrator heartbeat for setting %s: %s",
                            setting.id,
                            exc,
                        )

                # Load per-agent heartbeat configs
                agent_settings = (
                    db.query(SystemSetting)
                    .filter(SystemSetting.category == "agent_heartbeat")
                    .all()
                )
                for setting in agent_settings:
                    try:
                        agent_id = setting.key
                        cfg = setting.value if isinstance(setting.value, dict) else {}
                        if cfg.get("enabled"):
                            workspace_id = cfg.get("workspace_id", "")
                            self.schedule_agent_heartbeat(agent_id, workspace_id, cfg)
                    except Exception as exc:
                        logger.warning(
                            "Failed to schedule agent heartbeat for setting %s: %s",
                            setting.id,
                            exc,
                        )
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Could not load heartbeat configs from DB: %s", exc)

        scheduler.start()
        self._started = True
        logger.info("HeartbeatService started")

    async def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler and self._started:
            self._scheduler.shutdown(wait=True)
            self._started = False
            logger.info("HeartbeatService stopped")

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def schedule_orchestrator_heartbeat(
        self, workspace_id: str, cfg: Dict[str, Any]
    ) -> str:
        """Schedule a recurring orchestrator heartbeat for *workspace_id*.

        Config keys used:
            interval_minutes (int): cron interval, default 30
            timezone (str): IANA tz, default UTC
            active_hours_start (int): hour 0-23
            active_hours_end (int): hour 0-23
        """
        scheduler = self._init_scheduler()
        interval = int(cfg.get("interval_minutes", 30))
        tz = cfg.get("timezone", "UTC")
        job_id = f"orchestrator_heartbeat_{workspace_id}"

        scheduler.add_job(
            self._orchestrator_tick,
            trigger=CronTrigger(minute=f"*/{interval}", timezone=tz),
            id=job_id,
            replace_existing=True,
            kwargs={"workspace_id": workspace_id, "_config": cfg},
        )
        logger.info(
            "Scheduled orchestrator heartbeat for workspace %s every %d min (%s)",
            workspace_id,
            interval,
            tz,
        )
        return job_id

    def schedule_agent_heartbeat(
        self, agent_id: str, workspace_id: str, cfg: Dict[str, Any]
    ) -> str:
        """Schedule a recurring heartbeat for a specific agent."""
        scheduler = self._init_scheduler()
        interval = int(cfg.get("interval_minutes", 30))
        tz = cfg.get("timezone", "UTC")
        job_id = f"agent_heartbeat_{agent_id}"

        scheduler.add_job(
            self._agent_tick,
            trigger=CronTrigger(minute=f"*/{interval}", timezone=tz),
            id=job_id,
            replace_existing=True,
            kwargs={
                "agent_id": agent_id,
                "workspace_id": workspace_id,
                "_config": cfg,
            },
        )
        logger.info(
            "Scheduled agent heartbeat for agent %s every %d min (%s)",
            agent_id,
            interval,
            tz,
        )
        return job_id

    def unschedule_heartbeat(self, job_id: str) -> bool:
        """Remove a scheduled heartbeat job. Returns True if removed."""
        if not self._scheduler:
            return False
        try:
            self._scheduler.remove_job(job_id)
            logger.info("Removed heartbeat job %s", job_id)
            return True
        except Exception:
            logger.debug("Job %s not found for removal", job_id)
            return False

    # ------------------------------------------------------------------
    # Active-hours guard
    # ------------------------------------------------------------------

    @staticmethod
    def _is_within_active_hours(cfg: Dict[str, Any]) -> bool:
        """Return True if the current hour falls within configured active hours."""
        start = cfg.get("active_hours_start")
        end = cfg.get("active_hours_end")
        if start is None or end is None:
            return True  # no restriction

        tz_name = cfg.get("timezone", "UTC")
        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo(tz_name))
        except Exception:
            now = datetime.now(timezone.utc)

        hour = now.hour
        if start <= end:
            return start <= hour < end
        else:
            # Wraps midnight (e.g. 22 -> 6)
            return hour >= start or hour < end

    # ------------------------------------------------------------------
    # Tick implementations
    # ------------------------------------------------------------------

    async def _orchestrator_tick(
        self, workspace_id: str, _config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Periodic orchestrator heartbeat: health checks, unrouted events, summary."""
        cfg = _config or {}
        if not self._is_within_active_hours(cfg):
            logger.debug("Orchestrator heartbeat skipped (outside active hours)")
            return

        sem = _get_workspace_semaphore(workspace_id)
        if not sem._value:  # all slots taken
            logger.warning(
                "Workspace %s at max concurrent heartbeats, skipping orchestrator tick",
                workspace_id,
            )
            return

        async with sem:
            status = "success"
            findings: Dict[str, Any] = {}
            actions_taken: List[str] = []
            tokens_used = 0
            cost = 0.0

            try:
                from core.database.database import SessionLocal
                from core.models.core import Agent
                from core.models.routing import UnroutedEvent

                db = SessionLocal()
                try:
                    # 1. Check agent health
                    agents = (
                        db.query(Agent)
                        .filter(Agent.workspace_id == workspace_id)
                        .all()
                    )
                    agent_summary = []
                    for agent in agents:
                        agent_summary.append(
                            {
                                "id": agent.id,
                                "name": agent.name,
                                "status": getattr(agent, "status", "unknown"),
                            }
                        )
                    findings["agents"] = agent_summary

                    # 2. Review unrouted events
                    unrouted = (
                        db.query(UnroutedEvent)
                        .filter(UnroutedEvent.workspace_id == workspace_id)
                        .order_by(UnroutedEvent.created_at.desc())
                        .limit(20)
                        .all()
                    )
                    if unrouted:
                        findings["unrouted_events"] = len(unrouted)
                        actions_taken.append(
                            f"Found {len(unrouted)} unrouted events"
                        )
                finally:
                    db.close()

                # 3. Summarise via LLM
                try:
                    from core.llm import create_llm_manager

                    mgr = create_llm_manager(service_name="orchestrator")
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are the Automatos orchestrator heartbeat. "
                                "Summarize the workspace health."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Workspace {workspace_id} status:\n"
                                f"Agents: {len(agent_summary)}\n"
                                f"Unrouted events: {findings.get('unrouted_events', 0)}\n"
                                "Provide a brief health summary and any recommended actions."
                            ),
                        },
                    ]
                    response = await mgr.generate_response(messages)
                    findings["summary"] = response.content
                    if response.usage:
                        tokens_used = response.usage.get("total_tokens", 0)
                        # Rough cost estimate
                        cost = tokens_used * 0.000003
                    actions_taken.append("Generated health summary")
                except Exception as llm_exc:
                    logger.warning("LLM call in orchestrator tick failed: %s", llm_exc)
                    findings["summary"] = "LLM unavailable"

            except Exception as exc:
                logger.error("Orchestrator tick error for workspace %s: %s", workspace_id, exc)
                status = "error"
                findings["error"] = str(exc)

            await self._store_heartbeat_result(
                source_type="orchestrator",
                source_id=workspace_id,
                workspace_id=workspace_id,
                status=status,
                findings=findings,
                actions_taken=actions_taken,
                tokens_used=tokens_used,
                cost=cost,
            )

    async def _agent_tick(
        self,
        agent_id: str,
        workspace_id: str,
        _config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Periodic agent heartbeat: load agent, build context, call LLM."""
        cfg = _config or {}
        if not self._is_within_active_hours(cfg):
            logger.debug("Agent %s heartbeat skipped (outside active hours)", agent_id)
            return

        lock = _get_agent_lock(str(agent_id))
        if lock.locked():
            logger.debug("Agent %s heartbeat already running, skipping", agent_id)
            return

        sem = _get_workspace_semaphore(workspace_id)
        if not sem._value:
            logger.warning(
                "Workspace %s at max concurrent heartbeats, skipping agent %s",
                workspace_id,
                agent_id,
            )
            return

        async with sem:
            async with lock:
                status = "success"
                findings: Dict[str, Any] = {}
                actions_taken: List[str] = []
                tokens_used = 0
                cost = 0.0

                try:
                    from core.database.database import SessionLocal
                    from core.models.core import Agent

                    db = SessionLocal()
                    try:
                        agent = db.query(Agent).filter(Agent.id == int(agent_id)).first()
                        if not agent:
                            logger.warning("Agent %s not found for heartbeat", agent_id)
                            status = "error"
                            findings["error"] = "Agent not found"
                            await self._store_heartbeat_result(
                                "agent", str(agent_id), workspace_id,
                                status, findings, actions_taken, 0, 0.0,
                            )
                            return

                        agent_name = agent.name
                        agent_persona = getattr(agent, "persona", "") or ""
                        agent_instructions = getattr(agent, "instructions", "") or ""
                    finally:
                        db.close()

                    # Build heartbeat prompt
                    heartbeat_prompt = cfg.get(
                        "prompt",
                        (
                            "Review your recent activity. Summarize what you've done, "
                            "flag any issues, and suggest next actions."
                        ),
                    )

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                f"You are {agent_name}. {agent_persona}\n"
                                f"Instructions: {agent_instructions}\n\n"
                                "This is your periodic heartbeat check."
                            ),
                        },
                        {"role": "user", "content": heartbeat_prompt},
                    ]

                    model = cfg.get("model")

                    from core.llm import create_llm_manager

                    mgr = create_llm_manager(service_name="orchestrator")
                    response = await mgr.generate_response(messages)

                    findings["response"] = response.content
                    if response.usage:
                        tokens_used = response.usage.get("total_tokens", 0)
                        cost = tokens_used * 0.000003
                    actions_taken.append("Completed agent heartbeat check")

                except Exception as exc:
                    logger.error(
                        "Agent tick error for agent %s: %s", agent_id, exc
                    )
                    status = "error"
                    findings["error"] = str(exc)

                await self._store_heartbeat_result(
                    source_type="agent",
                    source_id=str(agent_id),
                    workspace_id=workspace_id,
                    status=status,
                    findings=findings,
                    actions_taken=actions_taken,
                    tokens_used=tokens_used,
                    cost=cost,
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    async def _store_heartbeat_result(
        source_type: str,
        source_id: str,
        workspace_id: str,
        status: str,
        findings: Dict[str, Any],
        actions_taken: List[Any],
        tokens_used: int,
        cost: float,
    ) -> None:
        """Persist a heartbeat result row."""
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import Column, DateTime, Float, Integer, String, Text, text
            from sqlalchemy.dialects.postgresql import JSONB

            db = SessionLocal()
            try:
                db.execute(
                    text(
                        "INSERT INTO heartbeat_results "
                        "(source_type, source_id, workspace_id, status, findings, "
                        "actions_taken, tokens_used, cost, created_at) "
                        "VALUES (:source_type, :source_id, :workspace_id::uuid, :status, "
                        ":findings::jsonb, :actions_taken::jsonb, :tokens_used, :cost, NOW())"
                    ),
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "workspace_id": workspace_id,
                        "status": status,
                        "findings": __import__("json").dumps(findings),
                        "actions_taken": __import__("json").dumps(actions_taken),
                        "tokens_used": tokens_used,
                        "cost": cost,
                    },
                )
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            logger.error("Failed to store heartbeat result: %s", exc)

    # ------------------------------------------------------------------
    # Public manual triggers
    # ------------------------------------------------------------------

    async def run_orchestrator_heartbeat(self, workspace_id: str) -> Dict[str, Any]:
        """Manually trigger an orchestrator heartbeat and return findings."""
        await self._orchestrator_tick(workspace_id)
        # Return the most recent result
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                row = db.execute(
                    text(
                        "SELECT id, status, findings, actions_taken, tokens_used, cost, created_at "
                        "FROM heartbeat_results "
                        "WHERE source_type = 'orchestrator' AND source_id = :wid "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"wid": workspace_id},
                ).fetchone()
                if row:
                    return {
                        "id": row[0],
                        "status": row[1],
                        "findings": row[2],
                        "actions_taken": row[3],
                        "tokens_used": row[4],
                        "cost": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                    }
            finally:
                db.close()
        except Exception as exc:
            logger.error("Failed to fetch orchestrator heartbeat result: %s", exc)
        return {"status": "completed"}

    async def run_agent_heartbeat(
        self, agent_id: str, workspace_id: str
    ) -> Dict[str, Any]:
        """Manually trigger an agent heartbeat and return findings."""
        await self._agent_tick(agent_id, workspace_id)
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                row = db.execute(
                    text(
                        "SELECT id, status, findings, actions_taken, tokens_used, cost, created_at "
                        "FROM heartbeat_results "
                        "WHERE source_type = 'agent' AND source_id = :aid "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"aid": str(agent_id)},
                ).fetchone()
                if row:
                    return {
                        "id": row[0],
                        "status": row[1],
                        "findings": row[2],
                        "actions_taken": row[3],
                        "tokens_used": row[4],
                        "cost": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                    }
            finally:
                db.close()
        except Exception as exc:
            logger.error("Failed to fetch agent heartbeat result: %s", exc)
        return {"status": "completed"}

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return information about all active heartbeat jobs."""
        if not self._scheduler:
            return {"running": False, "jobs": []}

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_at": (
                        job.next_run_time.isoformat() if job.next_run_time else None
                    ),
                    "trigger": str(job.trigger),
                }
            )
        return {"running": self._started, "jobs": jobs}


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[HeartbeatService] = None


def get_heartbeat_service() -> HeartbeatService:
    """Return the singleton HeartbeatService instance."""
    global _instance
    if _instance is None:
        _instance = HeartbeatService()
    return _instance
