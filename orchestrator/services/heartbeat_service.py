"""
HeartbeatService (PRD-55 US-007)
================================
Schedules and executes periodic heartbeat ticks using APScheduler.

Dependencies (add to requirements.txt):
    apscheduler>=3.10
    pytz
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive-mapped heartbeat findings (PRD-142 Wave 3 · WS-M · W3-S1)
#
# The Command Centre's per-primitive health tile reads ``heartbeat_results``
# rows whose JSONB ``findings`` carry the ``primitive_check`` shape. Each
# primitive's hardening story (S6 chat … S13 channels) wires its own caller
# of ``emit_primitive_finding`` once it has a real signal; until then the
# primitive emits NOTHING (the tile reads ``unknown`` — never a fake green).
# Canonical lowercase keys only (CLAUDE.md §10 — no legacy nouns).
# ---------------------------------------------------------------------------

PRIMITIVE_NAMES = frozenset({
    "chat",
    "memory",
    "rag",
    "nl2sql",
    "graph",
    "missions",
    "playbooks",
    "channels",
})
PRIMITIVE_STATUSES = frozenset({"green", "degraded", "down"})


# Board-task pickup priority now lives in the dispatch claim query
# (services/board_dispatcher.py, _PRIORITY_ORDER_SQL) — PRD-161 moved task
# dispatch out of the heartbeat.


def emit_primitive_finding(
    workspace_id: str,
    primitive: str,
    status: str,
    detail: str = "",
) -> bool:
    """Best-effort write of a primitive-mapped heartbeat finding.

    Mirrors the ``_store_heartbeat_result`` INSERT path — same table, same
    columns. The ``primitive`` / ``status`` / ``finding_type`` keys live
    inside the existing JSONB ``findings`` column, so NO schema change is
    required. Each call writes exactly one row whose ``findings`` payload is
    a single ``primitive_check`` dict; the W3-S2 analytics endpoint picks the
    latest-per-primitive when it reads.

    Returns True on a written row, False on validation reject or write
    failure. NEVER raises — a failure here cannot break the heartbeat cycle
    or the primitive's own code path.
    """
    try:
        if primitive not in PRIMITIVE_NAMES:
            logger.warning(
                "[Heartbeat] emit_primitive_finding rejected unknown primitive=%r",
                primitive,
            )
            return False
        if status not in PRIMITIVE_STATUSES:
            logger.warning(
                "[Heartbeat] emit_primitive_finding rejected invalid status=%r",
                status,
            )
            return False

        from core.database.database import SessionLocal
        from sqlalchemy import text

        db = SessionLocal()
        try:
            finding = {
                "finding_type": "primitive_check",
                "primitive": primitive,
                "status": status,
                "detail": (str(detail) if detail else "")[:500],
            }
            db.execute(
                text(
                    """
                    INSERT INTO heartbeat_results
                        (source_type, source_id, workspace_id, status,
                         findings, actions_taken, tokens_used, created_at)
                    VALUES
                        ('orchestrator', :source_id, :workspace_id, 'success',
                         :findings, '[]', 0, NOW())
                    """
                ),
                {
                    "source_id": str(workspace_id),
                    "workspace_id": str(workspace_id),
                    "findings": json.dumps([finding]),
                },
            )
            db.commit()
            return True
        finally:
            db.close()
    except Exception:
        logger.error(
            "[Heartbeat] emit_primitive_finding failed ws=%s primitive=%s",
            workspace_id, primitive, exc_info=True,
        )
        return False


class HeartbeatService:
    """
    Manages periodic heartbeat ticks for orchestrator and agents.

    - Uses APScheduler with memory job store (Redis store optional enhancement)
    - Rate limiting: max 1 concurrent heartbeat per agent, max 5 per workspace
    - Timezone-aware scheduling respecting active hours
    """

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._owns_scheduler: bool = False  # True when we created our own scheduler (tests)
        self._running_ticks: Dict[str, bool] = {}  # track concurrent ticks
        # PRD-185 S11: last emitted memory-primitive status, so the durable-store
        # probe writes a heartbeat_results row only on a state CHANGE (not every tick).
        self._last_durable_probe_status: Optional[str] = None
        self._max_concurrent_per_workspace = 5

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: Optional[AsyncIOScheduler] = None):
        """Initialize scheduler and load all active heartbeat configs.

        Args:
            scheduler: Shared APScheduler instance from UnifiedScheduler.
                        If None, creates a local scheduler (useful for tests).
        """
        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            # Standalone mode (tests / backwards compat)
            jobstores = {"default": MemoryJobStore()}
            try:
                from config import config as app_config

                if app_config.REDIS_URL:
                    from apscheduler.jobstores.redis import RedisJobStore

                    jobstores["default"] = RedisJobStore(url=app_config.REDIS_URL)
                    logger.info("[Heartbeat] Using Redis job store (standalone)")
            except Exception:
                pass
            self._scheduler = AsyncIOScheduler(jobstores=jobstores)
            self._scheduler.start()
            self._owns_scheduler = True

        # Load and schedule all active heartbeats
        await self._load_heartbeat_configs()

        # Schedule daily summary at 1 AM UTC
        self._scheduler.add_job(
            self._daily_summary_tick,
            "cron",
            hour=1,
            minute=0,
            id="daily_summary",
            replace_existing=True,
            max_instances=1,
        )

        # PRD-187 S1: durable-store health probe (was the PRD-141 US-006 mem0
        # HTTP probe). Pings the in-process Qdrant durable store on a fixed
        # interval and feeds the per-workspace memory primitive tile (W3-S1).
        try:
            from config import config as _app_config

            from apscheduler.triggers.interval import IntervalTrigger

            probe_interval = int(_app_config.DURABLE_MEMORY_PROBE_INTERVAL_SECONDS)
            self._scheduler.add_job(
                self._durable_memory_probe_tick,
                IntervalTrigger(seconds=probe_interval),
                id="durable_memory_probe",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            logger.info(
                "[Heartbeat] Durable-memory health probe scheduled every %ds", probe_interval
            )
        except Exception:
            logger.error("[Heartbeat] Failed to schedule Mem0 health probe", exc_info=True)

        # PRD-185 S2: per-lane telemetry canary. Alarms when organic tool-execution
        # rows stop landing (the 2-month type-poison outage S1 repaired had no such
        # signal). Registered here beside the Mem0 probe; the first run fires at boot
        # as the boot-probe, then every TELEMETRY_CANARY_INTERVAL_SECONDS.
        try:
            from config import config as _cfg

            if _cfg.TELEMETRY_CANARY_ENABLED:
                from apscheduler.triggers.interval import IntervalTrigger
                from services.telemetry_canary import telemetry_canary_tick

                canary_interval = int(_cfg.TELEMETRY_CANARY_INTERVAL_SECONDS)
                self._scheduler.add_job(
                    telemetry_canary_tick,
                    IntervalTrigger(seconds=canary_interval),
                    id="telemetry_canary",
                    replace_existing=True,
                    max_instances=1,
                    coalesce=True,
                    next_run_time=datetime.now(timezone.utc),  # boot-probe: run once now
                )
                logger.info(
                    "[Heartbeat] Telemetry canary scheduled every %ds (boot-probe now)",
                    canary_interval,
                )
        except Exception:
            logger.error("[Heartbeat] Failed to schedule telemetry canary", exc_info=True)

        logger.info("[Heartbeat] Service started (daily summary at 01:00 UTC)")

    async def stop(self):
        """Remove heartbeat jobs. Only shuts down scheduler if we own it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=True)
            logger.info("[Heartbeat] Standalone scheduler stopped")
        logger.info("[Heartbeat] Service stopped")

    async def _durable_memory_probe_tick(self) -> None:
        """Probe the durable store + emit memory primitive finding (PRD-187 S1 + W3-S1).

        Pings the shared in-process durable store (the one real traffic uses)
        via its ``health()`` Qdrant round-trip — an unreachable store surfaces
        here LOUDLY instead of skipping silently (the old mem0 failure mode).

        W3-S1 (pathfinder wiring): after the probe lands, emit a ``memory``
        primitive_check finding per workspace that has an active orchestrator
        heartbeat, so the per-workspace memory tile reflects the latest probe
        result. Richer multi-layer (L1/L2/L3) signal lands with W3-S7.
        """
        try:
            from modules.memory.unified_memory_service import get_unified_memory_service

            health = await get_unified_memory_service()._durable.health()
            healthy = bool(health.get("healthy"))
            if not healthy:
                logger.error(
                    "[Heartbeat] Durable-memory probe FAILED: %s", health.get("error")
                )
        except Exception:
            logger.error("[Heartbeat] Durable-memory probe tick errored", exc_info=True)
            return

        if self._scheduler is None:
            return
        primitive_status = "green" if healthy else "down"
        # PRD-185 S11: emit the memory primitive finding ONLY on a state change.
        # This probe previously wrote one heartbeat_results row per workspace
        # EVERY tick (~2880/ws/day) — pure noise that polluted the table and the
        # primitive-health read. The tile is latest-wins with no freshness gate
        # (api/analytics_real.get_primitive_health), so one row per transition
        # keeps it correct while ending the spam. First tick emits the baseline.
        if primitive_status == self._last_durable_probe_status:
            return
        detail = "durable store ok" if healthy else "durable store unreachable"
        try:
            for job in self._scheduler.get_jobs():
                jid = getattr(job, "id", "") or ""
                if not jid.startswith("orch_hb_"):
                    continue
                ws_id = jid[len("orch_hb_"):]
                emit_primitive_finding(ws_id, "memory", primitive_status, detail)
            self._last_durable_probe_status = primitive_status
        except Exception:
            logger.error(
                "[Heartbeat] memory primitive emit loop errored", exc_info=True
            )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    async def _load_heartbeat_configs(self):
        """Load all active heartbeat configs from DB and schedule jobs.

        Clears stale heartbeat jobs first (e.g. from Redis jobstore persistence)
        so disabled agents don't keep running after a restart.
        """
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace
        from core.models import Agent

        # Remove any leftover heartbeat jobs from previous runs before
        # re-adding only the currently-enabled ones.
        for job in list(self._scheduler.get_jobs()):
            if job.id.startswith("agent_hb_") or job.id.startswith("orch_hb_"):
                self._scheduler.remove_job(job.id)

        db = SessionLocal()
        try:
            # Schedule orchestrator heartbeats
            workspaces = db.query(Workspace).all()
            for ws in workspaces:
                settings = ws.settings or {}
                orch = settings.get("orchestrator", {})
                hb = orch.get("heartbeat", {})
                if hb.get("enabled"):
                    self.schedule_orchestrator_heartbeat(str(ws.id), hb)

            # Schedule agent heartbeats
            agents = db.query(Agent).all()
            for agent in agents:
                cfg = agent.configuration or {}
                hb = cfg.get("heartbeat", {})
                if hb.get("enabled"):
                    self.schedule_agent_heartbeat(
                        agent.id, str(agent.workspace_id), hb
                    )
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    @staticmethod
    def _interval_to_cron_trigger(minutes: int) -> CronTrigger:
        """Convert an interval in minutes to a CronTrigger firing at fixed times.

        The cron-field math is the single source of truth in
        ``schedule_util.interval_to_cron`` (PRD-162) — the calendar and the
        ``platform_get_schedule`` tool compute the same firing times from it.
        This just builds the APScheduler trigger from that one cron string:

            15→``0,15,30,45 * * * *``  30→``0,30 * * * *``  60→``0 * * * *``
            120→``0 */2 * * *``  1440→``0 9 * * *``  10080→``0 9 * * 1`` (Mon 9am)
        """
        from services.schedule_util import interval_to_cron

        return CronTrigger.from_crontab(interval_to_cron(minutes))

    def schedule_orchestrator_heartbeat(
        self, workspace_id: str, hb_config: dict
    ):
        """Schedule or reschedule an orchestrator heartbeat job."""
        job_id = f"orch_hb_{workspace_id}"
        interval_minutes = hb_config.get("interval_minutes", 30)

        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        trigger = self._interval_to_cron_trigger(interval_minutes)

        self._scheduler.add_job(
            self._orchestrator_tick,
            trigger,
            id=job_id,
            args=[workspace_id, hb_config],
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[Heartbeat] Scheduled orchestrator heartbeat for ws=%s every %dm (cron: %s)",
            workspace_id,
            interval_minutes,
            trigger,
        )

    def schedule_agent_heartbeat(
        self, agent_id: int, workspace_id: str, hb_config: dict
    ):
        """Schedule or reschedule an agent heartbeat job."""
        job_id = f"agent_hb_{agent_id}"
        interval_minutes = hb_config.get("interval_minutes", 60)

        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        trigger = self._interval_to_cron_trigger(interval_minutes)

        self._scheduler.add_job(
            self._agent_tick,
            trigger,
            id=job_id,
            args=[agent_id, workspace_id, hb_config],
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[Heartbeat] Scheduled agent heartbeat for agent=%s every %dm (cron: %s)",
            agent_id,
            interval_minutes,
            trigger,
        )

    def unschedule_heartbeat(self, job_id: str):
        """Remove a scheduled heartbeat job."""
        if self._scheduler and self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("[Heartbeat] Unscheduled job %s", job_id)

    # ------------------------------------------------------------------
    # Active-hours guard
    # ------------------------------------------------------------------

    async def _is_within_active_hours(
        self, hb_config: dict, workspace_id: str = ""
    ) -> bool:
        """Check if current time is within the heartbeat's active hours.

        When ``inherit_active_hours`` is True in *hb_config*, the active-hours
        window and timezone are loaded from the **orchestrator** workspace
        settings instead of the agent's own config.
        """
        import pytz

        active_cfg = dict(hb_config)

        # If inheriting, load orchestrator-level active hours
        if hb_config.get("inherit_active_hours") and workspace_id:
            try:
                from core.database.database import SessionLocal
                from core.models.workspaces import Workspace

                db = SessionLocal()
                try:
                    ws = db.query(Workspace).get(workspace_id)
                    if ws:
                        orch = (ws.settings or {}).get("orchestrator", {})
                        orch_hb = orch.get("heartbeat", {})
                        if orch_hb.get("active_hours_start"):
                            active_cfg["active_hours_start"] = orch_hb["active_hours_start"]
                        if orch_hb.get("active_hours_end"):
                            active_cfg["active_hours_end"] = orch_hb["active_hours_end"]
                        if orch_hb.get("timezone"):
                            active_cfg["timezone"] = orch_hb["timezone"]
                        logger.debug(
                            "[Heartbeat] Inherited active hours from orchestrator: %s–%s (%s)",
                            active_cfg.get("active_hours_start"),
                            active_cfg.get("active_hours_end"),
                            active_cfg.get("timezone"),
                        )
                finally:
                    db.close()
            except Exception as e:
                logger.warning(
                    "[Heartbeat] Failed to inherit orchestrator active hours for ws=%s: %s — assuming within hours",
                    workspace_id, e,
                )
                return True  # fail open: if we can't check, let the tick run

        tz_name = active_cfg.get("timezone", "UTC")
        try:
            tz = pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            tz = pytz.UTC

        now = datetime.now(tz)
        start_str = active_cfg.get("active_hours_start", "08:00")
        end_str = active_cfg.get("active_hours_end", "20:00")

        def _to_minutes(s: str) -> int:
            h, m = map(int, s.split(":"))
            return h * 60 + m

        current = now.hour * 60 + now.minute
        start = _to_minutes(start_str)
        end = _to_minutes(end_str)

        # Handle overnight windows (e.g. 22:00 → 06:00)
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    # ------------------------------------------------------------------
    # Period-based deduplication
    # ------------------------------------------------------------------

    async def _was_recently_executed(
        self, source_type: str, source_id: str, interval_minutes: int
    ) -> bool:
        """Check if a heartbeat already ran within the current interval window.

        Prevents duplicate executions after service restarts or scheduler
        re-initialisation.  Queries the most recent successful heartbeat
        result from the database and skips if it falls inside the current
        interval.
        """
        # Give a 10 % grace margin so we don't skip a tick that's just
        # barely on the boundary (clock skew, slow queries, etc.)
        cooldown_minutes = max(interval_minutes * 0.9, interval_minutes - 5)

        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                row = db.execute(
                    text(
                        """
                        SELECT created_at FROM heartbeat_results
                        WHERE source_type = :source_type
                          AND source_id   = :source_id
                          AND status      = 'success'
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"source_type": source_type, "source_id": source_id},
                ).fetchone()

                if row:
                    last_run: datetime = row[0]
                    elapsed = (datetime.utcnow() - last_run).total_seconds() / 60
                    if elapsed < cooldown_minutes:
                        logger.info(
                            "[Heartbeat] Skipping %s %s — last successful run %.0f min ago "
                            "(interval=%d min, cooldown=%.0f min)",
                            source_type,
                            source_id,
                            elapsed,
                            interval_minutes,
                            cooldown_minutes,
                        )
                        return True
                return False
            finally:
                db.close()
        except Exception as e:
            logger.warning("[Heartbeat] Dedup check failed (%s), allowing tick: %s", source_id, e)
            return False

    # ------------------------------------------------------------------
    # Tick implementations
    # ------------------------------------------------------------------

    def _trial_skip(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """PRD-222 US-005 — no background burn.

        Trial workspaces (onboarding.trial.state active/warned/exhausted) get NO
        heartbeat execution until the trial converts — an idle trial workspace
        must burn $0. Returns a VISIBLE skip result (never a silent no-op) or
        ``None`` to proceed. Converted / never-granted workspaces proceed.
        """
        # PRD-234 S3: the local edition has no platform-paid trial credit to protect —
        # keys are the operator's own and Claude Code agents run on their subscription —
        # yet Auto-led onboarding grants a trial record there too, which silently
        # switched every local heartbeat off (found 2026-09-03). Local runs.
        try:
            from config import config as _config
            if getattr(_config, "AUTH_EDITION", "saas") == "local":
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            from core.database.database import SessionLocal
            from core.models.workspaces import Workspace
            from services.trial_ledger import is_trial_active_workspace

            db = SessionLocal()
            try:
                ws = db.query(Workspace).get(workspace_id)
                if is_trial_active_workspace(ws):
                    logger.info(
                        "[Heartbeat] Skipping trial workspace %s — no background "
                        "burn until converted",
                        workspace_id,
                    )
                    return {"status": "skipped", "reason": "trial_workspace"}
            finally:
                db.close()
        except Exception as e:
            logger.debug("[Heartbeat] trial-skip check failed for ws=%s: %s", workspace_id, e)
        return None

    async def _orchestrator_tick(self, workspace_id: str, hb_config: dict) -> Dict[str, Any]:
        """Execute an LLM-powered orchestrator heartbeat tick."""
        tick_key = f"orch_{workspace_id}"
        if self._running_ticks.get(tick_key):
            logger.debug(
                "[Heartbeat] Orchestrator tick already running for ws=%s, skipping",
                workspace_id,
            )
            return {"status": "skipped", "reason": "already_running"}

        _trial = self._trial_skip(workspace_id)
        if _trial:
            return _trial

        if not await self._is_within_active_hours(hb_config, workspace_id):
            logger.debug(
                "[Heartbeat] Outside active hours for ws=%s, skipping",
                workspace_id,
            )
            return {"status": "skipped", "reason": "outside_active_hours"}

        interval_minutes = hb_config.get("interval_minutes", 30)
        # Manual UI triggers (run_orchestrator_heartbeat) set force_run=True to
        # bypass the cooldown — clicking "Run Now" should always execute.
        if not hb_config.get("force_run") and await self._was_recently_executed(
            "orchestrator", workspace_id, interval_minutes
        ):
            return {"status": "skipped", "reason": "already_ran_this_period"}

        self._running_ticks[tick_key] = True
        result: Dict[str, Any] = {
            "source_type": "orchestrator",
            "source_id": workspace_id,
            "workspace_id": workspace_id,
            "status": "success",
            "findings": [],
            "actions_taken": [],
            "tokens_used": 0,
        }

        # Resolve proactive_level once for both success and error paths
        try:
            from consumers.chatbot.personality import load_orchestrator_settings as _load_orch
            _orch = _load_orch(workspace_id)
            _proactive_level = hb_config.get("proactive_level") or _orch.get("proactive_level", "notify")
        except Exception:
            _proactive_level = "notify"

        try:
            logger.info(
                "[Heartbeat] Orchestrator tick starting for ws=%s", workspace_id
            )

            # Try LLM-powered tick; fall back to shallow analysis on failure
            try:
                await self._orchestrator_tick_llm(workspace_id, hb_config, result)
            except Exception as llm_err:
                logger.warning(
                    "[Heartbeat] LLM tick failed for ws=%s, falling back to shallow: %s",
                    workspace_id, llm_err, exc_info=True,
                )
                result["findings"].append(
                    {"check": "llm_error", "detail": f"LLM unavailable: {str(llm_err)[:200]}"}
                )
                await self._orchestrator_tick_shallow(workspace_id, hb_config, result)

            await self._store_heartbeat_result(result)

            if _proactive_level != "silent":
                await self._dispatch_heartbeat_notification(result)

            await self._auto_create_orchestrator_report(workspace_id, result)

            logger.info(
                "[Heartbeat] Orchestrator tick completed for ws=%s: %d findings, %d tokens",
                workspace_id, len(result["findings"]), result.get("tokens_used", 0),
            )

        except Exception as e:
            logger.error(
                "[Heartbeat] Orchestrator tick failed for ws=%s: %s",
                workspace_id, e, exc_info=True,
            )
            result["status"] = "error"
            result["findings"].append({"check": "error", "detail": str(e)})
            await self._store_heartbeat_result(result)
            if _proactive_level != "silent":
                await self._dispatch_heartbeat_notification(result)
            await self._auto_create_orchestrator_report(workspace_id, result)
        finally:
            self._running_ticks.pop(tick_key, None)

        return result

    async def _orchestrator_tick_llm(
        self, workspace_id: str, hb_config: dict, result: Dict[str, Any],
    ) -> None:
        """Run the LLM-powered orchestrator heartbeat with tool loop."""
        from consumers.chatbot.personality import load_orchestrator_settings
        from core.llm.manager import LLMManager
        from modules.context import ContextService, ContextMode
        from modules.tools.discovery.platform_executor import PlatformActionExecutor
        from core.database.database import SessionLocal, end_open_transaction
        from types import SimpleNamespace

        # 1. Load personality settings for heartbeat-specific instructions
        orch_settings = load_orchestrator_settings(workspace_id)
        personality_mode = orch_settings.get("personality_mode", "friendly")
        communication_style = orch_settings.get("communication_style", "balanced")
        proactive_level = hb_config.get("proactive_level") or orch_settings.get("proactive_level", "notify")

        # 2. Build heartbeat-specific instructions as task_description
        level_instructions = {
            "silent": "Report findings only. Do NOT take any corrective actions.",
            "notify": "Report findings only. Do NOT take any corrective actions.",
            "act_notify": "Take corrective action if needed using your tools. Report what you did.",
            "autonomous": "Act independently to resolve any issues you find. Report a summary of actions taken.",
        }
        level_instruction = level_instructions.get(proactive_level, level_instructions["notify"])

        style_suffix = {
            "concise": " Keep your response extremely short and direct.",
            "balanced": "",
            "detailed": " Provide thorough analysis with specifics.",
        }.get(communication_style, "")

        checklist = hb_config.get("checklist", "")
        checklist_block = ""
        if checklist and checklist.strip():
            checklist_block = f"\n\nChecklist to review:\n{checklist}"

        # Wave 4 — opt-in cadence loops (Daily Brief / Weekly Review /
        # Monday HARNESS / Post-change / Incident review). Default off.
        from core.services.auto_cadence import build_cadence_block
        cadence_block = build_cadence_block(hb_config)

        task_description = (
            f"Perform a scheduled health check for this workspace.\n"
            f"Personality: {personality_mode}.\n\n"
            f"Analyze your workspace using the tools provided.{checklist_block}{cadence_block}\n\n"
            f"{level_instruction}{style_suffix}\n\n"
            f"Reply with a SHORT plain-text summary (max 500 chars). No markdown."
        )

        # 3. Build context via ContextService
        orchestrator_agent = SimpleNamespace(
            id=None,
            name="Automatos Orchestrator",
            agent_type="orchestrator",
            description="Scheduled workspace health check agent",
            use_custom_persona=False,
            persona=None,
        )

        db = SessionLocal()
        try:
            context = await ContextService(db).build_context(
                mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
                agent=orchestrator_agent,
                workspace_id=workspace_id,
                task_description=task_description,
                # Narrow the dispatcher enum to heartbeat-relevant actions —
                # without a query this lane shipped all 137 on every run.
                query=task_description,
            )

            system_prompt = context.system_prompt
            platform_tools = context.tools

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Run the scheduled heartbeat check now. Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"},
            ]

            # 4. Create LLM manager (maps to orchestrator_llm settings)
            llm = LLMManager(
                service_name="heartbeat",
                workspace_id=workspace_id,
                request_type="heartbeat",
            )

            # 5. Tool loop — budget from system_settings (agent_heartbeat.max_tool_iterations)
            from config import config as _hb_config
            max_iterations = _hb_config.AGENT_HEARTBEAT_MAX_TOOL_ITERATIONS
            total_tokens = 0
            executor = PlatformActionExecutor(db, workspace_id)

            for iteration in range(max_iterations):
                # End the open transaction before each LLM call so the heartbeat
                # connection is not idle-in-transaction across the await (PRD-135
                # / W1-S9). First pass commits build_context()'s reads; later
                # passes commit the prior tool's writes.
                end_open_transaction(db)
                response = await llm.generate_response(messages, tools=platform_tools if platform_tools else None)

                # Track tokens
                usage = getattr(response, "usage", None) or {}
                total_tokens += usage.get("total_tokens", 0) or (
                    (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
                )

                # Check for tool calls
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    # No more tool calls — capture final response
                    content = getattr(response, "content", "") or ""
                    if content:
                        result["findings"].append(
                            {"check": "llm_analysis", "detail": str(content)[:1000]}
                        )
                    break

                # Build assistant message with all tool calls, then execute each
                assistant_msg = {"role": "assistant", "content": getattr(response, "content", "") or None, "tool_calls": []}
                tool_results_msgs = []
                for tc in tool_calls:
                    func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    fn_name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                    fn_args_raw = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")

                    assistant_msg["tool_calls"].append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": fn_args_raw},
                    })

                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        tool_result = await executor.execute(fn_name, fn_args)
                        result["actions_taken"].append({"tool": fn_name, "params": fn_args})
                    except Exception as tool_err:
                        tool_result = {"error": str(tool_err)[:500]}
                        logger.warning("[Heartbeat] Tool %s failed: %s", fn_name, tool_err)

                    tool_results_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_result, default=str)[:2000],
                    })

                messages.append(assistant_msg)
                messages.extend(tool_results_msgs)

                # Trim older tool exchanges if context is growing too large.
                # IMPORTANT: Trim at exchange boundaries — each exchange is
                # an assistant msg (with tool_calls) followed by N tool result msgs.
                # Slicing arbitrarily orphans tool results from their assistant msg.
                if len(messages) > 10:
                    preamble = messages[:2]  # system + user
                    exchanges = messages[2:]  # assistant+tool groups

                    # Walk backwards to find complete exchange boundaries
                    # Each exchange starts with role=assistant
                    exchange_starts = [
                        i for i, m in enumerate(exchanges)
                        if m.get("role") == "assistant"
                    ]

                    # Keep last 2 complete exchanges
                    if len(exchange_starts) > 2:
                        keep_from = exchange_starts[-2]
                        messages = preamble + exchanges[keep_from:]
            else:
                # Max iterations reached
                result["findings"].append(
                    {"check": "llm_analysis", "detail": "Heartbeat analysis completed (max tool iterations reached)."}
                )
        finally:
            db.close()

        result["tokens_used"] = total_tokens

    async def _orchestrator_tick_shallow(
        self, workspace_id: str, hb_config: dict, result: Dict[str, Any],
    ) -> None:
        """Shallow fallback when LLM is unavailable — counts agents and parses checklist."""
        from core.database.database import SessionLocal
        from core.models import Agent

        db = SessionLocal()
        try:
            agents = (
                db.query(Agent)
                .filter(Agent.workspace_id == workspace_id)
                .all()
            )
            active_agents = [a for a in agents if a.status == "active"]
            inactive_agents = [a for a in agents if a.status != "active"]

            result["findings"].append(
                {
                    "check": "agent_health",
                    "detail": (
                        f"{len(active_agents)} active, "
                        f"{len(inactive_agents)} inactive agents"
                    ),
                }
            )

            checklist = hb_config.get("checklist", "")
            if checklist:
                items = [
                    line.strip().lstrip("- ")
                    for line in checklist.split("\n")
                    if line.strip()
                ]
                result["findings"].append(
                    {
                        "check": "checklist",
                        "items": items,
                        "detail": f"Reviewed {len(items)} checklist items (shallow mode — LLM unavailable)",
                    }
                )
        finally:
            db.close()

    async def _agent_tick(
        self, agent_id: int, workspace_id: str, hb_config: dict
    ) -> Dict[str, Any]:
        """Execute an agent heartbeat tick. Returns result dict."""
        tick_key = f"agent_{agent_id}"
        if self._running_ticks.get(tick_key):
            logger.debug(
                "[Heartbeat] Agent tick already running for agent=%s, skipping",
                agent_id,
            )
            return {"status": "skipped", "reason": "already_running"}

        _trial = self._trial_skip(workspace_id)
        if _trial:
            return _trial

        if not await self._is_within_active_hours(hb_config, workspace_id):
            logger.info(
                "[Heartbeat] Outside active hours for agent=%s (ws=%s), skipping",
                agent_id, workspace_id,
            )
            return {"status": "skipped", "reason": "outside_active_hours"}

        interval_minutes = hb_config.get("interval_minutes", 60)
        # Manual UI triggers (run_agent_heartbeat) set force_run=True to bypass
        # the cooldown — without this, daily-interval agents could only be
        # tested once a day. Scheduled ticks never set force_run.
        if not hb_config.get("force_run") and await self._was_recently_executed(
            "agent", str(agent_id), interval_minutes
        ):
            return {"status": "skipped", "reason": "already_ran_this_period"}

        self._running_ticks[tick_key] = True
        run_started_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "source_type": "agent",
            "source_id": str(agent_id),
            "workspace_id": workspace_id,
            "status": "success",
            "findings": [],
            "actions_taken": [],
            "tokens_used": 0,
            "_run_started_at": run_started_at,
        }

        try:
            logger.info(
                "[Heartbeat] Agent tick starting for agent=%s", agent_id
            )

            from core.database.database import SessionLocal
            from core.models import Agent

            db = SessionLocal()
            try:
                agent = db.query(Agent).get(agent_id)
                if not agent:
                    result["status"] = "error"
                    result["findings"].append(
                        {"check": "error", "detail": "Agent not found"}
                    )
                    return

                heartbeat_prompt = hb_config.get(
                    "prompt",
                    "Check your domain for any issues or updates.",
                )
                auto_act = hb_config.get("auto_act", False)

                # PRD-161: board-task pickup moved OUT of the heartbeat into the
                # dedicated dispatch loop (services/board_dispatcher.py), which
                # claims each assigned task with FOR UPDATE SKIP LOCKED and runs
                # it individually. The heartbeat is monitoring/recurring only now
                # (Q40) — it no longer batches 3 tasks into one prompt.

                # PRD-140 Phase 1 — opt-in cadence blocks (e.g. team_review
                # when this agent has team_lead_enabled=True). Same module as
                # Auto's cadence so we don't run two parallel mechanisms.
                from core.services.auto_cadence import build_cadence_block
                cadence_block = build_cadence_block(hb_config)

                prompt = (
                    f"Scheduled heartbeat check. {heartbeat_prompt}\n"
                    + (cadence_block + "\n\n" if cadence_block else "")
                    + "Use your tools to check. Reply with a SHORT plain-text summary (max 500 chars), no markdown.\n"
                    + (
                        "You may take action if needed."
                        if auto_act
                        else "Report findings only."
                    )
                )

                # PRD-234 S3: a Claude Code agent's heartbeat is a board ticket the
                # paired host runs as the user's own session — never a factory call
                # (the factory refuses cli agents by design). One open heartbeat
                # ticket per agent at a time.
                from core.cli_runtime import RUNTIME_CLI, runtime_kind_of
                cli_ticket_filed = False
                if runtime_kind_of(agent.configuration or {}) == RUNTIME_CLI:
                    from services.cli_ticket_lane import file_cli_ticket, queued_line, source_id_for
                    ticket = file_cli_ticket(
                        db, workspace_id=workspace_id, agent_id=agent_id,
                        title=f"Heartbeat: {agent.name}", prompt=prompt,
                        source_type="heartbeat", source_id=source_id_for("agent", agent_id),
                        priority="low",
                    )
                    result["findings"].append({"check": "cli_ticket", "detail": queued_line(ticket)})
                    result["actions_taken"].append({"action": "file_cli_ticket", "task_id": ticket.id})
                    cli_ticket_filed = True
                else:
                    # Execute through AgentFactory so the agent has its full toolset
                    try:
                        from modules.agents.factory.agent_factory import AgentFactory
                        from modules.context.modes import ContextMode
                        from services.heartbeat_outcome import read_exec_outcome, tokens_of
                        factory = AgentFactory(db_session=db)
                        exec_result = await factory.execute_with_prompt(
                            agent=agent_id,
                            prompt=prompt,
                            context={"source": "heartbeat", "workspace_id": workspace_id},
                            context_mode=ContextMode.HEARTBEAT_AGENT,
                        )
                        # PRD-234 S3: an error dict is an error — it used to be filed as a
                        # green 'llm_analysis' finding, so a failing agent looked healthy.
                        llm_text, is_error, error_detail = read_exec_outcome(exec_result)
                        if is_error:
                            result["status"] = "error"
                            result["findings"].append({"check": "exec_error", "detail": error_detail})
                        else:
                            result["findings"].append(
                                {"check": "llm_analysis", "detail": llm_text or str(exec_result)[:500]}
                            )
                        result["tokens_used"] = tokens_of(exec_result)
                    except Exception as exec_err:
                        logger.warning(
                            "[Heartbeat] Agent execution failed for agent=%s: %s",
                            agent_id,
                            exec_err,
                        )
                        result["findings"].append(
                            {
                                "check": "exec_error",
                                "detail": f"Agent execution failed: {str(exec_err)[:200]}",
                            }
                        )
            finally:
                db.close()

            result["_run_completed_at"] = datetime.utcnow()
            await self._store_heartbeat_result(result)
            await self._dispatch_heartbeat_notification(result)
            await self._auto_create_report(agent_id, workspace_id, result)
            logger.info(
                "[Heartbeat] Agent tick completed for agent=%s", agent_id
            )

        except Exception as e:
            logger.error(
                "[Heartbeat] Agent tick failed for agent=%s: %s", agent_id, e
            )
            result["status"] = "error"
            result["findings"].append({"check": "error", "detail": str(e)})
            result["_run_completed_at"] = datetime.utcnow()
            await self._store_heartbeat_result(result)
            await self._dispatch_heartbeat_notification(result)
            await self._auto_create_report(agent_id, workspace_id, result)
        finally:
            self._running_ticks.pop(tick_key, None)

        return result

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def _daily_summary_tick(self):
        """Fetch last 24h of heartbeat_results, build summary, store in SmartMemoryManager."""
        from datetime import timedelta

        logger.info("[Heartbeat] Running daily summary")

        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                rows = db.execute(
                    text(
                        "SELECT source_type, source_id, workspace_id, status, findings, tokens_used, created_at "
                        "FROM heartbeat_results WHERE created_at >= :cutoff ORDER BY created_at DESC"
                    ),
                    {"cutoff": cutoff},
                ).fetchall()

                if not rows:
                    logger.info("[Heartbeat] No results in last 24h, skipping summary")
                    return

                # Group by workspace
                ws_summaries: Dict[str, list] = {}
                for r in rows:
                    ws_id = str(r.workspace_id) if r.workspace_id else "unknown"
                    ws_summaries.setdefault(ws_id, []).append(r)

                for ws_id, ws_rows in ws_summaries.items():
                    success_count = sum(1 for r in ws_rows if r.status == "success")
                    error_count = sum(1 for r in ws_rows if r.status == "error")
                    total_tokens = sum(r.tokens_used or 0 for r in ws_rows)

                    # PRD-185 S11: the daily digest is an OPERATIONAL LOG, not a
                    # memory. It used to be double-written into the memory plane —
                    # once as a fabricated user/assistant L3 turn (a fake summary
                    # "request" + digest reply) that then got injected into real
                    # client prompts, and once as an L2 heartbeat_log row. Both are
                    # removed: the raw ticks are the source of truth in
                    # heartbeat_results; the digest is logged here, never injected.
                    logger.info(
                        "[Heartbeat] Daily summary ws=%s: %d ticks, %d ok, %d errors, %d tokens",
                        ws_id, len(ws_rows), success_count, error_count, total_tokens,
                    )

            finally:
                db.close()

        except Exception as e:
            logger.error("[Heartbeat] Daily summary failed: %s", e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _store_heartbeat_result(self, result: dict) -> Optional[int]:
        """Store heartbeat result in the database. Returns the row ID."""
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                row = db.execute(
                    text(
                        """
                        INSERT INTO heartbeat_results
                            (source_type, source_id, workspace_id, status,
                             findings, actions_taken, tokens_used,
                             objective_met, evidence_ref, created_at)
                        VALUES
                            (:source_type, :source_id, :workspace_id, :status,
                             :findings, :actions_taken, :tokens_used,
                             :objective_met, :evidence_ref, NOW())
                        RETURNING id
                        """
                    ),
                    {
                        "source_type": result["source_type"],
                        "source_id": result["source_id"],
                        "workspace_id": result["workspace_id"],
                        "status": result["status"],
                        "findings": json.dumps(result["findings"]),
                        "actions_taken": json.dumps(result["actions_taken"]),
                        "tokens_used": result.get("tokens_used", 0),
                        "objective_met": self._infer_objective_met(result),
                        "evidence_ref": result.get("evidence_ref"),
                    },
                ).fetchone()
                db.commit()
                hb_id = row[0] if row else None
                result["_heartbeat_result_id"] = hb_id
                return hb_id
            finally:
                db.close()
        except Exception as e:
            logger.error("[Heartbeat] Failed to store result: %s", e)
            return None

    @staticmethod
    def _infer_objective_met(result: dict) -> Optional[bool]:
        """Best-effort objective completion classifier — Wave 1.B.

        ``True``  — status=success and there is observable output (actions
                    taken, findings recorded, or an explicit evidence_ref).
        ``False`` — status indicates failure (error / failed / timeout).
        ``None``  — cannot be determined (silent success with no output).

        Callers may override by setting ``result["objective_met"]`` directly
        before persistence.
        """
        if "objective_met" in result:
            return result.get("objective_met")
        status = (result.get("status") or "").lower()
        if status in {"error", "failed", "timeout"}:
            return False
        if status == "success":
            has_output = bool(
                result.get("actions_taken")
                or result.get("findings")
                or result.get("evidence_ref")
            )
            return True if has_output else None
        return None

    # ------------------------------------------------------------------
    # Notification delivery (PRD-128: unified dispatcher)
    # ------------------------------------------------------------------

    async def _dispatch_heartbeat_notification(self, result: dict) -> None:
        """Fan out a heartbeat completion event via ``NotificationDispatcher``.

        Reads ``notification_preferences`` for the workspace and delivers the
        event to every enabled destination (in_app / telegram / slack /
        webhook / silent). Opens a dedicated DB session so the in-app row is
        committed independently of the caller's transaction. Non-blocking:
        any failure is logged but never propagates back to the heartbeat tick.
        """
        workspace_id = result.get("workspace_id")
        if not workspace_id:
            return

        try:
            from core.database.database import SessionLocal
            from core.services.notification_dispatcher import NotificationDispatcher

            source_type = result.get("source_type", "orchestrator")
            hb_status = result.get("status", "success")
            dispatch_status = "ok" if hb_status == "success" else "error"

            # Extract LLM analysis (or fallback summary) for the message body
            findings = result.get("findings", []) or []
            message_body = ""
            for f in findings:
                if f.get("check") == "llm_analysis":
                    message_body = (f.get("detail") or "")[:2000]
                    break
            if not message_body:
                details = [
                    (f.get("detail") or "")[:300]
                    for f in findings
                    if f.get("detail")
                ]
                message_body = "\n".join(details) if details else "No findings."

            # Resolve agent metadata + per-agent report_to override (agent tick only)
            agent_id: Optional[int] = None
            agent_name: Optional[str] = None
            agent_hb_config: dict = {}
            if source_type == "agent":
                try:
                    agent_id = int(result.get("source_id"))
                except (TypeError, ValueError):
                    agent_id = None
                if agent_id is not None:
                    db_lookup = SessionLocal()
                    try:
                        from core.models import Agent

                        agent = db_lookup.query(Agent).get(agent_id)
                        agent_name = agent.name if agent else f"agent-{agent_id}"
                        if agent and agent.configuration:
                            agent_hb_config = (agent.configuration or {}).get("heartbeat", {}) or {}
                    finally:
                        db_lookup.close()
                title = f"{agent_name or 'Agent'} Heartbeat"
            else:
                agent_name = "Orchestrator"
                title = "Orchestrator Heartbeat"

            link_id = result.get("_heartbeat_result_id")

            # Per-agent ``report_to`` overrides the workspace dispatch path.
            #   - "auto"     → assign a board task to the workspace's Auto, so
            #                  Auto picks it up on its next tick (PRD-72 loop).
            #   - "webhook"  → POST directly to the agent's stored webhook_url,
            #                  bypassing workspace prefs.
            #   - anything else (orchestrator/telegram/slack/empty) falls
            #                  through to the workspace-level NotificationDispatcher.
            report_to = (agent_hb_config.get("report_to") or "").strip().lower()

            if report_to == "auto" and source_type == "agent":
                await self._route_heartbeat_to_auto(
                    workspace_id=workspace_id,
                    source_agent_id=agent_id,
                    source_agent_name=agent_name,
                    title=title,
                    message=message_body,
                    link_id=link_id,
                    status=dispatch_status,
                )
                return

            if report_to == "webhook" and agent_hb_config.get("webhook_url"):
                await self._route_heartbeat_to_webhook(
                    webhook_url=agent_hb_config["webhook_url"],
                    title=title,
                    message=message_body,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    link_id=link_id,
                    status=dispatch_status,
                )
                return

            db = SessionLocal()
            try:
                dispatcher = NotificationDispatcher(db, workspace_id)
                dispatched = await dispatcher.dispatch(
                    event_type="heartbeat_complete",
                    title=title,
                    message=message_body,
                    link_type="heartbeat",
                    link_id=str(link_id) if link_id is not None else None,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=dispatch_status,
                )
                db.commit()
                logger.info(
                    "[Heartbeat] Dispatched heartbeat_complete ws=%s → %s",
                    workspace_id,
                    dispatched.get("dispatched_to"),
                )
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        except Exception as e:
            logger.error(
                "[Heartbeat] Notification dispatch failed for ws=%s: %s",
                workspace_id, e, exc_info=True,
            )

    # ------------------------------------------------------------------
    # report_to routing — per-agent overrides (PRD-72 follow-up)
    # ------------------------------------------------------------------

    async def _route_heartbeat_to_auto(
        self,
        *,
        workspace_id: str,
        source_agent_id: Optional[int],
        source_agent_name: Optional[str],
        title: str,
        message: str,
        link_id: Optional[Any],
        status: str,
    ) -> None:
        """Create a BoardTask for Auto to pick up on its next tick.

        ``report_to=auto`` means "the manager should look at this." Auto's
        heartbeat already scans assigned board tasks
        (heartbeat_service:730-765) and ingests them as priority work, so
        creating a task here is enough — no new scheduler, no notification.
        """
        from core.database.database import SessionLocal
        from core.models import Agent
        from core.models.core import BoardTask

        db = SessionLocal()
        try:
            auto = (
                db.query(Agent)
                .filter(
                    Agent.workspace_id == workspace_id,
                    Agent.is_system_agent.is_(True),
                    Agent.name == "Auto",
                )
                .first()
            )
            if not auto:
                logger.warning(
                    "[Heartbeat] report_to=auto but no canonical Auto found for ws=%s — "
                    "falling back to silent (DB only).",
                    workspace_id,
                )
                return

            summary_line = (message or "").strip().splitlines()[0] if message else ""
            summary_line = summary_line[:200] or "no findings"

            description = (
                f"{source_agent_name or 'Agent'} heartbeat completed (status={status}).\n\n"
                f"Findings:\n{(message or '(no findings)')[:2000]}\n\n"
                f"Read the full report with `platform_get_latest_report` "
                f"(agent_name=\"{source_agent_name}\")."
            )

            priority = "high" if status in ("error", "critical") else "medium"

            task = BoardTask(
                workspace_id=workspace_id,
                title=f"Review {source_agent_name or 'agent'} heartbeat — {summary_line}"[:500],
                description=description,
                status="assigned",
                priority=priority,
                assigned_agent_id=auto.id,
                created_by_type="agent",
                created_by_id=str(source_agent_id) if source_agent_id else None,
                source_type="heartbeat",
                source_id=str(link_id) if link_id is not None else None,
            )
            db.add(task)
            db.commit()
            logger.info(
                "[Heartbeat] report_to=auto: created BoardTask id=%s assigned to Auto "
                "(agent_id=%s) for source_agent=%s",
                task.id, auto.id, source_agent_id,
            )
        except Exception as e:
            db.rollback()
            logger.error(
                "[Heartbeat] _route_heartbeat_to_auto failed for ws=%s: %s",
                workspace_id, e, exc_info=True,
            )
        finally:
            db.close()

    async def _route_heartbeat_to_webhook(
        self,
        *,
        webhook_url: str,
        title: str,
        message: str,
        agent_id: Optional[int],
        agent_name: Optional[str],
        link_id: Optional[Any],
        status: str,
    ) -> None:
        """POST the heartbeat result directly to a per-agent webhook URL.

        Bypasses the workspace-level NotificationDispatcher because the
        agent has its own destination. Failure is logged but never raised
        — heartbeat completion isn't blocked by a flaky downstream URL.
        """
        try:
            import httpx
            payload = {
                "event": "heartbeat_complete",
                "title": title,
                "message": (message or "")[:4000],
                "agent_id": agent_id,
                "agent_name": agent_name,
                "status": status,
                "report_id": str(link_id) if link_id is not None else None,
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload)
                logger.info(
                    "[Heartbeat] report_to=webhook: POST %s → %s (agent=%s)",
                    webhook_url, resp.status_code, agent_id,
                )
        except Exception as e:
            logger.warning(
                "[Heartbeat] Per-agent webhook delivery failed for agent=%s: %s",
                agent_id, e,
            )

    # ------------------------------------------------------------------
    # Public API (manual triggers)
    # ------------------------------------------------------------------

    async def run_orchestrator_heartbeat(self, workspace_id: str) -> dict:
        """Manually trigger an orchestrator heartbeat and return the result."""
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace

        db = SessionLocal()
        try:
            ws = db.query(Workspace).get(workspace_id)
            if not ws:
                return {"status": "error", "detail": "Workspace not found"}
            hb_config = (
                (ws.settings or {})
                .get("orchestrator", {})
                .get("heartbeat", {})
            )
        finally:
            db.close()

        # Force run regardless of active hours AND cooldown — clicking Run
        # Now from the UI should always trigger a tick.
        hb_config_override = {
            **hb_config,
            "active_hours_start": "00:00",
            "active_hours_end": "23:59",
            "inherit_active_hours": False,  # manual runs always execute
            "force_run": True,
        }
        return await self._orchestrator_tick(workspace_id, hb_config_override)

    async def run_agent_heartbeat(self, agent_id: int) -> dict:
        """Manually trigger an agent heartbeat and return the result."""
        from core.database.database import SessionLocal
        from core.models import Agent

        db = SessionLocal()
        try:
            agent = db.query(Agent).get(agent_id)
            if not agent:
                return {"status": "error", "detail": "Agent not found"}
            hb_config = (agent.configuration or {}).get("heartbeat", {})
            workspace_id = str(agent.workspace_id)
        finally:
            db.close()

        hb_config_override = {
            **hb_config,
            "active_hours_start": "00:00",
            "active_hours_end": "23:59",
            "inherit_active_hours": False,  # manual runs always execute
            "force_run": True,             # bypass cooldown so daily agents are testable
        }
        return await self._agent_tick(agent_id, workspace_id, hb_config_override)

    def get_status(self) -> dict:
        """Return status of all scheduled heartbeat jobs."""
        if not self._scheduler:
            return {"active": False, "jobs": []}

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "next_run_at": (
                        job.next_run_time.isoformat()
                        if job.next_run_time
                        else None
                    ),
                    "trigger": str(job.trigger),
                }
            )

        return {
            "active": (
                self._scheduler.running if self._scheduler else False
            ),
            "jobs": jobs,
            "running_ticks": list(self._running_ticks.keys()),
        }

    # ------------------------------------------------------------------
    # PRD-76: Auto-create report from heartbeat result
    # ------------------------------------------------------------------

    async def _auto_create_report(
        self, agent_id: int, workspace_id: str, result: dict
    ):
        """
        Auto-create a report row from heartbeat result data.
        Ensures every heartbeat run has a corresponding report —
        even if the agent didn't call platform_submit_report.
        """
        try:
            from core.database.database import SessionLocal
            from core.models import Agent
            from services.report_service import ReportService, compute_execution_metrics

            db = SessionLocal()
            try:
                agent = db.query(Agent).get(agent_id)
                agent_name = agent.name if agent else f"agent-{agent_id}"

                # Build markdown content from findings
                findings = result.get("findings", [])
                actions = result.get("actions_taken", [])
                hb_status = result.get("status", "success")
                tokens = result.get("tokens_used", 0)
                started_at = result.get("_run_started_at")
                completed_at = result.get("_run_completed_at")

                # Pull cost/model/duration rollup from llm_usage
                exec_metrics = compute_execution_metrics(
                    db,
                    workspace_id,
                    agent_id=agent_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    extra={
                        "findings_count": len(findings),
                        "actions_count": len(actions),
                        "trigger": "heartbeat",
                    },
                )
                if exec_metrics.get("tokens_used"):
                    tokens = exec_metrics["tokens_used"]

                lines = [
                    f"# {agent_name} — Heartbeat Report",
                    f"**Status:** {hb_status}",
                    "",
                ]

                if findings:
                    lines.append("## Findings")
                    for f in findings:
                        check = f.get("check", "unknown")
                        detail = f.get("detail", "")
                        lines.append(f"- **{check}:** {detail}")
                    lines.append("")

                if actions:
                    lines.append("## Actions Taken")
                    for a in actions:
                        if isinstance(a, dict):
                            lines.append(f"- {a.get('action', '')} → {a.get('result', '')}")
                        else:
                            lines.append(f"- {a}")
                    lines.append("")

                lines.append("## Execution Metrics")
                lines.append(f"- Model: {exec_metrics.get('model') or 'unknown'}")
                lines.append(f"- LLM calls: {exec_metrics.get('llm_calls', 0)}")
                lines.append(f"- Tokens (in/out/total): "
                             f"{exec_metrics.get('input_tokens', 0)} / "
                             f"{exec_metrics.get('output_tokens', 0)} / "
                             f"{exec_metrics.get('tokens_used', tokens)}")
                lines.append(f"- Cost: ${exec_metrics.get('cost_usd', 0):.4f}")
                duration_ms = exec_metrics.get('duration_ms')
                if duration_ms is not None:
                    lines.append(f"- Duration: {duration_ms} ms")
                lines.append(f"- Findings: {len(findings)}")
                lines.append(f"- Actions: {len(actions)}")

                content = "\n".join(lines)

                # Map heartbeat status to report status
                report_status = "ok" if hb_status == "success" else "warning"
                if any(f.get("check") == "error" for f in findings):
                    report_status = "critical"

                # Summary from first finding detail
                summary = None
                for f in findings:
                    detail = f.get("detail", "")
                    if detail and f.get("check") != "error":
                        summary = detail[:497] + "..." if len(detail) > 497 else detail
                        break

                svc = ReportService(db, workspace_id)
                report_result = await svc.create_report(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    title=f"{agent_name} Heartbeat",
                    content=content,
                    report_type="standup",
                    status=report_status,
                    summary=summary,
                    metrics=exec_metrics,
                    heartbeat_result_id=result.get("_heartbeat_result_id"),
                )

                if report_result.get("success"):
                    logger.info(
                        "[Heartbeat] Auto-created report %s for agent=%s",
                        report_result.get("report_id"), agent_id,
                    )
                else:
                    logger.warning(
                        "[Heartbeat] Auto-report creation failed for agent=%s: %s",
                        agent_id, report_result.get("error"),
                    )
            finally:
                db.close()

        except Exception as e:
            logger.warning(
                "[Heartbeat] Failed to auto-create report for agent=%s: %s",
                agent_id, e,
            )

    async def _auto_create_orchestrator_report(
        self, workspace_id: str, result: dict
    ):
        """
        Auto-create a report for orchestrator heartbeat ticks.
        Uses agent_id=None with agent_name='Orchestrator'.
        """
        try:
            from core.database.database import SessionLocal
            from services.report_service import ReportService

            findings = result.get("findings", [])
            actions = result.get("actions_taken", [])
            hb_status = result.get("status", "success")
            tokens = result.get("tokens_used", 0)

            lines = [
                "# Orchestrator — Heartbeat Report",
                f"**Status:** {hb_status}",
                "",
            ]

            if findings:
                lines.append("## Findings")
                for f in findings:
                    check = f.get("check", "unknown")
                    detail = f.get("detail", "")
                    lines.append(f"- **{check}:** {detail}")
                lines.append("")

            if actions:
                lines.append("## Actions Taken")
                for a in actions:
                    if isinstance(a, dict):
                        lines.append(f"- {a.get('tool', '')}({a.get('params', '')})")
                    else:
                        lines.append(f"- {a}")
                lines.append("")

            lines.append("## Metrics")
            lines.append(f"- Tokens used: {tokens}")
            lines.append(f"- Findings: {len(findings)}")
            lines.append(f"- Actions: {len(actions)}")

            content = "\n".join(lines)

            report_status = "ok" if hb_status == "success" else "warning"
            if any(f.get("check") == "error" for f in findings):
                report_status = "critical"

            summary = None
            for f in findings:
                detail = f.get("detail", "")
                if detail and f.get("check") != "error":
                    summary = detail[:497] + "..." if len(detail) > 497 else detail
                    break

            db = SessionLocal()
            try:
                svc = ReportService(db, workspace_id)
                report_result = await svc.create_report(
                    agent_id=None,
                    agent_name="Orchestrator",
                    title="Orchestrator Heartbeat",
                    content=content,
                    report_type="standup",
                    status=report_status,
                    summary=summary,
                    metrics={
                        "tokens_used": tokens,
                        "findings_count": len(findings),
                        "actions_count": len(actions),
                    },
                    heartbeat_result_id=result.get("_heartbeat_result_id"),
                )

                if report_result.get("success"):
                    logger.info(
                        "[Heartbeat] Auto-created orchestrator report %s for ws=%s",
                        report_result.get("report_id"), workspace_id,
                    )
                else:
                    logger.warning(
                        "[Heartbeat] Orchestrator report creation failed for ws=%s: %s",
                        workspace_id, report_result.get("error"),
                    )
            finally:
                db.close()

        except Exception as e:
            logger.warning(
                "[Heartbeat] Failed to auto-create orchestrator report for ws=%s: %s",
                workspace_id, e,
            )


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_heartbeat_service: Optional[HeartbeatService] = None


def get_heartbeat_service() -> HeartbeatService:
    global _heartbeat_service
    if _heartbeat_service is None:
        _heartbeat_service = HeartbeatService()
    return _heartbeat_service
