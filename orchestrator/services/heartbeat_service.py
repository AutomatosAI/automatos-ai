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
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


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
        logger.info("[Heartbeat] Service started (daily summary at 01:00 UTC)")

    async def stop(self):
        """Remove heartbeat jobs. Only shuts down scheduler if we own it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=True)
            logger.info("[Heartbeat] Standalone scheduler stopped")
        logger.info("[Heartbeat] Service stopped")

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    async def _load_heartbeat_configs(self):
        """Load all active heartbeat configs from DB and schedule jobs."""
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace
        from core.models import Agent

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

        Examples:
            15    → ``0,15,30,45 * * * *``
            30    → ``0,30 * * * *``
            60    → ``0 * * * *``  (top of every hour)
            120   → ``0 */2 * * *``  (every 2 hours)
            480   → ``0 */8 * * *``  (every 8 hours)
            1440  → ``0 9 * * *``   (daily at 9am)
            10080 → ``0 9 * * 1``   (weekly, Monday 9am)
        """
        if minutes <= 0:
            minutes = 60

        if minutes < 60:
            # Sub-hour: distribute evenly within the hour
            offsets = list(range(0, 60, minutes))
            minute_field = ",".join(str(o) for o in offsets)
            return CronTrigger(minute=minute_field)
        elif minutes >= 10080:
            # Weekly: Monday at 9am
            return CronTrigger(minute="0", hour="9", day_of_week="mon")
        elif minutes >= 1440:
            # Daily: at 9am
            return CronTrigger(minute="0", hour="9")
        else:
            # Hourly or multi-hour
            hours = minutes // 60
            if hours == 1:
                return CronTrigger(minute="0")
            return CronTrigger(minute="0", hour=f"*/{hours}")

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

    async def _orchestrator_tick(self, workspace_id: str, hb_config: dict) -> Dict[str, Any]:
        """Execute an LLM-powered orchestrator heartbeat tick."""
        tick_key = f"orch_{workspace_id}"
        if self._running_ticks.get(tick_key):
            logger.debug(
                "[Heartbeat] Orchestrator tick already running for ws=%s, skipping",
                workspace_id,
            )
            return {"status": "skipped", "reason": "already_running"}

        if not await self._is_within_active_hours(hb_config, workspace_id):
            logger.debug(
                "[Heartbeat] Outside active hours for ws=%s, skipping",
                workspace_id,
            )
            return {"status": "skipped", "reason": "outside_active_hours"}

        interval_minutes = hb_config.get("interval_minutes", 30)
        if await self._was_recently_executed("orchestrator", workspace_id, interval_minutes):
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
        from core.database.database import SessionLocal
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

        task_description = (
            f"Perform a scheduled health check for this workspace.\n"
            f"Personality: {personality_mode}.\n\n"
            f"Analyze your workspace using the tools provided.{checklist_block}\n\n"
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

            # 5. Tool loop (max 5 iterations)
            max_iterations = 5
            total_tokens = 0
            executor = PlatformActionExecutor(db, workspace_id)

            for iteration in range(max_iterations):
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

        if not await self._is_within_active_hours(hb_config, workspace_id):
            logger.info(
                "[Heartbeat] Outside active hours for agent=%s (ws=%s), skipping",
                agent_id, workspace_id,
            )
            return {"status": "skipped", "reason": "outside_active_hours"}

        interval_minutes = hb_config.get("interval_minutes", 60)
        if await self._was_recently_executed("agent", str(agent_id), interval_minutes):
            return {"status": "skipped", "reason": "already_ran_this_period"}

        self._running_ticks[tick_key] = True
        result: Dict[str, Any] = {
            "source_type": "agent",
            "source_id": str(agent_id),
            "workspace_id": workspace_id,
            "status": "success",
            "findings": [],
            "actions_taken": [],
            "tokens_used": 0,
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

                # --- PRD-72: Scan for assigned board tasks ---
                assigned_tasks = []
                task_context = ""
                try:
                    from core.models.core import BoardTask
                    assigned_tasks = (
                        db.query(BoardTask)
                        .filter(
                            BoardTask.assigned_agent_id == agent_id,
                            BoardTask.status == "assigned",
                            BoardTask.workspace_id == workspace_id,
                        )
                        .order_by(BoardTask.priority.desc(), BoardTask.created_at.asc())
                        .limit(3)
                        .all()
                    )
                    if assigned_tasks:
                        task_lines = []
                        for t in assigned_tasks:
                            t.status = "in_progress"
                            t.started_at = datetime.utcnow()
                            task_lines.append(
                                f"- [TASK-{t.id}] {t.title}: {t.description or ''}"
                            )
                        db.commit()
                        task_context = (
                            "\n\n## ASSIGNED TASKS (Priority Work)\n"
                            "You have tasks assigned to you. Complete them and report results.\n"
                            + "\n".join(task_lines)
                            + "\n\nAfter completing each task, use platform_submit_report or respond with results."
                        )
                        logger.info(
                            "[Heartbeat] Agent %s picked up %d assigned tasks",
                            agent_id,
                            len(assigned_tasks),
                        )
                except Exception as task_err:
                    logger.warning(
                        "[Heartbeat] Failed to scan board tasks for agent=%s: %s",
                        agent_id,
                        task_err,
                    )

                prompt = (
                    f"Scheduled heartbeat check. {heartbeat_prompt}\n"
                    "Use your tools to check. Reply with a SHORT plain-text summary (max 500 chars), no markdown.\n"
                    + (
                        "You may take action if needed."
                        if auto_act
                        else "Report findings only."
                    )
                    + task_context
                )

                # Execute through AgentFactory so the agent has its full toolset
                try:
                    from modules.agents.factory.agent_factory import AgentFactory

                    from modules.context.modes import ContextMode

                    factory = AgentFactory(db_session=db)
                    exec_result = await factory.execute_with_prompt(
                        agent=agent_id,
                        prompt=prompt,
                        context={"source": "heartbeat", "workspace_id": workspace_id},
                        context_mode=ContextMode.HEARTBEAT_AGENT,
                    )

                    # Extract the actual text from nested result
                    llm_text = ""
                    if isinstance(exec_result, dict):
                        llm_text = (
                            exec_result.get("result")
                            or exec_result.get("response")
                            or exec_result.get("output")
                            or exec_result.get("content")
                            or ""
                        )
                        # Handle nested dict in result
                        if isinstance(llm_text, dict):
                            llm_text = llm_text.get("result") or llm_text.get("response") or str(llm_text)
                    if not llm_text:
                        llm_text = str(exec_result)[:500]

                    result["findings"].append(
                        {"check": "llm_analysis", "detail": str(llm_text)[:1000]}
                    )
                    result["tokens_used"] = exec_result.get("tokens_used", 0) if isinstance(exec_result, dict) else 0

                    # PRD-72: Auto-complete board tasks after successful execution
                    if assigned_tasks:
                        try:
                            from core.models.core import BoardTask as BT
                            for t in assigned_tasks:
                                t_fresh = db.query(BT).get(t.id)
                                if t_fresh and t_fresh.status == "in_progress":
                                    t_fresh.status = "done" if t_fresh.review_mode == "auto" else "review"
                                    t_fresh.completed_at = datetime.utcnow()
                                    t_fresh.result = str(llm_text)[:2000]
                            db.commit()
                            logger.info(
                                "[Heartbeat] Agent %s completed %d tasks",
                                agent_id,
                                len(assigned_tasks),
                            )
                        except Exception as tc_err:
                            logger.warning(
                                "[Heartbeat] Failed to complete board tasks for agent=%s: %s",
                                agent_id,
                                tc_err,
                            )

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

                    summary_text = (
                        f"Heartbeat Daily Summary ({cutoff.strftime('%Y-%m-%d')} to {datetime.utcnow().strftime('%Y-%m-%d')}):\n"
                        f"- Total ticks: {len(ws_rows)}\n"
                        f"- Successful: {success_count}\n"
                        f"- Errors: {error_count}\n"
                        f"- Tokens used: {total_tokens}\n"
                    )

                    # Add notable findings
                    notable = []
                    for r in ws_rows:
                        findings = r.findings if isinstance(r.findings, list) else json.loads(r.findings or "[]")
                        for f in findings:
                            if f.get("check") in ("error", "llm_error"):
                                notable.append(f"[{r.source_type}/{r.source_id}] {f.get('detail', '')[:120]}")
                    if notable:
                        summary_text += "\nNotable issues:\n" + "\n".join(f"- {n}" for n in notable[:10])

                    # Store in SmartMemoryManager as long-term memory
                    try:
                        from consumers.chatbot.smart_memory import get_smart_memory_manager

                        mem_mgr = get_smart_memory_manager()
                        await mem_mgr.store_conversation(
                            workspace_id=ws_id,
                            agent_id=None,
                            user_message="Daily heartbeat summary request",
                            assistant_response=summary_text,
                            chat_id=None,
                        )
                        logger.info("[Heartbeat] Stored daily summary for ws=%s (%d ticks)", ws_id, len(ws_rows))
                    except Exception as mem_err:
                        logger.warning("[Heartbeat] Failed to store summary in memory for ws=%s: %s", ws_id, mem_err)

                    # L2: Store heartbeat summary in short-term memory (fire-and-forget)
                    try:
                        from modules.memory.unified_memory_service import get_unified_memory_service

                        unified = get_unified_memory_service()
                        asyncio.create_task(
                            unified.store_short_term(
                                workspace_id=ws_id,
                                content=summary_text[:1500],
                                content_type="heartbeat_log",
                                importance=0.4,
                                metadata={
                                    "type": "heartbeat_daily_summary",
                                    "date": cutoff.strftime("%Y-%m-%d"),
                                    "tick_count": len(ws_rows),
                                    "success_count": success_count,
                                    "error_count": error_count,
                                },
                            )
                        )
                    except Exception:
                        logger.debug(
                            "[Heartbeat] L2 store_short_term failed for ws=%s",
                            ws_id,
                            exc_info=True,
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
                             findings, actions_taken, tokens_used, created_at)
                        VALUES
                            (:source_type, :source_id, :workspace_id, :status,
                             :findings, :actions_taken, :tokens_used, NOW())
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

            # Resolve agent metadata (agent tick only)
            agent_id: Optional[int] = None
            agent_name: Optional[str] = None
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
                    finally:
                        db_lookup.close()
                title = f"{agent_name or 'Agent'} Heartbeat"
            else:
                agent_name = "Orchestrator"
                title = "Orchestrator Heartbeat"

            link_id = result.get("_heartbeat_result_id")

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

        # Force run regardless of active hours
        hb_config_override = {
            **hb_config,
            "active_hours_start": "00:00",
            "active_hours_end": "23:59",
            "inherit_active_hours": False,  # manual runs always execute
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
            from services.report_service import ReportService

            db = SessionLocal()
            try:
                agent = db.query(Agent).get(agent_id)
                agent_name = agent.name if agent else f"agent-{agent_id}"

                # Build markdown content from findings
                findings = result.get("findings", [])
                actions = result.get("actions_taken", [])
                hb_status = result.get("status", "success")
                tokens = result.get("tokens_used", 0)

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

                lines.append("## Metrics")
                lines.append(f"- Tokens used: {tokens}")
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
                    metrics={
                        "tokens_used": tokens,
                        "findings_count": len(findings),
                        "actions_count": len(actions),
                    },
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
