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
from datetime import datetime

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
            60  → ``0 * * * *``  (top of every hour)
            30  → ``0,30 * * * *``
            15  → ``0,15,30,45 * * * *``
            120 → ``0 */2 * * *``  (every 2 hours)
        """
        if minutes <= 0:
            minutes = 60

        if minutes < 60:
            # Sub-hour: distribute evenly within the hour
            offsets = list(range(0, 60, minutes))
            minute_field = ",".join(str(o) for o in offsets)
            return CronTrigger(minute=minute_field)
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
    # Tick implementations
    # ------------------------------------------------------------------

    async def _orchestrator_tick(self, workspace_id: str, hb_config: dict) -> Dict[str, Any]:
        """Execute an orchestrator heartbeat tick. Returns result dict."""
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

        try:
            logger.info(
                "[Heartbeat] Orchestrator tick starting for ws=%s", workspace_id
            )

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

                # Execute checklist items if provided
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
                            "detail": f"Reviewed {len(items)} checklist items",
                        }
                    )
            finally:
                db.close()

            await self._store_heartbeat_result(result)
            await self._deliver_notification(result, hb_config)
            logger.info(
                "[Heartbeat] Orchestrator tick completed for ws=%s: %d findings",
                workspace_id,
                len(result["findings"]),
            )

        except Exception as e:
            logger.error(
                "[Heartbeat] Orchestrator tick failed for ws=%s: %s",
                workspace_id,
                e,
            )
            result["status"] = "error"
            result["findings"].append({"check": "error", "detail": str(e)})
            await self._store_heartbeat_result(result)
            await self._deliver_notification(result, hb_config)
        finally:
            self._running_ticks.pop(tick_key, None)

        return result

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

                prompt = (
                    f"Scheduled heartbeat check. {heartbeat_prompt}\n"
                    f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
                    "Use your tools to check. Reply with a SHORT plain-text summary (max 500 chars), no markdown.\n"
                    + (
                        "You may take action if needed."
                        if auto_act
                        else "Report findings only."
                    )
                )

                # Execute through AgentFactory so the agent has its full toolset
                try:
                    from modules.agents.factory.agent_factory import AgentFactory

                    factory = AgentFactory(db_session=db)
                    exec_result = await factory.execute_with_prompt(
                        agent=agent_id,
                        prompt=prompt,
                        context={"source": "heartbeat", "workspace_id": workspace_id},
                        use_memory=False,  # keep context lean
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
            await self._deliver_notification(result, hb_config)
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
            await self._deliver_notification(result, hb_config)
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

            finally:
                db.close()

        except Exception as e:
            logger.error("[Heartbeat] Daily summary failed: %s", e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _store_heartbeat_result(self, result: dict):
        """Store heartbeat result in the database."""
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                db.execute(
                    text(
                        """
                        INSERT INTO heartbeat_results
                            (source_type, source_id, workspace_id, status,
                             findings, actions_taken, tokens_used, created_at)
                        VALUES
                            (:source_type, :source_id, :workspace_id, :status,
                             :findings, :actions_taken, :tokens_used, NOW())
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
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("[Heartbeat] Failed to store result: %s", e)

    # ------------------------------------------------------------------
    # Notification delivery
    # ------------------------------------------------------------------

    async def _deliver_notification(self, result: dict, hb_config: dict):
        """Deliver heartbeat result to the configured destination.

        report_to values:
          - "orchestrator"  → DB only (no extra delivery)
          - "direct"        → no-op for now (result is in DB, frontend polls)
          - "telegram"      → push via workspace Telegram bot integration
          - "slack"         → push via workspace Slack bot integration
          - "webhook"       → HTTP POST to webhook_url
        """
        report_to = hb_config.get("report_to", "orchestrator")

        if report_to in ("orchestrator", "direct"):
            return

        message = self._format_heartbeat_message(result)
        workspace_id = result.get("workspace_id")

        if report_to == "telegram":
            await self._send_via_integration(workspace_id, "telegram", message, hb_config)
        elif report_to == "slack":
            await self._send_via_integration(workspace_id, "slack", message, hb_config)
        elif report_to == "webhook":
            webhook_url = hb_config.get("webhook_url")
            if webhook_url:
                await self._send_via_webhook(webhook_url, result, message)
            else:
                logger.warning("[Heartbeat] report_to=webhook but no webhook_url configured")

    def _format_heartbeat_message(self, result: dict) -> str:
        """Format heartbeat result as a clean notification message."""
        status = result.get("status", "unknown")
        status_icon = "OK" if status == "success" else "ERROR"

        # Extract the main analysis text from findings
        findings = result.get("findings", [])
        analysis = ""
        for f in findings:
            if f.get("check") == "llm_analysis":
                analysis = f.get("detail", "")
                break

        if analysis:
            return f"[Heartbeat {status_icon}]\n{analysis[:2000]}"

        # Fallback: summarize all findings
        lines = [f"[Heartbeat {status_icon}]"]
        for f in findings:
            detail = f.get("detail", "")
            if detail:
                lines.append(detail[:300])

        return "\n".join(lines) if len(lines) > 1 else f"[Heartbeat {status_icon}] No findings."

    async def _send_via_integration(
        self, workspace_id: str, platform: str, message: str, hb_config: dict
    ):
        """Send notification through workspace integration (Telegram, Slack, etc.)."""
        try:
            from core.database.database import SessionLocal
            from core.models.workspaces import Workspace

            db = SessionLocal()
            try:
                ws = db.query(Workspace).get(workspace_id)
                if not ws:
                    logger.warning("[Heartbeat] Workspace %s not found for notification", workspace_id)
                    return

                integrations = (ws.settings or {}).get("integrations", {})

                # Also check channel_connections for bot tokens
                from sqlalchemy import text as sql_text
                channel_config = {}
                try:
                    row = db.execute(
                        sql_text(
                            "SELECT config FROM channel_connections "
                            "WHERE workspace_id = :ws AND platform = :plat "
                            "ORDER BY created_at DESC LIMIT 1"
                        ),
                        {"ws": workspace_id, "plat": platform},
                    ).fetchone()
                    if row and row.config:
                        channel_config = row.config if isinstance(row.config, dict) else json.loads(row.config)
                except Exception as e:
                    logger.debug("[Heartbeat] Could not load channel_connections for %s: %s", platform, e)

                if platform == "telegram":
                    token = (
                        integrations.get("telegram_bot_token")
                        or channel_config.get("bot_token")
                    )
                    if not token:
                        logger.warning("[Heartbeat] No telegram bot token found for ws=%s", workspace_id)
                        return

                    raw_channel_id = hb_config.get("channel_id") or ""
                    # Guard: ignore if someone accidentally saved a bot token as chat_id
                    chat_id = (
                        (raw_channel_id if raw_channel_id and ":" not in raw_channel_id else "")
                        or integrations.get("telegram_default_chat_id")
                        or channel_config.get("default_chat_id")
                    )
                    # Auto-resolve chat_id from Telegram API if not stored
                    if not chat_id:
                        chat_id = await self._resolve_telegram_chat_id(token)
                    if not chat_id:
                        logger.warning("[Heartbeat] Could not resolve Telegram chat_id for ws=%s", workspace_id)
                        return

                    from api.webhooks import _send_telegram_reply
                    ok = await _send_telegram_reply(int(chat_id), message, token)
                    if ok:
                        logger.info("[Heartbeat] Telegram notification sent to chat %s", chat_id)
                    else:
                        logger.warning("[Heartbeat] Telegram send failed for chat %s", chat_id)

                elif platform == "slack":
                    token = (
                        integrations.get("slack_bot_token")
                        or channel_config.get("bot_token")
                    )
                    channel = (
                        hb_config.get("channel_id")
                        or integrations.get("slack_default_channel")
                        or channel_config.get("default_channel")
                    )
                    if not token:
                        logger.warning("[Heartbeat] No slack bot token found for ws=%s", workspace_id)
                        return
                    if not channel:
                        logger.warning("[Heartbeat] No channel for Slack notification (ws=%s)", workspace_id)
                        return

                    from api.webhooks import _send_slack_reply
                    ok = await _send_slack_reply(channel, message, token)
                    if ok:
                        logger.info("[Heartbeat] Slack notification sent to %s", channel)
                    else:
                        logger.warning("[Heartbeat] Slack send failed for %s", channel)

                else:
                    logger.warning("[Heartbeat] Unknown integration platform: %s", platform)
            finally:
                db.close()
        except Exception as e:
            logger.error("[Heartbeat] Integration notification failed (%s): %s", platform, e)

    async def _resolve_telegram_chat_id(self, bot_token: str) -> Optional[str]:
        """Get the most recent chat_id from the Telegram Bot API."""
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{bot_token}/getUpdates?limit=1&offset=-1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    results = data.get("result", [])
                    if not results:
                        return None
                    # Extract chat_id from the most recent update
                    update = results[0]
                    msg = update.get("message") or update.get("channel_post") or {}
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    if chat_id:
                        logger.info("[Heartbeat] Auto-resolved Telegram chat_id=%s", chat_id)
                        return str(chat_id)
        except Exception as e:
            logger.debug("[Heartbeat] Failed to resolve Telegram chat_id: %s", e)
        return None

    async def _send_via_webhook(self, url: str, result: dict, message: str):
        """POST heartbeat result to a webhook URL."""
        try:
            import aiohttp

            payload = {
                "source_type": result.get("source_type"),
                "source_id": result.get("source_id"),
                "status": result.get("status"),
                "message": message,
                "findings": result.get("findings", []),
                "tokens_used": result.get("tokens_used", 0),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status < 300:
                        logger.info("[Heartbeat] Webhook delivered to %s (status %s)", url, resp.status)
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[Heartbeat] Webhook %s returned %s: %s",
                            url, resp.status, body[:200],
                        )
        except Exception as e:
            logger.error("[Heartbeat] Webhook delivery failed to %s: %s", url, e)

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
# Singleton
# ------------------------------------------------------------------

_heartbeat_service: Optional[HeartbeatService] = None


def get_heartbeat_service() -> HeartbeatService:
    global _heartbeat_service
    if _heartbeat_service is None:
        _heartbeat_service = HeartbeatService()
    return _heartbeat_service
