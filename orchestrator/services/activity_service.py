"""
Activity Service (PRD-72 US-004)
=================================

Merges chats, heartbeat executions, and recipe executions into a unified
activity feed. Provides get_feed() and get_stats() for the Activity
Command Centre page.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import desc, exists, func, text
from sqlalchemy.orm import Session

from core.models.core import (
    Agent,
    Chat,
    Message,
    RecipeExecution,
    WorkflowTemplate,
)

logger = logging.getLogger(__name__)


class ActivityService:
    """Request-scoped service — accepts a SQLAlchemy Session."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id
        self._ws_str = str(workspace_id)

    # ── Public API ────────────────────────────────────────────────

    def get_feed(
        self,
        *,
        types: Optional[List[str]] = None,
        status: Optional[str] = None,
        period: str = "7d",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Return a unified activity feed sorted by started_at DESC.

        Parameters
        ----------
        types : list of 'chat' | 'routine' | 'recipe'
            Filter to specific activity types.  None = all.
        status : str or None
            Filter by mapped status (working / done / attention / upcoming).
        period : str
            Time window — '1d', '7d', '30d', '90d'.
        limit / offset : pagination controls.
        """
        since = self._period_start(period)
        items: List[Dict[str, Any]] = []

        want_all = types is None or len(types) == 0
        want_types = set(types) if types else set()

        if want_all or "chat" in want_types:
            items.extend(self._fetch_chats(since, limit + offset))

        if want_all or "routine" in want_types:
            items.extend(self._fetch_routines(since, limit + offset))

        if want_all or "recipe" in want_types:
            items.extend(self._fetch_recipes(since, limit + offset))

        # Apply status filter after merge
        if status:
            mapped = self._map_status_filter(status)
            items = [i for i in items if i["status"] in mapped]

        # Sort merged results by started_at DESC
        items.sort(key=lambda x: x["started_at"] or "", reverse=True)

        total = len(items)
        page = items[offset : offset + limit]

        return {"items": page, "total": total, "limit": limit, "offset": offset}

    def get_stats(self, *, period: str = "1d") -> Dict[str, Any]:
        """Return hero-card stats for the Activity Command Centre.

        Returns working_now, channels_live, completed_today, needs_attention.
        """
        since = self._period_start(period)

        working_now = self._count_working_now()
        channels_live = self._count_channels_live()
        completed_today = self._count_completed(since)
        needs_attention = self._count_needs_attention(since)

        return {
            "working_now": working_now,
            "channels_live": channels_live,
            "completed_today": completed_today,
            "needs_attention": needs_attention,
            "period": period,
        }

    # ── Chat Fetching ─────────────────────────────────────────────

    def _fetch_chats(self, since: datetime, max_rows: int) -> List[Dict[str, Any]]:
        """Fetch recent chats scoped to workspace via messages join."""
        try:
            # Subquery: chats that have messages in this workspace
            ws_filter = exists().where(
                (Message.chat_id == Chat.id)
                & (Message.workspace_id == self.workspace_id)
            )
            chats = (
                self.db.query(Chat)
                .filter(ws_filter, Chat.updated_at >= since)
                .order_by(desc(Chat.updated_at))
                .limit(max_rows)
                .all()
            )

            items: List[Dict[str, Any]] = []
            for chat in chats:
                # Get first user message for summary
                first_msg = (
                    self.db.query(Message)
                    .filter(
                        Message.chat_id == chat.id,
                        Message.role == "user",
                    )
                    .order_by(Message.created_at)
                    .first()
                )
                summary = self._extract_message_text(first_msg)

                # Get the agent used (from last_context if available)
                agent_info = self._agent_from_chat_context(chat.last_context)

                items.append(
                    self._build_feed_item(
                        id=str(chat.id),
                        item_type="chat",
                        name=chat.title or "Untitled Chat",
                        status="completed",  # Chats don't have running state in feed
                        started_at=chat.created_at,
                        completed_at=chat.updated_at,
                        agent=agent_info,
                        summary=summary,
                        source_id=str(chat.id),
                        source_url=f"/chat/{chat.id}",
                        trigger=None,
                    )
                )
            return items
        except Exception as e:
            logger.error("Failed to fetch chats for activity feed: %s", e, exc_info=True)
            self.db.rollback()
            return []

    # ── Routine (Heartbeat) Fetching ──────────────────────────────

    def _fetch_routines(self, since: datetime, max_rows: int) -> List[Dict[str, Any]]:
        """Fetch recent heartbeat executions as routine activity items."""
        try:
            rows = self.db.execute(
                text("""
                    SELECT hr.id, hr.source_id, hr.source_type, hr.status, hr.findings,
                           hr.tokens_used, hr.created_at,
                           a.name AS agent_name, a.marketplace_icon AS agent_icon
                    FROM heartbeat_results hr
                    LEFT JOIN agents a ON hr.source_type = 'agent'
                        AND hr.source_id ~ '^\d+$'
                        AND a.id = CAST(hr.source_id AS INTEGER)
                    WHERE hr.workspace_id = :ws_id
                      AND hr.source_type IN ('agent', 'orchestrator')
                      AND hr.created_at >= :since
                    ORDER BY hr.created_at DESC
                    LIMIT :lim
                """),
                {"ws_id": self._ws_str, "since": since, "lim": max_rows},
            ).fetchall()

            items: List[Dict[str, Any]] = []
            for r in rows:
                # Map heartbeat status → feed status
                feed_status = "completed" if r.status == "success" else "failed"
                summary = self._routine_summary(r.findings)
                error_msg = self._routine_error(r.status, r.findings)

                agent_name = r.agent_name or ("Orchestrator" if r.source_type == "orchestrator" else "Unknown Agent")
                is_orchestrator = r.source_type == "orchestrator"
                agent_info = None if is_orchestrator else {
                    "id": int(r.source_id) if r.source_id else None,
                    "name": r.agent_name or "Unknown Agent",
                    "avatar_url": r.agent_icon,
                }
                source_url = None if is_orchestrator else f"/agents/{r.source_id}"

                items.append(
                    self._build_feed_item(
                        id=f"routine-{r.id}",
                        item_type="routine",
                        name=f"{agent_name} Routine",
                        status=feed_status,
                        started_at=r.created_at,
                        completed_at=r.created_at,  # Heartbeats are near-instant
                        agent=agent_info,
                        summary=summary,
                        source_id=r.source_id,
                        source_url=source_url,
                        trigger="heartbeat",
                        error_message=error_msg,
                        tokens_used=r.tokens_used,
                    )
                )
            return items
        except Exception as e:
            logger.error("Failed to fetch routines for activity feed: %s", e, exc_info=True)
            self.db.rollback()
            return []

    # ── Recipe Execution Fetching ─────────────────────────────────

    def _fetch_recipes(self, since: datetime, max_rows: int) -> List[Dict[str, Any]]:
        """Fetch recent recipe executions as activity items."""
        try:
            executions = (
                self.db.query(RecipeExecution)
                .filter(
                    RecipeExecution.workspace_id == self.workspace_id,
                    RecipeExecution.started_at >= since,
                )
                .order_by(desc(RecipeExecution.started_at))
                .limit(max_rows)
                .all()
            )

            # Batch-fetch recipe names
            recipe_ids = {e.recipe_id for e in executions}
            recipes_by_id: Dict[int, WorkflowTemplate] = {}
            if recipe_ids:
                recipes = (
                    self.db.query(WorkflowTemplate)
                    .filter(WorkflowTemplate.id.in_(recipe_ids))
                    .all()
                )
                recipes_by_id = {r.id: r for r in recipes}

            items: List[Dict[str, Any]] = []
            for ex in executions:
                recipe = recipes_by_id.get(ex.recipe_id)
                recipe_name = recipe.name if recipe else f"Recipe #{ex.recipe_id}"

                # Calculate duration
                duration = None
                if ex.completed_at and ex.started_at:
                    duration = (ex.completed_at - ex.started_at).total_seconds()

                # Build step progress from step_results
                step_progress = self._build_step_progress(ex)

                # Map trigger
                trigger = self._map_recipe_trigger(ex.triggered_by)

                # Build deep-link to ExecutionKitchen
                exec_id = getattr(ex, "execution_id", None) or str(ex.id)
                recipe_template_id = recipe.template_id if recipe else str(ex.recipe_id)
                source_url = f"/activity?openExecution={exec_id}&recipeId={recipe_template_id}"

                # Aggregate tokens and duration from step_results
                total_tokens = 0
                total_duration_ms = 0
                step_res = ex.step_results or []
                if isinstance(step_res, list):
                    for sr in step_res:
                        if isinstance(sr, dict):
                            total_tokens += sr.get("tokens", 0) or 0
                            total_duration_ms += sr.get("duration_ms", 0) or 0

                items.append(
                    self._build_feed_item(
                        id=f"recipe-{exec_id}",
                        item_type="recipe",
                        name=recipe_name,
                        status=ex.status or "pending",
                        started_at=ex.started_at,
                        completed_at=ex.completed_at,
                        duration_seconds=duration,
                        agent=None,  # Recipes may use multiple agents
                        summary=self._recipe_summary(ex, recipe_name),
                        source_id=str(ex.recipe_id),
                        source_url=source_url,
                        trigger=trigger,
                        step_progress=step_progress,
                        error_message=ex.error_message,
                        total_tokens=total_tokens if total_tokens > 0 else None,
                        total_duration_ms=total_duration_ms if total_duration_ms > 0 else None,
                    )
                )
            return items
        except Exception as e:
            logger.error("Failed to fetch recipes for activity feed: %s", e, exc_info=True)
            self.db.rollback()
            return []

    # ── Stats Helpers ─────────────────────────────────────────────

    def _count_working_now(self) -> int:
        """Count currently running executions (recipes + routines)."""
        try:
            recipe_running = (
                self.db.query(func.count(RecipeExecution.id))
                .filter(
                    RecipeExecution.workspace_id == self.workspace_id,
                    RecipeExecution.status == "running",
                )
                .scalar()
                or 0
            )
            return recipe_running
        except Exception as e:
            logger.error("Failed to count working_now: %s", e)
            return 0

    def _count_channels_live(self) -> int:
        """Count connected channel_connections for this workspace."""
        try:
            row = self.db.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM channel_connections
                    WHERE workspace_id = :ws_id AND status = 'connected'
                """),
                {"ws_id": self._ws_str},
            ).fetchone()
            return row.cnt if row else 0
        except Exception as e:
            # Table may not exist in all environments
            logger.debug("Failed to count channels_live: %s", e)
            return 0

    def _count_completed(self, since: datetime) -> int:
        """Count completed executions within the period."""
        try:
            recipe_done = (
                self.db.query(func.count(RecipeExecution.id))
                .filter(
                    RecipeExecution.workspace_id == self.workspace_id,
                    RecipeExecution.status == "completed",
                    RecipeExecution.completed_at >= since,
                )
                .scalar()
                or 0
            )

            hb_done_row = self.db.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM heartbeat_results
                    WHERE workspace_id = :ws_id
                      AND status = 'success'
                      AND created_at >= :since
                """),
                {"ws_id": self._ws_str, "since": since},
            ).fetchone()
            hb_done = hb_done_row.cnt if hb_done_row else 0

            return recipe_done + hb_done
        except Exception as e:
            logger.error("Failed to count completed: %s", e)
            return 0

    def _count_needs_attention(self, since: datetime) -> int:
        """Count failed executions + stale routines within the period."""
        try:
            recipe_failed = (
                self.db.query(func.count(RecipeExecution.id))
                .filter(
                    RecipeExecution.workspace_id == self.workspace_id,
                    RecipeExecution.status == "failed",
                    RecipeExecution.started_at >= since,
                )
                .scalar()
                or 0
            )

            hb_failed_row = self.db.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM heartbeat_results
                    WHERE workspace_id = :ws_id
                      AND status = 'error'
                      AND created_at >= :since
                """),
                {"ws_id": self._ws_str, "since": since},
            ).fetchone()
            hb_failed = hb_failed_row.cnt if hb_failed_row else 0

            return recipe_failed + hb_failed
        except Exception as e:
            logger.error("Failed to count needs_attention: %s", e)
            return 0

    # ── Feed Item Builder ─────────────────────────────────────────

    @staticmethod
    def _build_feed_item(
        *,
        id: str,
        item_type: str,
        name: str,
        status: str,
        started_at: Optional[datetime],
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
        agent: Optional[Dict[str, Any]] = None,
        summary: str = "",
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        trigger: Optional[str] = None,
        channel: Optional[Dict[str, str]] = None,
        step_progress: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        tokens_used: Optional[int] = None,
        total_tokens: Optional[int] = None,
        total_duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct a unified ActivityFeedItem dict matching PRD-72 schema."""
        if duration_seconds is None and completed_at and started_at:
            duration_seconds = (completed_at - started_at).total_seconds()

        return {
            "id": id,
            "type": item_type,
            "name": name,
            "status": status,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": completed_at.isoformat() if completed_at else None,
            "duration_seconds": round(duration_seconds, 1) if duration_seconds else None,
            "agent": agent,
            "agents": [agent] if agent else [],
            "summary": summary[:300] if summary else "",
            "source_id": source_id,
            "source_url": source_url,
            "trigger": trigger,
            "channel": channel,
            "step_progress": step_progress,
            "error_message": error_message,
            "tokens_used": tokens_used,
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration_ms,
        }

    # ── Internal Helpers ──────────────────────────────────────────

    @staticmethod
    def _period_start(period: str) -> datetime:
        """Convert a period string to a datetime threshold."""
        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(period, 7)
        return datetime.utcnow() - timedelta(days=days)

    @staticmethod
    def _map_status_filter(status: str) -> set:
        """Map user-facing status filter to internal statuses."""
        mapping = {
            "working": {"running", "pending"},
            "done": {"completed"},
            "attention": {"failed"},
            "upcoming": {"scheduled", "paused"},
        }
        return mapping.get(status, {status})

    @staticmethod
    def _extract_message_text(message) -> str:
        """Extract plain text from a Message's JSONB parts field."""
        if not message:
            return ""
        parts = message.parts or []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text", "")
                return text_val[:100]
            if isinstance(part, str):
                return part[:100]
        return ""

    @staticmethod
    def _agent_from_chat_context(last_context) -> Optional[Dict[str, Any]]:
        """Extract agent info from a chat's last_context JSONB."""
        if not last_context or not isinstance(last_context, dict):
            return None
        agent_id = last_context.get("agent_id")
        agent_name = last_context.get("agent_name")
        if agent_id:
            return {
                "id": agent_id,
                "name": agent_name or "Agent",
                "avatar_url": None,
            }
        return None

    @staticmethod
    def _routine_summary(findings) -> str:
        """Build a one-line summary from heartbeat findings JSONB."""
        if not findings:
            return "Routine check completed"
        if isinstance(findings, str):
            try:
                findings = json.loads(findings)
            except (json.JSONDecodeError, TypeError):
                return "Routine check completed"
        if isinstance(findings, list):
            count = len(findings)
            return f"Checked {count} item{'s' if count != 1 else ''}"
        return "Routine check completed"

    @staticmethod
    def _routine_error(status: str, findings) -> Optional[str]:
        """Extract error message from routine findings."""
        if status != "error":
            return None
        if not findings:
            return "Unknown error"
        if isinstance(findings, str):
            try:
                findings = json.loads(findings)
            except (json.JSONDecodeError, TypeError):
                return "Unknown error"
        if isinstance(findings, list):
            for f in findings:
                if isinstance(f, dict) and f.get("check") in ("error", "llm_error"):
                    return f.get("detail", "")[:500]
        return "Heartbeat execution failed"

    @staticmethod
    def _build_step_progress(execution: RecipeExecution) -> Optional[Dict[str, Any]]:
        """Build step_progress dict from recipe execution data."""
        meta = execution.execution_metadata or {}
        total_steps = meta.get("total_steps", 0)
        if not total_steps:
            return None

        step_results = execution.step_results or []
        steps = []
        for i in range(total_steps):
            if i < len(step_results):
                sr = step_results[i] if isinstance(step_results, list) else {}
                step_status = sr.get("status", "completed") if isinstance(sr, dict) else "completed"
                step_name = sr.get("name", f"Step {i + 1}") if isinstance(sr, dict) else f"Step {i + 1}"
            else:
                step_status = "pending"
                step_name = f"Step {i + 1}"
            steps.append({"name": step_name, "status": step_status})

        current = execution.current_step or 0
        return {"current": current, "total": total_steps, "steps": steps}

    @staticmethod
    def _map_recipe_trigger(triggered_by: Optional[str]) -> Optional[str]:
        """Map recipe triggered_by to PRD trigger types."""
        if not triggered_by:
            return "manual"
        triggered_lower = triggered_by.lower()
        if "cron" in triggered_lower or "schedule" in triggered_lower:
            return "scheduled"
        if "webhook" in triggered_lower:
            return "webhook"
        if "event" in triggered_lower:
            return "event"
        return "manual"

    # ── Schedule Endpoint ────────────────────────────────────────

    def get_schedule(self, *, range_days: int = 7) -> Dict[str, Any]:
        """Return upcoming scheduled routines and recipes for the calendar widget."""
        scheduled: List[Dict[str, Any]] = []
        scheduler_active = False

        # 1) Agent heartbeat routines (from APScheduler via heartbeat_service)
        try:
            from services.heartbeat_service import get_heartbeat_service
            hb_svc = get_heartbeat_service()
            hb_status = hb_svc.get_status()
            if hb_status.get("active"):
                scheduler_active = True

            # Fetch agent names for the jobs
            agent_ids = []
            for job in hb_status.get("jobs", []):
                # Job IDs are like "agent_42_heartbeat" or "heartbeat_agent_42"
                parts = job["id"].split("_")
                for p in parts:
                    if p.isdigit():
                        agent_ids.append(int(p))
                        break

            agents_by_id: Dict[int, Any] = {}
            if agent_ids:
                agents = (
                    self.db.query(Agent)
                    .filter(
                        Agent.id.in_(agent_ids),
                        Agent.workspace_id == self.workspace_id,
                    )
                    .all()
                )
                agents_by_id = {a.id: a for a in agents}

            for job in hb_status.get("jobs", []):
                agent_id = None
                parts = job["id"].split("_")
                for p in parts:
                    if p.isdigit():
                        agent_id = int(p)
                        break

                agent = agents_by_id.get(agent_id) if agent_id else None
                if not agent:
                    continue  # Skip jobs not in this workspace

                hb_config = (agent.configuration or {}).get("heartbeat", {})
                interval = hb_config.get("interval_minutes", 60)

                scheduled.append({
                    "id": f"routine-{agent_id}",
                    "name": f"{agent.name} Routine",
                    "type": "routine",
                    "next_run_at": job.get("next_run_at"),
                    "frequency": f"Every {interval}m" if interval < 60 else f"Every {interval // 60}h",
                    "agent_name": agent.name,
                    "agent_id": agent_id,
                })
        except Exception as e:
            logger.error("Failed to fetch routine schedule: %s", e, exc_info=True)

        # 2) Cron-scheduled recipes (from recipe_scheduler)
        try:
            from services.recipe_scheduler import get_recipe_scheduler
            sched = get_recipe_scheduler()
            if sched:
                sched_status = sched.get_status()
                recipe_ids_from_jobs = []
                for job in sched_status.get("jobs", []):
                    # Job IDs like "recipe_cron_42"
                    parts = job["id"].split("_")
                    for p in parts:
                        if p.isdigit():
                            recipe_ids_from_jobs.append(int(p))
                            break

                recipes_by_id: Dict[int, WorkflowTemplate] = {}
                if recipe_ids_from_jobs:
                    recipes = (
                        self.db.query(WorkflowTemplate)
                        .filter(
                            WorkflowTemplate.id.in_(recipe_ids_from_jobs),
                            WorkflowTemplate.workspace_id == self.workspace_id,
                        )
                        .all()
                    )
                    recipes_by_id = {r.id: r for r in recipes}

                for job in sched_status.get("jobs", []):
                    recipe_id = None
                    parts = job["id"].split("_")
                    for p in parts:
                        if p.isdigit():
                            recipe_id = int(p)
                            break

                    recipe = recipes_by_id.get(recipe_id) if recipe_id else None
                    if not recipe:
                        continue

                    sc = recipe.schedule_config or {}
                    cron_expr = sc.get("cron_expression", "")

                    scheduled.append({
                        "id": f"recipe-{recipe_id}",
                        "name": recipe.name,
                        "type": "recipe",
                        "next_run_at": job.get("next_run_at"),
                        "frequency": cron_expr or "Scheduled",
                        "agent_name": None,
                        "agent_id": None,
                    })
        except Exception as e:
            logger.error("Failed to fetch recipe schedule: %s", e, exc_info=True)

        # Sort by next_run_at
        scheduled.sort(key=lambda x: x.get("next_run_at") or "9999")

        return {"scheduled": scheduled, "scheduler_active": scheduler_active}

    # ── Agent Reports Endpoint ─────────────────────────────────

    def get_agent_reports(self, *, agent_ids: List[int]) -> Dict[str, Any]:
        """Return latest execution summaries for pinned agents."""
        reports: List[Dict[str, Any]] = []

        if not agent_ids:
            return {"reports": reports}

        # Fetch agent metadata
        agents = (
            self.db.query(Agent)
            .filter(
                Agent.id.in_(agent_ids),
                Agent.workspace_id == self.workspace_id,
            )
            .all()
        )
        agents_by_id = {a.id: a for a in agents}

        # Get latest heartbeat result per agent
        for aid in agent_ids:
            agent = agents_by_id.get(aid)
            if not agent:
                continue

            try:
                row = self.db.execute(
                    text("""
                        SELECT hr.id, hr.status, hr.findings, hr.tokens_used, hr.created_at
                        FROM heartbeat_results hr
                        WHERE hr.workspace_id = :ws_id
                          AND hr.source_type = 'agent'
                          AND hr.source_id = :agent_id
                        ORDER BY hr.created_at DESC
                        LIMIT 1
                    """),
                    {"ws_id": self._ws_str, "agent_id": str(aid)},
                ).fetchone()

                summary = ""
                status = "no_data"
                last_run = None
                execution_id = None

                if row:
                    status = "completed" if row.status == "success" else "failed"
                    summary = self._routine_summary(row.findings)
                    last_run = row.created_at.isoformat() if row.created_at else None
                    execution_id = str(row.id)

                reports.append({
                    "agent_id": aid,
                    "agent_name": agent.name,
                    "agent_icon": agent.marketplace_icon,
                    "status": status,
                    "summary": summary[:300],
                    "last_run": last_run,
                    "execution_id": execution_id,
                })
            except Exception as e:
                logger.error("Failed to fetch report for agent %d: %s", aid, e, exc_info=True)
                reports.append({
                    "agent_id": aid,
                    "agent_name": agent.name if agent else f"Agent #{aid}",
                    "agent_icon": None,
                    "status": "error",
                    "summary": "",
                    "last_run": None,
                    "execution_id": None,
                })

        return {"reports": reports}

    @staticmethod
    def _recipe_summary(execution: RecipeExecution, recipe_name: str) -> str:
        """Build a context-line summary for a recipe execution."""
        meta = execution.execution_metadata or {}
        total_steps = meta.get("total_steps", 0)
        current = execution.current_step or 0

        if execution.status == "running" and total_steps:
            return f"Step {current} of {total_steps}"
        if execution.status == "completed" and total_steps:
            return f"All {total_steps} steps completed"
        if execution.status == "failed":
            return execution.error_message[:100] if execution.error_message else "Execution failed"
        return recipe_name
