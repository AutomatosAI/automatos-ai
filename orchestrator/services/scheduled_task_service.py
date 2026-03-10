"""
ScheduledTaskService (PRD-77)
==============================
Manages agent-initiated scheduled tasks.
Agents call platform_schedule_task → this service creates DB records and
registers jobs with the UnifiedScheduler (APScheduler).

When a job fires it creates a new chat session with the target agent,
injecting the task description as the opening message.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)

# Limits
MAX_TASKS_PER_AGENT = 10
MAX_RECURRING_PER_WORKSPACE = 25

# Cron validation — 5-field standard cron (minute hour dom month dow)
_CRON_RE = re.compile(
    r"^("
    r"(\*|[0-9]{1,2}(-[0-9]{1,2})?(,[0-9]{1,2}(-[0-9]{1,2})?)*(/[0-9]{1,2})?)"
    r"\s+){4}"
    r"(\*|[0-9]{1,2}(-[0-9]{1,2})?(,[0-9]{1,2}(-[0-9]{1,2})?)*(/[0-9]{1,2})?)"
    r"$"
)


class ScheduledTaskService:
    """Creates, lists, cancels, and executes agent-scheduled tasks."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_task(
        self,
        created_by_agent_id: int,
        target_agent_id: int,
        task_type: str,
        description: str,
        schedule: str,
        max_runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new scheduled task.

        Args:
            created_by_agent_id: Agent requesting the task.
            target_agent_id: Agent that will execute it.
            task_type: 'one_shot' or 'recurring'.
            description: What the agent should do when the task fires.
            schedule: ISO datetime for one_shot, cron expression for recurring.
            max_runs: Max executions for recurring tasks (None = unlimited).
        """
        # Validate task_type
        if task_type not in ("one_shot", "recurring"):
            return {"success": False, "error": "task_type must be 'one_shot' or 'recurring'"}

        # Validate schedule format
        if task_type == "one_shot":
            try:
                run_at = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
                if run_at <= datetime.now(timezone.utc):
                    return {"success": False, "error": "Schedule datetime must be in the future"}
            except (ValueError, TypeError):
                return {"success": False, "error": f"Invalid ISO datetime: {schedule}. Use format: 2026-03-11T09:00:00Z"}
        else:
            if not _CRON_RE.match(schedule.strip()):
                return {"success": False, "error": f"Invalid cron expression: {schedule}. Use 5-field format: '0 9 * * 1' (minute hour dom month dow)"}

        # Validate agents exist in workspace
        agent_check = self.db.execute(
            text("""
                SELECT id, name FROM agents
                WHERE id IN (:created_by, :target) AND workspace_id = :ws_id
            """),
            {"created_by": created_by_agent_id, "target": target_agent_id, "ws_id": str(self.workspace_id)},
        ).fetchall()

        agent_ids_found = {row.id for row in agent_check}
        if created_by_agent_id not in agent_ids_found:
            return {"success": False, "error": f"Creator agent {created_by_agent_id} not found in workspace"}
        if target_agent_id not in agent_ids_found:
            return {"success": False, "error": f"Target agent {target_agent_id} not found in workspace"}

        # Rate limits
        active_count = self.db.execute(
            text("""
                SELECT COUNT(*) FROM agent_scheduled_tasks
                WHERE created_by_agent_id = :agent_id
                  AND workspace_id = :ws_id
                  AND status = 'active'
            """),
            {"agent_id": created_by_agent_id, "ws_id": str(self.workspace_id)},
        ).scalar() or 0

        if active_count >= MAX_TASKS_PER_AGENT:
            return {"success": False, "error": f"Agent has reached the limit of {MAX_TASKS_PER_AGENT} active tasks"}

        if task_type == "recurring":
            recurring_count = self.db.execute(
                text("""
                    SELECT COUNT(*) FROM agent_scheduled_tasks
                    WHERE workspace_id = :ws_id
                      AND task_type = 'recurring'
                      AND status = 'active'
                """),
                {"ws_id": str(self.workspace_id)},
            ).scalar() or 0

            if recurring_count >= MAX_RECURRING_PER_WORKSPACE:
                return {"success": False, "error": f"Workspace has reached the limit of {MAX_RECURRING_PER_WORKSPACE} recurring tasks"}

        # Compute next_run_at
        if task_type == "one_shot":
            next_run_at = run_at
        else:
            next_run_at = self._next_cron_run(schedule)

        # Insert
        result = self.db.execute(
            text("""
                INSERT INTO agent_scheduled_tasks
                    (workspace_id, created_by_agent_id, target_agent_id,
                     task_type, description, schedule, max_runs, next_run_at)
                VALUES
                    (:ws_id, :created_by, :target,
                     :task_type, :description, :schedule, :max_runs, :next_run_at)
                RETURNING id, created_at
            """),
            {
                "ws_id": str(self.workspace_id),
                "created_by": created_by_agent_id,
                "target": target_agent_id,
                "task_type": task_type,
                "description": description,
                "schedule": schedule,
                "max_runs": max_runs,
                "next_run_at": next_run_at,
            },
        )
        row = result.fetchone()
        self.db.commit()

        task_id = row.id

        # Register with APScheduler
        self._register_with_scheduler(task_id, task_type, schedule, target_agent_id)

        target_name = next((r.name for r in agent_check if r.id == target_agent_id), "unknown")
        logger.info(
            "[ScheduledTask] Created task %d: %s → agent '%s' (%s @ %s)",
            task_id, task_type, target_name, task_type, schedule,
        )

        return {
            "success": True,
            "task_id": task_id,
            "task_type": task_type,
            "target_agent": target_name,
            "schedule": schedule,
            "next_run_at": next_run_at.isoformat() if next_run_at else None,
            "message": f"Scheduled {task_type} task #{task_id} for agent '{target_name}'",
        }

    # ------------------------------------------------------------------
    # List / Get
    # ------------------------------------------------------------------

    async def list_tasks(
        self,
        agent_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List scheduled tasks for the workspace."""
        conditions = ["t.workspace_id = :ws_id"]
        params: Dict[str, Any] = {"ws_id": str(self.workspace_id), "limit": limit, "offset": offset}

        if agent_id:
            conditions.append("(t.created_by_agent_id = :agent_id OR t.target_agent_id = :agent_id)")
            params["agent_id"] = agent_id
        if status:
            conditions.append("t.status = :status")
            params["status"] = status

        where = " AND ".join(conditions)

        rows = self.db.execute(
            text(f"""
                SELECT t.*,
                       ca.name as creator_name,
                       ta.name as target_name
                FROM agent_scheduled_tasks t
                LEFT JOIN agents ca ON ca.id = t.created_by_agent_id
                LEFT JOIN agents ta ON ta.id = t.target_agent_id
                WHERE {where}
                ORDER BY t.created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()

        total = self.db.execute(
            text(f"SELECT COUNT(*) FROM agent_scheduled_tasks t WHERE {where}"),
            params,
        ).scalar() or 0

        return {
            "success": True,
            "tasks": [
                {
                    "id": r.id,
                    "task_type": r.task_type,
                    "description": r.description,
                    "schedule": r.schedule,
                    "status": r.status,
                    "creator_agent": r.creator_name,
                    "target_agent": r.target_name,
                    "run_count": r.run_count,
                    "max_runs": r.max_runs,
                    "next_run_at": r.next_run_at.isoformat() if r.next_run_at else None,
                    "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
                    "last_error": r.last_error,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
            "total": total,
        }

    # ------------------------------------------------------------------
    # Cancel / Pause / Resume
    # ------------------------------------------------------------------

    async def update_task_status(self, task_id: int, new_status: str) -> Dict[str, Any]:
        """Cancel, pause, or resume a task."""
        valid = {"cancelled", "paused", "active"}
        if new_status not in valid:
            return {"success": False, "error": f"status must be one of: {', '.join(sorted(valid))}"}

        row = self.db.execute(
            text("""
                UPDATE agent_scheduled_tasks
                SET status = :status, updated_at = NOW()
                WHERE id = :id AND workspace_id = :ws_id
                RETURNING id, status
            """),
            {"id": task_id, "status": new_status, "ws_id": str(self.workspace_id)},
        ).fetchone()

        if not row:
            return {"success": False, "error": f"Task {task_id} not found"}

        self.db.commit()

        # Update scheduler
        job_id = f"scheduled_task_{task_id}"
        try:
            from services.scheduler import get_unified_scheduler
            scheduler = get_unified_scheduler()
            if scheduler.apscheduler:
                if new_status in ("cancelled", "paused"):
                    try:
                        scheduler.apscheduler.remove_job(job_id)
                    except Exception:
                        pass  # Job may not exist yet
                elif new_status == "active":
                    # Re-read task and re-register
                    task = self.db.execute(
                        text("SELECT * FROM agent_scheduled_tasks WHERE id = :id"),
                        {"id": task_id},
                    ).fetchone()
                    if task:
                        self._register_with_scheduler(
                            task.id, task.task_type, task.schedule, task.target_agent_id,
                        )
        except Exception as e:
            logger.warning("[ScheduledTask] Failed to update scheduler for task %d: %s", task_id, e)

        logger.info("[ScheduledTask] Task %d → %s", task_id, new_status)
        return {"success": True, "task_id": task_id, "status": new_status}

    # ------------------------------------------------------------------
    # Execution (called by APScheduler when job fires)
    # ------------------------------------------------------------------

    @staticmethod
    async def execute_task(task_id: int) -> None:
        """
        Execute a scheduled task: create a chat session with the target agent.
        Called by APScheduler job trigger.
        """
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            task = db.execute(
                text("SELECT * FROM agent_scheduled_tasks WHERE id = :id AND status = 'active'"),
                {"id": task_id},
            ).fetchone()

            if not task:
                logger.warning("[ScheduledTask] Task %d not found or not active", task_id)
                return

            logger.info(
                "[ScheduledTask] Firing task %d: '%s' → agent %d",
                task_id, task.description[:80], task.target_agent_id,
            )

            # Update run tracking
            db.execute(
                text("""
                    UPDATE agent_scheduled_tasks
                    SET run_count = run_count + 1,
                        last_run_at = NOW(),
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {"id": task_id},
            )

            # Check if we've hit max_runs → mark completed
            if task.max_runs and (task.run_count + 1) >= task.max_runs:
                db.execute(
                    text("""
                        UPDATE agent_scheduled_tasks
                        SET status = 'completed', updated_at = NOW()
                        WHERE id = :id
                    """),
                    {"id": task_id},
                )
                # Remove from scheduler
                try:
                    from services.scheduler import get_unified_scheduler
                    sched = get_unified_scheduler()
                    if sched.apscheduler:
                        sched.apscheduler.remove_job(f"scheduled_task_{task_id}")
                except Exception:
                    pass

            # Mark one_shot as completed after execution
            if task.task_type == "one_shot":
                db.execute(
                    text("""
                        UPDATE agent_scheduled_tasks
                        SET status = 'completed', updated_at = NOW()
                        WHERE id = :id
                    """),
                    {"id": task_id},
                )

            db.commit()

            # Trigger agent chat via internal API
            await ScheduledTaskService._trigger_agent_chat(
                workspace_id=str(task.workspace_id),
                agent_id=task.target_agent_id,
                message=f"[Scheduled Task #{task_id}] {task.description}",
                db=db,
            )

        except Exception as e:
            logger.error("[ScheduledTask] Task %d execution failed: %s", task_id, e, exc_info=True)
            db.execute(
                text("""
                    UPDATE agent_scheduled_tasks
                    SET last_error = :error, updated_at = NOW()
                    WHERE id = :id
                """),
                {"id": task_id, "error": str(e)[:500]},
            )
            db.commit()
        finally:
            db.close()

    @staticmethod
    async def _trigger_agent_chat(
        workspace_id: str,
        agent_id: int,
        message: str,
        db: Session,
    ) -> None:
        """
        Execute a task on the target agent via AgentFactory.
        Same pattern as HeartbeatService._agent_tick().
        """
        try:
            from modules.agents.factory.agent_factory import AgentFactory

            factory = AgentFactory(db_session=db)
            result = await factory.execute_with_prompt(
                agent=agent_id,
                prompt=message,
                context={"source": "scheduled_task", "workspace_id": workspace_id},
                use_memory=True,
            )

            llm_text = ""
            if isinstance(result, dict):
                llm_text = (
                    result.get("result")
                    or result.get("response")
                    or result.get("output")
                    or ""
                )
            logger.info(
                "[ScheduledTask] Agent %d completed task: %s",
                agent_id, str(llm_text)[:200],
            )
        except Exception as e:
            logger.error("[ScheduledTask] Failed to trigger agent chat: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_with_scheduler(
        self,
        task_id: int,
        task_type: str,
        schedule: str,
        target_agent_id: int,
    ) -> None:
        """Register task with UnifiedScheduler (APScheduler)."""
        try:
            from services.scheduler import get_unified_scheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.date import DateTrigger

            scheduler = get_unified_scheduler()
            if not scheduler.apscheduler or not scheduler.apscheduler.running:
                logger.warning("[ScheduledTask] Scheduler not running — task %d will be picked up on restart", task_id)
                return

            job_id = f"scheduled_task_{task_id}"

            if task_type == "one_shot":
                run_at = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
                trigger = DateTrigger(run_date=run_at)
            else:
                parts = schedule.strip().split()
                trigger = CronTrigger(
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                )

            # APScheduler needs a sync wrapper for async execute_task
            import asyncio

            def _sync_execute():
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(ScheduledTaskService.execute_task(task_id))
                finally:
                    loop.close()

            scheduler.apscheduler.add_job(
                _sync_execute,
                trigger,
                id=job_id,
                replace_existing=True,
                max_instances=1,
            )
            logger.info("[ScheduledTask] Registered job %s with scheduler", job_id)

        except Exception as e:
            logger.warning("[ScheduledTask] Could not register task %d with scheduler: %s", task_id, e)

    @staticmethod
    def _next_cron_run(cron_expr: str) -> Optional[datetime]:
        """Compute next run time from a cron expression."""
        try:
            from apscheduler.triggers.cron import CronTrigger

            parts = cron_expr.strip().split()
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
            return trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        except Exception:
            return None

    async def load_active_tasks_to_scheduler(self) -> int:
        """
        Load all active tasks from DB into APScheduler.
        Called on startup after UnifiedScheduler.start().
        """
        rows = self.db.execute(
            text("""
                SELECT id, task_type, schedule, target_agent_id
                FROM agent_scheduled_tasks
                WHERE status = 'active'
            """),
        ).fetchall()

        count = 0
        for row in rows:
            self._register_with_scheduler(row.id, row.task_type, row.schedule, row.target_agent_id)
            count += 1

        if count:
            logger.info("[ScheduledTask] Loaded %d active tasks into scheduler", count)
        return count
