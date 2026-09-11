"""
ScheduledTaskService (PRD-77)
==============================
Manages agent-initiated scheduled tasks.
Agents call platform_schedule_task → this service creates DB records and
registers jobs with the UnifiedScheduler (APScheduler).

When a job fires it creates a new chat session with the target agent,
injecting the task description as the opening message — or, for a row with
``deliver_as='board_task'`` (the Command Centre calendar's "schedule a board
task for later", from the board's Create Task dialog or ``platform_schedule_task``),
files the ticket the row describes on the board, where the dispatcher runs it
like any other.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from services.schedule_util import is_valid_cron, next_run as _util_next_run

logger = logging.getLogger(__name__)

# Limits
MAX_TASKS_PER_AGENT = 10
MAX_RECURRING_PER_WORKSPACE = 25
# Operator-scheduled rows (no creator agent) share one workspace-wide cap.
MAX_OPERATOR_TASKS_PER_WORKSPACE = 50

# APScheduler job id prefix for a scheduled task; the reconcile tick keys on it.
JOB_ID_PREFIX = "scheduled_task_"
# The scheduler worker catches up with the DB within this many seconds
# (services.schedule_reconcile). Creates, pauses, resumes and cancels land on
# ANY uvicorn worker; only the one holding the scheduler lock has APScheduler.
RECONCILE_INTERVAL_SECONDS = 60

# How a fired task is delivered.
DELIVER_CHAT = "chat"              # PRD-77: open a chat with the target agent
DELIVER_BOARD_TASK = "board_task"  # file a board ticket (assigned, or Inbox)
DELIVERY_MODES = (DELIVER_CHAT, DELIVER_BOARD_TASK)
_REVIEW_MODES = ("auto", "human", "llm")

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
        created_by_agent_id: Optional[int],
        target_agent_id: Optional[int],
        task_type: str,
        description: str,
        schedule: str,
        max_runs: Optional[int] = None,
        origin_chat_id: Optional[str] = None,
        *,
        deliver_as: str = DELIVER_CHAT,
        payload: Optional[Dict[str, Any]] = None,
        created_by_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new scheduled task.

        Args:
            created_by_agent_id: Agent requesting the task (None when an operator
                scheduled it — then created_by_user_id is required).
            target_agent_id: Agent that will execute it. Required for chat
                delivery; a board ticket with no target is filed into the Inbox.
            task_type: 'one_shot' or 'recurring'.
            description: What the agent should do when the task fires.
            schedule: ISO datetime for one_shot, cron expression for recurring.
            max_runs: Max executions for recurring tasks (None = unlimited).
            deliver_as: 'chat' opens a chat with the target agent when it fires
                (PRD-77); 'board_task' files a board ticket instead.
            payload: board_task only — title (required), priority, review_mode,
                tags of the ticket to file.
            created_by_user_id: the operator who scheduled it (the board dialog,
                or the human driving a chat turn); the consent actor at fire time.
        """
        # Validate task_type
        if task_type not in ("one_shot", "recurring"):
            return {"success": False, "error": "task_type must be 'one_shot' or 'recurring'"}
        if deliver_as not in DELIVERY_MODES:
            return {"success": False, "error": f"deliver_as must be one of: {', '.join(DELIVERY_MODES)}"}
        if created_by_agent_id is None and not created_by_user_id:
            return {"success": False, "error": "A creator is required (an agent or a user)"}
        if deliver_as == DELIVER_CHAT and target_agent_id is None:
            return {"success": False, "error": "Chat delivery needs a target agent"}
        payload = dict(payload or {})
        if deliver_as == DELIVER_BOARD_TASK and not str(payload.get("title") or "").strip():
            return {"success": False, "error": "A board task needs a title"}

        # Validate schedule format
        if task_type == "one_shot":
            try:
                run_at = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
                if run_at <= datetime.now(timezone.utc):
                    return {"success": False, "error": "Schedule datetime must be in the future"}
            except (ValueError, TypeError):
                return {"success": False, "error": f"Invalid ISO datetime: {schedule}. Use format: 2026-03-11T09:00:00Z"}
        else:
            # Validate with the shared cron util (croniter — standard crontab
            # semantics, the same the calendar and firing use post-PRD-162).
            if not is_valid_cron(schedule):
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
        if created_by_agent_id is not None and created_by_agent_id not in agent_ids_found:
            return {"success": False, "error": f"Creator agent {created_by_agent_id} not found in workspace"}
        if target_agent_id is not None and target_agent_id not in agent_ids_found:
            return {"success": False, "error": f"Target agent {target_agent_id} not found in workspace"}

        # Rate limits: per creator agent, or one workspace-wide cap for operator rows
        if created_by_agent_id is not None:
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
        else:
            operator_count = self.db.execute(
                text("""
                    SELECT COUNT(*) FROM agent_scheduled_tasks
                    WHERE workspace_id = :ws_id
                      AND created_by_user_id IS NOT NULL
                      AND status = 'active'
                """),
                {"ws_id": str(self.workspace_id)},
            ).scalar() or 0
            if operator_count >= MAX_OPERATOR_TASKS_PER_WORKSPACE:
                return {
                    "success": False,
                    "error": f"Workspace has reached the limit of {MAX_OPERATOR_TASKS_PER_WORKSPACE} active scheduled board tasks",
                }

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
                     task_type, description, schedule, max_runs, next_run_at,
                     origin_chat_id, deliver_as, payload, created_by_user_id)
                VALUES
                    (:ws_id, :created_by, :target,
                     :task_type, :description, :schedule, :max_runs, :next_run_at,
                     CAST(:origin_chat_id AS uuid), :deliver_as, CAST(:payload AS jsonb),
                     :created_by_user_id)
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
                "origin_chat_id": str(origin_chat_id) if origin_chat_id else None,
                "deliver_as": deliver_as,
                "payload": json.dumps(payload) if payload else None,
                "created_by_user_id": str(created_by_user_id) if created_by_user_id else None,
            },
        )
        row = result.fetchone()
        self.db.commit()

        task_id = row.id

        # Register with APScheduler
        self._register_with_scheduler(task_id, task_type, schedule, target_agent_id)

        target_name = next((r.name for r in agent_check if r.id == target_agent_id), None)
        logger.info(
            "[ScheduledTask] Created task %d: %s → %s '%s' (%s @ %s)",
            task_id, task_type, deliver_as, target_name or "unassigned", task_type, schedule,
        )

        if deliver_as == DELIVER_BOARD_TASK:
            where = f"assigned to '{target_name}'" if target_name else "in the Inbox"
            message = f"Scheduled {task_type} board task #{task_id} '{payload['title']}' — filed {where} when it fires"
        else:
            message = f"Scheduled {task_type} task #{task_id} for agent '{target_name or 'unknown'}'"

        return {
            "success": True,
            "task_id": task_id,
            "task_type": task_type,
            "deliver_as": deliver_as,
            "title": payload.get("title"),
            "target_agent": target_name,
            "schedule": schedule,
            "next_run_at": next_run_at.isoformat() if next_run_at else None,
            "message": message,
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
                    "deliver_as": getattr(r, "deliver_as", None) or DELIVER_CHAT,
                    "title": (r.payload or {}).get("title") if isinstance(getattr(r, "payload", None), dict) else None,
                    "payload": r.payload if isinstance(getattr(r, "payload", None), dict) else None,
                    "created_by_user_id": getattr(r, "created_by_user_id", None),
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

            # PRD-222 US-005 — no background burn: trial workspaces get no
            # scheduled execution until converted. Visible skip, never silent.
            try:
                from core.models.workspaces import Workspace
                from services.trial_ledger import is_trial_active_workspace
                # PRD-234 S3: the local edition has no platform-paid trial credit to
                # protect (operator keys / the user's own Claude subscription) — the
                # onboarding trial record must not switch scheduled work off there.
                from config import config as _config
                _local_edition = getattr(_config, "AUTH_EDITION", "saas") == "local"

                _ws = db.query(Workspace).get(task.workspace_id)
                if not _local_edition and is_trial_active_workspace(_ws):
                    logger.warning(
                        "[ScheduledTask] Skipping task %d — trial workspace %s "
                        "(no background burn until converted)",
                        task_id, task.workspace_id,
                    )
                    return
            except Exception as _e:
                logger.debug("[ScheduledTask] trial-skip check failed for task %d: %s", task_id, _e)

            logger.info(
                "[ScheduledTask] Firing task %d: '%s' → %s / agent %s",
                task_id, task.description[:80], getattr(task, "deliver_as", DELIVER_CHAT),
                task.target_agent_id,
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

            if getattr(task, "deliver_as", DELIVER_CHAT) == DELIVER_BOARD_TASK:
                # Board delivery: the ticket goes on the board, the dispatcher runs it.
                ScheduledTaskService._file_board_task(db, task, task_id)
            else:
                # Trigger agent chat via internal API
                await ScheduledTaskService._trigger_agent_chat(
                    workspace_id=str(task.workspace_id),
                    agent_id=task.target_agent_id,
                    message=f"[Scheduled Task #{task_id}] {task.description}",
                    db=db,
                    origin_chat_id=getattr(task, "origin_chat_id", None),
                    task_id=task_id,
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
    def _file_board_task(db: Session, task: Any, task_id: int) -> None:
        """Board delivery: file the ticket the row describes.

        The same three steps the HTTP create path takes — insert, PRD-234 D16
        consent when the operator who scheduled it also assigned it (local
        edition), then the board SSE push and the dispatcher wake — so the
        ticket runs exactly like one filed by hand. A recurring row files a fresh
        ticket per fire (``source_id`` carries the fire time).
        """
        from core.models.core import BoardTask
        from services.board_consent import (
            WHY_SCHEDULED_AND_ASSIGNED, actor_from_user_id, consent_for_created_ticket,
        )
        from services.board_dispatcher import notify_task_available
        from services.board_events import notify_board_event
        from services.board_sla import PRIORITY_SLA_HOURS, sla_deadline_for
        from services.cli_ticket_lane import source_id_for

        payload = task.payload if isinstance(getattr(task, "payload", None), dict) else {}
        description = str(task.description or "").strip()
        first_line = description.splitlines()[0] if description else "Scheduled task"
        title = (str(payload.get("title") or "").strip() or first_line)[:255]
        priority = payload.get("priority") if payload.get("priority") in PRIORITY_SLA_HOURS else "medium"
        review_mode = payload.get("review_mode") if payload.get("review_mode") in _REVIEW_MODES else "auto"
        tags = [str(t) for t in (payload.get("tags") or []) if t]
        assigned_agent_id = getattr(task, "target_agent_id", None)
        created_by_user_id = getattr(task, "created_by_user_id", None)
        now = datetime.now(timezone.utc)

        ticket = BoardTask(
            workspace_id=task.workspace_id,
            title=title,
            description=description or title,
            priority=priority,
            review_mode=review_mode,
            assigned_agent_id=assigned_agent_id,
            status="assigned" if assigned_agent_id else "inbox",
            created_by_type="user" if created_by_user_id else "agent",
            created_by_id=str(created_by_user_id or getattr(task, "created_by_agent_id", None) or ""),
            source_type="scheduled_task",
            source_id=source_id_for("task", task_id, now),
            tags=tags,
            sla_deadline=sla_deadline_for(priority, now=now),
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)

        if created_by_user_id:
            consent_for_created_ticket(
                db, workspace_id=task.workspace_id, task=ticket,
                actor=actor_from_user_id(created_by_user_id), why=WHY_SCHEDULED_AND_ASSIGNED,
            )
        notify_board_event(
            db, workspace_id=str(task.workspace_id), task_id=ticket.id,
            status=ticket.status, event="task_created",
        )
        if ticket.status == "assigned":
            notify_task_available(db, workspace_id=str(task.workspace_id), task_id=ticket.id)
        logger.info("[ScheduledTask] Task %d filed board ticket #%s (%s)", task_id, ticket.id, ticket.status)

        origin_chat_id = getattr(task, "origin_chat_id", None)
        if origin_chat_id:
            from services.chat_messenger import deliver_background_message

            deliver_background_message(
                db, workspace_id=str(task.workspace_id),
                text=f"Filed board ticket #{ticket.id}: {title}",
                source={"origin": "scheduled_task"}, chat_id=str(origin_chat_id),
                link_type="board_task", link_id=str(ticket.id),
            )

    @staticmethod
    async def _trigger_agent_chat(
        workspace_id: str,
        agent_id: int,
        message: str,
        db: Session,
        origin_chat_id: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> None:
        """
        Execute a task on the target agent via AgentFactory.
        Same pattern as HeartbeatService._agent_tick().
        """
        try:
            # PRD-234 S3: a Claude Code agent's scheduled task is a board ticket the
            # paired host runs as the user's own session; the origin conversation
            # hears that it is queued (the factory refuses cli agents by design).
            from services.cli_ticket_lane import file_cli_ticket, is_cli_agent, queued_line, source_id_for
            if is_cli_agent(db, agent_id):
                from datetime import datetime as _dt, timezone as _tz
                _first = str(message).strip().splitlines()[0][:80] if str(message).strip() else "task"
                ticket = file_cli_ticket(
                    db, workspace_id=workspace_id, agent_id=agent_id,
                    title=f"Scheduled: {_first}", prompt=str(message), source_type="scheduled_task",
                    source_id=source_id_for("task", task_id if task_id is not None else "adhoc", _dt.now(_tz.utc)),
                )
                logger.info("[ScheduledTask] cli agent %d — filed ticket #%s", agent_id, ticket.id)
                if origin_chat_id:
                    from services.chat_messenger import deliver_background_message
                    deliver_background_message(
                        db, workspace_id=workspace_id, text=queued_line(ticket),
                        source={"origin": "scheduled_task"}, chat_id=str(origin_chat_id),
                        link_type="board_task", link_id=str(ticket.id),
                    )
                return
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

            # PRD-205 S6: the output is DELIVERED, not discarded (the PRD-77
            # defect: this used to be a 200-char logger.info and nothing
            # else). Target = the conversation the task was created from
            # (origin_chat_id, captured at platform_schedule_task time) --
            # scheduled-task rows are agent-created, so there is no user to
            # fall back to; a task with no captured origin (pre-205 rows, API
            # creations) keeps the log-only behaviour honestly. Fail-soft: a
            # chat failure never fails the scheduled run.
            if str(llm_text).strip() and origin_chat_id:
                from services.chat_messenger import deliver_background_message

                deliver_background_message(
                    db,
                    workspace_id=workspace_id,
                    text=str(llm_text),
                    source={"origin": "scheduled_task"},
                    chat_id=str(origin_chat_id),
                    link_type="scheduled_task",
                    link_id=str(task_id) if task_id is not None else None,
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
                # This worker does not host APScheduler (another one holds the lock):
                # the DB row is the truth and the scheduler worker's reconcile tick
                # registers it within RECONCILE_INTERVAL_SECONDS.
                logger.info(
                    "[ScheduledTask] Scheduler not on this worker — task %d is registered by the reconcile tick within %ds",
                    task_id, RECONCILE_INTERVAL_SECONDS,
                )
                return

            job_id = f"{JOB_ID_PREFIX}{task_id}"

            if task_type == "one_shot":
                run_at = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
                trigger = DateTrigger(run_date=run_at)
            else:
                # Standard crontab semantics so firing matches the calendar's
                # croniter next_run (PRD-162 — one schedule truth).
                trigger = CronTrigger.from_crontab(schedule)

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
        """Next run from a cron expression via the shared schedule util (croniter)."""
        return _util_next_run(cron_expr, now=datetime.now(timezone.utc))

    def reconcile_with_scheduler(self, scheduler: Any) -> Dict[str, Any]:
        """Make the scheduler worker's APScheduler match the DB, all workspaces.

        Active rows with no job get registered; jobs whose row is no longer active
        (paused, cancelled, completed, deleted) are removed. Idempotent — an active
        row with its job already present is left alone (re-adding a cron job would
        recompute its next fire time). Runs on the leader every
        RECONCILE_INTERVAL_SECONDS (services.schedule_reconcile) and once at boot.
        """
        if scheduler is None or not getattr(scheduler, "running", False):
            return {"added": 0, "removed": 0, "skipped": True}
        rows = self.db.execute(
            text("""
                SELECT id, task_type, schedule, target_agent_id
                FROM agent_scheduled_tasks
                WHERE status = 'active'
            """),
        ).fetchall()
        active = {int(row.id): row for row in rows}
        existing = {
            str(job.id) for job in scheduler.get_jobs()
            if str(getattr(job, "id", "")).startswith(JOB_ID_PREFIX)
        }
        added = 0
        for task_id, row in active.items():
            if f"{JOB_ID_PREFIX}{task_id}" not in existing:
                self._register_with_scheduler(row.id, row.task_type, row.schedule, row.target_agent_id)
                added += 1
        removed = 0
        for job_id in existing:
            try:
                task_id = int(job_id[len(JOB_ID_PREFIX):])
            except ValueError:
                continue
            if task_id not in active:
                try:
                    scheduler.remove_job(job_id)
                except Exception:  # noqa: BLE001 — already gone
                    pass
                removed += 1
        if added or removed:
            logger.info("[ScheduledTask] reconcile: %d job(s) added, %d removed", added, removed)
        return {"added": added, "removed": removed, "skipped": False}

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
