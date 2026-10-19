"""
HARNESS Service (PRD-121)
==========================

Self-Optimizing Organization Loop.  Runs weekly, invisible to users.
Reads each agent's real weekly counts, judges them against a clean reference
(services.harness_diagnose, PRD-263), lists the configuration changes it would
suggest with their risk scores, and submits the audit report. It changes nothing:
every workspace is report_only through beta (PRD-263 D2, services.harness_mode).

Registered at server startup alongside HeartbeatService / CoordinatorService.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from core.models.harness import RUN_COMPLETED, RUN_DORMANT_AGENTS, RUN_FAILED, RUN_RUNNING, HarnessRun

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from core.models.workspaces import Workspace
    from services.harness_metrics import HarnessMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The sweep's schedule is config.HARNESS_SWEEP_CRON, read in UTC (services.harness_runs).
_MIN_AGENTS = 3

# DIAGNOSE's thresholds are the HARNESS_* statistics settings (services.harness_diagnose).

# The audit report's section for PRESCRIBE's suggestions, which HARNESS reports
# and never applies (PRD-263 US-016).
SUGGESTIONS_HEADING = "## Suggested changes (reported only; nothing was changed)"

# What get_status says for a workspace whose mode is disabled (PRD-263 US-016).
DISABLED_MESSAGE = (
    "HARNESS is disabled for this workspace. "
    "Choose Report only in Settings > Orchestrator to turn it back on."
)

# PRD-263 US-019: a "reconnect <app>" suggestion changes no agent setting (the owner
# reconnects the app), so it carries the lowest risk score.
RECONNECT_RISK = 1

# The weekly audit report, written straight through ReportService and attributed to
# Auto (PRD-263 US-048): its title (also the report's heading) and its report type.
AUDIT_REPORT_TITLE = "HARNESS Weekly Org Review — Run #{iteration}"
AUDIT_REPORT_TYPE = "audit"


class HarnessService:
    """Weekly self-optimizing loop for workspace agent configuration."""

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._owns_scheduler: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: Optional[AsyncIOScheduler] = None) -> None:
        """Register the weekly sweep for all active workspaces."""
        from config import config

        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            jobstores = {"default": MemoryJobStore()}
            self._scheduler = AsyncIOScheduler(jobstores=jobstores)
            self._scheduler.start()
            self._owns_scheduler = True

        await self._register_workspace_jobs()
        logger.info("[HARNESS] Service started (cron: %s UTC)", config.HARNESS_SWEEP_CRON)

    async def _register_workspace_jobs(self) -> None:
        """Register a single cron job that iterates all active workspaces.

        Single-job design avoids thundering herd (N workspaces all firing at
        once) and automatically picks up new workspaces without restart. The
        job fires on HARNESS_SWEEP_CRON in UTC (US-012).
        """
        from config import config
        from services import harness_runs

        self._scheduler.add_job(
            self._harness_sweep,
            trigger=harness_runs.sweep_trigger(),
            id="harness_sweep",
            replace_existing=True,
            max_instances=1,
        )
        logger.info("[HARNESS] Registered sweep job (cron: %s UTC)", config.HARNESS_SWEEP_CRON)

    async def _harness_sweep(self) -> None:
        """Iterate all active, opted-in workspaces sequentially.

        Each tick takes its workspace's advisory lock and skips a workspace whose
        judged week already has a completed or dormant run, so replicas that each
        start a scheduler still run a workspace once per week (US-011).
        """
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace

        db = SessionLocal()
        try:
            workspaces = db.query(Workspace).filter(Workspace.is_active.is_(True)).all()
            eligible = [ws.id for ws in workspaces if self._workspace_opted_in(ws)]
            logger.info(
                "[HARNESS] Sweep starting — %d/%d workspaces opted in",
                len(eligible), len(workspaces),
            )
        finally:
            db.close()
        for workspace_id in eligible:
            await self._harness_tick(workspace_id=workspace_id, sweep=True)

    # ------------------------------------------------------------------
    # Public helpers (called by platform tool handlers)
    # ------------------------------------------------------------------

    def get_status(
        self, workspace_id: UUID, db: Optional["Session"] = None
    ) -> Dict[str, Any]:
        """Return current HARNESS state with granular, observable status.

        Possible statuses:
          - running                       — a run holds the workspace's advisory lock
          - disabled                      — the workspace's HARNESS mode is disabled
          - failed                        — the latest run failed, or was left
                                            unfinished with no lock holder (interrupted)
          - dormant_insufficient_agents   — fewer than _MIN_AGENTS active
          - scheduled_not_run_yet         — eligible but cron hasn't fired
          - completed                     — the latest run completed; with its
                                            coverage, convergence and verdict counts

        ``active_agents`` counts two different sets. On ``completed`` it is the run's
        own count: the agents DIAGNOSE judged or could not judge, never Auto (the
        denominator of ``coverage``). On the dormant, scheduled and unfinished
        statuses it is the dormancy count from _sufficiency_breakdown, which
        includes Auto. Compare it only with figures from the same status.

        Runs are read from harness_runs (US-010); ``running`` from pg_locks (US-011).
        """
        own_db = db is None
        if own_db:
            from core.database.database import SessionLocal
            db = SessionLocal()
        try:
            return self._status(db, workspace_id)
        finally:
            if own_db:
                db.close()

    def _status(self, db: "Session", workspace_id: UUID) -> Dict[str, Any]:
        from services import harness_diagnose, harness_runs

        next_scheduled = harness_runs.next_sweep_run()
        latest = harness_runs.latest_run(db, workspace_id)
        if harness_runs.lock_is_held(db, workspace_id):
            in_flight = latest is not None and latest.finished_at is None
            return {
                "status": RUN_RUNNING,
                "started_at": _iso(latest.started_at) if in_flight else None,
                "next_scheduled_run": next_scheduled,
                "iteration_count": latest.iteration if latest else 0,
            }

        if not self.workspace_enabled(db, workspace_id):
            return {
                "status": "disabled",
                "message": DISABLED_MESSAGE,
                "iteration_count": 0,
                "next_scheduled_run": None,
            }

        if latest is not None and latest.status == RUN_COMPLETED:
            return {
                "status": RUN_COMPLETED,
                "iteration_count": latest.iteration,
                "cur_window": latest.cur_window,
                **harness_diagnose.status_fields(latest.coverage, latest.counts, latest.verdicts),
                "last_run_at": _iso(latest.finished_at),
                "next_scheduled_run": next_scheduled,
                "artifacts": {"audit_report": latest.report_status},
            }

        sufficiency = self._sufficiency_breakdown(db, workspace_id)
        if latest is not None:
            return {**self._unfinished_status(latest), "next_scheduled_run": next_scheduled, **sufficiency}
        return {
            "status": self._dormancy(sufficiency) or "scheduled_not_run_yet",
            "iteration_count": 0,
            "next_scheduled_run": next_scheduled,
            **sufficiency,
        }

    @staticmethod
    def _unfinished_status(run: "HarnessRun") -> Dict[str, Any]:
        """The latest run did not complete: dormant, failed, or interrupted (left
        ``running`` while no run holds the lock)."""
        from services import harness_runs

        status = {
            "status": run.status,
            "last_run_at": _iso(run.finished_at or run.started_at),
            "iteration_count": run.iteration,
        }
        if run.finished_at is None:
            return {**status, "status": RUN_FAILED, "error": harness_runs.INTERRUPTED}
        if run.status == RUN_FAILED:
            return {**status, "error": harness_runs.failure_reason(run.report_status)}
        return status

    async def trigger_now(self, workspace_id: UUID) -> None:
        """Kick off a HARNESS run outside the weekly cron schedule.

        The run takes the workspace's advisory lock like the sweep (a held lock
        makes it a logged no-op) but is exempt from the once-per-week skip.
        """
        asyncio.ensure_future(self._harness_tick(workspace_id=workspace_id))

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def _harness_tick(self, workspace_id: UUID, sweep: bool = False) -> None:
        """One run for one workspace, under its advisory lock (US-011).

        A held lock (a run in another worker or replica) makes this a logged
        no-op. The sweep (``sweep=True``) also skips a workspace whose judged
        week already has a completed or dormant run; trigger_now does not. A
        failure to take the lock or reach the run table is logged and ends this
        tick only, so the sweep goes on to the next workspace.
        """
        from core.database.database import engine
        from services import harness_runs

        try:
            with harness_runs.workspace_lock(engine, workspace_id) as held:
                if not held:
                    logger.info("[HARNESS] A run already holds the lock for %s, skipping", workspace_id)
                    return
                await self._run_locked(workspace_id, sweep)
        except Exception:
            logger.exception("[HARNESS] Tick for %s could not run or be recorded", workspace_id)

    async def _run_locked(self, workspace_id: UUID, sweep: bool) -> None:
        """Holding the lock: skip a disabled workspace (sweep and trigger_now
        alike), close interrupted runs, apply the weekly skip, then run the
        pipeline on its own harness_runs row."""
        from core.database.database import SessionLocal
        from services import harness_runs

        db = SessionLocal()
        try:
            if not self.workspace_enabled(db, workspace_id):
                logger.info("[HARNESS] %s is disabled, skipping", workspace_id)
                return
            cur_window = harness_runs.last_complete_iso_week()
            harness_runs.close_interrupted(db, workspace_id)
            if sweep and harness_runs.week_is_settled(db, workspace_id, cur_window):
                logger.info("[HARNESS] %s already has a run for %s, skipping", workspace_id, cur_window)
                return
            run_id = harness_runs.open_run(db, workspace_id, cur_window)
            await self._run_recorded(db, workspace_id, run_id, cur_window)
        finally:
            db.close()

    async def _run_recorded(self, db: "Session", workspace_id: UUID, run_id: int, cur_window: str) -> None:
        """Run the pipeline for the judged week and close its row as completed, dormant or failed."""
        from services import harness_runs

        t0 = time.monotonic()
        logger.info("[HARNESS] Tick started for workspace %s (run %s, week %s)", workspace_id, run_id, cur_window)
        try:
            outcome = await self._run_pipeline(db, workspace_id, run_id, cur_window)
            harness_runs.close_run(db, run_id, **outcome)
        except Exception as exc:
            logger.exception("[HARNESS] Tick failed for %s", workspace_id)
            self._record_failure(db, run_id, exc)
            return
        logger.info(
            "[HARNESS] Tick %s for %s in %.1fs — %s",
            outcome["status"], workspace_id, time.monotonic() - t0, outcome.get("counts"),
        )

    @staticmethod
    def _record_failure(db: "Session", run_id: int, exc: Exception) -> None:
        """Close the run's row as failed; if even that fails, the row stays
        ``running`` and the next run closes it as interrupted."""
        from services import harness_runs

        try:
            db.rollback()
            harness_runs.close_run(db, run_id, RUN_FAILED, report_status=harness_runs.failure(str(exc)))
        except Exception:
            logger.exception("[HARNESS] Could not record run %s as failed", run_id)

    async def _run_pipeline(
        self, db: "Session", workspace_id: UUID, run_id: int, cur_window: str
    ) -> Dict[str, Any]:
        """The dormancy check, then the four phases; returns the row's closing fields.

        Report only (PRD-263 US-016): PRESCRIBE's suggestions become lines of the
        audit report, and nothing is applied or queued. ``propose`` runs exactly
        like ``report_only`` until proposals exist (US-054).
        """
        from core.database.database import end_open_transaction
        from services import harness_diagnose, harness_runs

        sufficiency = self._sufficiency_breakdown(db, workspace_id)
        dormant = self._dormancy(sufficiency)
        if dormant:
            logger.info("[HARNESS] Dormant for %s — %s (%s)", workspace_id, dormant, sufficiency)
            return {"status": dormant, "counts": sufficiency}

        iteration = harness_runs.completed_runs(db, workspace_id) + 1

        # Commit before each phase so the connection is not idle-in-transaction
        # across the phase's long awaits — PRD-135 / W1-S9. Phases that mutate
        # state persist through their own commits; these only release the
        # connection.
        end_open_transaction(db)
        metrics = await self._phase_collect(workspace_id, db, cur_window)
        end_open_transaction(db)
        diagnosis = await self._phase_diagnose(workspace_id, metrics, db)
        end_open_transaction(db)
        prescriptions = await self._phase_prescribe(workspace_id, diagnosis, run_id)
        end_open_transaction(db)
        report_md, report_status = await self._phase_report(
            workspace_id, diagnosis, prescriptions, db, iteration,
        )
        counts = {**diagnosis["counts"], "prescriptions": len(prescriptions)}
        return {
            "status": RUN_COMPLETED, "iteration": iteration, "coverage": diagnosis["coverage"], "counts": counts,
            "verdicts": harness_diagnose.stored_verdicts(diagnosis), "prescriptions": prescriptions,
            "report_md": report_md, "report_status": report_status,
        }

    @staticmethod
    def _dormancy(sufficiency: Dict[str, Any]) -> Optional[str]:
        """The dormant status the sufficiency breakdown calls for, or None."""
        return None if sufficiency["agents_ok"] else RUN_DORMANT_AGENTS

    # ------------------------------------------------------------------
    # Workspace opt-in / settings
    # ------------------------------------------------------------------

    @staticmethod
    def _workspace_opted_in(workspace: "Workspace") -> bool:
        """Decide whether HARNESS should run for this workspace.

        PRD-121 says "no setup required" — HARNESS runs unless the workspace's
        mode (``orchestrator.harness.mode``, read through services.harness_mode,
        which maps the retired ``enabled``/``disabled`` keys) is ``disabled``.
        """
        from services.harness_mode import DISABLED, harness_mode

        return harness_mode(workspace.settings) != DISABLED

    def workspace_enabled(self, db: "Session", workspace_id: UUID) -> bool:
        """False when the workspace's mode is ``disabled``; a workspace with no
        row is left to the run, which records what it finds. The run, get_status
        and the platform_harness_trigger handler all ask it."""
        from core.models.workspaces import Workspace as WsModel

        ws = db.query(WsModel).get(workspace_id)
        return ws is None or self._workspace_opted_in(ws)

    # ------------------------------------------------------------------
    # Dormancy check
    # ------------------------------------------------------------------

    def _sufficiency_breakdown(
        self, db: "Session", workspace_id: UUID
    ) -> Dict[str, Any]:
        """Return the dormancy detail used by status + tick: the active-agent count.

        Coverage (judged agents over active ones) replaced the heartbeat-age check
        (PRD-263 US-014a); this count still includes Auto, as it always has.
        """
        from core.models import Agent

        # Agent has no `is_active` column — use the canonical `status` field
        # (matches semantic_indexer + AgentService throughout the codebase).
        agent_count = (
            db.query(Agent)
            .filter(Agent.workspace_id == workspace_id, Agent.status == "active")
            .count()
        )
        return {
            "active_agents": agent_count,
            "min_required_agents": _MIN_AGENTS,
            "agents_ok": agent_count >= _MIN_AGENTS,
        }

    # ------------------------------------------------------------------
    # Phase 1: COLLECT
    # ------------------------------------------------------------------

    async def _phase_collect(self, workspace_id: UUID, db: "Session", cur_window: str) -> "HarnessMetrics":
        """Read each agent's real counts for the judged week and its reference (US-013).

        One workspace-bound SQL read on a worker thread (services.harness_metrics);
        no platform action. The reference is HARNESS_REF_DAYS in whole ISO weeks.
        """
        from services import harness_diagnose, harness_metrics

        logger.info("[HARNESS] Phase 1 COLLECT — workspace %s, week %s", workspace_id, cur_window)
        metrics = await harness_metrics.read(
            db, workspace_id, cur_window, ref_weeks=harness_diagnose.reference_weeks(),
        )
        logger.info("[HARNESS] COLLECT done — %d agent-weeks", len(metrics.agent_rows))
        return metrics

    # ------------------------------------------------------------------
    # Phase 2: DIAGNOSE
    # ------------------------------------------------------------------

    async def _phase_diagnose(
        self, workspace_id: UUID, metrics: "HarnessMetrics", db: "Session"
    ) -> Dict[str, Any]:
        """Judge each agent x metric on its real counts (US-014a) against the clean
        reference, with the week-by-week verdict history (US-014b).

        The history is the workspace's completed harness_runs rows; post-change
        weeks come from harness_task_ledger. Each agent's tool failures in the
        judged week, from its production tool logs, are issues (US-019).
        """
        from services import harness_diagnose, harness_runs

        window = metrics.window
        logger.info("[HARNESS] Phase 2 DIAGNOSE — workspace %s, week %s", workspace_id, window.cur_week)
        history = harness_diagnose.verdict_history(harness_runs.judged_verdicts(db, workspace_id, window.weeks))
        changes = harness_diagnose.applied_change_weeks(db, workspace_id, window)
        settings = harness_diagnose.settings_from_config()
        diagnosis = harness_diagnose.diagnose(metrics, history, changes, settings)
        issues = harness_diagnose.tool_failure_issues(metrics, settings)
        logger.info(
            "[HARNESS] DIAGNOSE done — coverage %s, convergence %s, %s, %d issues",
            diagnosis["coverage"], diagnosis["convergence"], diagnosis["counts"], len(issues),
        )
        return {**diagnosis, "issues": issues}

    # ------------------------------------------------------------------
    # Phase 3: PRESCRIBE
    # ------------------------------------------------------------------

    async def _phase_prescribe(
        self, workspace_id: UUID, diagnosis: Dict[str, Any], run_id: int
    ) -> List[Dict[str, Any]]:
        """Generate prioritized, risk-scored configuration change suggestions for
        the audit report; nothing is applied or queued (PRD-263 US-016).

        Ids are ``rx-<run_id>-<seq>``: the run's harness_runs id keeps two runs on
        one day apart. "Reconnect <app>" is the one suggestion left (US-019). The
        heartbeat, cost and description rules read list_agents fields and
        week-over-week deltas that never existed; they went with those inputs
        (PRD-263 PR-1D), and the change-kind registry brings real detectors
        (US-049, US-050).
        """
        logger.info("[HARNESS] Phase 3 PRESCRIBE — workspace %s", workspace_id)
        prescriptions = self._prescribe_reconnects(diagnosis.get("issues", []), run_id)
        logger.info("[HARNESS] PRESCRIBE done — %d prescriptions generated", len(prescriptions))
        return prescriptions

    @staticmethod
    def _prescribe_reconnects(issues: List[Dict[str, Any]], run_id: int) -> List[Dict[str, Any]]:
        """A "reconnect <app>" suggestion for each reconnect_app issue (PRD-263 US-019): a
        line of the audit report for the owner, applied by nothing. Its id is
        ``rx-<run_id>-<seq>``, from the run's harness_runs row. A tool failure gets no
        removal suggestion; tool-removal proposals come with US-049 and US-057."""
        from services.harness_diagnose import RECONNECT_APP

        reconnects = [issue for issue in issues if issue.get("root_cause") == RECONNECT_APP]
        return [{
            "prescription_id": f"rx-{run_id}-{seq:03d}",
            "target_type": "agent",
            "target_id": int(issue["agent_id"]),
            "target_name": issue["agent_name"],
            "change_type": RECONNECT_APP,
            "current_value": {"app_name": issue["app_name"]},
            "proposed_value": {"app_name": issue["app_name"]},
            "risk_score": RECONNECT_RISK,
            "expected_improvement": f"{issue['agent_name']}'s {issue['app_name']} calls stop failing",
            "rationale": issue["detail"],
        } for seq, issue in enumerate(reconnects, 1)]

    # ------------------------------------------------------------------
    # Phase 4: REPORT
    # ------------------------------------------------------------------

    async def _phase_report(
        self,
        workspace_id: UUID,
        diagnosis: Dict[str, Any],
        prescriptions: List[Dict[str, Any]],
        db: "Session",
        iteration: int,
    ) -> "tuple[str, str]":
        """Build the audit report and submit it, attributed to Auto.

        ``iteration`` is the run's harness_runs iteration (US-010) and
        ``prescriptions`` are PRESCRIBE's suggestions, reported and not applied
        (PRD-263 US-016). Returns
        ``(report_md, report_status)``: the audit report and "ok" or
        "failed: <reason>", both stored on the run's row. Nothing is written to
        the workspace volume: the run's row is its record (PRD-263 FR-2).
        """
        from services import harness_diagnose

        logger.info("[HARNESS] Phase 4 REPORT — workspace %s", workspace_id)
        report_content = self._build_changelog_markdown(iteration, diagnosis, prescriptions)
        report_status = await self._submit_audit_report(
            db, workspace_id, iteration, report_content,
            warning=harness_diagnose.needs_attention(diagnosis) or bool(diagnosis.get("issues")),
        )
        logger.info(
            "[HARNESS] REPORT done — iteration %d, convergence=%s, audit_report=%s",
            iteration, diagnosis.get("convergence"), report_status,
        )
        return report_content, report_status

    async def _submit_audit_report(
        self,
        db: "Session",
        workspace_id: UUID,
        iteration: int,
        report_content: str,
        warning: bool,
    ) -> str:
        """Submit HARNESS audit report attributed to Auto.

        One ReportService.create_report call, the write the platform_submit_report
        handler makes, without the platform action executor (PRD-263 US-048, FR-12).
        The PRD-248 shadow triage still wraps create_report, and runs only when the
        decision engine's report_triage_mode dial is on (FR-28).

        Returns "ok" or "failed: <reason>", which the tick stores as the run's
        report_status and get_status() reports.
        """
        from services.report_service import ReportService

        auto_agent = self._resolve_auto_agent(db, workspace_id)
        if auto_agent is None:
            msg = "no system agent (Auto) found in workspace — report not attributable"
            logger.error(
                "[HARNESS] %s — workspace %s. Audit report skipped.",
                msg, workspace_id,
            )
            return f"failed: {msg}"

        try:
            result = await ReportService(db, workspace_id).create_report(
                agent_id=auto_agent["id"],
                agent_name=auto_agent["name"],
                title=AUDIT_REPORT_TITLE.format(iteration=iteration),
                content=report_content,
                report_type=AUDIT_REPORT_TYPE,
                status="warning" if warning else "ok",
            )
        except Exception as exc:
            logger.error("[HARNESS] Failed to submit report: %s", exc, exc_info=True)
            return f"failed: {exc}"

        # Surface create_report's failures (it returns dicts, not raises)
        if result.get("success") is False:
            err = result.get("error", "unknown")
            logger.error("[HARNESS] The audit report was not created: %s", err)
            return f"failed: {err}"

        return "ok"

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _resolve_auto_agent(
        db: "Session", workspace_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Resolve the workspace's canonical reporting agent.

        Fallback chain:
          1. Agent.slug == "auto-{workspace_id}" + is_system_agent
          2. Any system agent named "Auto" in workspace
          3. Any system agent in workspace
        Returns {"id": int, "name": str} or None.
        """
        from core.models import Agent

        slug = f"auto-{workspace_id}"

        candidates = [
            (Agent.slug == slug, Agent.is_system_agent.is_(True)),
            (Agent.is_system_agent.is_(True), Agent.name == "Auto"),
            (Agent.is_system_agent.is_(True),),
        ]

        for clauses in candidates:
            agent = (
                db.query(Agent)
                .filter(Agent.workspace_id == workspace_id, *clauses)
                .first()
            )
            if agent is not None:
                return {"id": agent.id, "name": agent.name}
        return None

    def _build_changelog_markdown(
        self,
        iteration: int,
        diagnosis: Dict[str, Any],
        prescriptions: List[Dict[str, Any]],
    ) -> str:
        """Build the human-readable audit report: coverage and every agent's verdict
        with its n first (services.harness_diagnose.report_lines), then the issues
        and the changes HARNESS suggests, each a report line and nothing more
        (PRD-263 US-016). It never calls an unjudged agent stable (PRD-263 FR-7)."""
        from services import harness_diagnose

        lines = [
            f"# {AUDIT_REPORT_TITLE.format(iteration=iteration)}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            *harness_diagnose.report_lines(diagnosis),
            "",
        ]

        # Issues
        issues = diagnosis.get("issues", [])
        if issues:
            lines.append("## Issues Detected")
            for issue in issues:
                lines.append(f"- **{issue['agent_name']}**: {issue['detail']} (root cause: {issue['root_cause']})")
            lines.append("")

        # Suggested changes: reported, never applied or queued
        if prescriptions:
            lines.append(SUGGESTIONS_HEADING)
            for rx in prescriptions:
                lines.append(
                    f"- {rx['target_name']}: {rx['change_type']} ({rx['prescription_id']}, "
                    f"risk {rx['risk_score']}/5). {rx['rationale']}"
                )
            lines.append("")

        return "\n".join(lines)


def _iso(moment: Optional[datetime]) -> Optional[str]:
    """A timestamp as ISO 8601, or None."""
    return moment.isoformat() if moment else None


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_harness_service: Optional[HarnessService] = None


def get_harness_service() -> HarnessService:
    global _harness_service
    if _harness_service is None:
        _harness_service = HarnessService()
    return _harness_service
