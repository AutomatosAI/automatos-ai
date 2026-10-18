"""
HARNESS Service (PRD-121)
==========================

Self-Optimizing Organization Loop.  Runs weekly, invisible to users.
Reads each agent's real weekly counts, judges them against a clean reference
(services.harness_diagnose, PRD-263), prescribes configuration changes with
risk scores, auto-applies safe ones (risk ≤ 2), queues risky ones (risk ≥ 3)
as board tasks for human review, and submits the audit report.

Registered at server startup alongside HeartbeatService / CoordinatorService.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from sqlalchemy import text

from core.models.harness import RUN_COMPLETED, RUN_DORMANT_AGENTS, RUN_FAILED, RUN_RUNNING, HarnessRun

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from services.harness_metrics import HarnessMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The sweep's schedule is config.HARNESS_SWEEP_CRON, read in UTC (services.harness_runs).
_MIN_AGENTS = 3

# Risk thresholds live in config (Railway-overridable env vars), read at use:
#   HARNESS_AUTO_APPLY_MAX_RISK_STANDARD / _FULL — auto-apply ceilings (§12.3)
#   HARNESS_HIGH_PRIORITY_RISK — escalation / high-priority threshold
# DIAGNOSE's thresholds are the HARNESS_* statistics settings (services.harness_diagnose).

# Sentinel proposed_value meaning "a human must supply the real value". PRESCRIBE
# emitted it (model, description) until PRD-263 PR-1D, so [HARNESS] cards filed
# before then can still carry it. The apply layer MUST refuse it: writing it
# literally would set an agent's model/description to the string "review_needed".
_PLACEHOLDER_PROPOSED_VALUE = "review_needed"
# W4-S10: a fails_for_intent affinity must reach this many samples before HARNESS
# treats it as a *sustained* tool failure worth surfacing (not a one-off).
_TOOL_FAILURE_MIN_SAMPLES = 5

# F156: the ledger file kept before the move to the database, and the one card
# that says when it cannot be read (_file_unreadable_ledger_card).
LEGACY_LEDGER_FILE = "harness/applied_tasks.json"
UNREADABLE_LEDGER_TAG = "harness-ledger-unreadable"
UNREADABLE_LEDGER_TITLE = "[HARNESS] Self-management is paused: its old ledger can't be read"
UNREADABLE_LEDGER_TEXT = (
    "HARNESS keeps a record of which [HARNESS] changes it has applied. This workspace's older record is "
    f"the file {LEGACY_LEDGER_FILE} on its volume, which is read once to move it into the database, and it "
    "can't be read (it isn't valid JSON, or it can't be opened). Without it, a change applied before could "
    "be applied twice, so HARNESS applies no change here.\n\n"
    "To restart it, make the file valid JSON again: {\"applied_task_ids\": [...], "
    "\"needs_approve_task_ids\": [...]}, listing the board task ids of the [HARNESS] changes already "
    "applied and of those waiting for /approve. HARNESS reads it on its next run (the weekly tick, or an "
    "/approve). Deleting the file restarts HARNESS too, but then every done [HARNESS] task counts as not "
    "yet applied, and the changes they describe can be applied again."
)


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
          - disabled                      — workspace explicitly opted out
          - failed                        — the latest run failed, or was left
                                            unfinished with no lock holder (interrupted)
          - dormant_insufficient_agents   — fewer than _MIN_AGENTS active
          - scheduled_not_run_yet         — eligible but cron hasn't fired
          - completed                     — the latest run completed; with its
                                            coverage, convergence and verdict counts

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
        from core.models.workspaces import Workspace as WsModel
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

        ws = db.query(WsModel).get(workspace_id)
        if ws is not None and not self._workspace_opted_in(ws):
            return {
                "status": "disabled",
                "message": (
                    "HARNESS is disabled for this workspace. "
                    "Set workspace.settings.orchestrator.harness.disabled = false to re-enable."
                ),
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
        """Holding the lock: close interrupted runs, apply the weekly skip, then
        run the pipeline on its own harness_runs row."""
        from core.database.database import SessionLocal
        from services import harness_runs

        db = SessionLocal()
        try:
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
            outcome = await self._run_pipeline(db, workspace_id, cur_window)
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

    async def _run_pipeline(self, db: "Session", workspace_id: UUID, cur_window: str) -> Dict[str, Any]:
        """The dormancy check, then the five phases; returns the row's closing fields."""
        from core.database.database import end_open_transaction
        from core.models.workspaces import Workspace as WsModel
        from services import harness_diagnose, harness_runs

        sufficiency = self._sufficiency_breakdown(db, workspace_id)
        dormant = self._dormancy(sufficiency)
        if dormant:
            logger.info("[HARNESS] Dormant for %s — %s (%s)", workspace_id, dormant, sufficiency)
            return {"status": dormant, "counts": sufficiency}

        ws = db.query(WsModel).get(workspace_id)
        allow_auto = self._workspace_allows_auto_apply(ws.settings if ws else None)
        # §12.3: full-autonomy workspaces get the higher auto-apply risk ceiling.
        max_risk = self._auto_apply_max_risk(db, workspace_id)
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
        prescriptions = await self._phase_prescribe(workspace_id, diagnosis)
        end_open_transaction(db)
        changelog = await self._phase_apply(
            workspace_id, prescriptions, db,
            allow_auto_apply=allow_auto, max_risk=max_risk,
        )
        end_open_transaction(db)
        report_md, report_status = await self._phase_report(
            workspace_id, diagnosis, changelog, db, iteration,
        )
        counts = {
            **diagnosis["counts"], "prescriptions": len(prescriptions),
            **{key: len(rows) for key, rows in changelog.items()},
        }
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

        PRD-121 says "no setup required" — HARNESS is on by default.
        Workspaces opt out via `orchestrator.harness.disabled = true` (or
        the legacy explicit `enabled = false`); anything else means enabled.
        """
        settings = workspace.settings or {}
        harness = settings.get("orchestrator", {}).get("harness", {})
        if harness.get("disabled") is True:
            return False
        return harness.get("enabled", True)

    @staticmethod
    def _get_harness_config(workspace_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return harness config merged onto defaults (enabled by default)."""
        defaults = {
            "enabled": True,
            "disabled": False,
            "schedule": "weekly",
            "mode": "full_auto",
        }
        if not workspace_settings:
            return defaults
        return {
            **defaults,
            **workspace_settings.get("orchestrator", {}).get("harness", {}),
        }

    @staticmethod
    def _workspace_allows_auto_apply(workspace_settings: Optional[Dict[str, Any]]) -> bool:
        """Return False if the user wants all changes queued for review (mode=manual)."""
        if not workspace_settings:
            return True
        harness = workspace_settings.get("orchestrator", {}).get("harness", {})
        return harness.get("mode", "full_auto") == "full_auto"

    @staticmethod
    def _auto_apply_max_risk(db: "Session", workspace_id: UUID) -> int:
        """§12.3: the auto-apply risk ceiling for this workspace.

        Workspaces running at autonomy=full get the higher ``_FULL`` ceiling;
        everyone else gets ``_STANDARD``. Both are Railway-overridable config
        (HARNESS_AUTO_APPLY_MAX_RISK_*). The level is read via auto_autonomy, which
        fails safe to standard on a missing/corrupt setting — and a failed read
        (DB error mid-tick) fails safe here too, so a lookup problem can only
        NARROW the ceiling, never widen it, and never aborts the tick.
        """
        from config import config
        from core.services.auto_autonomy import is_full_autonomy

        try:
            full = is_full_autonomy(db, workspace_id)
        except Exception:
            logger.warning(
                "[HARNESS] autonomy lookup failed for %s — using standard ceiling",
                workspace_id, exc_info=True,
            )
            full = False
        if full:
            return config.HARNESS_AUTO_APPLY_MAX_RISK_FULL
        return config.HARNESS_AUTO_APPLY_MAX_RISK_STANDARD

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
        weeks come from harness_task_ledger. Sustained tool failures are surfaced
        as issues (W4-S10).
        """
        from services import harness_diagnose, harness_runs

        window = metrics.window
        logger.info("[HARNESS] Phase 2 DIAGNOSE — workspace %s, week %s", workspace_id, window.cur_week)
        history = harness_diagnose.verdict_history(harness_runs.judged_verdicts(db, workspace_id, window.weeks))
        changes = harness_diagnose.applied_change_weeks(db, workspace_id, window)
        diagnosis = harness_diagnose.diagnose(metrics, history, changes, harness_diagnose.settings_from_config())
        issues = self._diagnose_tool_failures(workspace_id, db, harness_diagnose.agent_names(metrics))
        logger.info(
            "[HARNESS] DIAGNOSE done — coverage %s, convergence %s, %s, %d issues",
            diagnosis["coverage"], diagnosis["convergence"], diagnosis["counts"], len(issues),
        )
        return {**diagnosis, "issues": issues}

    @staticmethod
    def _derive_app_name(action_name: str) -> Optional[str]:
        """Best-effort map a recorded action_name to a removable Composio app.

        Composio app actions are uppercase APP_ACTION (e.g. 'GMAIL_SEND_EMAIL' ->
        'GMAIL'). Returns None for meta-tools / platform_* / workspace_* tools,
        whose failures are surfaced as a diagnosis but never auto-proposed for
        removal — the assignment they map to is ambiguous (the tool hot path often
        records the 'composio_execute' meta-tool, not the specific app action).
        """
        if not action_name or "_" not in action_name or not action_name.isupper():
            return None
        return action_name.split("_", 1)[0]

    def _diagnose_tool_failures(
        self, workspace_id: UUID, db: "Session", name_by_id: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Surface SUSTAINED tool failures from the tool-routing learning graph
        (PRD-138/139 fails_for_intent) as inefficiency issues HARNESS can act on
        (W4-S10 — the learning cross-link). Learning-only: reads ONLY
        tool_routing_affinities, never a business entity. Workspace-scoped.
        ``name_by_id`` maps agent ids (as strings) to names.
        Best-effort — never raises (telemetry must not break the weekly tick)."""
        issues: List[Dict[str, Any]] = []
        try:
            from core.models.tool_routing import ToolRoutingAffinity

            rows = (
                db.query(ToolRoutingAffinity)
                .filter(
                    ToolRoutingAffinity.affinity_type == "fails_for_intent",
                    ToolRoutingAffinity.workspace_id == workspace_id,
                    ToolRoutingAffinity.agent_id.isnot(None),
                    ToolRoutingAffinity.sample_count >= _TOOL_FAILURE_MIN_SAMPLES,
                )
                .all()
            )
            for row in rows:
                issues.append({
                    "agent_id": str(row.agent_id),
                    "agent_name": name_by_id.get(str(row.agent_id), "unknown"),
                    "root_cause": "tool_failure",
                    "severity": "medium",
                    "detail": (
                        f"tool '{row.action_name}' failed {row.sample_count}x for this agent "
                        f"(tool-routing fails_for_intent)"
                    ),
                    "tool_failure_spec": {
                        "agent_id": row.agent_id,
                        "action_name": row.action_name,
                        "app_name": self._derive_app_name(row.action_name),
                        "sample_count": row.sample_count,
                    },
                })
        except Exception as exc:
            logger.warning("[HARNESS] tool-failure diagnosis skipped: %s", exc)
        return issues

    def _prescribe_tool_removals(self, issues, today, start_seq):
        """Turn tool_failure issues (W4-S10) into risk-3 tool_assignment_remove
        prescriptions — only when the failing action cleanly maps to a removable
        Composio app. Removing a tool is consequential: at standard autonomy it is
        queued for human review (risk 3 > the standard auto-apply ceiling); at full
        autonomy the ceiling rises to 3 (§12.3) so it auto-applies. When the action
        has no clean app mapping the issue is surfaced in the diagnosis but no
        removal is proposed. Returns (prescriptions, next_seq)."""
        out: List[Dict[str, Any]] = []
        seq = start_seq
        for issue in issues:
            if issue.get("root_cause") != "tool_failure":
                continue
            spec = issue.get("tool_failure_spec") or {}
            app_name = spec.get("app_name")
            if not app_name or spec.get("agent_id") is None:
                continue  # surfaced as a diagnosis, but not a clean removable assignment
            seq += 1
            out.append({
                "prescription_id": f"rx-{today}-{seq:03d}",
                "target_type": "agent",
                "target_id": spec.get("agent_id"),
                "target_name": issue.get("agent_name", "unknown"),
                "change_type": "tool_assignment_remove",
                "current_value": {"app_name": app_name},
                "proposed_value": {"app_name": app_name},
                "risk_score": 3,
                "expected_improvement": f"Stop assigning '{app_name}' — it keeps failing for this agent",
                "rationale": issue.get("detail", "Sustained tool failure from the tool-routing graph."),
            })
        return out, seq

    # ------------------------------------------------------------------
    # Phase 3: PRESCRIBE
    # ------------------------------------------------------------------

    async def _phase_prescribe(self, workspace_id: UUID, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized, risk-scored configuration change proposals.

        Sustained tool failures are the one input left (W4-S10). The heartbeat,
        cost and description rules read list_agents fields and week-over-week
        deltas that never existed; they went with those inputs (PRD-263 PR-1D),
        and the change-kind registry brings real detectors (US-049, US-050).
        """
        logger.info("[HARNESS] Phase 3 PRESCRIBE — workspace %s", workspace_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_rxs, _seq = self._prescribe_tool_removals(diagnosis.get("issues", []), today, 0)
        # Sort: risk ascending, then by prescription_id
        prescriptions = sorted(tool_rxs, key=lambda p: (p["risk_score"], p["prescription_id"]))
        logger.info("[HARNESS] PRESCRIBE done — %d prescriptions generated", len(prescriptions))
        return prescriptions

    # ------------------------------------------------------------------
    # Phase 4: APPLY
    # ------------------------------------------------------------------

    async def _phase_apply(
        self,
        workspace_id: UUID,
        prescriptions: List[Dict[str, Any]],
        db: "Session",
        allow_auto_apply: bool = True,
        max_risk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute safe changes (risk <= the workspace ceiling), queue risky ones.

        ``max_risk`` is this workspace's auto-apply ceiling (§12.3 — higher at
        autonomy=full); it defaults to the standard ceiling when a direct caller
        omits it. When allow_auto_apply is False (manual mode), ALL prescriptions
        are queued as board tasks regardless of risk score.
        """
        from config import config
        if max_risk is None:
            max_risk = config.HARNESS_AUTO_APPLY_MAX_RISK_STANDARD
        high_priority_risk = config.HARNESS_HIGH_PRIORITY_RISK
        logger.info(
            "[HARNESS] Phase 4 APPLY — workspace %s (auto_apply=%s)",
            workspace_id, allow_auto_apply,
        )

        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        executor = PlatformActionExecutor(db=db, workspace_id=workspace_id)

        changelog: Dict[str, List[Dict[str, Any]]] = {
            "applied": [],
            "queued": [],
            "failed": [],
            "escalated": [],
        }

        if not prescriptions:
            logger.info("[HARNESS] APPLY skipped — no prescriptions")
            return changelog

        for rx in prescriptions:
            risk = rx.get("risk_score", 5)
            rx_id = rx["prescription_id"]
            target_name = rx.get("target_name", "unknown")
            change_type = rx.get("change_type", "unknown")

            if allow_auto_apply and risk <= max_risk:
                # Auto-apply
                result = await self._auto_apply_prescription(executor, rx)
                if result.get("success"):
                    changelog["applied"].append({
                        "prescription_id": rx_id,
                        "target_id": rx.get("target_id"),
                        "target_name": target_name,
                        "change_type": change_type,
                        # Snapshot so US-022 can revert this change if the agent
                        # regresses on the next tick.
                        "current_value_before": self._snapshot_current_value(rx),
                        "result": "applied",
                    })
                else:
                    changelog["failed"].append({
                        "prescription_id": rx_id,
                        "target_name": target_name,
                        "change_type": change_type,
                        "error": result.get("error", "unknown"),
                    })
            else:
                # Queue as board task
                priority = "high" if risk >= high_priority_risk else "medium"
                try:
                    task_result = await executor.execute("platform_create_task", {
                        "title": f"[HARNESS] {change_type} for {target_name}",
                        "description": (
                            f"**Risk Score:** {risk}/5\n\n"
                            f"**Change Type:** {change_type}\n\n"
                            f"**Current:** {json.dumps(rx.get('current_value', {}))}\n\n"
                            f"**Proposed:** {json.dumps(rx.get('proposed_value', {}))}\n\n"
                            f"**Rationale:** {rx.get('rationale', '')}\n\n"
                            f"**Expected Improvement:** {rx.get('expected_improvement', '')}"
                        ),
                        # rx:{rx_id} lets US-025 /approve resolve the board task
                        # back to its prescription by tag.
                        "tags": ["harness", "org-review", f"risk-{risk}", f"rx:{rx_id}"],
                        "priority": priority,
                    })
                    changelog["queued"].append({
                        "prescription_id": rx_id,
                        "target_name": target_name,
                        "change_type": change_type,
                        "board_task_id": task_result.get("data", {}).get("id") if isinstance(task_result, dict) else None,
                    })
                    # US-024: nudge a human for high-risk changes (board task is
                    # the durable record; the notification is best-effort).
                    await self._maybe_escalate(db, workspace_id, rx, changelog)
                except Exception as exc:
                    logger.error("[HARNESS] Failed to queue rx %s: %s", rx_id, exc, exc_info=True)
                    changelog["failed"].append({
                        "prescription_id": rx_id,
                        "error": str(exc),
                    })

        # Apply previously approved board tasks (status=done, tag=harness)
        await self._apply_approved_board_tasks(executor, workspace_id, changelog)

        logger.info(
            "[HARNESS] APPLY done — %d applied, %d queued, %d failed, %d escalated",
            len(changelog["applied"]), len(changelog["queued"]),
            len(changelog["failed"]), len(changelog.get("escalated", [])),
        )
        return changelog

    async def _maybe_escalate(
        self,
        db: "Session",
        workspace_id: UUID,
        rx: Dict[str, Any],
        changelog: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Notify a human to approve/reject a high-risk queued prescription.

        Only fires for risk >= the escalation threshold (config
        HARNESS_HIGH_PRIORITY_RISK), and only when the workspace has a connected
        channel — otherwise there is nobody to notify and the
        board task already stands for in-app review (so we skip silently rather
        than record a phantom escalation). Records an 'escalated' changelog
        entry with whether delivery actually succeeded. Never raises.
        """
        from config import config
        risk = rx.get("risk_score", 5)
        if risk < config.HARNESS_HIGH_PRIORITY_RISK:
            return
        if not self._workspace_has_channel(db, workspace_id):
            return
        from core.services.notification_service import send_workspace_notification

        notified = await send_workspace_notification(
            str(workspace_id), self._build_escalation_message(rx, risk), channel="default"
        )
        changelog.setdefault("escalated", []).append({
            "prescription_id": rx.get("prescription_id"),
            "target_name": rx.get("target_name", "unknown"),
            "change_type": rx.get("change_type", "unknown"),
            "risk_score": risk,
            "notified": notified,
        })

    def _build_escalation_message(self, rx: Dict[str, Any], risk: int) -> str:
        """Plain-text approval request pointing to the Command Center.

        Approval happens on the authenticated Command Center surface
        (``api/harness.py``), never via a channel reply: the inbound webhook has
        no sender authorization, so a channel command can't be trusted to mutate
        state (PRD-142 Wave 4, W4-S1). The notice carries the prescription id so
        the admin can find it in the queued-prescriptions list.
        """
        rx_id = rx.get("prescription_id", "unknown")
        return (
            f"HARNESS needs approval (risk {risk}/5)\n"
            f"{rx.get('change_type', 'unknown')} for {rx.get('target_name', 'unknown')}\n"
            f"Current: {json.dumps(rx.get('current_value', {}))}\n"
            f"Proposed: {json.dumps(rx.get('proposed_value', {}))}\n"
            f"Review and approve or reject in the Command Center (prescription {rx_id})."
        )

    def _workspace_has_channel(self, db: "Session", workspace_id: UUID) -> bool:
        """True if the workspace has any channel_connections row.

        Matches how channels.sender resolves a channel — by row presence, not
        by the status column (which is 'active' in channels.manager but
        'connected' elsewhere) — so 'can escalate' agrees with 'can deliver'.
        """
        try:
            from sqlalchemy import text
            row = db.execute(
                text("SELECT 1 FROM channel_connections WHERE workspace_id = :ws LIMIT 1"),
                {"ws": str(workspace_id)},
            ).fetchone()
            return row is not None
        except Exception:
            logger.debug(
                "[HARNESS] channel-presence check failed for ws=%s", workspace_id, exc_info=True
            )
            return False

    # ------------------------------------------------------------------
    # Phase 5: REPORT
    # ------------------------------------------------------------------

    async def _phase_report(
        self,
        workspace_id: UUID,
        diagnosis: Dict[str, Any],
        changelog: Dict[str, Any],
        db: "Session",
        iteration: int,
    ) -> "tuple[str, str]":
        """Build the audit report and submit it, attributed to Auto.

        ``iteration`` is the run's harness_runs iteration (US-010). Returns
        ``(report_md, report_status)``: the audit report and "ok" or
        "failed: <reason>", both stored on the run's row. Nothing is written to
        the workspace volume: the run's row is its record (PRD-263 FR-2).
        """
        from modules.tools.discovery.platform_executor import PlatformActionExecutor
        from services import harness_diagnose

        logger.info("[HARNESS] Phase 5 REPORT — workspace %s", workspace_id)
        executor = PlatformActionExecutor(db=db, workspace_id=workspace_id)
        report_content = self._build_changelog_markdown(iteration, diagnosis, changelog)
        report_status = await self._submit_audit_report(
            executor, db, workspace_id, iteration, report_content,
            warning=harness_diagnose.needs_attention(diagnosis) or bool(diagnosis.get("issues")),
        )
        logger.info(
            "[HARNESS] REPORT done — iteration %d, convergence=%s, audit_report=%s",
            iteration, diagnosis.get("convergence"), report_status,
        )
        return report_content, report_status

    async def _submit_audit_report(
        self,
        executor: "PlatformActionExecutor",
        db: "Session",
        workspace_id: UUID,
        iteration: int,
        report_content: str,
        warning: bool,
    ) -> str:
        """Submit HARNESS audit report attributed to Auto.

        Returns "ok" or "failed: <reason>", which the tick stores as the run's
        report_status and get_status() reports.
        """
        auto_agent = self._resolve_auto_agent(db, workspace_id)
        if auto_agent is None:
            msg = "no system agent (Auto) found in workspace — report not attributable"
            logger.error(
                "[HARNESS] %s — workspace %s. Audit report skipped.",
                msg, workspace_id,
            )
            return f"failed: {msg}"

        try:
            result = await executor.execute("platform_submit_report", {
                "title": f"HARNESS Weekly Org Review — Run #{iteration}",
                "content": report_content,
                "report_type": "audit",
                "status": "warning" if warning else "ok",
                "_agent_id": auto_agent["id"],
                "_agent_name": auto_agent["name"],
            })
        except Exception as exc:
            logger.error("[HARNESS] Failed to submit report: %s", exc, exc_info=True)
            return f"failed: {exc}"

        # Surface handler-level failures (the handler returns dicts, not raises)
        if isinstance(result, dict) and result.get("success") is False:
            err = result.get("error", "unknown")
            logger.error("[HARNESS] platform_submit_report returned failure: %s", err)
            return f"failed: {err}"

        return "ok"

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _resolve_auto_agent(
        db: "Session", workspace_id: UUID
    ) -> Optional[Dict[str, str]]:
        """Resolve the workspace's canonical reporting agent.

        Fallback chain:
          1. Agent.slug == "auto-{workspace_id}" + is_system_agent
          2. Any system agent named "Auto" in workspace
          3. Any system agent in workspace
        Returns {"id": str, "name": str} or None.
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
                return {"id": str(agent.id), "name": agent.name}
        return None

    def _extract_agents_list(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract flat agent list from metrics."""
        agents_raw = metrics.get("agents")
        if not agents_raw:
            return []
        if isinstance(agents_raw, dict):
            return agents_raw.get("data", agents_raw.get("agents", []))
        if isinstance(agents_raw, list):
            return agents_raw
        return []

    async def _auto_apply_prescription(
        self,
        executor: "PlatformActionExecutor",
        rx: Dict[str, Any],
        caller_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single auto-apply prescription.

        ``caller_context`` names the person the change is made for: an owner's
        or admin's explicit /approve passes theirs (F151), so an admin_only
        action applies as that person. HARNESS's own ticks pass none — a lane
        acting for nobody, which the executor's admin gate refuses.
        """
        change_type = rx.get("change_type", "")
        target_id = rx.get("target_id")
        proposed = rx.get("proposed_value", {})

        # Refuse placeholder prescriptions — _phase_prescribe emits
        # {"model"|"description": "review_needed"} when it can flag a problem but
        # not decide the fix. Applying the sentinel literally would corrupt the
        # agent. This guard protects BOTH the low-risk auto-apply path and the
        # approved-board-task path, which both flow through here.
        if isinstance(proposed, dict) and _PLACEHOLDER_PROPOSED_VALUE in proposed.values():
            return {
                "success": False,
                "error": (
                    f"proposed_value is a placeholder ('{_PLACEHOLDER_PROPOSED_VALUE}') "
                    "requiring human input — not auto-applicable"
                ),
            }

        try:
            if change_type == "heartbeat_tune":
                result = await executor.execute("platform_configure_agent_heartbeat", {
                    "agent_id": target_id,
                    "interval_minutes": proposed.get("interval_minutes"),
                }, caller_context)
            elif change_type == "temperature_adjust":
                # update_agent reads temperature as a top-level param (it folds it
                # into model_config itself); a nested model_config is ignored.
                result = await executor.execute("platform_update_agent", {
                    "agent_id": target_id,
                    "temperature": proposed.get("temperature"),
                }, caller_context)
            elif change_type in ("tag_update", "description_update"):
                update_params = {"agent_id": target_id}
                if change_type == "tag_update":
                    update_params["tags"] = proposed.get("tags", [])
                else:
                    update_params["description"] = proposed.get("description", "")
                result = await executor.execute("platform_update_agent", update_params, caller_context)
            elif change_type == "model_change_same_tier":
                # update_agent reads the new model as the top-level model_id param
                # (a nested model_config is ignored). The prescription carries it
                # under "model", so map model -> model_id here.
                result = await executor.execute("platform_update_agent", {
                    "agent_id": target_id,
                    "model_id": proposed.get("model"),
                }, caller_context)
            elif change_type == "tool_assignment_add":
                # Verified against actions_assignments.py: the param is app_name
                # (the Composio app identifier), NOT tool_name. Idempotent —
                # re-activates a previously deactivated assignment.
                result = await executor.execute("platform_assign_tool_to_agent", {
                    "agent_id": target_id,
                    "app_name": proposed.get("app_name"),
                }, caller_context)
            elif change_type == "tool_assignment_remove":
                # Deactivates the assignment (is_active=False) by default, keeping
                # the audit trail. app_name is accepted by both assign/unassign.
                result = await executor.execute("platform_unassign_tool_from_agent", {
                    "agent_id": target_id,
                    "app_name": proposed.get("app_name"),
                }, caller_context)
            elif change_type == "routing_rule_add":
                # routing_rules is read by the UniversalRouter at Tier 2a
                # (core/routing/engine.py); the rule is workspace-scoped by the
                # executor context. The target is a routing rule, not an agent, so
                # target_id is unused here. (PRD-142 Wave 4, W4-S6.)
                result = await executor.execute("platform_create_routing_rule", {
                    "source_pattern": proposed.get("source_pattern"),
                    "source_channel": proposed.get("source_channel"),
                    "intent_keywords": proposed.get("intent_keywords", []),
                    "target_agent_id": proposed.get("target_agent_id"),
                    "target_workflow_id": proposed.get("target_workflow_id"),
                    "priority": proposed.get("priority", 0),
                }, caller_context)
            elif change_type in ("power_mode_upgrade", "power_mode_downgrade"):
                # Sets the workspace default power mode (workspace.settings['power_mode']),
                # which a Mission run inherits when its run_config doesn't pin one
                # (coordinator_service._workspace_power_mode_default). Workspace-scoped
                # by the executor context; target_id is unused (this is a workspace
                # knob, not an agent attribute). (PRD-142 Wave 4, W4-S5.)
                result = await executor.execute("platform_set_power_mode", {
                    "power_mode": proposed.get("power_mode"),
                }, caller_context)
            else:
                # routing_rule_add (W4-S6) and power_mode_* (W4-S5) are handled above.
                return {"success": False, "error": f"Unknown auto-apply change_type: {change_type}"}
            return result if isinstance(result, dict) else {"success": True}
        except Exception as exc:
            logger.error("[HARNESS] Auto-apply failed for %s: %s", rx.get("prescription_id"), exc)
            return {"success": False, "error": str(exc)}

    async def _apply_approved_board_tasks(
        self,
        executor: "PlatformActionExecutor",
        workspace_id: UUID,
        changelog: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Auto-apply previously approved (done) HARNESS board tasks.

        Gated on HARNESS_SELF_MANAGEMENT_ENABLED — when off this is a pure no-op,
        so Phase 5 stays inert by default. When on, each done [HARNESS] task that
        has not already been applied is parsed, its pre-change value snapshotted
        (for US-022 rollback), and applied via _auto_apply_prescription. Applied
        and held task ids are recorded in the task ledger (harness_task_ledger,
        F156); that ledger is the idempotency key because board tasks have no
        tag-mutation tool, so a task is never re-applied on a later weekly tick.
        """
        from config import config

        if not config.HARNESS_SELF_MANAGEMENT_ENABLED:
            return

        try:
            result = await executor.execute("platform_list_tasks", {
                "tags": ["harness"],
                "status": "done",
            })
            if not result or not isinstance(result, dict):
                return
            tasks = result.get("data", result.get("tasks", []))
            if not isinstance(tasks, list):
                return

            ledger = self._read_applied_tasks(executor.db, workspace_id)
            if ledger is None:
                logger.error("[HARNESS] Approved board tasks not applied for %s: the task ledger is unreadable",
                             workspace_id)
                return
            # Normalise ids to str: the ledger stores integers and the task API
            # may return either, so one canonical type keeps the membership
            # check exact.
            applied_ids = {str(i) for i in ledger.get("applied_task_ids", [])}
            # F151: tasks held for an owner's or admin's /approve are not retried.
            held_ids = {str(i) for i in ledger.get("needs_approve_task_ids", [])}
            agents_by_name = await self._resolve_agents_by_name(executor)

            newly_applied: List[Dict[str, Any]] = []
            newly_held: List[str] = []
            for task in tasks:
                raw_id = task.get("id")
                if raw_id is None:
                    continue
                task_id = str(raw_id)
                if task_id in applied_ids or task_id in held_ids:
                    continue

                rx = self._parse_harness_task(task, agents_by_name=agents_by_name)
                if not rx or rx.get("target_id") is None:
                    # Never apply a change to an unresolved / guessed target.
                    changelog.setdefault("skipped", []).append({
                        "task_id": task_id,
                        "title": task.get("title", ""),
                        "reason": "unparseable or unresolved target",
                    })
                    continue

                current_before = self._snapshot_current_value(rx)
                apply_result = await self._auto_apply_prescription(executor, rx)

                if apply_result.get("success"):
                    entry = {
                        "task_id": task_id,
                        "prescription_id": rx.get("prescription_id"),
                        "target_id": rx.get("target_id"),
                        "target_name": rx.get("target_name"),
                        "change_type": rx.get("change_type"),
                        "current_value_before": current_before,
                        "proposed_value": rx.get("proposed_value"),
                        "applied_at": datetime.now(timezone.utc).isoformat(),
                    }
                    applied_ids.add(task_id)
                    newly_applied.append(entry)
                    changelog.setdefault("applied_from_approved", []).append(entry)
                elif apply_result.get("required_role") == "owner_or_admin":
                    # F151: completing a board task is not an owner's or admin's
                    # approval of an admin_only change — say so once, then wait.
                    newly_held.append(task_id)
                    self._note_needs_an_approve(executor, workspace_id, task, rx)
                    changelog.setdefault("needs_approve", []).append({
                        "task_id": task_id,
                        "title": task.get("title", ""),
                        "change_type": rx.get("change_type"),
                    })
                else:
                    changelog.setdefault("failed", []).append({
                        "task_id": task_id,
                        "title": task.get("title", ""),
                        "error": apply_result.get("error", "unknown"),
                    })

            if newly_applied or newly_held:
                self._write_applied_tasks(executor.db, workspace_id, newly_applied, newly_held)
        except Exception as exc:
            # A failure here means human-approved changes were silently dropped —
            # surface the trace rather than bury it in a warning.
            logger.error(
                "[HARNESS] Failed to apply approved board tasks: %s",
                exc, exc_info=True,
            )

    @staticmethod
    def _note_needs_an_approve(
        executor: "PlatformActionExecutor", workspace_id: UUID, task: Dict[str, Any], rx: Dict[str, Any]
    ) -> None:
        """F151: write once, on a done [HARNESS] task, that its change was not applied.

        The status action carries no result field, so the row is set directly, as
        api.harness_commands does for an applied change. Best-effort: the ledger
        already holds the task, so a failed note only loses the explanation.
        """
        from core.models.core import BoardTask

        rx_tag = next((t for t in task.get("tags") or [] if isinstance(t, str) and t.startswith("rx:")), None)
        how = f"/approve {rx_tag[3:]}" if rx_tag else "/approve"
        note = {
            "actuated": False,
            "needs": "an owner's or admin's /approve",
            "message": (
                f"Not applied: {rx.get('change_type')} is an owner's or admin's change, and completing "
                f"this task does not approve it. An owner or admin applies it with {how}."
            ),
        }
        try:
            row = (
                executor.db.query(BoardTask)
                .filter(BoardTask.id == int(task.get("id")), BoardTask.workspace_id == workspace_id)
                .first()
            )
            if row is not None:
                row.result = json.dumps(note)
                executor.db.commit()
        except Exception:  # noqa: BLE001 — the ledger hold stands without the note
            logger.warning("[HARNESS] Could not note task %s as needing /approve", task.get("id"), exc_info=True)

    async def _resolve_agents_by_name(
        self, executor: "PlatformActionExecutor"
    ) -> Dict[str, Any]:
        """Build a {agent_name: agent_id} map for resolving board-task targets."""
        try:
            result = await executor.execute("platform_list_agents", {})
        except Exception as exc:
            logger.warning("[HARNESS] Failed to list agents for self-management: %s", exc)
            return {}
        agents = self._extract_agents_list({"agents": result})
        return {
            a.get("name"): a.get("id")
            for a in agents
            if isinstance(a, dict) and a.get("name") and a.get("id") is not None
        }

    def _snapshot_current_value(self, rx: Dict[str, Any]) -> Dict[str, Any]:
        """Capture the value an approved change is about to overwrite.

        Uses the prescription's parsed current_value — the producer wrote it in
        the same shape _auto_apply_prescription consumes, so it round-trips
        exactly with no agent-schema coupling. Recorded as current_value_before
        so US-022 can roll back to it on regression.
        """
        current = rx.get("current_value")
        if isinstance(current, dict):
            return current
        # Rollback prescriptions legitimately carry current_value={} (a dict), so
        # reaching here means a non-dict slipped through (e.g. a mis-parsed board
        # task). The change stays safe — it just becomes ineligible for rollback —
        # but log it so the dropped snapshot is visible rather than silent.
        if current is not None:
            logger.warning(
                "[HARNESS] prescription %s has non-dict current_value (%s); "
                "recording empty snapshot — change will be ineligible for rollback",
                rx.get("prescription_id"), type(current).__name__,
            )
        return {}

    @staticmethod
    def _read_applied_tasks(db: Any, workspace_id: UUID) -> Optional[Dict[str, Any]]:
        """The task ledger: the done [HARNESS] board tasks already applied, and
        those held for an owner's or admin's /approve (F156: harness_task_ledger,
        written only by this service — no workspace tool reaches it).

        None when it cannot be read, and callers then apply nothing: not every
        change re-applies harmlessly (a routing rule is added again), and a
        human may have reverted one since. A workspace with nothing in the
        table has its pre-F156 file imported first (_import_legacy_ledger).
        """
        from core.models.harness import LEDGER_APPLIED, LEDGER_HELD

        try:
            with db.begin_nested():
                rows = db.execute(
                    text("SELECT board_task_id, state FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid)"),
                    {"ws": str(workspace_id)},
                ).fetchall()
        except Exception:  # noqa: BLE001 — unreadable: apply nothing (see the docstring)
            logger.error("[HARNESS] Task ledger unreadable for %s — nothing is applied", workspace_id, exc_info=True)
            return None
        if not rows:
            rows = HarnessService._import_legacy_ledger(db, workspace_id)
            if rows is None:
                return None
        return {
            "applied_task_ids": sorted(str(task) for task, state in rows if state == LEDGER_APPLIED),
            "needs_approve_task_ids": sorted(str(task) for task, state in rows if state == LEDGER_HELD),
        }

    @staticmethod
    def _import_legacy_ledger(db: Any, workspace_id: UUID) -> Optional[List[Any]]:
        """F156, once per workspace: the ledger it kept in
        harness/applied_tasks.json on the workspace volume before the move to
        the database. Its applied tasks (with their entries) and held tasks are
        written to harness_task_ledger when the table has none for the
        workspace, so nothing applied before the move is applied again.

        Returns the (task id, state) pairs it found: [] when there is no file,
        None when the file cannot be read (callers then apply nothing).
        """
        import os

        from config import config
        from core.models.harness import LEDGER_APPLIED, LEDGER_HELD

        path = os.path.join(config.WORKSPACE_VOLUME_PATH, str(workspace_id), LEGACY_LEDGER_FILE)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            applied = [str(int(task)) for task in data.get("applied_task_ids") or []]
            held = [str(int(task)) for task in data.get("needs_approve_task_ids") or [] if str(int(task)) not in applied]
            entries = {str(e.get("task_id")): e for e in data.get("entries") or [] if isinstance(e, dict)}
        except Exception:  # noqa: BLE001 — unreadable: apply nothing until it is fixed or removed
            logger.error("[HARNESS] The pre-F156 ledger for %s cannot be read; nothing is applied until it is "
                         "fixed or removed (%s)", workspace_id, path, exc_info=True)
            HarnessService._file_unreadable_ledger_card(db, workspace_id)
            return None
        HarnessService._write_applied_tasks(
            db, workspace_id, [{**entries.get(task, {}), "task_id": task} for task in applied], held,
        )
        logger.info("[HARNESS] Imported the pre-F156 ledger for %s: %d applied, %d held",
                    workspace_id, len(applied), len(held))
        return [(int(task), LEDGER_APPLIED) for task in applied] + [(int(task), LEDGER_HELD) for task in held]

    @staticmethod
    def _file_unreadable_ledger_card(db: Any, workspace_id: UUID) -> None:
        """F156: the one blocked [HARNESS] card saying why self-management
        applies nothing in this workspace and how to restart it. Filed once per
        workspace, never again; best-effort (a card that cannot be filed is
        logged, and nothing is applied either way)."""
        from datetime import datetime, timezone

        from core.models.core import BoardTask

        try:
            # One card whoever gets here first: a concurrent caller (an /approve
            # during the weekly tick, another worker) waits on this lock, then
            # finds the card. The lock ends with the transaction.
            db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
                       {"key": f"{UNREADABLE_LEDGER_TAG}:{workspace_id}"})
            if db.query(BoardTask.id).filter(BoardTask.workspace_id == workspace_id,
                                             BoardTask.tags.contains([UNREADABLE_LEDGER_TAG])).first():
                db.commit()  # the lock goes with it
                return
            with db.begin_nested():
                db.add(BoardTask(
                    workspace_id=workspace_id, title=UNREADABLE_LEDGER_TITLE, description=UNREADABLE_LEDGER_TEXT,
                    status="blocked", blocked_at=datetime.now(timezone.utc),
                    blocked_reason=f"{LEGACY_LEDGER_FILE} can't be read", priority="high", review_mode="human",
                    created_by_type="system", created_by_id="harness", tags=["harness", UNREADABLE_LEDGER_TAG],
                ))
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
            logger.warning("[HARNESS] Could not file the unreadable-ledger card for %s", workspace_id, exc_info=True)

    @staticmethod
    def _write_applied_tasks(
        db: Any,
        workspace_id: UUID,
        newly_applied: List[Dict[str, Any]],
        newly_held: List[str],
    ) -> None:
        """Record newly applied tasks (each with its entry: the value it
        replaced, for US-022) and newly held ones. A held task that is later
        applied becomes applied; an applied one never goes back to held."""
        from core.models.harness import LEDGER_APPLIED, LEDGER_HELD

        rows = [(entry.get("task_id"), LEDGER_APPLIED, json.dumps(entry, default=str)) for entry in newly_applied]
        rows += [(task_id, LEDGER_HELD, None) for task_id in newly_held]
        try:
            with db.begin_nested():
                for task_id, state, entry in rows:
                    db.execute(
                        text(
                            "INSERT INTO harness_task_ledger (workspace_id, board_task_id, state, entry) "
                            "VALUES (CAST(:ws AS uuid), :task, :state, CAST(:entry AS jsonb)) "
                            "ON CONFLICT (workspace_id, board_task_id) DO UPDATE SET "
                            "state = CASE WHEN harness_task_ledger.state = :applied THEN :applied "
                            "ELSE EXCLUDED.state END, "
                            "entry = COALESCE(EXCLUDED.entry, harness_task_ledger.entry), updated_at = now()"
                        ),
                        {"ws": str(workspace_id), "task": int(task_id), "state": state, "entry": entry,
                         "applied": LEDGER_APPLIED},
                    )
            db.commit()
        except Exception as exc:  # noqa: BLE001 — the next tick finds the tasks again
            logger.warning("[HARNESS] Failed to record the task ledger for %s: %s", workspace_id, exc)

    def _parse_harness_task(
        self,
        task: Dict[str, Any],
        agents_by_name: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct a prescription from an approved [HARNESS] board task.

        Inverse of the producer in _phase_apply(): parses the
        '[HARNESS] {change_type} for {target_name}' title and the
        '**Current:** {json}' / '**Proposed:** {json}' description sections,
        and resolves target_id from target_name via agents_by_name (name -> id).
        Returns None for any task that is not a well-formed HARNESS task.

        target_id is None when the name cannot be resolved — callers MUST treat
        an unresolved (None) target as non-applicable and skip it, never apply a
        change to a guessed/null target.
        """
        title = (task.get("title") or "").strip()
        prefix = "[HARNESS] "
        if not title.startswith(prefix):
            return None

        # change_type is a space-free snake_case identifier, so partitioning on
        # the first " for " cleanly splits it from target_name (which may itself
        # contain spaces or even " for ").
        change_type, sep, target_name = title[len(prefix):].partition(" for ")
        change_type = change_type.strip()
        target_name = target_name.strip()
        if not sep or not change_type or not target_name:
            return None

        description = task.get("description") or ""
        agents_by_name = agents_by_name or {}

        return {
            "prescription_id": f"rx-task-{task.get('id')}",
            "target_type": "agent",
            "target_id": agents_by_name.get(target_name),
            "target_name": target_name,
            "change_type": change_type,
            "current_value": self._extract_json_section(description, "Current"),
            "proposed_value": self._extract_json_section(description, "Proposed"),
            "risk_score": self._extract_risk_score(description, task.get("tags")),
            "rationale": self._extract_text_section(description, "Rationale"),
            "expected_improvement": self._extract_text_section(description, "Expected Improvement"),
        }

    @staticmethod
    def _extract_json_section(description: str, label: str) -> Dict[str, Any]:
        """Pull a single-line '**{label}:** {json}' section and json.loads it."""
        match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", description)
        if not match:
            return {}
        try:
            value = json.loads(match.group(1).strip())
        except (ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_text_section(description: str, label: str) -> str:
        """Pull the trailing text of a '**{label}:** {text}' section."""
        match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", description)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_risk_score(description: str, tags: Optional[List[str]]) -> int:
        """Recover the risk score from the description, falling back to a
        'risk-{n}' tag. Unknown -> 5 (max risk; never qualifies for auto-apply)."""
        match = re.search(r"\*\*Risk Score:\*\*\s*(\d+)", description)
        if match:
            return int(match.group(1))
        for tag in (tags or []):
            if isinstance(tag, str) and tag.startswith("risk-"):
                try:
                    return int(tag.split("-", 1)[1])
                except ValueError:
                    continue
        return 5

    def _build_changelog_markdown(
        self,
        iteration: int,
        diagnosis: Dict[str, Any],
        changelog: Dict[str, Any],
    ) -> str:
        """Build the human-readable audit report: coverage and every agent's verdict
        with its n first (services.harness_diagnose.report_lines), then the issues
        and the changes. It never calls an unjudged agent stable (PRD-263 FR-7)."""
        from services import harness_diagnose

        lines = [
            f"# HARNESS Weekly Org Review — Run #{iteration}",
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

        # Applied changes
        applied = changelog.get("applied", [])
        if applied:
            lines.append("## Applied Changes (Auto)")
            for change in applied:
                lines.append(f"- {change['target_name']}: {change['change_type']}")
            lines.append("")

        # Queued changes
        queued = changelog.get("queued", [])
        if queued:
            lines.append("## Queued for Review")
            for change in queued:
                lines.append(f"- {change['target_name']}: {change['change_type']}")
            lines.append("")

        # Failed
        failed = changelog.get("failed", [])
        if failed:
            lines.append("## Failed")
            for f_item in failed:
                lines.append(f"- {f_item.get('prescription_id', '?')}: {f_item.get('error', 'unknown')}")
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
