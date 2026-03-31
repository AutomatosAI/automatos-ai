"""
HARNESS Service (PRD-121)
==========================

Self-Optimizing Organization Loop.  Runs weekly, invisible to users.
Collects org-wide metrics, diagnoses regressions, prescribes configuration
changes with risk scores, auto-applies safe ones (risk ≤ 2), queues risky
ones (risk ≥ 3) as board tasks for human review, and snapshots a new
baseline for next week's comparison.

Registered at server startup alongside HeartbeatService / CoordinatorService.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRON_EXPRESSION = {"day_of_week": "sun", "hour": 2, "minute": 0}  # Sunday 2AM UTC
_MIN_AGENTS = 3
_MIN_DATA_DAYS = 7

# Risk thresholds
_AUTO_APPLY_MAX_RISK = 2
_HIGH_PRIORITY_RISK = 4

# Convergence thresholds
_CONVERGED_DELTA = 2.0
_CONVERGED_VARIANCE = 0.02
_CONVERGED_MIN_RUNS = 2
_EXPLORING_MAX_ITERATION = 3

# Delta classification thresholds (percentage)
_REGRESSION_THRESHOLD = 10.0
_IMPROVEMENT_THRESHOLD = 10.0

# Pareto: don't optimise cost when quality is poor
_MIN_SUCCESS_RATE_FOR_COST_OPT = 85.0


class HarnessService:
    """Weekly self-optimizing loop for workspace agent configuration."""

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._owns_scheduler: bool = False
        self._running: Dict[str, bool] = {}  # per-workspace lock

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: Optional[AsyncIOScheduler] = None) -> None:
        """Register weekly cron jobs for all active workspaces."""
        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            jobstores = {"default": MemoryJobStore()}
            self._scheduler = AsyncIOScheduler(jobstores=jobstores)
            self._scheduler.start()
            self._owns_scheduler = True

        await self._register_workspace_jobs()
        logger.info("[HARNESS] Service started (cron: Sunday 02:00 UTC)")

    async def _register_workspace_jobs(self) -> None:
        """Query active workspaces and register a cron job for each."""
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace

        db = SessionLocal()
        try:
            workspaces = db.query(Workspace.id).filter(Workspace.is_active.is_(True)).all()
            for (ws_id,) in workspaces:
                job_id = f"harness_{ws_id}"
                self._scheduler.add_job(
                    self._harness_tick,
                    "cron",
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    kwargs={"workspace_id": ws_id},
                    **_CRON_EXPRESSION,
                )
            logger.info("[HARNESS] Registered jobs for %d workspaces", len(workspaces))
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Public helpers (called by platform tool handlers)
    # ------------------------------------------------------------------

    def get_status(self, workspace_id: UUID) -> Dict[str, Any]:
        """Return current HARNESS state by reading baseline_latest.json."""
        baseline = self._read_baseline(None, workspace_id)
        if baseline is None:
            return {
                "status": "dormant",
                "message": "No HARNESS runs yet",
                "iteration_count": 0,
            }
        conv = baseline.get("convergence", {})
        return {
            "status": conv.get("status", "unknown"),
            "iteration_count": conv.get("iteration_count", 0),
            "last_run": baseline.get("created_at"),
            "total_delta_magnitude": conv.get("total_delta_magnitude"),
            "next_scheduled_run": "Sunday 02:00 UTC",
        }

    async def trigger_now(self, workspace_id: UUID) -> None:
        """Kick off a HARNESS run outside the weekly cron schedule."""
        asyncio.ensure_future(self._harness_tick(workspace_id=workspace_id))

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def _harness_tick(self, workspace_id: UUID) -> None:
        ws_key = str(workspace_id)
        if self._running.get(ws_key):
            logger.warning("[HARNESS] Tick already running for %s, skipping", ws_key)
            return

        self._running[ws_key] = True
        t0 = time.monotonic()
        logger.info("[HARNESS] Tick started for workspace %s", ws_key)

        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            # ----- Dormancy check -----
            if not self._has_sufficient_data(db, workspace_id):
                logger.info("[HARNESS] Dormant for %s — insufficient data", ws_key)
                return

            # ----- 5-phase pipeline -----
            metrics = await self._phase_collect(workspace_id, db)
            diagnosis = await self._phase_diagnose(workspace_id, metrics, db)
            prescriptions = await self._phase_prescribe(workspace_id, diagnosis, metrics, db)
            changelog = await self._phase_apply(workspace_id, prescriptions, db)
            await self._phase_baseline(workspace_id, metrics, diagnosis, prescriptions, changelog, db)

            elapsed = time.monotonic() - t0
            logger.info(
                "[HARNESS] Tick completed for %s in %.1fs — %d prescriptions, %d applied, %d queued",
                ws_key, elapsed,
                len(prescriptions),
                len(changelog.get("applied", [])),
                len(changelog.get("queued", [])),
            )
        except Exception:
            logger.error("[HARNESS] Tick failed for %s", ws_key, exc_info=True)
        finally:
            db.close()
            self._running[ws_key] = False

    # ------------------------------------------------------------------
    # Dormancy check
    # ------------------------------------------------------------------

    def _has_sufficient_data(self, db: "Session", workspace_id: UUID) -> bool:
        from core.models import Agent
        from sqlalchemy import text

        agent_count = (
            db.query(Agent)
            .filter(Agent.workspace_id == workspace_id, Agent.is_active.is_(True))
            .count()
        )
        if agent_count < _MIN_AGENTS:
            return False

        # Check earliest heartbeat_result for this workspace (raw SQL — no ORM model)
        row = db.execute(
            text(
                "SELECT MIN(created_at) FROM heartbeat_results "
                "WHERE workspace_id = :ws_id"
            ),
            {"ws_id": str(workspace_id)},
        ).fetchone()

        if row is None or row[0] is None:
            return False

        earliest_dt = row[0]
        if earliest_dt.tzinfo is None:
            earliest_dt = earliest_dt.replace(tzinfo=timezone.utc)
        days_of_data = (datetime.now(timezone.utc) - earliest_dt).days
        return days_of_data >= _MIN_DATA_DAYS

    # ------------------------------------------------------------------
    # Phase 1: COLLECT
    # ------------------------------------------------------------------

    async def _phase_collect(self, workspace_id: UUID, db: "Session") -> Dict[str, Any]:
        """Gather comprehensive raw metrics across all agents and systems."""
        logger.info("[HARNESS] Phase 1 COLLECT — workspace %s", workspace_id)
        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        executor = PlatformActionExecutor(db=db, workspace_id=workspace_id)

        tool_calls = {
            "agents": ("platform_list_agents", {}),
            "ranking": ("platform_get_agent_ranking", {}),
            "success_rate": ("platform_get_success_rate", {}),
            "error_rates": ("platform_get_error_rates", {}),
            "cost_breakdown": ("platform_get_cost_breakdown", {}),
            "sla": ("platform_get_sla_compliance", {}),
            "efficiency": ("platform_get_efficiency_score", {}),
            "bottlenecks": ("platform_get_bottlenecks", {}),
            "llm_usage": ("platform_get_llm_usage", {}),
            "board_summary": ("platform_board_summary", {}),
        }

        metrics: Dict[str, Any] = {}

        async def _call(key: str, action: str, params: dict) -> None:
            try:
                result = await executor.execute(action, params)
                metrics[key] = result
            except Exception as exc:
                logger.warning("[HARNESS] COLLECT %s failed: %s", action, exc)
                metrics[key] = None

        # Run all independent calls concurrently
        await asyncio.gather(*[
            _call(k, action, params) for k, (action, params) in tool_calls.items()
        ])

        # Read previous baseline (sequential — needs result for later phases)
        metrics["previous_baseline"] = self._read_baseline(db, workspace_id)

        # Prior HARNESS board tasks
        try:
            tasks_result = await executor.execute("platform_list_tasks", {"tags": ["harness"]})
            metrics["prior_harness_tasks"] = tasks_result
        except Exception:
            metrics["prior_harness_tasks"] = None

        agent_count = 0
        if metrics.get("agents") and isinstance(metrics["agents"], dict):
            agent_count = len(metrics["agents"].get("data", metrics["agents"].get("agents", [])))
        logger.info("[HARNESS] COLLECT done — %d agents found", agent_count)

        return metrics

    # ------------------------------------------------------------------
    # Phase 2: DIAGNOSE
    # ------------------------------------------------------------------

    async def _phase_diagnose(
        self, workspace_id: UUID, metrics: Dict[str, Any], db: "Session"
    ) -> Dict[str, Any]:
        """Compare current metrics against previous baseline, produce health cards."""
        logger.info("[HARNESS] Phase 2 DIAGNOSE — workspace %s", workspace_id)

        baseline = metrics.get("previous_baseline")
        is_first_run = baseline is None

        # Extract agents list from metrics
        agents_data = self._extract_agents_list(metrics)

        health_cards: Dict[str, Dict[str, Any]] = {}
        issues: List[Dict[str, Any]] = []
        total_delta_magnitude = 0.0

        baseline_agents = {}
        if not is_first_run:
            baseline_agents = baseline.get("per_agent", {})

        for agent in agents_data:
            agent_id = str(agent.get("id", ""))
            agent_name = agent.get("name", "unknown")
            prev = baseline_agents.get(agent_id, {})

            card = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "model": agent.get("model", "unknown"),
            }

            if is_first_run:
                card["classification"] = "NEW"
                card["success_rate_delta"] = 0
                card["cost_delta"] = 0
                card["efficiency_delta"] = 0
                card["error_rate_delta"] = 0
            else:
                deltas = self._compute_deltas(agent, prev)
                card.update(deltas)

                # Classify
                classification = "STABLE"
                worst_regression = max(
                    deltas.get("cost_delta", 0),        # higher cost = bad
                    -deltas.get("success_rate_delta", 0),  # lower success = bad
                    -deltas.get("efficiency_delta", 0),    # lower efficiency = bad
                    deltas.get("error_rate_delta", 0),     # higher errors = bad
                )
                best_improvement = max(
                    deltas.get("success_rate_delta", 0),
                    deltas.get("efficiency_delta", 0),
                    -deltas.get("cost_delta", 0),
                    -deltas.get("error_rate_delta", 0),
                )

                if worst_regression > _REGRESSION_THRESHOLD:
                    classification = "REGRESSION"
                elif best_improvement > _IMPROVEMENT_THRESHOLD:
                    classification = "IMPROVEMENT"

                card["classification"] = classification

                # Track issues
                if classification == "REGRESSION":
                    root_cause = self._infer_root_cause(deltas, agent)
                    issues.append({
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "root_cause": root_cause,
                        "severity": "high" if worst_regression > 25 else "medium",
                        "detail": f"{agent_name}: {classification} — worst delta {worst_regression:.1f}%",
                    })

                # Sum of absolute deltas for convergence tracking
                for d_key in ("success_rate_delta", "cost_delta", "efficiency_delta", "error_rate_delta"):
                    total_delta_magnitude += abs(deltas.get(d_key, 0))

            health_cards[agent_id] = card

        # Org-level diagnosis
        org_diagnosis = self._compute_org_diagnosis(metrics, baseline)

        diagnosis = {
            "health_cards": health_cards,
            "org_diagnosis": org_diagnosis,
            "total_delta_magnitude": total_delta_magnitude,
            "issues": issues,
            "is_first_run": is_first_run,
        }

        logger.info(
            "[HARNESS] DIAGNOSE done — %d agents, %d issues, delta_magnitude=%.1f, first_run=%s",
            len(health_cards), len(issues), total_delta_magnitude, is_first_run,
        )
        return diagnosis

    # ------------------------------------------------------------------
    # Phase 3: PRESCRIBE
    # ------------------------------------------------------------------

    async def _phase_prescribe(
        self,
        workspace_id: UUID,
        diagnosis: Dict[str, Any],
        metrics: Dict[str, Any],
        db: "Session",
    ) -> List[Dict[str, Any]]:
        """Generate prioritized, risk-scored configuration change proposals."""
        logger.info("[HARNESS] Phase 3 PRESCRIBE — workspace %s", workspace_id)

        if diagnosis.get("is_first_run"):
            logger.info("[HARNESS] PRESCRIBE skipped — first run, no baseline to compare against")
            return []

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prescriptions: List[Dict[str, Any]] = []
        seq = 0

        # Collect rejected prescription signatures from prior board tasks
        rejected_signatures = self._get_rejected_signatures(metrics)

        agents_data = self._extract_agents_list(metrics)
        health_cards = diagnosis.get("health_cards", {})

        # Compute workspace averages for comparison
        heartbeat_costs = [a.get("heartbeat_cost_7d", 0) for a in agents_data if a.get("heartbeat_cost_7d")]
        avg_heartbeat_cost = sum(heartbeat_costs) / len(heartbeat_costs) if heartbeat_costs else 0

        for agent in agents_data:
            agent_id = str(agent.get("id", ""))
            agent_name = agent.get("name", "unknown")
            card = health_cards.get(agent_id, {})
            success_rate = agent.get("success_rate", card.get("success_rate", 100))

            # --- Heartbeat tuning ---
            hb_cost = agent.get("heartbeat_cost_7d", 0)
            hb_interval = agent.get("heartbeat_interval", agent.get("heartbeat_interval_minutes", 0))
            task_volume_delta = card.get("task_volume_delta", 0)

            if (
                hb_cost > avg_heartbeat_cost * 2
                and task_volume_delta < -25
                and hb_interval > 0
                and hb_interval < 360
            ):
                new_interval = min(hb_interval + 60, 360)
                sig = f"heartbeat_tune:{agent_name}"
                if sig not in rejected_signatures:
                    seq += 1
                    prescriptions.append({
                        "prescription_id": f"rx-{today}-{seq:03d}",
                        "target_type": "agent",
                        "target_id": agent.get("id"),
                        "target_name": agent_name,
                        "change_type": "heartbeat_tune",
                        "current_value": {"interval_minutes": hb_interval},
                        "proposed_value": {"interval_minutes": new_interval},
                        "risk_score": 2,
                        "expected_improvement": f"Save heartbeat token costs by extending interval from {hb_interval}min to {new_interval}min",
                        "rationale": (
                            f"{agent_name}'s heartbeat cost is {hb_cost:.2f} (>{avg_heartbeat_cost*2:.2f} = 2x avg) "
                            f"while task volume dropped {abs(task_volume_delta):.0f}%. "
                            f"Extending interval reduces cost without impacting low activity."
                        ),
                    })

            # --- Cost optimisation (only if quality is healthy) ---
            if success_rate >= _MIN_SUCCESS_RATE_FOR_COST_OPT:
                cost_delta = card.get("cost_delta", 0)
                if cost_delta > 20:  # cost increased >20%
                    sig = f"cost_review:{agent_name}"
                    if sig not in rejected_signatures:
                        seq += 1
                        prescriptions.append({
                            "prescription_id": f"rx-{today}-{seq:03d}",
                            "target_type": "agent",
                            "target_id": agent.get("id"),
                            "target_name": agent_name,
                            "change_type": "model_change_same_tier",
                            "current_value": {"model": agent.get("model", "unknown")},
                            "proposed_value": {"model": "review_needed"},
                            "risk_score": 3,
                            "expected_improvement": f"Potential cost reduction — cost up {cost_delta:.0f}% while success rate is healthy at {success_rate:.0f}%",
                            "rationale": (
                                f"{agent_name} cost increased {cost_delta:.0f}% week-over-week "
                                f"with success_rate at {success_rate:.0f}%. "
                                f"Consider a cheaper model in the same capability tier."
                            ),
                        })

            # --- Description/tag staleness ---
            if card.get("classification") == "REGRESSION" and card.get("error_rate_delta", 0) > 15:
                root_cause = self._infer_root_cause(card, agent)
                if root_cause == "config_stale":
                    sig = f"config_review:{agent_name}"
                    if sig not in rejected_signatures:
                        seq += 1
                        prescriptions.append({
                            "prescription_id": f"rx-{today}-{seq:03d}",
                            "target_type": "agent",
                            "target_id": agent.get("id"),
                            "target_name": agent_name,
                            "change_type": "description_update",
                            "current_value": {"description": agent.get("description", "")[:100]},
                            "proposed_value": {"description": "review_needed"},
                            "risk_score": 1,
                            "expected_improvement": "Align agent description with actual workload to improve routing",
                            "rationale": (
                                f"{agent_name} shows regression with error_rate up {card.get('error_rate_delta', 0):.0f}%. "
                                f"Root cause: config_stale — description may not match current responsibilities."
                            ),
                        })

        # Sort: risk ascending, then by prescription_id
        prescriptions.sort(key=lambda p: (p["risk_score"], p["prescription_id"]))

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
    ) -> Dict[str, Any]:
        """Execute safe changes (risk ≤ 2), queue risky ones as board tasks."""
        logger.info("[HARNESS] Phase 4 APPLY — workspace %s", workspace_id)

        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        executor = PlatformActionExecutor(db=db, workspace_id=workspace_id)

        changelog: Dict[str, List[Dict[str, Any]]] = {
            "applied": [],
            "queued": [],
            "failed": [],
        }

        if not prescriptions:
            logger.info("[HARNESS] APPLY skipped — no prescriptions")
            return changelog

        for rx in prescriptions:
            risk = rx.get("risk_score", 5)
            rx_id = rx["prescription_id"]
            target_name = rx.get("target_name", "unknown")
            change_type = rx.get("change_type", "unknown")

            if risk <= _AUTO_APPLY_MAX_RISK:
                # Auto-apply
                result = await self._auto_apply_prescription(executor, rx)
                if result.get("success"):
                    changelog["applied"].append({
                        "prescription_id": rx_id,
                        "target_name": target_name,
                        "change_type": change_type,
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
                priority = "high" if risk >= _HIGH_PRIORITY_RISK else "medium"
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
                        "tags": ["harness", "org-review", f"risk-{risk}"],
                        "priority": priority,
                    })
                    changelog["queued"].append({
                        "prescription_id": rx_id,
                        "target_name": target_name,
                        "change_type": change_type,
                        "board_task_id": task_result.get("data", {}).get("id") if isinstance(task_result, dict) else None,
                    })
                except Exception as exc:
                    logger.error("[HARNESS] Failed to queue rx %s: %s", rx_id, exc, exc_info=True)
                    changelog["failed"].append({
                        "prescription_id": rx_id,
                        "error": str(exc),
                    })

        # Apply previously approved board tasks (status=done, tag=harness)
        await self._apply_approved_board_tasks(executor, changelog)

        logger.info(
            "[HARNESS] APPLY done — %d applied, %d queued, %d failed",
            len(changelog["applied"]), len(changelog["queued"]), len(changelog["failed"]),
        )
        return changelog

    # ------------------------------------------------------------------
    # Phase 5: BASELINE
    # ------------------------------------------------------------------

    async def _phase_baseline(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        diagnosis: Dict[str, Any],
        prescriptions: List[Dict[str, Any]],
        changelog: Dict[str, Any],
        db: "Session",
    ) -> Dict[str, Any]:
        """Snapshot new org state, publish artifacts, submit audit report."""
        logger.info("[HARNESS] Phase 5 BASELINE — workspace %s", workspace_id)

        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        executor = PlatformActionExecutor(db=db, workspace_id=workspace_id)

        prev_baseline = metrics.get("previous_baseline")
        prev_iteration = prev_baseline.get("iteration", 0) if prev_baseline else 0
        iteration = prev_iteration + 1

        prev_delta = (
            prev_baseline.get("convergence", {}).get("total_delta_magnitude", 0)
            if prev_baseline
            else 0
        )
        current_delta = diagnosis.get("total_delta_magnitude", 0)

        # Determine convergence status
        convergence_status = self._compute_convergence_status(
            iteration, current_delta, prev_delta, prev_baseline
        )

        # Build per-agent snapshot from metrics
        per_agent = self._build_per_agent_snapshot(metrics, diagnosis)

        # Build org-level snapshot
        org_level = diagnosis.get("org_diagnosis", {})

        baseline = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "per_agent": per_agent,
            "org_level": org_level,
            "convergence": {
                "iteration_count": iteration,
                "total_delta_magnitude": current_delta,
                "prev_delta_magnitude": prev_delta,
                "status": convergence_status,
            },
            "applied_changes": changelog.get("applied", []),
            "queued_changes": changelog.get("queued", []),
        }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Write workspace files
        files_to_write = {
            "/harness/baseline_latest.json": json.dumps(baseline, indent=2),
            f"/harness/baselines/{today}.json": json.dumps(baseline, indent=2),
            f"/harness/traces/{today}_trace.json": json.dumps({
                "metrics": self._make_serializable(metrics),
                "diagnosis": self._make_serializable(diagnosis),
            }, indent=2),
            f"/harness/prescriptions/{today}_rx.json": json.dumps(prescriptions, indent=2),
            f"/harness/changelog/{today}.md": self._build_changelog_markdown(
                iteration, diagnosis, prescriptions, changelog, convergence_status
            ),
        }

        for path, content in files_to_write.items():
            try:
                await executor.execute("workspace_write_file", {
                    "path": path,
                    "content": content,
                })
            except Exception as exc:
                logger.error("[HARNESS] Failed to write %s: %s", path, exc)

        # Submit audit report
        try:
            report_content = self._build_changelog_markdown(
                iteration, diagnosis, prescriptions, changelog, convergence_status
            )
            await executor.execute("platform_submit_report", {
                "title": f"HARNESS Weekly Org Review — Run #{iteration}",
                "content": report_content,
                "report_type": "audit",
                "status": "ok" if not diagnosis.get("issues") else "warning",
            })
        except Exception as exc:
            logger.error("[HARNESS] Failed to submit report: %s", exc)

        logger.info("[HARNESS] BASELINE done — iteration %d, status=%s", iteration, convergence_status)
        return baseline

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _read_baseline(self, db: "Session", workspace_id: UUID) -> Optional[Dict[str, Any]]:
        """Read /harness/baseline_latest.json from workspace file storage."""
        try:
            from config import config
            import os

            baseline_path = os.path.join(
                config.WORKSPACE_VOLUME_PATH,
                str(workspace_id),
                "harness",
                "baseline_latest.json",
            )
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    return json.load(f)
        except Exception:
            logger.warning("[HARNESS] Failed to read baseline for %s", workspace_id, exc_info=True)
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

    def _compute_deltas(self, current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        """Compute percentage deltas between current and previous agent metrics."""
        def _pct_delta(curr_val: float, prev_val: float) -> float:
            if prev_val == 0:
                return 0 if curr_val == 0 else 100.0
            return ((curr_val - prev_val) / abs(prev_val)) * 100

        return {
            "success_rate_delta": _pct_delta(
                current.get("success_rate", 0), previous.get("success_rate", 0)
            ),
            "cost_delta": _pct_delta(
                current.get("cost_7d", 0), previous.get("cost_7d", 0)
            ),
            "efficiency_delta": _pct_delta(
                current.get("efficiency_score", 0), previous.get("efficiency_score", 0)
            ),
            "error_rate_delta": _pct_delta(
                current.get("error_rate", 0), previous.get("error_rate", 0)
            ),
        }

    def _infer_root_cause(self, deltas: Dict[str, Any], agent: Dict[str, Any]) -> str:
        """Classify root cause from deltas and agent config."""
        cost_up = deltas.get("cost_delta", 0) > 20
        success_down = deltas.get("success_rate_delta", 0) < -10
        error_up = deltas.get("error_rate_delta", 0) > 15

        if cost_up and not success_down:
            return "cost_inefficient"
        if success_down and error_up:
            return "model_mismatch"
        if error_up:
            return "config_stale"
        if cost_up:
            return "overload"
        return "config_stale"

    def _compute_org_diagnosis(
        self, metrics: Dict[str, Any], baseline: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute org-level aggregate metrics."""
        agents_data = self._extract_agents_list(metrics)
        total_cost = sum(a.get("cost_7d", 0) for a in agents_data)
        avg_success = (
            sum(a.get("success_rate", 0) for a in agents_data) / len(agents_data)
            if agents_data
            else 0
        )
        return {
            "total_cost_7d": total_cost,
            "avg_success_rate": avg_success,
            "active_agents": len(agents_data),
        }

    def _get_rejected_signatures(self, metrics: Dict[str, Any]) -> set:
        """Extract signatures of previously rejected HARNESS prescriptions.

        Signatures use agent_name (matching board task title format) because
        prescriptions in _phase_prescribe also build signatures with agent_name.
        """
        rejected = set()
        tasks_data = metrics.get("prior_harness_tasks")
        if not tasks_data or not isinstance(tasks_data, dict):
            return rejected
        tasks = tasks_data.get("data", tasks_data.get("tasks", []))
        if not isinstance(tasks, list):
            return rejected
        for task in tasks:
            if task.get("status") in ("blocked", "rejected"):
                title = task.get("title", "")
                # Extract signature from title pattern: [HARNESS] {change_type} for {agent_name}
                if "[HARNESS]" in title:
                    parts = title.replace("[HARNESS] ", "").split(" for ")
                    if len(parts) == 2:
                        rejected.add(f"{parts[0].strip()}:{parts[1].strip()}")
        return rejected

    async def _auto_apply_prescription(
        self, executor: "PlatformActionExecutor", rx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single auto-apply prescription."""
        change_type = rx.get("change_type", "")
        target_id = rx.get("target_id")
        proposed = rx.get("proposed_value", {})

        try:
            if change_type == "heartbeat_tune":
                result = await executor.execute("platform_configure_agent_heartbeat", {
                    "agent_id": target_id,
                    "interval_minutes": proposed.get("interval_minutes"),
                })
            elif change_type == "temperature_adjust":
                result = await executor.execute("platform_update_agent", {
                    "agent_id": target_id,
                    "model_config": {"temperature": proposed.get("temperature")},
                })
            elif change_type in ("tag_update", "description_update"):
                update_params = {"agent_id": target_id}
                if change_type == "tag_update":
                    update_params["tags"] = proposed.get("tags", [])
                else:
                    update_params["description"] = proposed.get("description", "")
                result = await executor.execute("platform_update_agent", update_params)
            elif change_type == "model_change_same_tier":
                result = await executor.execute("platform_update_agent", {
                    "agent_id": target_id,
                    "model_config": {"model": proposed.get("model")},
                })
            else:
                return {"success": False, "error": f"Unknown auto-apply change_type: {change_type}"}
            return result if isinstance(result, dict) else {"success": True}
        except Exception as exc:
            logger.error("[HARNESS] Auto-apply failed for %s: %s", rx.get("prescription_id"), exc)
            return {"success": False, "error": str(exc)}

    async def _apply_approved_board_tasks(
        self, executor: "PlatformActionExecutor", changelog: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Apply previously approved (done) HARNESS board tasks."""
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
            for task in tasks:
                logger.warning(
                    "[HARNESS] Approved board task not yet auto-applied (v1 limitation): %s",
                    task.get("title", ""),
                )
                changelog.setdefault("approved_pending", []).append({
                    "task_id": task.get("id"),
                    "title": task.get("title", ""),
                    "note": "Approved but not auto-applied — manual action required (v1)",
                })
        except Exception as exc:
            logger.warning("[HARNESS] Failed to check approved board tasks: %s", exc)

    def _compute_convergence_status(
        self,
        iteration: int,
        current_delta: float,
        prev_delta: float,
        prev_baseline: Optional[Dict[str, Any]],
    ) -> str:
        """Determine convergence state."""
        if iteration < _EXPLORING_MAX_ITERATION:
            return "exploring"

        if current_delta < _CONVERGED_DELTA:
            # Check if we've been below threshold for consecutive runs
            if prev_baseline:
                prev_status = prev_baseline.get("convergence", {}).get("status", "")
                if prev_status in ("converged", "converging") and prev_delta < _CONVERGED_DELTA:
                    return "converged"
            return "converging"

        if prev_delta > 0 and current_delta > prev_delta:
            return "diverging"

        if prev_delta > 0 and current_delta < prev_delta:
            return "converging"

        return "exploring"

    def _build_per_agent_snapshot(
        self, metrics: Dict[str, Any], diagnosis: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Build per-agent baseline snapshot."""
        agents_data = self._extract_agents_list(metrics)
        health_cards = diagnosis.get("health_cards", {})
        snapshot: Dict[str, Dict[str, Any]] = {}

        for agent in agents_data:
            agent_id = str(agent.get("id", ""))
            card = health_cards.get(agent_id, {})
            snapshot[agent_id] = {
                "name": agent.get("name", "unknown"),
                "model": agent.get("model", card.get("model", "unknown")),
                "success_rate": agent.get("success_rate", 0),
                "cost_7d": agent.get("cost_7d", 0),
                "efficiency_score": agent.get("efficiency_score", 0),
                "error_rate": agent.get("error_rate", 0),
                "token_usage_7d": agent.get("token_usage_7d", 0),
                "heartbeat_interval": agent.get("heartbeat_interval", 0),
                "tools_assigned": agent.get("tools_assigned", 0),
                "tools_used_7d": agent.get("tools_used_7d", 0),
            }

        return snapshot

    def _build_changelog_markdown(
        self,
        iteration: int,
        diagnosis: Dict[str, Any],
        prescriptions: List[Dict[str, Any]],
        changelog: Dict[str, Any],
        convergence_status: str,
    ) -> str:
        """Build human-readable changelog markdown."""
        lines = [
            f"# HARNESS Weekly Org Review — Run #{iteration}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Convergence:** {convergence_status}",
            f"**Delta Magnitude:** {diagnosis.get('total_delta_magnitude', 0):.1f}",
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

        if not issues and not applied and not queued:
            lines.append("No changes needed — organization is stable.")
            lines.append("")

        return "\n".join(lines)

    def _make_serializable(self, obj: Any) -> Any:
        """Make an object JSON-serializable by converting non-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_harness_service: Optional[HarnessService] = None


def get_harness_service() -> HarnessService:
    global _harness_service
    if _harness_service is None:
        _harness_service = HarnessService()
    return _harness_service
