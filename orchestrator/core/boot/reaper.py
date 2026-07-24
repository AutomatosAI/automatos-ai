"""Boot-time orphaned-run reaper (PRD-142 Wave 1 · WS-C · W1-S6).

On restart, an in-flight row whose background ``asyncio`` executor died with the
old process is stranded forever — nothing remains to move it to a terminal
state. ``reap_orphaned_runs`` runs once per deploy (under the boot leader lock)
and sweeps the three durable-launch surfaces:

  - **board task** stuck ``in_progress`` → ``done`` + ``error_message``
    (the board has no 'failed' Kanban column; this mirrors its own failure path);
  - **wizard profile** stuck ``scraping``/``scanning`` → ``failed`` +
    ``quality_findings`` (the wizard's own failure convention);
  - **workflow execution** stuck ``pending``/``running`` → ``failed`` +
    ``error_message`` + ``completed_at``.

A row is reaped only once it has been in-flight longer than
``BOOT_REAPER_STALE_MINUTES`` — long enough that no legitimately running job
(the wizard scrape, ~10–20 min, is the slowest) could still own it. Staleness is
filtered in Python: the row volume is tiny pre-launch and the cutoff logic stays
unit-testable. Each reaped surface fires
``record_error(subsystem=<surface>, operation="boot_reap")`` so the sweep
surfaces on the ERRORS-by-subsystem dashboard tile (the WS-A sink).

Missions are deliberately EXCLUDED: ``OrchestrationRun`` rows are owned by the
coordinator's reconcile loop, which already resumes/cleans RUNNING runs. Reaping
them here would race that state machine.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from config import config
from core.models.business_profiles import BusinessProfile
from core.models.core import BoardTask, RecipeExecution, WorkflowExecution
from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)

_ORPHAN_REASON = "orphaned_on_restart"


class OrphanedRunError(RuntimeError):
    """Synthetic error recorded when the reaper marks an orphaned run terminal.

    The reaper is proactive (it isn't catching a live exception), but
    ``record_error`` takes an ``Exception`` — this gives the ``error_events`` row
    a meaningful ``error_type`` instead of a generic ``RuntimeError``.
    """


def _coerce_aware(ts: datetime) -> datetime:
    """Treat a tz-naive timestamp as UTC so it compares against an aware cutoff.

    ``WorkflowExecution.started_at`` is tz-NAIVE; ``BoardTask`` and
    ``BusinessProfile`` timestamps are tz-AWARE. Normalising to aware-UTC lets a
    single cutoff serve all three surfaces.
    """
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _is_stale(ts: Optional[datetime], cutoff: datetime) -> bool:
    """True iff ``ts`` is older than ``cutoff``.

    A ``None`` timestamp can't be proven stale, so it is left alone — a later
    deploy will catch the row once it carries a real timestamp.
    """
    if ts is None:
        return False
    return _coerce_aware(ts) < cutoff


def _reap_board_tasks(db, cutoff: datetime, now: datetime) -> int:
    rows = db.query(BoardTask).filter(BoardTask.status == "in_progress").all()
    stale = [r for r in rows if _is_stale(r.started_at or r.updated_at, cutoff)]
    for r in stale:
        # The board has no 'failed' column; its own failure path marks 'done'
        # + error_message, so we mirror that exactly.
        r.status = "done"
        r.error_message = f"{_ORPHAN_REASON}: executor lost on restart"
        r.completed_at = now
    if stale:
        record_error(
            subsystem="board",
            operation="boot_reap",
            error=OrphanedRunError(f"reaped {len(stale)} orphaned board task(s)"),
            extra={"reaped_ids": [r.id for r in stale], "reason": _ORPHAN_REASON},
        )
    return len(stale)


def _reap_business_profiles(db, cutoff: datetime, now: datetime) -> int:
    rows = (
        db.query(BusinessProfile)
        .filter(BusinessProfile.status.in_(["scraping", "scanning"]))
        .all()
    )
    # updated_at is bumped on each status transition, so it marks when the row
    # entered the in-flight state.
    stale = [r for r in rows if _is_stale(r.updated_at, cutoff)]
    for r in stale:
        r.status = "failed"  # wizard's own failure convention
        findings = dict(r.quality_findings or {})
        errors = list(findings.get("errors") or [])
        errors.append(f"{_ORPHAN_REASON}: scrape executor lost on restart")
        findings["errors"] = errors
        r.quality_findings = findings
    if stale:
        record_error(
            subsystem="wizard",
            operation="boot_reap",
            error=OrphanedRunError(f"reaped {len(stale)} orphaned wizard profile(s)"),
            extra={"reaped_ids": [str(r.id) for r in stale], "reason": _ORPHAN_REASON},
        )
    return len(stale)


def _reap_workflow_executions(db, cutoff: datetime, now: datetime) -> int:
    rows = (
        db.query(WorkflowExecution)
        .filter(WorkflowExecution.status.in_(["pending", "running"]))
        .all()
    )
    stale = [r for r in rows if _is_stale(r.started_at, cutoff)]
    # completed_at is a tz-NAIVE column here, unlike the aware board column.
    naive_now = now.replace(tzinfo=None) if now.tzinfo is not None else now
    for r in stale:
        r.status = "failed"
        r.error_message = f"{_ORPHAN_REASON}: executor lost on restart"
        r.completed_at = naive_now
    if stale:
        record_error(
            subsystem="workflow",
            operation="boot_reap",
            error=OrphanedRunError(
                f"reaped {len(stale)} orphaned workflow execution(s)"
            ),
            extra={"reaped_ids": [r.id for r in stale], "reason": _ORPHAN_REASON},
        )
    return len(stale)


def _reap_recipe_executions(db, cutoff: datetime, now: datetime) -> int:
    """Sweep stuck ``RecipeExecution`` rows (the Playbook execution log) —
    PRD-142 Wave 3 W3-S12.

    Mirrors the workflow surface: ``pending``/``running`` past the staleness
    window is an orphan (process crashed mid-execution), and the row would
    otherwise stay ``running`` forever. ``RecipeExecution.started_at`` is
    tz-NAIVE (column default ``func.now()``) and ``completed_at`` is also
    tz-NAIVE — same naive_now handling as the workflow surface keeps the
    timestamp comparable to existing rows.

    The subsystem tag ``"playbook"`` (not ``"recipe"``) matches the
    canonical noun (CLAUDE.md §10 / GUARDRAILS C1) so the WS-A
    ERRORS-by-subsystem tile groups under the right name.
    """
    rows = (
        db.query(RecipeExecution)
        .filter(RecipeExecution.status.in_(["pending", "running"]))
        .all()
    )
    # Defensive Python-side status guard — belt-and-braces on top of the SQL
    # filter so a future refactor that drops the IN clause can NOT silently
    # re-mark an already-terminal row (§H DoD #6 — no double-writes to a
    # row whose status is the final word).
    in_flight = [r for r in rows if r.status in ("pending", "running")]
    stale = [r for r in in_flight if _is_stale(r.started_at, cutoff)]
    naive_now = now.replace(tzinfo=None) if now.tzinfo is not None else now
    for r in stale:
        r.status = "failed"
        r.error_message = f"{_ORPHAN_REASON}: playbook executor lost on restart"
        r.completed_at = naive_now
    if stale:
        record_error(
            subsystem="playbook",
            operation="boot_reap",
            error=OrphanedRunError(
                f"reaped {len(stale)} orphaned playbook execution(s)"
            ),
            extra={
                "reaped_execution_ids": [r.execution_id for r in stale],
                "reason": _ORPHAN_REASON,
            },
        )
    return len(stale)


def _run_surface(
    db,
    cutoff: datetime,
    now: datetime,
    subsystem: str,
    fn: Callable[[object, datetime, datetime], int],
) -> int:
    """Run one surface reaper in isolation — a failure is recorded, not raised.

    One broken surface must not stop the others from being swept.
    """
    try:
        return fn(db, cutoff, now)
    except Exception as exc:  # noqa: BLE001 — surface isolation by design
        logger.exception("Boot reaper: %s surface failed", subsystem)
        record_error(subsystem=subsystem, operation="boot_reap", error=exc)
        return 0


def reap_orphaned_runs(db, *, now: Optional[datetime] = None) -> int:
    """Sweep orphaned in-flight rows across board / wizard / workflow surfaces.

    Returns the number of rows marked terminal. Mutations are committed once at
    the end (tiny row volume pre-launch). Never raises out of a surface — each is
    isolated so one failure can't abort the rest.
    """
    if not config.BOOT_REAPER_ENABLED:
        logger.info("Boot reaper disabled (BOOT_REAPER_ENABLED=false) — skipping")
        return 0

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=config.BOOT_REAPER_STALE_MINUTES)

    reaped = 0
    reaped += _run_surface(db, cutoff, now, "board", _reap_board_tasks)
    reaped += _run_surface(db, cutoff, now, "wizard", _reap_business_profiles)
    reaped += _run_surface(db, cutoff, now, "workflow", _reap_workflow_executions)
    # PRD-142 Wave 3 · W3-S12: Playbook restart-durability — port the Mission
    # durability primitive (boot-time terminal-transition sweep) onto the
    # RecipeExecution log so an in-flight playbook cannot silently die when
    # the process restarts (§H DoD #3 + §A E1).
    reaped += _run_surface(db, cutoff, now, "playbook", _reap_recipe_executions)

    if reaped:
        try:
            db.commit()
        except Exception:  # noqa: BLE001 — don't let a commit failure crash boot
            logger.exception("Boot reaper: commit failed; rolling back")
            db.rollback()
            return 0
        logger.warning("Boot reaper: marked %d orphaned run(s) terminal", reaped)

    return reaped
