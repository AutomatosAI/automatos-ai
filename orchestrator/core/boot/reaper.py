"""Boot-time orphaned-run reaper (PRD-142 Wave 1 · WS-C · W1-S6).

On restart, an in-flight row whose background ``asyncio`` executor died with the
old process is stranded forever — nothing remains to move it to a terminal
state. ``reap_orphaned_runs`` runs once per deploy (under the boot leader lock)
and sweeps the three durable-launch surfaces:

  - **board task** stuck ``in_progress`` → ``failed`` with the reason, through
    the one completion writer (``finalize_board_task_run``): the owner is told
    (task_failed) and the report written, as for any failed run (F175's rule);
  - **wizard profile** stuck ``scraping``/``scanning`` → ``failed`` +
    ``quality_findings`` (the wizard's own failure convention);
  - **workflow execution** stuck ``pending``/``running`` → ``failed`` +
    ``error_message`` + ``completed_at``;
  - **social post** stuck ``rendering`` (PRD-251 S1.1c) → ``failed``, the reason
    in its ``review_log`` (the post lifecycle's own render failure), so it can
    be edited and rendered again.

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

import asyncio
import inspect
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Union

from config import config
from core.models.business_profiles import BusinessProfile
from core.models.core import BoardTask, RecipeExecution, WorkflowExecution
from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)

_ORPHAN_REASON = "orphaned_on_restart"
# Boot waits on each orphan's close: its task_failed notice can reach Telegram or
# Slack. The status is committed before the notice, so a slow channel only cuts
# the notice short (review MEDIUM on adc84365e). The bound cuts awaits, not the
# row lock the close takes first; nothing else holds an orphan's row at boot (the
# old process is gone and the dispatcher starts after the reaper).
ORPHAN_CLOSE_SECONDS = 15


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


async def _reap_board_tasks(db, cutoff: datetime, now: datetime) -> int:
    """An orphaned ticket is a failed run (F175's rule, night 6 #1078/#1091 closed
    'done'): the one completion writer closes it 'failed' with the reason, tells
    the owner and writes its report. A ticket no longer in progress is left alone."""
    from api.board_tasks import finalize_board_task_run

    rows = db.query(BoardTask).filter(BoardTask.status == "in_progress").all()
    stale = [(r.id, str(r.workspace_id), r.assigned_agent_id)
             for r in rows if _is_stale(r.started_at or r.updated_at, cutoff)]
    closed = []
    for task_id, workspace_id, agent_id in stale:
        try:
            if await asyncio.wait_for(finalize_board_task_run(
                db, task_id=task_id, workspace_id=workspace_id, agent_id=agent_id,
                exec_result={"status": "error", "error": f"{_ORPHAN_REASON}: executor lost on restart"},
            ), timeout=ORPHAN_CLOSE_SECONDS):
                closed.append(task_id)
        except Exception:  # noqa: BLE001 — one ticket never stops the rest of the sweep
            db.rollback()
            logger.exception("Boot reaper: closing orphaned board task %s did not finish", task_id)
            if db.query(BoardTask.status).filter(BoardTask.id == task_id).scalar() == "failed":
                closed.append(task_id)  # closed; only its notice or report was cut short
    if closed:
        record_error(
            subsystem="board",
            operation="boot_reap",
            error=OrphanedRunError(f"reaped {len(closed)} orphaned board task(s)"),
            extra={"reaped_ids": closed, "reason": _ORPHAN_REASON},
        )
    return len(closed)


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


def _reap_social_renders(db, cutoff: datetime, now: datetime) -> int:
    """Sweep ``social_posts`` stuck in ``rendering`` — PRD-251 S1.1c.

    The render runs as a background task of the process that started it; a
    restart kills it and nothing else ends the render. ``updated_at`` is bumped
    when the post enters ``rendering``, and a live render never outlasts
    ``SOCIALS_RENDER_MAX_WAIT_SECONDS``, which stays under the stale cutoff, so
    a reaped post has no task left to finish it. The failure goes through the
    post lifecycle (``fail_render``), which writes the reason to ``review_log``.
    """
    from core.models.socials import SocialPost
    from modules.socials import service as socials

    rows = db.query(SocialPost).filter(SocialPost.status == socials.RENDERING).all()
    stale = [r for r in rows if r.status == socials.RENDERING and _is_stale(r.updated_at, cutoff)]
    for r in stale:
        socials.fail_render(
            r,
            _ORPHAN_REASON,
            "The render was lost when the server restarted. Render again.",
            report={"code": _ORPHAN_REASON},
        )
    if stale:
        record_error(
            subsystem="socials",
            operation="boot_reap",
            error=OrphanedRunError(f"reaped {len(stale)} orphaned social render(s)"),
            extra={"reaped_ids": [str(r.id) for r in stale], "reason": _ORPHAN_REASON},
        )
    return len(stale)


async def _run_surface(
    db,
    cutoff: datetime,
    now: datetime,
    subsystem: str,
    fn: Callable[[object, datetime, datetime], Union[int, Awaitable[int]]],
) -> int:
    """Run one surface reaper in isolation — a failure is recorded, not raised.

    One broken surface must not stop the others from being swept.
    """
    try:
        reaped = fn(db, cutoff, now)
        return await reaped if inspect.isawaitable(reaped) else reaped
    except Exception as exc:  # noqa: BLE001 — surface isolation by design
        db.rollback()  # a failed statement would poison the session for the next surface
        logger.exception("Boot reaper: %s surface failed", subsystem)
        record_error(subsystem=subsystem, operation="boot_reap", error=exc)
        return 0


async def reap_orphaned_runs(db, *, now: Optional[datetime] = None) -> int:
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
    reaped += await _run_surface(db, cutoff, now, "board", _reap_board_tasks)
    reaped += await _run_surface(db, cutoff, now, "wizard", _reap_business_profiles)
    reaped += await _run_surface(db, cutoff, now, "workflow", _reap_workflow_executions)
    # PRD-142 Wave 3 · W3-S12: Playbook restart-durability — port the Mission
    # durability primitive (boot-time terminal-transition sweep) onto the
    # RecipeExecution log so an in-flight playbook cannot silently die when
    # the process restarts (§H DoD #3 + §A E1).
    reaped += await _run_surface(db, cutoff, now, "playbook", _reap_recipe_executions)
    # PRD-251 S1.1c: a post whose render task died with the old process.
    reaped += await _run_surface(db, cutoff, now, "socials", _reap_social_renders)

    if reaped:
        try:
            db.commit()
        except Exception:  # noqa: BLE001 — don't let a commit failure crash boot
            logger.exception("Boot reaper: commit failed; rolling back")
            db.rollback()
            return 0
        logger.warning("Boot reaper: marked %d orphaned run(s) terminal", reaped)

    return reaped
