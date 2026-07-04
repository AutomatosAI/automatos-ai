"""PRD-185 S2 — per-lane telemetry canary.

S1 repaired a type-poisoned telemetry write that had left ``tool_execution_logs``
with **zero** organic rows for ~two months. Nothing alarmed, because the platform
had no signal for "organic rows/day = 0". This module is that guardrail: it counts
production (``telemetry_source == 'production'``) ``ToolExecutionLog`` rows per lane
(``app_name``) over a window and reports LOUD when the platform — or a lane — has
gone silent, so S1 can never silently regress again.

It reuses the W10 telemetry source-of-truth (``ToolExecutionLog`` + the
``telemetry_source`` discriminator, PRD-180) — it does NOT stand up a parallel
metrics stack. The decision core (``evaluate_telemetry_canary``) is pure so it
unit-tests with plain dicts; ``run_telemetry_canary`` is the thin DB + log wrapper
the scheduled job and the boot-probe both call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import config
from core.models.composio_cache import ToolExecutionLog

logger = logging.getLogger(__name__)

# The organic discriminator: only real production traffic counts. Eval/replay
# rows must never mask a silent production lane. (Mirrors slo_metrics.py.)
_ORGANIC_SOURCE = "production"


def evaluate_telemetry_canary(
    per_lane_counts: Dict[str, int],
    *,
    window_seconds: int,
    min_rows: int,
) -> Dict[str, Any]:
    """Pure decision core: given per-lane organic row counts, decide the alarm.

    ``alert`` is True when the platform total is at or below ``min_rows`` (default
    0 → alarm only on a totally silent platform; raise it to catch partial
    silence). Per-lane counts ride along for diagnosis. No I/O — takes counts,
    returns a verdict dict.
    """
    per_lane = {str(k): int(v) for k, v in (per_lane_counts or {}).items()}
    total = sum(per_lane.values())
    return {
        "alert": total <= min_rows,
        "organic_rows": total,
        "per_lane": per_lane,
        "window_seconds": window_seconds,
        "min_rows": min_rows,
    }


def _count_organic_rows_by_lane(db: Session, *, since: datetime) -> Dict[str, int]:
    """Count production ``ToolExecutionLog`` rows since ``since``, grouped by lane
    (``app_name``). Thin DB read — the decision lives in
    ``evaluate_telemetry_canary``."""
    rows = (
        db.query(ToolExecutionLog.app_name, func.count().label("n"))
        .filter(
            ToolExecutionLog.executed_at >= since,
            ToolExecutionLog.telemetry_source == _ORGANIC_SOURCE,
        )
        .group_by(ToolExecutionLog.app_name)
        .all()
    )
    return {(app or "unknown"): int(n) for app, n in rows}


def run_telemetry_canary(
    db: Session,
    *,
    window_seconds: Optional[int] = None,
    min_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Query organic rows over the window, evaluate, and log the verdict.

    On alarm logs at WARNING (the loud signal the platform lacked for two
    months); otherwise a terse INFO. Returns the verdict dict so callers (the
    boot-probe, the scheduled job, a future health tile) can act on it. NEVER
    raises — a canary that crashes the scheduler is worse than a silent lane.
    """
    win = int(window_seconds if window_seconds is not None else config.TELEMETRY_CANARY_WINDOW_SECONDS)
    floor_rows = int(min_rows if min_rows is not None else config.TELEMETRY_CANARY_MIN_ROWS)
    try:
        since = datetime.utcnow() - timedelta(seconds=win)
        per_lane = _count_organic_rows_by_lane(db, since=since)
    except Exception:
        logger.warning(
            "[TelemetryCanary] query failed — cannot judge organic-row flow",
            exc_info=True,
        )
        return {
            "alert": False,
            "organic_rows": None,
            "per_lane": {},
            "window_seconds": win,
            "min_rows": floor_rows,
            "error": True,
        }

    verdict = evaluate_telemetry_canary(per_lane, window_seconds=win, min_rows=floor_rows)
    hours = round(win / 3600.0, 1)
    if verdict["alert"]:
        logger.warning(
            "[TelemetryCanary] ALARM — %d organic tool-execution rows in the last "
            "%sh (<= %d). The learning plane may be starving (S1 regression? telemetry "
            "write broken?). per_lane=%s",
            verdict["organic_rows"], hours, floor_rows, verdict["per_lane"],
        )
    else:
        logger.info(
            "[TelemetryCanary] ok — %d organic rows in the last %sh across %d lane(s)",
            verdict["organic_rows"], hours, len(verdict["per_lane"]),
        )
    return verdict


async def telemetry_canary_tick() -> Dict[str, Any]:
    """Scheduled-job + boot-probe entrypoint: open a session, run the canary,
    close it. Registered on the shared scheduler by ``HeartbeatService.start()``
    and fired once at boot as the boot-probe. Honours the enable flag so it can
    be switched off without unscheduling."""
    if not config.TELEMETRY_CANARY_ENABLED:
        return {"alert": False, "skipped": True}

    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return run_telemetry_canary(db)
    finally:
        db.close()
