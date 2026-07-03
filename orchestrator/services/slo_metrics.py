"""PRD-180 S5 — three tracked SLOs computed from existing telemetry.

Observability is only honest if there are real, measured objectives. This module
defines **three** concrete SLIs, each with a target and a measurement window, and
computes them from telemetry the platform already writes — it does NOT stand up a
parallel metrics stack (it reuses ``ToolExecutionLog``, ``BoardTask`` and the
existing ``StatisticalAnalysis`` percentile helper).

The three SLIs:

1. **tool_call_success_rate** — of all production tool executions in the window,
   the fraction that succeeded (``ToolExecutionLog.status == 'success'``).
   *Target: ≥ 99.0%.* Measures whether Auto's actions actually land.
2. **board_dispatch_latency_p95_seconds** — the 95th-percentile lag from a board
   task being created to a worker starting it (``started_at − created_at``).
   *Target: ≤ 5.0s.* Measures whether the dispatch spine is keeping up.
3. **board_event_freshness_seconds** — age of the most recent board-task mutation
   (``now − max(updated_at)``); with the LISTEN/NOTIFY SSE (S1) every mutation is
   pushed, so a stale source means a stale stream. *Target: ≤ 30.0s* while the
   board is active (returns ``None`` value when there is simply no activity).

Each computation returns an immutable dict: ``{sli, description, value, unit,
target, target_comparator, window_seconds, sample_size, meets_target}``. Values
are honest — ``None`` when there is no data to measure (never a fabricated number).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import config
from core.math.statistical_analysis import StatisticalAnalysis
from core.models.composio_cache import ToolExecutionLog
from core.models.core import BoardTask

# ── SLO definitions (targets + windows are config-as-data, not magic numbers) ──

# Default measurement window for rate/latency SLIs (config-driven, not hardcoded).
DEFAULT_WINDOW_SECONDS = config.SLO_DEFAULT_WINDOW_SECONDS

# Targets: (value_target, comparator) — comparator is how ``value`` is judged.
SUCCESS_RATE_TARGET = 99.0          # percent, value must be >=
DISPATCH_LATENCY_P95_TARGET = 5.0   # seconds, value must be <=
EVENT_FRESHNESS_TARGET = 30.0       # seconds, value must be <=


def _meets(value: Optional[float], target: float, comparator: str) -> Optional[bool]:
    """Judge ``value`` against ``target``. ``None`` value → ``None`` (no data)."""
    if value is None:
        return None
    if comparator == ">=":
        return value >= target
    if comparator == "<=":
        return value <= target
    raise ValueError(f"unknown comparator: {comparator}")


def tool_call_success_rate(
    db: Session,
    *,
    workspace_id,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """SLI 1 — production tool-call success rate (%) over the window.

    Only ``telemetry_source == 'production'`` rows count, so eval/replay traffic
    never skews the objective. ``value`` is ``None`` when no tool ran (no data to
    judge), never 0% — an honest empty measurement.
    """
    since = datetime.utcnow() - timedelta(seconds=window_seconds)
    base = db.query(ToolExecutionLog).filter(
        ToolExecutionLog.workspace_id == workspace_id,
        ToolExecutionLog.executed_at >= since,
        ToolExecutionLog.telemetry_source == "production",
    )
    total = base.count()
    value: Optional[float] = None
    if total > 0:
        succeeded = base.filter(ToolExecutionLog.status == "success").count()
        value = round(succeeded / total * 100.0, 2)

    return {
        "sli": "tool_call_success_rate",
        "description": "Production tool executions that succeeded",
        "value": value,
        "unit": "percent",
        "target": SUCCESS_RATE_TARGET,
        "target_comparator": ">=",
        "window_seconds": window_seconds,
        "sample_size": total,
        "meets_target": _meets(value, SUCCESS_RATE_TARGET, ">="),
    }


def board_dispatch_latency_p95_seconds(
    db: Session,
    *,
    workspace_id,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """SLI 2 — p95 board-task dispatch latency (created → started), in seconds.

    Reuses ``StatisticalAnalysis.calculate_percentile``. Only tasks that actually
    started in the window contribute (``started_at`` not null); ``value`` is
    ``None`` when nothing dispatched.
    """
    since = _aware_utc_now() - timedelta(seconds=window_seconds)
    rows = (
        db.query(BoardTask.created_at, BoardTask.started_at)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.started_at.isnot(None),
            BoardTask.created_at >= since,
        )
        .all()
    )
    latencies: List[float] = [
        (started - created).total_seconds()
        for created, started in rows
        if created is not None and started is not None and started >= created
    ]

    value: Optional[float] = None
    if latencies:
        value = round(StatisticalAnalysis.calculate_percentile(latencies, 95), 2)

    return {
        "sli": "board_dispatch_latency_p95_seconds",
        "description": "95th-percentile lag from board-task creation to worker start",
        "value": value,
        "unit": "seconds",
        "target": DISPATCH_LATENCY_P95_TARGET,
        "target_comparator": "<=",
        "window_seconds": window_seconds,
        "sample_size": len(latencies),
        "meets_target": _meets(value, DISPATCH_LATENCY_P95_TARGET, "<="),
    }


def board_event_freshness_seconds(
    db: Session,
    *,
    workspace_id,
) -> Dict[str, Any]:
    """SLI 3 — age (s) of the most recent board-task mutation.

    Proxy for SSE event-delivery freshness: with the LISTEN/NOTIFY SSE (S1) every
    board mutation is pushed, so the age of the newest ``updated_at`` is how fresh
    the event source is. ``value`` is ``None`` when the board has no tasks at all
    (nothing to be fresh about) — not a fabricated zero.
    """
    latest: Optional[datetime] = (
        db.query(func.max(BoardTask.updated_at))
        .filter(BoardTask.workspace_id == workspace_id)
        .scalar()
    )
    value: Optional[float] = None
    if latest is not None:
        value = round((_aware_utc_now() - _as_aware(latest)).total_seconds(), 2)
        value = max(value, 0.0)

    return {
        "sli": "board_event_freshness_seconds",
        "description": "Age of the most recent board-task mutation (SSE source freshness)",
        "value": value,
        "unit": "seconds",
        "target": EVENT_FRESHNESS_TARGET,
        "target_comparator": "<=",
        "window_seconds": None,
        "sample_size": 1 if latest is not None else 0,
        "meets_target": _meets(value, EVENT_FRESHNESS_TARGET, "<="),
    }


def compute_slos(
    db: Session,
    *,
    workspace_id,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """Compute all three SLIs for a workspace and wrap them for the dashboard.

    Returns ``{generated_at, window_seconds, slos: [<sli dict>, ...]}`` — a stable
    envelope the frontend can render as a dashboard.
    """
    slos = [
        tool_call_success_rate(db, workspace_id=workspace_id, window_seconds=window_seconds),
        board_dispatch_latency_p95_seconds(
            db, workspace_id=workspace_id, window_seconds=window_seconds
        ),
        board_event_freshness_seconds(db, workspace_id=workspace_id),
    ]
    return {
        "generated_at": _aware_utc_now().isoformat(),
        "window_seconds": window_seconds,
        "slos": slos,
    }


# ── time helpers (BoardTask is tz-aware; ToolExecutionLog.executed_at is naive) ─

def _aware_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC so tz-aware arithmetic never raises."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
