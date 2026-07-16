"""PRD-197 S4 — aggregate substrate_metric_events into per-seam health.

Read side of the substrate telemetry seam: for each retrieval seam
(documents / memory / field) over a window, the search count, error rate,
empty rate, and p95 latency, rolled into one green/degraded/down/unknown
status per seam. Served by ``GET /api/analytics/substrate-health``
(workspace-admin, own-workspace only) and rendered as a Command Center
"is-it-working" cell.

Honest by construction: a seam with no rows reports ``unknown`` ("awaiting
searches"), never a fabricated green.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from core.models.substrate_metrics import SubstrateMetricEvent
from core.observability.substrate_metrics import (
    SEAM_DOCUMENTS,
    SEAM_FIELD,
    SEAM_MEMORY,
    STATUS_EMPTY,
    STATUS_ERROR,
)

SEAMS = (SEAM_DOCUMENTS, SEAM_MEMORY, SEAM_FIELD)

# Status thresholds. The tile's job is catching dark planes: a seam that
# errors on >1/4 of searches is down; elevated errors or a sustained
# all-empty pattern (the "silently returns nothing" failure mode this
# review kept finding) is degraded. Legitimate zero-result queries exist,
# so all-empty only degrades once there are enough searches to mean it.
DOWN_ERROR_RATE = 0.25
DEGRADED_ERROR_RATE = 0.05
ALL_EMPTY_MIN_SEARCHES = 10
ALL_EMPTY_RATE = 0.98


def seam_status(searches: int, error_rate: float, empty_rate: float) -> str:
    """Pure roll-up of one seam's window into the tile vocabulary."""
    if searches == 0:
        return "unknown"
    if error_rate > DOWN_ERROR_RATE:
        return "down"
    if error_rate > DEGRADED_ERROR_RATE:
        return "degraded"
    if searches >= ALL_EMPTY_MIN_SEARCHES and empty_rate >= ALL_EMPTY_RATE:
        return "degraded"
    return "green"


def compute_substrate_health(
    db: Session,
    workspace_id: Any,
    window_seconds: int = 86400,
) -> Dict[str, Any]:
    """Per-seam health for one workspace over the window."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=window_seconds)

    rows = (
        db.query(
            SubstrateMetricEvent.seam.label("seam"),
            func.count().label("searches"),
            func.sum(
                case((SubstrateMetricEvent.status == STATUS_ERROR, 1), else_=0)
            ).label("errors"),
            func.sum(
                case((SubstrateMetricEvent.status == STATUS_EMPTY, 1), else_=0)
            ).label("empties"),
            func.avg(SubstrateMetricEvent.latency_ms).label("avg_latency_ms"),
            func.percentile_cont(0.95)
            .within_group(SubstrateMetricEvent.latency_ms.asc())
            .label("p95_latency_ms"),
        )
        .filter(
            SubstrateMetricEvent.created_at >= cutoff,
            SubstrateMetricEvent.workspace_id == str(workspace_id),
        )
        .group_by(SubstrateMetricEvent.seam)
        .all()
    )
    by_seam = {row.seam: row for row in rows}

    seams: List[Dict[str, Any]] = []
    for seam in SEAMS:
        row = by_seam.get(seam)
        searches = int(row.searches) if row else 0
        errors = int(row.errors or 0) if row else 0
        empties = int(row.empties or 0) if row else 0
        error_rate = (errors / searches) if searches else 0.0
        empty_rate = (empties / searches) if searches else 0.0
        seams.append(
            {
                "seam": seam,
                "searches": searches,
                "error_rate": round(error_rate, 4),
                "empty_rate": round(empty_rate, 4),
                "avg_latency_ms": round(float(row.avg_latency_ms), 1)
                if row and row.avg_latency_ms is not None
                else None,
                "p95_latency_ms": round(float(row.p95_latency_ms), 1)
                if row and row.p95_latency_ms is not None
                else None,
                "status": seam_status(searches, error_rate, empty_rate),
            }
        )

    return {
        "generated_at": now.isoformat(),
        "window_seconds": window_seconds,
        "seams": seams,
    }


def prune_substrate_metrics(db: Session, retention_days: int) -> int:
    """Delete rows past retention. Called by the memory-jobs sweep — the
    heartbeat_results lesson: telemetry tables never grow unbounded."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = (
        db.query(SubstrateMetricEvent)
        .filter(SubstrateMetricEvent.created_at < cutoff)
        .delete(synchronize_session=False)
    )
    db.commit()
    return int(deleted or 0)
