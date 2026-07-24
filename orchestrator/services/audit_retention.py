"""PRD-196 S5 (P2-15, governance J.9 / appendix §2.3) — audit-log retention.

``audit_logs`` grows unboundedly — nothing ever deleted a row — while EU-AI-Act
Art.12 mandates a >= 6-month floor and GDPR data-minimisation forbids forever.
This is the retention policy: a config-driven window with a HARD 180-day legal
floor enforced at read (a configured value below it is clamped up, never honoured),
a bounded batched delete over the S3 composite index, and one summary audit row
per affected workspace so the deletion is itself Art.12-traceable.

Hard-delete only — no soft-delete column, no archive table (the requirement is
deletion; CLAUDE.md §4). The S7 export path already produces a bundle if an
export-before-delete is ever wanted.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# EU-AI-Act Art.12: >= 6 months. The hard legal floor a config can never dip
# under — enforced at read, not just documented.
AUDIT_RETENTION_FLOOR_DAYS = 180

# Delete in bounded batches so a large backlog never locks the table.
_DEFAULT_BATCH = 5000


def effective_retention_days(configured: Optional[int] = None) -> int:
    """The retention window in days, floor-enforced.

    A configured value below the 180-day Art.12 floor is CLAMPED UP to 180 — the
    floor binds no matter what the config says. Reads ``config.AUDIT_RETENTION_DAYS``
    when no explicit value is given; fail-safe to the 365 default on a bad read.
    """
    if configured is None:
        try:
            from config import config

            configured = int(config.AUDIT_RETENTION_DAYS)
        except Exception:
            logger.warning("[audit_retention] config read failed — 365-day default", exc_info=True)
            configured = 365
    return max(int(configured), AUDIT_RETENTION_FLOOR_DAYS)


def compute_cutoff(now: datetime, retention_days: int) -> datetime:
    """Rows with ``created_at < cutoff`` are eligible for deletion (pure).

    The floor is re-applied here too, so no caller can compute a cutoff younger
    than the Art.12 minimum even by passing a small ``retention_days``.
    """
    days = max(int(retention_days), AUDIT_RETENTION_FLOOR_DAYS)
    return now - timedelta(days=days)


def sweep_expired_audit_logs(
    db: Any,
    *,
    now: Optional[datetime] = None,
    batch_size: int = _DEFAULT_BATCH,
) -> Dict[str, Any]:
    """Delete ``audit_logs`` rows older than the retention cutoff, in bounded
    batches over the ``ix_audit_logs_workspace_created`` index.

    Writes ONE ``audit:retention_sweep`` summary row per affected workspace
    (system actor) — ``audit_logs.workspace_id`` is NOT NULL, so the honest,
    per-tenant-traceable record is one row per workspace whose rows were swept
    (rather than a single unattributable global row). Returns counts.
    """
    from sqlalchemy import func

    from core.workspaces.audit import AuditLog, AuditService

    now = now or datetime.now(timezone.utc)
    retention_days = effective_retention_days()
    cutoff = compute_cutoff(now, retention_days)

    # Per-workspace pre-count for the attributable summary rows.
    counts: Dict[Any, int] = dict(
        db.query(AuditLog.workspace_id, func.count(AuditLog.id))
        .filter(AuditLog.created_at < cutoff)
        .group_by(AuditLog.workspace_id)
        .all()
    )

    total_deleted = 0
    while True:
        ids = [
            row[0]
            for row in db.query(AuditLog.id)
            .filter(AuditLog.created_at < cutoff)
            .limit(batch_size)
            .all()
        ]
        if not ids:
            break
        deleted = (
            db.query(AuditLog)
            .filter(AuditLog.id.in_(ids))
            .delete(synchronize_session=False)
        )
        db.commit()
        total_deleted += int(deleted or 0)
        if not deleted or deleted < batch_size:
            break

    audit = AuditService(db)
    for workspace_id, cnt in counts.items():
        if not cnt:
            continue
        try:
            audit.log(
                workspace_id=str(workspace_id),
                user_id=None,
                actor_type="system",
                action="audit:retention_sweep",
                resource_type="audit_logs",
                details={
                    "cutoff": cutoff.isoformat(),
                    "rows_deleted": int(cnt),
                    "retention_days": retention_days,
                },
            )
        except Exception:
            logger.warning(
                "[audit_retention] summary row failed for ws=%s", workspace_id, exc_info=True
            )

    logger.info(
        "[audit_retention] swept %d row(s) older than %s across %d workspace(s)",
        total_deleted, cutoff.isoformat(), len(counts),
    )
    return {
        "cutoff": cutoff.isoformat(),
        "retention_days": retention_days,
        "total_deleted": total_deleted,
        "workspaces_affected": len(counts),
    }
