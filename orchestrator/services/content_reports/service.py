"""File a content report: one audit row and the owner's in-app notice (PRD-266 US-016a).

``file_report`` runs these steps, in this order, and stops at the first that fails:

1. the reporter: the request's person (``resolve_user_pk``); an API key or an SDK
   key has none and cannot report (``ReportNeedsAPerson``);
2. the per-reporter rate limit (``check_rate_limit``; over it the limiter raises
   its own 429, before anything is written);
3. the target, found in the caller's workspace (``targets``), else
   ``ReportTargetNotFound``;
4. a duplicate check: the same reporter's earlier report of the same target
   answers with that report's id and nothing else happens (best effort: two
   requests in the same instant can both pass it, as ``audit_logs`` has no unique
   key for it and this PRD adds no migration; the rate limit bounds the burst);
5. ONE audit row through ``AuditService.log``, which commits it; its ``None``
   return is ``ReportNotRecorded``;
6. the owner's in-app notice, committed after it. It names the target kind and the
   reason only, never the excerpt or the note. No owner means no notice (a
   notice with no user would reach every member); a failed notice is logged and
   the report still stands.

The moderation post (SaaS only) is not sent here: the route hands the result to
``moderation`` after the response (US-016b).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional
from uuid import UUID

from config import config
from core.auth.principal import resolve_user_pk
from core.models.workspaces import Workspace
from core.security import rate_limiter
from core.services.notification_dispatcher import NotificationDispatcher
from core.workspaces.audit import AuditLog, AuditService
from core.workspaces.owner import workspace_owner
from services.content_reports import AUDIT_ACTION
from services.content_reports.targets import canonical_target_id, resolve_target

logger = logging.getLogger(__name__)

# The limiter's operation name, and each reporter's (max_requests, window_seconds):
# 20 an hour unless RATE_LIMIT_CONTENT_REPORT_MAX / _WINDOW_SECONDS say otherwise.
RATE_LIMIT_OPERATION = "content_report"
CONTENT_REPORT_RATE_LIMIT: tuple[int, int] = config.rate_limit_for(RATE_LIMIT_OPERATION, 20, 3600)

REPORTER_ACTOR_TYPE = "user"

# The owner's notice (K9). The bell opens Command Center -> Governance for this link type.
NOTICE_TITLE = "Content reported"
NOTICE_LINK_TYPE = "content_report"
NOTICE_STATUS = "warning"
TARGET_KIND_LABELS: dict[str, str] = {"message": "An Auto message", "deliverable": "A Deliverable"}


class ReportNeedsAPerson(Exception):
    """The caller is no person (an API key, an SDK key): only a person can report."""


class ReportTargetNotFound(Exception):
    """The target is not in the caller's workspace, or is not the reporter's to report."""


class ReportNotRecorded(Exception):
    """The audit row could not be written; nothing else happened."""


@dataclass(frozen=True)
class FiledReport:
    """A report as filed: what the route answers and what the moderation post needs."""

    report_id: int
    duplicate: bool
    workspace_id: str
    target_type: str
    target_id: str
    reason: str
    note: Optional[str]
    excerpt: str


def notice_message(target_type: str, reason: str) -> str:
    """The owner's notice text: the target kind and the reason, nothing of the content."""
    kind = TARGET_KIND_LABELS.get(target_type, "Content")
    return f"{kind} was reported as {reason}. Open Governance to review it."


async def _within_rate_limit(workspace_id: UUID, reporter_pk: int) -> None:
    """Count this report against the reporter's bucket; the limiter raises 429 over it."""
    max_requests, window_seconds = CONTENT_REPORT_RATE_LIMIT
    await rate_limiter.check_rate_limit(
        str(workspace_id),
        RATE_LIMIT_OPERATION,
        max_requests=max_requests,
        window_seconds=window_seconds,
        subject_id=str(reporter_pk),
    )


def _earlier_report_id(
    db: Any, *, workspace_id: UUID, reporter_pk: int, target_type: str, target_id: str
) -> Optional[int]:
    """The id of this reporter's earlier report of this target, or ``None``."""
    row = (
        db.query(AuditLog.id)
        .filter(
            AuditLog.workspace_id == workspace_id,
            AuditLog.user_id == reporter_pk,
            AuditLog.action == AUDIT_ACTION,
            AuditLog.resource_type == target_type,
            AuditLog.resource_id == target_id,
        )
        .order_by(AuditLog.id)
        .first()
    )
    return int(row.id) if row is not None else None


def _record(db: Any, *, reporter_pk: int, report: FiledReport) -> int:
    """Write the report's one audit row (``AuditService.log`` commits it); its id."""
    entry = AuditService(db).log(
        workspace_id=report.workspace_id,
        user_id=reporter_pk,
        actor_type=REPORTER_ACTOR_TYPE,
        action=AUDIT_ACTION,
        resource_type=report.target_type,
        resource_id=report.target_id,
        details={"reason": report.reason, "note": report.note, "excerpt": report.excerpt},
    )
    if entry is None:
        raise ReportNotRecorded()
    return int(entry.id)


async def _notify_owner(db: Any, report: FiledReport) -> None:
    """Raise the owner's in-app notice and commit it; a failure leaves the report standing."""
    try:
        owner = workspace_owner(db, db.get(Workspace, UUID(report.workspace_id)))
        if owner is None:
            logger.warning(
                "[content_reports] report %s: workspace %s has no owner to notify",
                report.report_id, report.workspace_id,
            )
            return
        await NotificationDispatcher(db, report.workspace_id).dispatch(
            event_type=AUDIT_ACTION,
            title=NOTICE_TITLE,
            message=notice_message(report.target_type, report.reason),
            link_type=NOTICE_LINK_TYPE,
            link_id=str(report.report_id),
            user_id=owner.id,
            status=NOTICE_STATUS,
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "[content_reports] report %s: the owner's notice failed (the report stands)",
            report.report_id,
        )


async def file_report(
    db: Any, ctx: Any, *, target_type: str, target_id: str, reason: str, note: Optional[str]
) -> FiledReport:
    """File one report of ``target_type``/``target_id`` by the request's person.

    Raises ``ReportNeedsAPerson``, ``ReportTargetNotFound`` or ``ReportNotRecorded``;
    the rate limiter raises its own ``HTTPException(429)``. A duplicate answers the
    earlier report (``duplicate=True``) and writes and sends nothing.
    """
    reporter_pk = resolve_user_pk(db, ctx)
    if reporter_pk is None:
        raise ReportNeedsAPerson()
    workspace_id = getattr(ctx, "workspace_id", None)
    await _within_rate_limit(workspace_id, reporter_pk)

    canonical_id = canonical_target_id(target_id)
    excerpt = None
    if workspace_id is not None and canonical_id is not None:
        excerpt = await resolve_target(
            db, workspace_id=workspace_id, reporter_pk=reporter_pk,
            target_type=target_type, target_id=canonical_id,
        )
    if excerpt is None:
        raise ReportTargetNotFound()

    report = FiledReport(
        report_id=0, duplicate=False, workspace_id=str(workspace_id), target_type=target_type,
        target_id=canonical_id, reason=reason, note=note, excerpt=excerpt,
    )
    earlier = _earlier_report_id(
        db, workspace_id=workspace_id, reporter_pk=reporter_pk,
        target_type=target_type, target_id=canonical_id,
    )
    if earlier is not None:
        return _with_id(report, earlier, duplicate=True)

    filed = _with_id(report, _record(db, reporter_pk=reporter_pk, report=report), duplicate=False)
    await _notify_owner(db, filed)
    return filed


def _with_id(report: FiledReport, report_id: int, *, duplicate: bool) -> FiledReport:
    """A copy of ``report`` carrying its id."""
    return replace(report, report_id=report_id, duplicate=duplicate)


__all__ = [
    "CONTENT_REPORT_RATE_LIMIT",
    "FiledReport",
    "NOTICE_LINK_TYPE",
    "NOTICE_TITLE",
    "RATE_LIMIT_OPERATION",
    "ReportNeedsAPerson",
    "ReportNotRecorded",
    "ReportTargetNotFound",
    "file_report",
    "notice_message",
]
