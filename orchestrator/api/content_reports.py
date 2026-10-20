"""Content reports: ``POST /api/v1/reports`` (PRD-266 §4.8, US-016a, D15, FR-15).

A member reports an Auto message or a Deliverable of their workspace. The report is
one audit row (action ``content_reported``) and an in-app notice to the workspace's
owner (``services.content_reports.service``); the owner reads reports in the
existing audit view (Command Center -> Governance).

This router adds no public route. The one route is authenticated and gated by
``require_workspace_permission("members:read")``, which every workspace role holds,
so any member may report, a viewer included. A caller with no person (an API key,
an SDK key) is 403. A target the reporter cannot see is 404, never 403.

Not to be confused with ``api/reports.py``: PRD-76's agent reports at ``/api/reports``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.content_reports import (
    NOTE_MAX_CHARS,
    REPORT_REASONS,
    REPORT_TARGET_TYPES,
    TARGET_ID_MAX_CHARS,
)
from services.content_reports.service import (
    ReportNeedsAPerson,
    ReportNotRecorded,
    ReportTargetNotFound,
    file_report,
)

CONTENT_REPORTS_PREFIX = "/api/v1/reports"
# Every workspace role holds it (modules/policy/roles.py): any member may report.
REPORT_PERMISSION = "members:read"

STATUS_RECEIVED = "received"
STATUS_ALREADY_RECEIVED = "already_received"
TARGET_NOT_FOUND_CODE = "report_target_not_found"
TARGET_NOT_FOUND_MESSAGE = "That message or Deliverable was not found in this workspace."
NEEDS_A_PERSON_DETAIL = "Only a signed-in member can report content."
NOT_RECORDED_DETAIL = "The report could not be saved. Try again in a minute."

router = APIRouter(prefix=CONTENT_REPORTS_PREFIX, tags=["Content reports"])


class ReportRequest(BaseModel):
    """What is reported and why; any other field is refused (422)."""

    model_config = ConfigDict(extra="forbid")

    target_type: Literal[REPORT_TARGET_TYPES]  # type: ignore[valid-type]
    target_id: str = Field(min_length=1, max_length=TARGET_ID_MAX_CHARS)
    reason: Literal[REPORT_REASONS]  # type: ignore[valid-type]
    note: Optional[str] = Field(default=None, max_length=NOTE_MAX_CHARS)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_workspace_permission(REPORT_PERMISSION))],
)
async def report_content(
    body: ReportRequest,
    response: Response,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """File a report: 201 for a new one, 200 for one this member already made.

    ``async def``: it awaits the rate limiter and the notice dispatcher.
    """
    try:
        filed = await file_report(
            db, ctx,
            target_type=body.target_type, target_id=body.target_id,
            reason=body.reason, note=body.note,
        )
    except ReportNeedsAPerson:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NEEDS_A_PERSON_DETAIL) from None
    except ReportTargetNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": TARGET_NOT_FOUND_CODE, "message": TARGET_NOT_FOUND_MESSAGE},
        ) from None
    except ReportNotRecorded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=NOT_RECORDED_DETAIL) from None
    if filed.duplicate:
        response.status_code = status.HTTP_200_OK
        return {"report_id": filed.report_id, "status": STATUS_ALREADY_RECEIVED}
    return {"report_id": filed.report_id, "status": STATUS_RECEIVED}


__all__ = ["CONTENT_REPORTS_PREFIX", "REPORT_PERMISSION", "ReportRequest", "router"]
