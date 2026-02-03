"""
Bug Reports API — Pilot Helper Widget
======================================

Creates Jira tickets via Composio for user-submitted bug reports.
Uses the admin pattern (agent_id=0, skip_validation=True) from CloudSyncService.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bug-reports", tags=["Bug Reports"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BugReportContext(BaseModel):
    url: Optional[str] = None
    page: Optional[str] = None
    user_agent: Optional[str] = None
    viewport: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    console_errors: Optional[List[str]] = Field(default_factory=list)
    timestamp: Optional[str] = None


class BugReportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    severity: str = Field(default="Minor")  # Critical | Major | Minor
    category: str = Field(default="Other")  # UI Bug | Data Issue | Performance | Other
    screenshot_base64: Optional[str] = None
    context: BugReportContext = Field(default_factory=BugReportContext)


class BugReportResponse(BaseModel):
    success: bool
    jira_key: Optional[str] = None
    jira_url: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_TO_PRIORITY = {
    "Critical": "Highest",
    "Major": "High",
    "Minor": "Medium",
}


def _build_jira_description(req: BugReportRequest) -> str:
    """Format a rich Jira description from the bug report + auto-captured context."""
    ctx = req.context
    lines = [
        req.description,
        "",
        "----",
        "*Auto-captured context*",
        "",
        f"| *Field* | *Value* |",
        f"| URL | {ctx.url or 'N/A'} |",
        f"| Page | {ctx.page or 'N/A'} |",
        f"| User Agent | {ctx.user_agent or 'N/A'} |",
        f"| Viewport | {ctx.viewport or 'N/A'} |",
        f"| User | {ctx.user_name or 'N/A'} ({ctx.user_email or 'N/A'}) |",
        f"| Timestamp | {ctx.timestamp or datetime.now(timezone.utc).isoformat()} |",
    ]

    if ctx.console_errors:
        lines.append("")
        lines.append("*Console errors (last 20):*")
        lines.append("{noformat}")
        for err in ctx.console_errors[:20]:
            lines.append(err[:500])
        lines.append("{noformat}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=BugReportResponse)
async def create_bug_report(
    req: BugReportRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Submit a bug report that creates a Jira ticket via Composio.

    Requires a connected Jira integration in the workspace.
    """
    # Feature flag check
    if not config.JIRA_BUG_REPORTS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Bug reporting is currently disabled.",
        )

    # Build Jira fields (Composio JIRA_CREATE_ISSUE parameter names)
    project_key = config.JIRA_PROJECT_KEY
    summary = f"[{req.category}] {req.title}"
    description = _build_jira_description(req)

    jira_params: Dict[str, Any] = {
        "project_key": project_key,
        "summary": summary,
        "description": description,
        "issue_type": "Bug",
    }

    try:
        from core.composio.tool_executor import ComposioToolExecutor

        executor = ComposioToolExecutor(db)
        result = await executor.execute(
            action="JIRA_CREATE_ISSUE",
            params=jira_params,
            agent_id=0,
            workspace_id=ctx.workspace_id,
            app_name="JIRA",
            skip_validation=True,
        )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error from Composio")
            logger.error(f"Jira ticket creation failed: {error_msg}")
            return BugReportResponse(
                success=False,
                message=f"Failed to create Jira ticket: {error_msg}",
            )

        # Extract key and URL from Composio response
        data = result.get("data", {})
        jira_key = data.get("key") or data.get("id")
        jira_url = data.get("browser_url") or data.get("self")

        # Build a browser-friendly URL if we only got the REST self link
        if jira_url and "/rest/api/" in jira_url:
            base = jira_url.split("/rest/api/")[0]
            jira_url = f"{base}/browse/{jira_key}" if jira_key else jira_url
        elif jira_key and not jira_url:
            jira_url = None

        logger.info(f"Bug report created: {jira_key} for workspace {ctx.workspace_id}")

        return BugReportResponse(
            success=True,
            jira_key=jira_key,
            jira_url=jira_url,
            message=f"Bug report created: {jira_key}",
        )

    except ImportError:
        logger.error("Composio module not available")
        raise HTTPException(
            status_code=503,
            detail="Bug reporting service is not available (Composio not installed).",
        )
    except Exception as exc:
        logger.exception(f"Bug report submission failed: {exc}")
        return BugReportResponse(
            success=False,
            message="An unexpected error occurred while creating the bug report. Please try again.",
        )
