"""
Workspace GitHub Integration
==============================
PRD-66 Phase 2: GitHub repo browsing + cloning via Composio

  GET  /api/workspaces/{workspace_id}/github/repos  — list user's GitHub repos
  POST /api/workspaces/{workspace_id}/github/clone  — clone a repo into workspace
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/github",
    tags=["workspace-github"],
)

# Allowed hosts for clone URLs
_ALLOWED_CLONE_HOSTS = {"github.com", "gitlab.com", "bitbucket.org"}

# Safe branch name pattern (git ref chars, no .., @{, leading/trailing whitespace)
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CloneRequest(BaseModel):
    repo_url: str = Field(..., description="Git clone URL (HTTPS)")
    branch: Optional[str] = Field(None, description="Branch to clone (default: repo default)")

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme != "https":
            raise ValueError("Only HTTPS clone URLs are allowed")
        if parsed.hostname not in _ALLOWED_CLONE_HOSTS:
            raise ValueError(f"Host not allowed: {parsed.hostname}")
        if parsed.username or parsed.password:
            raise ValueError("Clone URL must not contain embedded credentials")
        return v

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if ".." in v or "@{" in v or not _BRANCH_RE.match(v):
            raise ValueError("Invalid branch name")
        return v


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/github/repos
# ---------------------------------------------------------------------------
@router.get("/repos")
async def list_github_repos(
    workspace_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List GitHub repos accessible to the user via Composio."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    try:
        from core.composio.client import get_composio_client
    except ImportError as err:
        raise HTTPException(status_code=501, detail="Composio SDK not installed") from err

    client = get_composio_client()
    entity_id = str(ctx.workspace_id)

    result = await asyncio.to_thread(
        client.execute_action,
        action="GITHUB_LIST_REPOS_FOR_AUTHENTICATED_USER",
        params={"per_page": per_page, "page": page},
        entity_id=entity_id,
    )

    if not result.get("success"):
        error_msg = result.get("error", "Failed to list GitHub repos")
        raise HTTPException(status_code=502, detail=f"GitHub API error: {error_msg}")

    # Extract repo list from Composio response
    raw_data = result.get("data", {})

    # Composio wraps results — navigate to the actual list
    repos_raw = raw_data
    if isinstance(raw_data, dict):
        repos_raw = raw_data.get("data", raw_data.get("response_data", raw_data))

    # If still a dict (single-level Composio wrapper), try deeper
    if isinstance(repos_raw, dict):
        repos_raw = repos_raw.get("data", repos_raw.get("items", []))

    if not isinstance(repos_raw, list):
        repos_raw = []

    repos = []
    for repo in repos_raw:
        if not isinstance(repo, dict):
            continue
        repos.append({
            "name": repo.get("name", ""),
            "full_name": repo.get("full_name", ""),
            "url": repo.get("clone_url") or repo.get("html_url", ""),
            "description": repo.get("description") or "",
            "default_branch": repo.get("default_branch", "main"),
            "private": repo.get("private", False),
            "language": repo.get("language"),
            "updated_at": repo.get("updated_at"),
        })

    return {"repos": repos, "page": page, "per_page": per_page}


# ---------------------------------------------------------------------------
# POST /api/workspaces/{workspace_id}/github/clone
# ---------------------------------------------------------------------------
@router.post("/clone")
async def clone_github_repo(
    workspace_id: str,
    body: CloneRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Clone a GitHub repo into the workspace via task submission.

    Attempts to retrieve the GitHub OAuth token from Composio to
    authenticate the clone (required for private repos). Falls back
    to unauthenticated HTTPS clone for public repos.
    """
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    from core.task_runner import get_task_runner

    runner = get_task_runner()
    if runner.backend_name != "queued":
        raise HTTPException(
            status_code=400,
            detail="Repo cloning requires TASK_RUNNER_BACKEND=queued. "
                   f"Current backend: {runner.backend_name}",
        )

    # Try to get GitHub token from Composio for authenticated clone
    clone_url = body.repo_url
    try:
        from core.composio.client import get_composio_client
        client = get_composio_client()
        token = await asyncio.to_thread(
            client.get_app_access_token, str(ctx.workspace_id), "GITHUB"
        )
        if token:
            # Inject token into HTTPS URL: https://x-access-token:{token}@github.com/...
            if clone_url.startswith("https://github.com"):
                clone_url = clone_url.replace(
                    "https://github.com",
                    f"https://x-access-token:{token}@github.com",
                )
                logger.info("Injected GitHub token for authenticated clone")
    except Exception as e:
        logger.warning("Could not retrieve GitHub token (public clone only): %s", e)

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)

    # Build a git_clone step
    step = {
        "action": "git_clone",
        "repo": clone_url,
        "description": f"Clone {body.repo_url}",
    }
    if body.branch:
        step["branch"] = body.branch

    payload = {
        "task_id": task_id,
        "task_type": "background_job",
        "workspace_id": workspace_id,
        "agent_id": None,
        "priority": "normal",
        "timeout_seconds": 300,
        "steps": [step],
        "created_at": now.isoformat(),
    }

    # Insert DB record first (before enqueue) for atomicity
    from sqlalchemy import text
    db.execute(text("""
        INSERT INTO task_executions (
            id, workspace_id, task_type, status, priority,
            runner_backend, configuration, submitted_at
        ) VALUES (
            :id, :workspace_id, :task_type, :status, :priority,
            :runner_backend, :configuration, :submitted_at
        )
    """), {
        "id": task_id,
        "workspace_id": workspace_id,
        "task_type": "background_job",
        "status": "queued",
        "priority": "normal",
        "runner_backend": "queued",
        "configuration": json.dumps({"steps": [step]}),
        "submitted_at": now,
    })
    db.commit()

    # Enqueue to Redis (after DB commit succeeds)
    redis = await runner._get_redis()
    try:
        status_key = f"workspace:task:{task_id}:status"
        await redis.hset(status_key, mapping={
            "status": "queued",
            "workspace_id": workspace_id,
            "submitted_at": now.isoformat(),
            "priority": "normal",
            "task_type": "background_job",
        })
        await redis.expire(status_key, 7200)

        ws_active_key = f"workspace:ws:{workspace_id}:active_tasks"
        await redis.sadd(ws_active_key, task_id)

        await redis.lpush("workspace:tasks:normal", json.dumps(payload))
    except Exception as enqueue_err:
        # Redis enqueue failed after DB commit — mark DB row as failed
        logger.error("Redis enqueue failed for task %s: %s", task_id[:8], enqueue_err)
        db.execute(text(
            "UPDATE task_executions SET status = 'failed', error_message = :err WHERE id = :id"
        ), {"id": task_id, "err": f"Enqueue failed: {enqueue_err}"})
        db.commit()
        raise HTTPException(status_code=503, detail="Failed to enqueue task to worker") from enqueue_err

    logger.info(
        "Clone task %s submitted (workspace=%s, repo=%s)",
        task_id[:8], workspace_id[:8], body.repo_url,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "events_url": f"/api/tasks/{task_id}/events",
    }
