"""
Workspace GitHub Integration
==============================
PRD-66 Phase 2: GitHub repo browsing + cloning via Composio

  GET  /api/workspaces/{workspace_id}/github/repos  — list user's GitHub repos
  POST /api/workspaces/{workspace_id}/github/clone  — clone a repo into workspace
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
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


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CloneRequest(BaseModel):
    repo_url: str = Field(..., description="Git clone URL (HTTPS)")
    branch: Optional[str] = Field(None, description="Branch to clone (default: repo default)")


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
    except ImportError:
        raise HTTPException(status_code=501, detail="Composio SDK not installed")

    client = get_composio_client()
    entity_id = str(ctx.workspace_id)

    result = client.execute_action(
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
    """Clone a GitHub repo into the workspace via task submission."""
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

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)

    # Build a git_clone step
    step = {
        "action": "git_clone",
        "repo": body.repo_url,
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

    redis = await runner._get_redis()

    # Write initial status
    status_key = f"workspace:task:{task_id}:status"
    await redis.hset(status_key, mapping={
        "status": "queued",
        "workspace_id": workspace_id,
        "submitted_at": now.isoformat(),
        "priority": "normal",
        "task_type": "background_job",
    })
    await redis.expire(status_key, 7200)

    # Track active tasks
    ws_active_key = f"workspace:ws:{workspace_id}:active_tasks"
    await redis.sadd(ws_active_key, task_id)

    # Push to normal queue
    await redis.lpush("workspace:tasks:normal", json.dumps(payload))

    # DB record
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

    logger.info(
        "Clone task %s submitted (workspace=%s, repo=%s)",
        task_id[:8], workspace_id[:8], body.repo_url,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "events_url": f"/api/tasks/{task_id}/events",
    }
