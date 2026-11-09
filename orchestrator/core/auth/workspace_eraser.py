"""Who may finish erasing a closed workspace (#1442, CodeRabbit on #1443).

The GDPR erase closes the workspace (soft-deletes it) before it erases anything, and
answers 503 when a step failed, so the caller asks again and the retry finishes it.
Every request lane refuses a closed workspace (``hybrid._workspace_exists``), so on
hosted the owner's retry was a 400 and the erasure could never finish.

This is the one exception, on the erase route only: the workspace's owner
(``workspaces.owner_id``), signed in with Clerk, reaches their own closed workspace.
Anyone else falls through to the usual lanes, which refuse a closed workspace as they
always have, so a closed workspace's existence is told to no one but its owner.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from core.auth.bearer_registry import bearer_token
from core.auth.clerk import get_clerk_auth
from core.auth.dependencies import RequestContext, clerk_user
from core.auth.hybrid import _get_workspace_id_from_request, get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db

logger = logging.getLogger(__name__)


def closed_workspace_owner_context(request: Request, db) -> Optional[RequestContext]:
    """The owner's context in the closed workspace the request names, or None: the
    workspace is open, unnamed, or the Clerk user is not its owner (the usual lanes
    decide). 401 for a token that does not verify."""
    bearer = bearer_token(request)
    workspace_id = _get_workspace_id_from_request(request)
    if not bearer or not workspace_id:
        return None
    row = db.execute(
        text("SELECT w.deleted_at IS NOT NULL AS closed, u.clerk_user_id AS owner FROM workspaces w "
             "LEFT JOIN users u ON u.id = w.owner_id WHERE w.id = :id"),
        {"id": str(workspace_id)},
    ).fetchone()
    if row is None or not row.closed:
        return None

    clerk = get_clerk_auth()
    claims = clerk.verify_token(bearer)
    if not claims:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    info = clerk.extract_user_info(claims)
    if not row.owner or info.get("clerk_user_id") != row.owner:
        return None
    logger.info("The owner of closed workspace %s reached it to finish its erasure", workspace_id)
    return RequestContext(workspace_id=workspace_id, user=clerk_user(info, claims), auth_type="clerk")


async def require_workspace_eraser(request: Request, db=Depends(get_db)) -> RequestContext:
    """The erase route's caller: a closed workspace's owner finishing its erasure, else a
    workspace admin, exactly as ``require_workspace_admin`` decides it."""
    owner = await run_in_threadpool(closed_workspace_owner_context, request, db)
    if owner is not None:
        return owner
    ctx = await get_request_context_hybrid(request)
    return await run_in_threadpool(require_workspace_admin, ctx, db)
