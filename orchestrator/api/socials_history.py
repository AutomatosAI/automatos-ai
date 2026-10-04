"""
Socials history and Posted (PRD-251C C5, C7; US-C103, US-C408)
==============================================================

``GET /api/socials/history?days&limit``: what the workspace posted, newest first: every post
that went out, is approved or scheduled, or waits for a person, across every plan and the
posts people made by hand (``modules/socials/history.py``). ``days`` and ``limit`` default to
``SOCIALS_HISTORY_DAYS`` and ``SOCIALS_HISTORY_LIMIT``; more than ``MAX_DAYS`` or
``MAX_LIMIT`` is a 422. Scoped to the caller's workspace. A plain ``def`` (F105).

``GET /api/socials/posted?plan_id&channel&format&limit``: Posted, what went out, newest
first, each with its receipts, its numbers and its topic (``modules/socials/posted.py``),
filtered by plan, channel (a toolkit) and format.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from modules.socials import history, posted

router = APIRouter()


@router.get("/history")
def list_social_history(
    days: Optional[int] = Query(None, ge=1, le=history.MAX_DAYS, description="How many days back"),
    limit: Optional[int] = Query(None, ge=1, le=history.MAX_LIMIT, description="How many posts at most"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The workspace's history, newest first: ``{posts, total}``."""
    posts = history.history(db, ctx.workspace_id, days=days, limit=limit)
    return {"posts": posts, "total": len(posts)}


@router.get("/posted")
def list_social_posted(
    plan_id: Optional[UUID] = Query(None, description="Only this plan's posts"),
    channel: Optional[str] = Query(None, max_length=64, pattern=r"^[a-z][a-z0-9_]*$", description="Only posts that went out here"),
    format: Optional[str] = Query(None, max_length=32, description="Only posts of this format"),  # noqa: A002 — the query's name
    limit: Optional[int] = Query(None, ge=1, le=posted.MAX_POSTED, description="How many posts at most"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Posted (PRD-251C US-C408): what went out, newest first: ``{posts, total}``."""
    posts = posted.posted(db, ctx.workspace_id, plan_id=plan_id, channel=channel, fmt=format, limit=limit)
    return {"posts": posts, "total": len(posts)}
