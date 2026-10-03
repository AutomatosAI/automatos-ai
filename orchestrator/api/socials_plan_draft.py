"""
Plan with Auto (PRD-251B, 3 Oct 2026 pass)
==========================================

``POST /api/socials/plans/draft``: a plan drafted from what the person says, NOT saved
(``modules/socials/plan_draft.py``). The body is their words, their timezone and the sources
research may read (their knowledge, their website, their Deliverables). The answer is the
plan as the Plan page opens it, the suggestions it heard as content bank topics, and a
warning for everything Auto's answer had that a plan cannot carry. The Plan page saves it
with ``POST /plans``, adds the topics and starts research.

Drawn from the caller's workspace only: its connected channels and what each can post now
(``modules/socials/capabilities.py``), its social templates' names and formats, and its
brand kit's voice and style. One call to the workspace's model, tracked as
``socials_plan_draft``: 504 when it does not answer in time, 502 when its answer cannot be
read twice or the model cannot be reached.

``api/socials_plans.py`` includes this router, so it takes the Socials router's prefix and
gate. A plain ``def`` (F105): the database reads run in the threadpool, and the model call
on the event loop from there, as the composer's does.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Tuple
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api import socials_compose
from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.socials import plan_draft
from modules.socials.capabilities import social_channels

logger = logging.getLogger(__name__)

router = APIRouter()

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
DEFAULT_TIMEZONE = "UTC"
TIMEZONE_MAX_CHARS = 64


class DraftSourcesIn(BaseModel):
    """What research may read for the plan: the person's choice."""

    model_config = ConfigDict(extra="forbid")

    knowledge: bool = True
    website: bool = True
    deliverables: bool = True


class PlanDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request: str = Field(..., min_length=1, max_length=plan_draft.REQUEST_MAX_CHARS)
    timezone: str = Field(DEFAULT_TIMEZONE, max_length=TIMEZONE_MAX_CHARS)
    sources: DraftSourcesIn = Field(default_factory=DraftSourcesIn)


def today_in(timezone_name: str) -> Tuple[date, str]:
    """Today in the person's timezone, and the timezone; an unknown one is UTC."""
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone, timezone_name = ZoneInfo(DEFAULT_TIMEZONE), DEFAULT_TIMEZONE
    return datetime.now(zone).date(), timezone_name


def draft_context(db: Session, workspace_id: UUID, body: PlanDraftRequest) -> plan_draft.DraftContext:
    """Everything the model is given, from the caller's workspace only."""
    today, timezone_name = today_in(body.timezone)
    channels = [
        {"toolkit": c.toolkit, "label": c.label, "kinds": [k.kind for k in c.post_kinds if k.available]}
        for c in social_channels(db, workspace_id)
    ]
    templates = [{"name": t["name"], "format": t["format"]} for t in socials_compose.social_templates(db, workspace_id, None)]
    return plan_draft.DraftContext(
        request=body.request, today=today, timezone=timezone_name, channels=channels, templates=templates,
        voice=socials_compose.brand_voice(db, workspace_id), style=socials_compose.brand_style_text(db, workspace_id),
    )


@router.post("/plans/draft", dependencies=[CAN_CREATE])
def draft_social_plan(
    body: PlanDraftRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """A plan drafted from ``body.request`` (not saved): ``{plan, topics, warnings}``."""
    context = draft_context(db, ctx.workspace_id, body)
    sources = plan_draft.DraftSources(**body.sources.model_dump())
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    try:
        return anyio.from_thread.run(plan_draft.draft, context, sources, plan_draft.llm_factory(ctx.workspace_id), timeout)
    except plan_draft.DraftTimedOut as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except plan_draft.DraftFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[Socials] the plan draft failed for workspace %s", ctx.workspace_id)
        raise HTTPException(status_code=502, detail="Auto could not be reached. Try again.") from exc
