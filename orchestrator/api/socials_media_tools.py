"""
Socials AI tools and AI-made visuals (PRD-251B B10; US-B304, US-B305)
=====================================================================

* ``GET /api/socials/media-tools``: the Brand kit's AI tools section: each media toolkit
  with its state (connected, to connect in Composio, unavailable and why, built in), the
  choices per media type, the workspace's defaults, its monthly and per-post caps and this
  month's media spend (``modules/socials/media_tools.py``).
* ``PUT /api/socials/media-tools``: new defaults (each one the workspace is offered now),
  a monthly cap and a per-post cap, in dollars; owners and admins (``workspace:manage``).
* ``POST /api/socials/posts/{post_id}/ai-options``: four AI options for one of the post's
  template image slots, made in the background (202); the slot says ``making``, then the
  options or why none were made (``modules/socials/ai_options.py``).
* ``PUT /api/socials/posts/{post_id}/ai-options/{slot}``: the option picked becomes the
  slot's file, as a render's would. A render setting: an approval stands.

``api/socials_templates.py`` includes this router, so it takes the Socials router's prefix
and gate. Plain ``def`` routes (F105); the options run on the event loop from there.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import SessionLocal, get_db
from core.models.workspaces import Workspace
from api import socials_brand
from core.social_templates import IMAGE_SLOT, SocialTemplateError, validate_social_blocks
from modules.socials import ai_options, media_caps, media_tools, render, service
from modules.socials.capabilities import media_capabilities
from modules.socials.media_store import MediaStore
from modules.socials.recipes.footage import route_for
from modules.socials.settings import KEY_MEDIA_MONTHLY_CAP, KEY_MEDIA_POST_CAP, WORKSPACE_SOCIALS_SETTINGS_KEY

logger = logging.getLogger(__name__)
router = APIRouter()

CAN_MANAGE = Depends(require_workspace_permission("workspace:manage"))
CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
MAX_CAP_USD = 100_000.0
PROMPT_MAX_CHARS = 1000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaToolsChange(_Strict):
    defaults: Optional[Dict[str, str]] = None
    monthly_cap_usd: Optional[float] = Field(None, ge=0, le=MAX_CAP_USD)
    per_post_cap_usd: Optional[float] = Field(None, ge=0, le=MAX_CAP_USD)


class OptionsRequest(_Strict):
    slot: str = Field(..., min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=PROMPT_MAX_CHARS)


class PickRequest(_Strict):
    name: str = Field(..., min_length=1, max_length=200)


def _posts_api() -> Any:
    from api import socials

    return socials


def _view(db: Session, workspace: Workspace) -> Dict[str, Any]:
    caps = media_capabilities(db, workspace.id)
    spend = media_caps.media_spend(db, workspace)
    return {
        "toolkits": media_tools.toolkit_rows(caps),
        "offered": media_tools.offered(caps),
        "defaults": media_tools.defaults_of(workspace.settings),
        "caps": {"monthly_usd": spend.monthly_cap_usd, "per_post_usd": spend.post_cap_usd, "problem": spend.problem},
        "spend": {"month_usd": spend.month_usd, "period_end": spend.period_end.isoformat()},
    }


@router.get("/media-tools")
def get_media_tools(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    return _view(db, _posts_api()._workspace(db, ctx))


@router.put("/media-tools", dependencies=[CAN_MANAGE])
def update_media_tools(body: MediaToolsChange, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """New defaults and caps (the module docstring); 422 for a default the workspace is not offered."""
    workspace = _posts_api()._workspace(db, ctx)
    settings = dict(workspace.settings or {})
    if body.defaults is not None:
        choices = media_tools.offered(media_capabilities(db, workspace.id))
        try:
            settings[media_tools.MEDIA_TOOLS_KEY] = media_tools.validate_defaults(body.defaults, choices, media_tools.defaults_of(settings))
        except service.InvalidPost as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    socials = dict(settings.get(WORKSPACE_SOCIALS_SETTINGS_KEY) or {})
    if "monthly_cap_usd" in body.model_fields_set:
        socials[KEY_MEDIA_MONTHLY_CAP] = body.monthly_cap_usd
    if "per_post_cap_usd" in body.model_fields_set:
        socials[KEY_MEDIA_POST_CAP] = body.per_post_cap_usd
    workspace.settings = {**settings, WORKSPACE_SOCIALS_SETTINGS_KEY: socials}
    db.commit()
    return _view(db, workspace)


def _image_route(db: Session, workspace: Workspace) -> Any:
    caps = media_capabilities(db, workspace.id)
    route = route_for(IMAGE_SLOT, caps, media_tools.prefer_for(workspace.settings).get(IMAGE_SLOT))
    if isinstance(route, str):
        raise HTTPException(status_code=422, detail=f"No AI image can be made: {route}")
    return route


def _options_plan(db: Session, workspace: Workspace, post: Any, body: OptionsRequest) -> Any:
    template = _posts_api()._render_template(db, workspace.id, post)
    try:
        blocks = validate_social_blocks(render.composition_of(template), template.format)
        spec = ai_options.image_slot(blocks, body.slot)
    except (SocialTemplateError, service.InvalidPost) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    width, height = render.render_size(blocks)
    brand = socials_brand.generation_inputs(db, workspace)
    return ai_options.plan_options(spec, body.slot, body.prompt.strip(), _image_route(db, workspace), width=width, height=height,
                                   style=brand["style"], references=brand["references"])


async def _make_options(plan: Any, workspace_id: UUID, post_id: UUID, title: str, slot: str, prompt: str) -> None:
    """The background work: make the options, then record them (or why none) on the post."""
    records, error = await ai_options.make(plan, workspace_id=workspace_id, post_id=post_id, title=title, slot=slot, prompt=prompt,
                                           session_factory=SessionLocal, store=MediaStore())
    await anyio.to_thread.run_sync(_settle, workspace_id, post_id, slot, prompt, records, error)


def _settle(workspace_id: UUID, post_id: UUID, slot: str, prompt: str, records: Any, error: Optional[str]) -> None:
    with SessionLocal() as db:
        post = service.get_post(db, workspace_id, post_id)
        if post is not None and ai_options.settle(post, slot, prompt, records, error):
            db.commit()


@router.post("/posts/{post_id}/ai-options", status_code=202, dependencies=[CAN_UPDATE])
def make_ai_options(post_id: UUID, body: OptionsRequest, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """Four AI options for one of the template's image slots, made in the background."""
    from core.utils.background_tasks import launch_guarded

    posts_api = _posts_api()
    post, workspace = posts_api._load(db, ctx, post_id), posts_api._workspace(db, ctx)
    try:
        ai_options.assert_editable(post)
        plan = _options_plan(db, workspace, post, body)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)
    status, content_hash = post.status, post.content_hash
    ai_options.ask(post, body.slot, body.prompt.strip())
    answer = posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    work = _make_options(plan, workspace.id, post.id, post.title or "", body.slot, body.prompt.strip())
    anyio.from_thread.run_sync(functools.partial(launch_guarded, work, subsystem="socials", operation="ai_options", workspace_id=workspace.id))
    return answer


@router.put("/posts/{post_id}/ai-options/{slot}", dependencies=[CAN_UPDATE])
def pick_ai_option(post_id: UUID, slot: str, body: PickRequest, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """The option picked becomes the slot's file (a render setting: an approval stands)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    try:
        ai_options.assert_editable(post)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)
    status, content_hash = post.status, post.content_hash
    if not ai_options.pick(post, slot, body.name):
        raise HTTPException(status_code=404, detail="No such AI option for this slot")
    return posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
