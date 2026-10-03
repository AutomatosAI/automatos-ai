"""
Socials composer (PRD-251 S2.2a, US-207)
========================================

``POST /api/socials/compose``: a brief becomes a draft proposal, which is NOT
saved. The person reviews it in the composer and saves the draft with
``POST /posts`` and ``PUT /posts/{id}/targets``.

The proposal is drawn from the caller's workspace only: the brand kit's voice
(D5), its social templates (``document_templates`` of format social_image or
social_video, with their variables and sizes), the connected channels the
brief is for (``modules/socials/capabilities.py``) and their copy limits,
candidate sources (``modules/socials/sources.py``) and the built-in Socials
skills (global ``skills`` rows). One model call, through the platform's LLM
manager (``modules/socials/compose.py``), checked field by field
(``modules/socials/compose_checks.py``).

F253 ("Let Auto pick"): a post saved with its template left to Auto is given one when it
renders (``auto_template``): the composer picks the template for the post's format and
writes its fields from the post's own words, and ``api/socials.py`` saves the pick before
the render starts.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it
in the Socials router, whose ``require_socials_enabled`` answers 404 unless both
switches are on (D1). Never mount it in the app directly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate, Skill
from core.models.socials import SOCIAL_POST_FORMATS, SocialPost
from core.models.workspaces import Workspace
from core.social_templates import SOCIAL_TEMPLATE_FORMATS
from modules.documents.brand_kit import get_brand_kit
from modules.socials import compose, service
from modules.socials import sources as post_sources
from modules.socials.capabilities import social_channels
from modules.socials.compose_checks import template_kind
from modules.socials.render import NotRenderable
from modules.socials.template_gallery import durations_of
from api.socials_media_upload import UPLOAD_ASPECT

TEXT_FORMAT = "text"

logger = logging.getLogger(__name__)

router = APIRouter()

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
BRIEF_MAX_CHARS = 4000
MAX_CHANNELS = 10
# Candidate sources: the newest of each kind, and each address the brief quotes.
CANDIDATES_PER_KIND = 8
MAX_BRIEF_URLS = 5
_URL = re.compile(r"https?://[^\s<>\"')]+")


class ComposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: str = Field(..., min_length=1, max_length=BRIEF_MAX_CHARS)
    channels: Optional[List[str]] = Field(None, max_length=MAX_CHANNELS)
    format: Optional[str] = None
    # PRD-251B (B5, US-B103): the editor's choices. The template must be one of the
    # workspace's of the format's kind; the length one the template declares.
    template_id: Optional[UUID] = None
    length_seconds: Optional[int] = Field(None, ge=1)


class ChoiceRefused(ValueError):
    """A choice the editor sent that the workspace cannot honour (422 with a code)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------------------
# What the composer is given (each a seam the tests replace)
# ---------------------------------------------------------------------------


def social_templates(db: Session, workspace_id: UUID, post_format: Optional[str]) -> List[Dict[str, Any]]:
    """The workspace's social templates (of the post's kind when a format is set)."""
    kinds = [template_kind(post_format)] if post_format else list(SOCIAL_TEMPLATE_FORMATS)
    rows = (
        db.query(DocumentTemplate.id, DocumentTemplate.name, DocumentTemplate.format, DocumentTemplate.blocks)
        .filter(DocumentTemplate.workspace_id == workspace_id, DocumentTemplate.format.in_(kinds))
        .order_by(DocumentTemplate.name)
        .all()
    )
    out = []
    for row in rows:
        blocks = row.blocks if isinstance(row.blocks, dict) else {}
        out.append({
            "id": str(row.id), "name": row.name, "format": row.format,
            "sizes": blocks.get("sizes") or [], "variables_schema": blocks.get("variables_schema") or {},
            # PRD-251B (B5): the lengths a video declares (US-B104), for the editor and the model.
            "durations": durations_of(blocks, row.format),
        })
    return out


def chosen_templates(templates: List[Dict[str, Any]], body: ComposeRequest) -> List[Dict[str, Any]]:
    """The templates the composer is given: all of them, or only the chosen one (B5).
    A text post has none."""
    if body.format == TEXT_FORMAT:
        if body.template_id is not None:
            raise ChoiceRefused("template_not_allowed", "A text post has no template.")
        return []
    if body.template_id is None:
        return templates
    chosen = next((t for t in templates if t["id"] == str(body.template_id)), None)
    if chosen is None:
        raise ChoiceRefused(
            "template_not_allowed",
            "template_id is not one of this workspace's social templates of the post's format.",
        )
    return [chosen]


def check_length(templates: List[Dict[str, Any]], body: ComposeRequest) -> None:
    """A chosen length must be one the chosen template declares, or, with the template
    left to Auto, one some video template of the workspace declares (B5, US-B104)."""
    if body.length_seconds is None:
        return
    declared = sorted({d for t in templates for d in (t.get("durations") or [])})
    if body.length_seconds not in declared:
        where = "this template" if body.template_id is not None else "any video template of this workspace"
        raise ChoiceRefused(
            "length_not_declared",
            f"length_seconds must be a length {where} declares ({declared or 'none'}), got {body.length_seconds}.",
        )


def text_channels(channels: List[Dict[str, str]], requested: Optional[Sequence[str]], warnings: List[str]) -> List[Dict[str, str]]:
    """For a text post, the channels that take a text post kind: a named channel that
    does not is refused; an unnamed one is left out with a warning."""
    kept, refused = [], []
    for channel in channels:
        if channel.get("takes_text"):
            kept.append(channel)
        elif requested is not None and channel["toolkit"] in requested:
            refused.append(channel["toolkit"])
        else:
            warnings.append(f"{channel['toolkit']} takes no text-only post; it was left out")
    if refused:
        raise ChoiceRefused("channel_not_text", f"{', '.join(refused)}: no text-only post kind on this channel.")
    return kept


def connected_channels(
    db: Session, workspace_id: UUID, requested: Optional[Sequence[str]]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """The channels the brief is for (every connected one when none is named),
    and a warning for each named channel that is not connected."""
    connected = {c.toolkit: {"toolkit": c.toolkit, "label": c.label, "takes_text": _takes_text(c)} for c in social_channels(db, workspace_id)}
    if requested is None:
        return list(connected.values()), []
    wanted = list(dict.fromkeys(requested))
    missing = [t for t in wanted if t not in connected]
    warnings = [f"{t} is not connected in this workspace; it was left out" for t in missing]
    return [connected[t] for t in wanted if t in connected], warnings


def _takes_text(channel: Any) -> bool:
    """Whether the channel offers an available text-only post kind (the registry, US-203)."""
    kinds = getattr(channel, "post_kinds", None) or ()
    return any(getattr(k, "kind", None) == TEXT_FORMAT and getattr(k, "available", True) for k in kinds)


def candidate_sources(db: Session, workspace_id: UUID, brief: str) -> List[Dict[str, Any]]:
    """What a claim may be bound to: the workspace's newest of each kind, and each
    address the brief quotes."""
    found = post_sources.search(db, workspace_id, q="", limit=CANDIDATES_PER_KIND)
    for url in _URL.findall(brief)[:MAX_BRIEF_URLS]:
        found += post_sources.search(db, workspace_id, kind="url", q=url, limit=1)
    return found


def builtin_skills(db: Session) -> Dict[str, str]:
    """The built-in Socials skills by name, when the platform has them (global rows)."""
    rows = (
        db.query(Skill.name, Skill.prompt_template)
        .filter(Skill.name.in_(compose.SKILL_NAMES), Skill.workspace_id.is_(None), Skill.is_active.is_(True))
        .all()
    )
    return {row.name: row.prompt_template for row in rows if row.prompt_template}


def brand_voice(db: Session, workspace_id: UUID) -> Dict[str, Any]:
    """The brand kit's voice: its tone words and banned phrases (D5)."""
    workspace = db.get(Workspace, workspace_id)
    return dict(get_brand_kit(workspace.settings if workspace is not None else None).get("voice") or {})


def brand_style_text(db: Session, workspace_id: UUID) -> str:
    """The brand kit's style profile as one paragraph (PRD-251B US-B303); empty without one."""
    from modules.documents.brand_style import style_prompt

    workspace = db.get(Workspace, workspace_id)
    return style_prompt(workspace.settings if workspace is not None else None)


def compose_context(db: Session, workspace_id: UUID, body: ComposeRequest) -> compose.ComposeContext:
    """Everything the composer is given, from the caller's workspace only."""
    channels, warnings = connected_channels(db, workspace_id, body.channels)
    if body.format == TEXT_FORMAT:
        channels = text_channels(channels, body.channels, warnings)
    templates = chosen_templates(social_templates(db, workspace_id, body.format), body)
    check_length(templates, body)
    return compose.ComposeContext(
        brief=body.brief.strip(),
        format=body.format,
        channels=[{"toolkit": c["toolkit"], "label": c["label"]} for c in channels],
        templates=templates,
        candidates=candidate_sources(db, workspace_id, body.brief),
        voice=brand_voice(db, workspace_id),
        skills=builtin_skills(db),
        warnings=warnings,
        template_id=str(body.template_id) if body.template_id is not None else None,
        length_seconds=body.length_seconds,
        style=brand_style_text(db, workspace_id),
    )


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@router.post("/compose", dependencies=[CAN_CREATE])
def compose_social_post(
    body: ComposeRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """A draft proposal for ``brief`` (not saved): title, copy (base and per
    channel), format, template, variables, sources, channels and warnings. A plain
    ``def``: its database reads run in the threadpool (F105), and the model call
    runs on the event loop from there. 422 for an unknown format; 502 when the
    model's answer cannot be read twice; 504 when it does not answer in time."""
    if body.format is not None and body.format not in SOCIAL_POST_FORMATS:
        raise HTTPException(status_code=422, detail=f"format must be one of {list(SOCIAL_POST_FORMATS)}")
    try:
        context = compose_context(db, ctx.workspace_id, body)
    except ChoiceRefused as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)}) from exc
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    try:
        return anyio.from_thread.run(compose.propose, context, compose.llm_factory(ctx.workspace_id), timeout)
    except compose.ComposeTimedOut as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except compose.ComposeFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[Socials] compose failed for workspace %s", ctx.workspace_id)
        raise HTTPException(status_code=502, detail="The model could not be reached. Try again.") from exc


# ---------------------------------------------------------------------------
# F253: "Let Auto pick" when the post renders
# ---------------------------------------------------------------------------
# The editor's Template card offers "Let Auto pick" first, and the post is saved with no
# template. Every template has fields a render needs, so picking a template alone would
# still be refused: the composer picks it and writes those fields, as Redraft with Auto
# does, from the post's own words. The person's copy and title stay as they are.

VISUAL_FORMATS = tuple(name for name in SOCIAL_POST_FORMATS if name != TEXT_FORMAT)
NO_TEMPLATE_FOR = "Auto found no template for this {format} post: add one in Templates, or choose a format that has one."


def left_to_auto(post: Any) -> bool:
    """A post whose template is left to Auto: a visual format, no template yet, and no file
    of the person's own as its visual (an upload or a Library pick, ``media.original``)."""
    media = post.media if isinstance(post.media, dict) else {}
    return post.template_id is None and post.format in VISUAL_FORMATS and UPLOAD_ASPECT not in media


def auto_brief(post: Any) -> str:
    """What Auto writes the fields from: the post's brief and its copy, or its title alone."""
    copy = post.copy if isinstance(post.copy, dict) else {}
    words = [text.strip() for text in (post.brief, copy.get("base")) if isinstance(text, str) and text.strip()]
    return "\n\n".join(words or [post.title])[:BRIEF_MAX_CHARS]


def _auto_context(db: Session, workspace_id: UUID, post: Any) -> compose.ComposeContext:
    """The composer's context for ``post``; a chosen length keeps the templates that offer it."""
    body = ComposeRequest(brief=auto_brief(post), format=post.format, length_seconds=post.length_seconds)
    context = compose_context(db, workspace_id, body)
    if post.length_seconds is None:
        return context
    return replace(context, templates=[t for t in context.templates if post.length_seconds in (t.get("durations") or [])])


def _auto_edit(proposal: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """The edit that records the pick: the template, its fields and the sources Auto bound,
    where the post's own value wins for a field the template has, and its own sources stay."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    own = {name: spec for name, spec in (post.variables or {}).items() if name in schema}
    return {
        "template_id": proposal["template_id"],
        "variables": {**(proposal.get("variables") or {}), **own},
        "sources": {**(proposal.get("sources") or {}), **(post.sources or {})},
    }


async def _auto_proposal(context: compose.ComposeContext, workspace_id: UUID, post_id: Any) -> Dict[str, Any]:
    """The composer's checked proposal; a model that gives no usable answer is NotRenderable."""
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    try:
        return await compose.propose(context, compose.llm_factory(workspace_id), timeout)
    except (compose.ComposeTimedOut, compose.ComposeFailed) as exc:
        raise NotRenderable(f"Auto could not pick a template: {exc}") from exc
    except Exception as exc:
        logger.exception("[Socials] Auto's template pick failed for post %s", post_id)
        raise NotRenderable("Auto could not pick a template: the model could not be reached. Try again.") from exc


async def auto_template(db: Session, workspace_id: UUID, post: Any) -> Tuple[Dict[str, Any], str]:
    """Auto's pick for ``post``: the edit that records it, and the template's name.
    :class:`NotRenderable` (422, saying why) when no template of the workspace fits the
    post's format and length, or the model gave no usable answer."""
    try:
        context = await asyncio.to_thread(_auto_context, db, workspace_id, post)
    except ChoiceRefused as exc:
        raise NotRenderable(f"Auto could not pick a template: {exc}") from exc
    if not context.templates:
        raise NotRenderable(NO_TEMPLATE_FOR.format(format=post.format))
    proposal = await _auto_proposal(context, workspace_id, post.id)
    if not proposal.get("template_id"):
        raise NotRenderable(NO_TEMPLATE_FOR.format(format=post.format))
    return _auto_edit(proposal, post), str(proposal["template"]["name"])


def _posts_api() -> Any:
    """``api/socials.py``, for its compare-and-set commit: it includes this module's router,
    so it is imported when a request runs."""
    from api import socials

    return socials


async def let_auto_pick(
    db: Session, workspace: Workspace, post: SocialPost, actor: str, check: Callable[[Any], None]
) -> None:
    """Give a post left to Auto (``left_to_auto``) its template as it first renders or
    previews: Auto picks it and writes its fields, and the pick is saved before the render
    starts, by the compare-and-set commit, with a line in the history naming the template.
    The render, the preview and the editor then all use it. ``check`` refuses a post that
    cannot render or preview now, before the model is asked. Any other post is left as it is."""
    if not left_to_auto(post):
        return
    check(post)
    status, content_hash = post.status, post.content_hash
    changes, template_name = await auto_template(db, workspace.id, post)
    service.record_auto_pick(post, actor, changes, template_name)
    _posts_api()._commit_unchanged(db, post, status=status, content_hash=content_hash)
