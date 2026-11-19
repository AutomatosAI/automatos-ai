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

PRD-251D (US-D103, E2): at the person's own render or preview (``write_missing_fields``), an
empty caption is Auto's to write as well, from the post's brief and for its channels, in the
same composer call that writes the empty fields or picks the template, and saved in the same
edit. A render a plan or an agent starts (``render_post`` itself) writes no caption.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it
in the Socials router, whose ``require_socials_enabled`` answers 404 unless both
switches are on (D1). Never mount it in the app directly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from config import config
from core.social_cuts import fields_cut_out, schema_for_length
from core.social_field_order import ordered_schema
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate, Skill
from core.models.socials import SOCIAL_POST_FORMATS, SocialPost
from core.models.workspaces import Workspace
from core.social_templates import SOCIAL_TEMPLATE_FORMATS, claim_names, made_for
from modules.documents.brand_kit import get_brand_kit
from modules.socials import compose, compose_facts, compose_photos, compose_sources, empty_fields, history, preview, service
from modules.socials import voice_examples
from modules.socials import sources as post_sources
from modules.socials.capabilities import social_channels
from modules.socials.compose_checks import template_kind
from modules.socials.template_claims import with_template_claims
from modules.socials.render import NotRenderable
from modules.socials.template_gallery import durations_of
from api.socials_media_upload import UPLOAD_ASPECT

TEXT_FORMAT = "text"

logger = logging.getLogger(__name__)

router = APIRouter()

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
BRIEF_MAX_CHARS = 4000
MAX_CHANNELS = 10
# Candidate sources (F378): those matching each of the brief's words, per kind, and each address it quotes.
CANDIDATES_PER_TERM = 3
WEB_SCHEME = re.compile(r"^https?://", re.IGNORECASE)  # an end card writes a web address without it
MAX_CANDIDATES = 24
MAX_BRIEF_URLS = 5
_URL = re.compile(r"https?://[^\s<>\"')]+")
# The fields the person wrote themselves, sent with a redraft: at most the largest template's worth.
MAX_GIVEN_FIELDS = 200
GIVEN_NAME_MAX_CHARS = 120
GIVEN_VALUE_MAX_CHARS = 2000
GivenName = Annotated[str, StringConstraints(min_length=1, max_length=GIVEN_NAME_MAX_CHARS)]
GivenValue = Union[bool, int, float, Annotated[str, StringConstraints(max_length=GIVEN_VALUE_MAX_CHARS)]]
# What the composer reads of a template row (``template_entry``).
ENTRY_COLUMNS = (
    DocumentTemplate.id, DocumentTemplate.name, DocumentTemplate.description, DocumentTemplate.format, DocumentTemplate.blocks,
)


class ComposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: str = Field(..., min_length=1, max_length=BRIEF_MAX_CHARS)
    channels: Optional[List[str]] = Field(None, max_length=MAX_CHANNELS)
    format: Optional[str] = None
    # PRD-251B (B5, US-B103): the editor's choices. The template must be one of the
    # workspace's of the format's kind; the length one the template declares.
    template_id: Optional[UUID] = None
    length_seconds: Optional[int] = Field(None, ge=1)
    # The template fields the person wrote themselves (name → value): each is kept as written
    # and Auto writes the rest (``modules/socials/empty_fields.py``).
    given_variables: Optional[Dict[GivenName, GivenValue]] = Field(None, max_length=MAX_GIVEN_FIELDS)


class ChoiceRefused(ValueError):
    """A choice the editor sent that the workspace cannot honour (422 with a code)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------------------
# What the composer is given (each a seam the tests replace)
# ---------------------------------------------------------------------------


def social_templates(db: Session, workspace_id: UUID, post_format: Optional[str]) -> List[Dict[str, Any]]:
    """The workspace's active social templates (of the post's kind when a format is set), as
    the gallery lists them (``template_gallery.gallery``: an archived one is never offered)."""
    kinds = [template_kind(post_format)] if post_format else list(SOCIAL_TEMPLATE_FORMATS)
    rows = (
        db.query(*ENTRY_COLUMNS)
        .filter(DocumentTemplate.workspace_id == workspace_id, DocumentTemplate.format.in_(kinds),
                DocumentTemplate.is_active.isnot(False))
        .order_by(DocumentTemplate.name)
        .all()
    )
    return [template_entry(row) for row in rows]


def template_entry(row: Any) -> Dict[str, Any]:
    """One template as the composer sees it: its fields, sizes and lengths, and (F378) what it
    is for: its description, its photo spots and (F377) whom it is made for."""
    blocks = row.blocks if isinstance(row.blocks, dict) else {}
    return {
        "id": str(row.id), "name": row.name, "description": row.description, "format": row.format,
        "sizes": blocks.get("sizes") or [], "variables_schema": ordered_schema(blocks),
        # PRD-251B (B5): the lengths a video declares (US-B104), for the editor and the model.
        "durations": durations_of(blocks, row.format),
        "photo_slots": compose_photos.photo_slots(blocks),
        # F377: the fields each shorter cut never shows, so a post at that length is never asked
        # for them; and whom the template is for ("software": a software brief only).
        "fields_cut_out": fields_cut_out(blocks),
        "made_for": made_for(blocks),
    }


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
    """What a claim may be bound to: the workspace's sources that match the brief's words
    (F378, B1: never the newest of each kind whatever the brief says), and each address the
    brief quotes; each once, ``MAX_CANDIDATES`` at most."""
    found: List[Dict[str, Any]] = []
    for term in compose_sources.brief_terms(brief):
        found += post_sources.search(db, workspace_id, q=term, limit=CANDIDATES_PER_TERM)
    for url in _URL.findall(brief)[:MAX_BRIEF_URLS]:
        found += post_sources.search(db, workspace_id, kind="url", q=url, limit=1)
    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for candidate in found:
        unique.setdefault((str(candidate.get("kind")), str(candidate.get("ref"))), candidate)
    return list(unique.values())[:MAX_CANDIDATES]


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


def brand_handles(db: Session, workspace_id: UUID) -> Dict[str, str]:
    """F378 (night 11): the brand kit's social handles (toolkit → handle), the only ones a post carries."""
    workspace = db.get(Workspace, workspace_id)
    return dict(get_brand_kit(workspace.settings if workspace is not None else None).get("social_handles") or {})


def brand_website(db: Session, workspace_id: UUID) -> str:
    """The brand kit's company website as an end card writes it: no scheme and no trailing slash
    (``https://automatos.app/`` → ``automatos.app``); empty without one."""
    workspace = db.get(Workspace, workspace_id)
    company = get_brand_kit(workspace.settings if workspace is not None else None).get("company") or {}
    address = str(company.get("website") or "").strip()
    return WEB_SCHEME.sub("", address).rstrip("/")


def brand_style_text(db: Session, workspace_id: UUID) -> str:
    """The brand kit's style profile as one paragraph (PRD-251B US-B303); empty without one."""
    from modules.documents.brand_style import style_prompt

    workspace = db.get(Workspace, workspace_id)
    return style_prompt(workspace.settings if workspace is not None else None)


def compose_context(
    db: Session, workspace_id: UUID, body: ComposeRequest, *, current_take: Optional[Mapping[str, Any]] = None,
    given_fields: Optional[Mapping[str, Any]] = None,
) -> compose.ComposeContext:
    """Everything the composer is given, from the caller's workspace only; a retake's
    ``current_take`` (F378) is the post as it is, the take to start from; ``given_fields`` are
    the template fields the person wrote themselves, which stay as written."""
    channels, warnings = connected_channels(db, workspace_id, body.channels)
    if body.format == TEXT_FORMAT:
        channels = text_channels(channels, body.channels, warnings)
    templates = chosen_templates(social_templates(db, workspace_id, body.format), body)
    if body.template_id is None and compose_photos.brief_says_no_photo(body.brief):
        templates = compose_photos.without_required_photos(templates)  # F378 (B19): no photo, no photo-only card
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
        recent_openings=tuple(history.recent_openings(db, workspace_id)),  # PRD-251C US-C105
        voice_examples=tuple(voice_examples.for_composer(db, workspace_id)),  # PRD-251C US-C406
        handles=brand_handles(db, workspace_id),  # F378: the only handles a post carries
        website=brand_website(db, workspace_id),  # an end card's web address
        current_take=dict(current_take or {}),
        given_fields=dict(given_fields or {}),
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
    """A draft proposal for ``brief`` (not saved): title, copy (``{"base", "channels"}``,
    the shape a save takes), format, template, variables, sources, channels and warnings. A plain
    ``def``: its database reads run in the threadpool (F105), and the model call
    runs on the event loop from there. 422 for an unknown format; 502 when the
    model's answer cannot be read twice; 504 when it does not answer in time."""
    if body.format is not None and body.format not in SOCIAL_POST_FORMATS:
        raise HTTPException(status_code=422, detail=f"format must be one of {list(SOCIAL_POST_FORMATS)}")
    try:
        context = compose_context(db, ctx.workspace_id, body, given_fields=body.given_variables)
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
PICK_FAILED = "Auto could not pick a template"
WRITE_FAILED = "Auto could not write the words on the image"
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


def post_channels(post: Any) -> Optional[List[str]]:
    """The channels the post goes to, for the composer to write its caption for; ``None``, every
    connected channel, when it has none yet (as ``api/socials_retake.py`` asks)."""
    targets = getattr(post, "targets", None) or ()
    return sorted({target.toolkit for target in targets}) or None


def _auto_context(db: Session, workspace_id: UUID, post: Any, caption: bool = False) -> compose.ComposeContext:
    """The composer's context for ``post``; a chosen length keeps the templates that offer it.
    With ``caption``, the composer writes the caption for the post's own channels."""
    body = ComposeRequest(
        brief=auto_brief(post), format=post.format, length_seconds=post.length_seconds,
        channels=post_channels(post) if caption else None,
    )
    context = compose_context(db, workspace_id, body)
    if post.length_seconds is None:
        return context
    return replace(context, templates=[t for t in context.templates if post.length_seconds in (t.get("durations") or [])])


def _auto_edit(proposal: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """The edit that records the pick: the template, its fields and the sources Auto bound,
    where the post's own value wins for a field the template has, and its own sources stay.
    A value in a field the template marks a claim is one, the post's own included (F433)."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    own = {name: spec for name, spec in (post.variables or {}).items() if name in schema}
    return {
        "template_id": proposal["template_id"],
        "variables": with_template_claims({**(proposal.get("variables") or {}), **own}, set(claim_names(schema))),
        "sources": {**(proposal.get("sources") or {}), **(post.sources or {})},
    }


async def _auto_proposal(
    context: compose.ComposeContext, workspace_id: UUID, post_id: Any, failed: str = PICK_FAILED,
) -> Dict[str, Any]:
    """The composer's checked proposal; a model that gives no usable answer is NotRenderable,
    saying what Auto could not do (``failed``)."""
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    try:
        return await compose.propose(context, compose.llm_factory(workspace_id), timeout)
    except (compose.ComposeTimedOut, compose.ComposeFailed) as exc:
        raise NotRenderable(f"{failed}: {exc}") from exc
    except Exception as exc:
        logger.exception("[Socials] Auto's proposal failed for post %s (%s)", post_id, failed)
        raise NotRenderable(f"{failed}: the model could not be reached. Try again.") from exc


async def auto_template(
    db: Session, workspace_id: UUID, post: Any, *, caption: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Auto's pick for ``post``: the edit that records it, and the template's name.
    :class:`NotRenderable` (422, saying why) when no template of the workspace fits the
    post's format and length, or the model gave no usable answer. With ``caption`` (the
    person's own render of a post whose caption is empty, PRD-251D US-D103), the same call
    writes the caption for the post's channels, and it joins the edit."""
    try:
        context = await asyncio.to_thread(_auto_context, db, workspace_id, post, caption)
    except ChoiceRefused as exc:
        raise NotRenderable(f"Auto could not pick a template: {exc}") from exc
    if not context.templates:
        raise NotRenderable(NO_TEMPLATE_FOR.format(format=post.format))
    proposal = await _auto_proposal(context, workspace_id, post.id)
    if not proposal.get("template_id"):
        raise NotRenderable(NO_TEMPLATE_FOR.format(format=post.format))
    edit = _auto_edit(proposal, post)
    if caption:
        edit = {**edit, **empty_fields.caption_edit(proposal, post)}
    return edit, str(proposal["template"]["name"])


def _posts_api() -> Any:
    """``api/socials.py``, for its compare-and-set commit: it includes this module's router,
    so it is imported when a request runs."""
    from api import socials

    return socials


async def let_auto_pick(
    db: Session, workspace: Workspace, post: SocialPost, actor: str, check: Callable[[Any], None],
    *, caption: bool = False,
) -> None:
    """Give a post left to Auto (``left_to_auto``) its template as it first renders or
    previews: Auto picks it and writes its fields, and the pick is saved before the render
    starts, by the compare-and-set commit, with a line in the history naming the template.
    The render, the preview and the editor then all use it. ``check`` refuses a post that
    cannot render or preview now, before the model is asked. Any other post is left as it is.
    With ``caption`` (``write_missing_fields``, the person's own render), the empty caption is
    written in the same call and the same edit, and the history line names it too."""
    if not left_to_auto(post):
        return
    check(post)
    status, content_hash = post.status, post.content_hash
    # ``render_post`` and ``preview_post`` (a plan's, an agent's and the person's renders) ask for
    # the pick alone, in the call they always made.
    pick = auto_template(db, workspace.id, post, caption=True) if caption else auto_template(db, workspace.id, post)
    changes, template_name = await pick
    empty_fields.record_pick(post, actor, changes, template_name)
    _posts_api()._commit_unchanged(db, post, status=status, content_hash=content_hash)


# ---------------------------------------------------------------------------
# A chosen template's empty fields, written when the post renders
# ---------------------------------------------------------------------------
# Choosing a template used to hand the person its whole form: a render with a required field
# empty was refused. A field left empty is now Auto's to write, as "Let Auto pick" already
# does for a post with no template; a field that holds a value stays as it is.


def _chosen_template(db: Session, workspace_id: UUID, post: Any) -> Optional[Dict[str, Any]]:
    """The post's chosen template as the composer sees it, when it is one of the workspace's
    social templates; ``None`` for a post without one (``let_auto_pick``'s)."""
    if post.template_id is None:
        return None
    row = (
        db.query(*ENTRY_COLUMNS)
        .filter(DocumentTemplate.id == post.template_id, DocumentTemplate.workspace_id == workspace_id)
        .first()
    )
    return template_entry(row) if row is not None and row.format in SOCIAL_TEMPLATE_FORMATS else None


async def auto_fields(db: Session, workspace_id: UUID, post: Any, *, caption: bool = False) -> Dict[str, Any]:
    """The edit that fills the chosen template's empty fields: the composer, with that template
    and the post's own fields given, writes the rest from the post's own words. The post's own
    value stays in every field that holds one (``empty_fields.fields_edit``). With ``caption``
    (PRD-251D US-D103), the composer writes for the post's own channels and the caption it
    wrote joins the edit as ``copy``, for a post whose caption is empty."""
    body = ComposeRequest(
        brief=auto_brief(post), format=post.format, template_id=post.template_id, length_seconds=post.length_seconds,
        channels=post_channels(post) if caption else None,
    )
    given = empty_fields.own_values(post)
    try:
        context = await asyncio.to_thread(compose_context, db, workspace_id, body, given_fields=given)
    except ChoiceRefused as exc:
        raise NotRenderable(f"{WRITE_FAILED}: {exc}") from exc
    # A value the chosen template refuses (an earlier template's words) is not the owner's to keep: Auto writes it.
    schema = schema_for_length(context.templates[0], post.length_seconds) if len(context.templates) == 1 else None
    fitting = empty_fields.own_values(post, schema)
    if fitting != given:
        context = replace(context, given_fields=fitting)
    proposal = await _auto_proposal(context, workspace_id, post.id, WRITE_FAILED)
    edit = empty_fields.fields_edit(proposal, post)
    return {**edit, **empty_fields.caption_edit(proposal, post)} if caption else edit


def _still_needed(template: Mapping[str, Any], post: Any) -> None:
    """:class:`NotRenderable`, naming by its label each required field Auto could not write
    and the render would refuse (a fact the brief does not give: a customer's name, their words)."""
    missing = empty_fields.render_refuses(template, post)
    if not missing:
        return
    schema = template.get("variables_schema") or {}
    labels = [compose_facts.field_label(name, schema.get(name)) for name in missing]
    raise NotRenderable(empty_fields.STILL_NEEDED.format(fields=", ".join(labels)), missing=tuple(missing))


def _record_auto_words(post: Any, actor: str, template: Mapping[str, Any], changes: Mapping[str, Any]) -> bool:
    """Apply what Auto wrote (the fields it newly gave a value, the caption) as one edit with one
    history line naming them; ``False``, and nothing applied, when it wrote neither."""
    schema = template.get("variables_schema") or {}
    wrote = empty_fields.newly_written(changes, post, schema_for_length(template, getattr(post, "length_seconds", None)))
    caption = empty_fields.COPY in changes
    if not (wrote or caption):
        return False
    labels = [compose_facts.field_label(name, schema.get(name)) for name in wrote]
    empty_fields.record_written(post, actor, changes, empty_fields.written_line(labels, caption))
    return True


async def write_missing_fields(db: Session, workspace: Workspace, post: SocialPost, actor: str, *, preview_only: bool) -> None:
    """Before a person's render or preview: when the post's chosen template still has required
    fields empty, or its caption is empty and it has a brief to write one from (PRD-251D US-D103,
    ``empty_fields.caption_wanted``), Auto writes them from the post's own words in one composer
    call, saved as one edit by the compare-and-set commit with a line
    in the history naming them. Every field that holds a value stays as it is, and so does a
    caption that holds words; with only the caption empty, only the caption is written. A post
    left to "Let Auto pick" with an empty caption gets its template, fields and caption here
    (``let_auto_pick``); one with a caption, or with nothing empty, is left alone, and one that
    cannot render now is refused before the model is asked. What Auto could not write is asked
    of the person by its label (422), with what it did write kept."""
    if empty_fields.holds_an_approval(post):
        return  # nothing is written under an approval: the preview or the render answers for itself
    check = preview.assert_can_preview if preview_only else service.assert_can_render
    caption = empty_fields.caption_wanted(post)
    if left_to_auto(post):
        if caption:
            await let_auto_pick(db, workspace, post, actor, check, caption=True)
        return  # with a caption, ``render_post`` and ``preview_post`` pick as they always have
    template = await asyncio.to_thread(_chosen_template, db, workspace.id, post)
    if template is None:
        return
    missing = empty_fields.empty_required(template, post)
    if not (missing or caption):
        return
    check(post)
    status, content_hash = post.status, post.content_hash
    edit = await auto_fields(db, workspace.id, post, caption=caption)
    # With every required field filled, only the caption is written: an optional field the person
    # left empty stays empty, as it does today.
    changes = edit if missing else empty_fields.only_caption(edit)
    if _record_auto_words(post, actor, template, changes):
        _posts_api()._commit_unchanged(db, post, status=status, content_hash=content_hash)
    _still_needed(template, post)
