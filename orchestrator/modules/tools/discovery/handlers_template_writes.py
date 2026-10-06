"""``platform_create_template`` / ``platform_update_template``: agents build templates (PRD-255 US-013).

The Brand designer builds the owner's templates (a quote, a price list) in the
studio's block format. Both tools save through the studio's own rules:

* **One validator.** The block tree passes ``template_validation.validated_blocks``,
  the check ``POST`` / ``PUT /api/documents/templates`` run, and the service's save;
  a refusal comes back as the tool's failure with the field-level errors.
* **Document templates only** (Decision Q3): pdf, docx, xlsx. A social format is
  refused by both tools; the designer never edits social layouts.
* **A starter is never changed** (``created_by == STARTER_CREATOR``): copy to
  customise, as the studio does. ``create_template(copy_of=<id>)`` starts from any
  document template of the workspace, starters included. There is no delete tool.
* **Made by whom.** A created template is tagged ``made-by:<agent>`` from the
  server-minted caller (``_agent_id``, which ``exec_platform`` sets and strips from
  the call), never from a parameter; a ``made-by:`` tag the call sends is dropped,
  and an update keeps the template's own.
* **The caller's workspace only.** Another workspace's template answers "not found".
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MADE_BY_PREFIX = "made-by:"
AGENT_CREATOR = "agent"
DEFAULT_CATEGORY = "general"
COPY_SUFFIX = " (copy)"
MAX_NAME_LENGTH = 255
MAX_CATEGORY_LENGTH = 100
MAX_TAGS = 20
MAX_TAG_LENGTH = 64
DOCUMENT_FORMATS = ("pdf", "docx", "xlsx")
CREATE_FIELDS = ("name", "description", "category", "format", "blocks", "sample_data", "tags")
UPDATE_FIELDS = ("name", "description", "category", "blocks", "sample_data", "tags")

NEEDS_A_NAME = "create_template needs a name (or copy_of, whose copy is named '<name> (copy)')."
NEEDS_BLOCKS = "create_template needs blocks, the studio's block tree (or copy_of, to start from a template's)."
FORMAT_REFUSED = (
    "format {fmt!r}: create_template makes document templates only ({formats}); social templates are "
    "made in the studio. Nothing was saved."
)
SOCIAL_REFUSED = (
    "{name!r} is a {fmt} template: the template tools change document templates only ({formats}). "
    "Nothing was saved."
)
STARTER_REFUSED = (
    "{name!r} is a platform starter, which is never changed: copy it first with "
    "create_template(copy_of={id!r}), then update the copy. Nothing was saved."
)
NO_LAYOUT = "{name!r} has no block layout to copy. Nothing was saved."
NAME_TAKEN = "A template named {name!r} already exists in this workspace: pick another name. Nothing was saved."
NOT_AN_ID = "{param} {ref!r} is not a template id: list_templates gives it."
NOT_FOUND = "No template {ref} in this workspace."
NEEDS_AN_ID = "update_template needs the template_id, from list_templates."
NOTHING_TO_CHANGE = "update_template needs at least one of: " + ", ".join(UPDATE_FIELDS) + "."
BAD_FIELD = "{field} must be {kind}."
INVALID = "The template was not saved: {message}. Fix these and send it again."
CREATED_NOTE = (
    "Saved: it is listed in the studio like any template. Look at it with render_preview(template_id={id!r}) "
    "before you report."
)
UPDATED_NOTE = "Saved. Look at it again with render_preview(template_id={id!r})."


class Refused(ValueError):
    """A call the tool will not save, with the reason (and field-level errors) it hands back."""

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors


def _failed(e: Refused) -> Dict[str, Any]:
    answer: Dict[str, Any] = {"success": False, "error": str(e)}
    return {**answer, "errors": e.errors} if e.errors is not None else answer


def _text_field(params: Dict[str, Any], field: str, limit: int) -> Optional[str]:
    value = params.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise Refused(BAD_FIELD.format(field=field, kind=f"a non-empty text of at most {limit} characters"))
    return value.strip()


def _tags(raw: Any) -> List[str]:
    """The caller's tags, without any ``made-by:`` tag (those are the server's)."""
    if raw is None:
        return []
    if not (isinstance(raw, list) and len(raw) <= MAX_TAGS and all(map(_fits_as_tag, raw))):
        raise Refused(BAD_FIELD.format(field="tags", kind=f"a list of at most {MAX_TAGS} short texts"))
    return list(dict.fromkeys(t.strip() for t in raw if not t.strip().casefold().startswith(MADE_BY_PREFIX)))


# Each typed field: the type it must be, and how the refusal names it.
_TYPED_FIELDS = {
    "description": (str, "text"),
    "sample_data": (dict, "an object of the data the template shows"),
    "blocks": ((dict, list), "the block tree: an object with blocks, or a list"),
}


def _fits_as_tag(tag: Any) -> bool:
    return isinstance(tag, str) and bool(tag.strip()) and len(tag.strip()) <= MAX_TAG_LENGTH


def _fields(params: Dict[str, Any], allowed: Tuple[str, ...]) -> Dict[str, Any]:
    """The call's own fields among ``allowed``, typed; keys starting with "_" are the server's."""
    fields = {k: params[k] for k in allowed if params.get(k) is not None}
    for field, limit in (("name", MAX_NAME_LENGTH), ("category", MAX_CATEGORY_LENGTH)):
        if field in fields:
            fields[field] = _text_field(params, field, limit)
    for field, (kind, words) in _TYPED_FIELDS.items():
        if field in fields and not isinstance(fields[field], kind):
            raise Refused(BAD_FIELD.format(field=field, kind=words))
    if "tags" in fields:
        fields["tags"] = _tags(fields["tags"])
    return fields


def _template_id(params: Dict[str, Any], param: str) -> UUID:
    ref = str(params.get(param) or "").strip()
    if not ref:
        raise Refused(NEEDS_AN_ID if param == "template_id" else NOT_AN_ID.format(param=param, ref=ref))
    try:
        return UUID(ref)
    except ValueError:
        raise Refused(NOT_AN_ID.format(param=param, ref=ref)) from None


def _owned_template(db: Session, workspace_id: UUID, params: Dict[str, Any], param: str) -> Any:
    """The workspace's document template ``params[param]`` names; another workspace's is not found."""
    from core.social_templates import is_social_format
    from modules.documents.template_service import DocumentTemplateService

    template_id = _template_id(params, param)
    template = DocumentTemplateService(db).get_template(template_id, workspace_id)
    if template is None:
        raise Refused(NOT_FOUND.format(ref=template_id))
    if is_social_format(template.format):
        raise Refused(SOCIAL_REFUSED.format(name=template.name, fmt=template.format, formats=", ".join(DOCUMENT_FORMATS)))
    return template


def maker_tag(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> str:
    """``made-by:<agent>``: the calling agent's slug (else its name), as the server minted it."""
    from core.models import Agent

    agent_id = _agent_id(params)
    agent = None
    if agent_id is not None:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    who = (getattr(agent, "slug", None) or getattr(agent, "name", None) or AGENT_CREATOR).strip()
    return f"{MADE_BY_PREFIX}{who}"


def _agent_id(params: Dict[str, Any]) -> Optional[int]:
    """The calling agent's id, as ``exec_platform`` minted it (never the model's)."""
    raw = params.get("_agent_id")
    if isinstance(raw, bool):
        return None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _creator(params: Dict[str, Any]) -> str:
    """``created_by`` for an agent's template: never ``STARTER_CREATOR``, so it is the workspace's own."""
    agent_id = _agent_id(params)
    return f"{AGENT_CREATOR}:{agent_id}" if agent_id else AGENT_CREATOR


def _validated(fmt: str, blocks: Any) -> Any:
    """The block tree through the studio's own check (``template_validation``), or the refusal."""
    from modules.documents.blocks import BlockValidationError
    from modules.documents.template_validation import INVALID_BLOCKS, validated_blocks

    try:
        return validated_blocks(fmt, blocks)
    except BlockValidationError as e:
        raise Refused(INVALID.format(message=INVALID_BLOCKS), e.errors) from None


def _saved(db: Session, name: str, save: Callable[[], Any]) -> Any:
    """``save()``, with the service's refusals handed back as the tool's (field-level errors kept)."""
    from sqlalchemy.exc import IntegrityError

    from modules.documents.template_validation import INVALID_TEMPLATE, TEMPLATE_SAVE_ERRORS, save_errors

    try:
        return save()
    except TEMPLATE_SAVE_ERRORS as e:
        raise Refused(INVALID.format(message=INVALID_TEMPLATE), save_errors(e)) from None
    except IntegrityError:
        # The name is unique per workspace and version, removed templates included.
        db.rollback()
        logger.info("[template-tools] %r not saved: the name is taken", name)
        raise Refused(NAME_TAKEN.format(name=name)) from None


def _name_free(db: Session, workspace_id: UUID, name: str) -> None:
    from modules.documents.template_service import DocumentTemplateService

    if DocumentTemplateService(db).get_template_by_name(workspace_id, name) is not None:
        raise Refused(NAME_TAKEN.format(name=name))


def _copied(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """What ``copy_of`` starts from: the template's layout, sample and category, named '<name> (copy)'."""
    source = _owned_template(db, workspace_id, params, "copy_of")
    if not source.blocks:
        raise Refused(NO_LAYOUT.format(name=source.name))
    return {
        "name": f"{source.name}{COPY_SUFFIX}"[:MAX_NAME_LENGTH],
        "description": source.description,
        "category": source.category or DEFAULT_CATEGORY,
        "format": source.format,
        "blocks": source.blocks,
        "sample_data": source.sample_data or {},
        "tags": _tags([t for t in (source.tags or []) if _fits_as_tag(t)][:MAX_TAGS]),
    }


def create_fields(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The new template's fields, checked as the studio checks them, or the refusal."""
    fields = _fields(params, CREATE_FIELDS)
    if params.get("copy_of") is not None:
        fields = {**_copied(db, workspace_id, params), **fields}
    if "name" not in fields:
        raise Refused(NEEDS_A_NAME)
    if "blocks" not in fields:
        raise Refused(NEEDS_BLOCKS)
    fmt = str(fields.get("format") or "").strip()
    if fmt not in DOCUMENT_FORMATS:
        raise Refused(FORMAT_REFUSED.format(fmt=fmt, formats=", ".join(DOCUMENT_FORMATS)))
    return {**fields, "format": fmt, "blocks": _validated(fmt, fields["blocks"])}


def _summary(template: Any, note: str) -> Dict[str, Any]:
    return {
        "success": True,
        "template_id": str(template.id),
        "name": template.name,
        "format": template.format,
        "category": template.category,
        "tags": list(template.tags or []),
        "note": note.format(id=template.id),
    }


async def create_template(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """A new document template of this workspace, saved through the studio's own checks."""
    from modules.documents.template_service import DocumentTemplateService

    try:
        fields = create_fields(db, workspace_id, params)
        _name_free(db, workspace_id, fields["name"])
        tags = list(dict.fromkeys([*fields.get("tags", []), maker_tag(db, workspace_id, params)]))
        template = _saved(db, fields["name"], lambda: DocumentTemplateService(db).create_template(
            workspace_id=workspace_id,
            name=fields["name"],
            format=fields["format"],
            description=fields.get("description"),
            sample_data=fields.get("sample_data") or {},
            category=fields.get("category") or DEFAULT_CATEGORY,
            tags=tags,
            created_by=_creator(params),
            blocks=fields["blocks"],
        ))
    except Refused as e:
        return _failed(e)
    return _summary(template, CREATED_NOTE)


def update_fields(template: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The changes to ``template``, checked as the studio checks them, or the refusal."""
    from modules.documents.template_summary import STARTER_CREATOR

    if (getattr(template, "created_by", None) or "") == STARTER_CREATOR:
        raise Refused(STARTER_REFUSED.format(name=template.name, id=str(template.id)))
    updates = _fields(params, UPDATE_FIELDS)
    if not updates:
        raise Refused(NOTHING_TO_CHANGE)
    if "blocks" in updates:
        updates["blocks"] = _validated(template.format, updates["blocks"])
    if "tags" in updates:
        made_by = [t for t in (template.tags or []) if str(t).casefold().startswith(MADE_BY_PREFIX)]
        updates["tags"] = list(dict.fromkeys([*updates["tags"], *made_by]))
    return updates


async def update_template(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Change a document template of this workspace (never a starter), through the studio's own checks."""
    from modules.documents.template_service import DocumentTemplateService

    try:
        template = _owned_template(db, workspace_id, params, "template_id")
        updates = update_fields(template, params)
        name = updates.get("name", template.name)
        if name != template.name:
            _name_free(db, workspace_id, name)
        saved = _saved(db, name, lambda: DocumentTemplateService(db).update_template(
            template.id, workspace_id, **updates))
    except Refused as e:
        return _failed(e)
    if saved is None:
        return {"success": False, "error": NOT_FOUND.format(ref=template.id)}
    return _summary(saved, UPDATED_NOTE)


__all__ = ["MADE_BY_PREFIX", "create_fields", "create_template", "maker_tag", "update_fields", "update_template"]
