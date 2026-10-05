"""The template tools agents and Auto read before generate_document (F346, night 10b).

``platform_list_templates`` and ``platform_get_template_schema`` (and the session
tools ``list_templates`` / ``get_template_schema``, which run them).

F346 (night 10b):

* **The list stopped at "Executive Summary".** Each template came back as an object
  with its description, social templates included (18 of them), and a chat turn
  keeps about 2,000 tokens of a tool's answer: the list was cut alphabetically and
  the owner's own templates (Invoice, Meeting Notes, anything after "E") were never
  seen. Now a template is one line (``name | format | category | makes … | id``,
  with F348's supported formats), the document templates are listed by default
  (social ones with ``format``), the count comes first, and ``name`` searches by
  part of a name.
* **A template made by POST had guessed columns.** The schema answer said a table
  was ``data.line_items`` and nothing more, so its columns were read off the
  sample data, and a template with no sample had none. Now the answer carries the
  template's ``tables`` (each one's columns, from its blocks or its schema, and the
  columns that may stay empty) with ``required_fields`` and ``fallback_fields``
  (``modules.documents.field_requirements``, F345): what blocks when empty and
  what fills itself.
* **Only an id named a template.** A name works too: ``template_name``, or a
  ``template_id`` that is not an id (matched exactly, then ignoring case).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

TEMPLATE_ROW = "{name} | {format} | {category} | makes {makes} | {id}"
ROW_FORMAT = "name | format | category | makes (the formats generate_document may ask of it) | id"
FORMAT_JOINER = ", "
SOCIAL_NOTE = (
    "Document templates only: pass format social_image or social_video for the social ones. "
    "get_template_schema (by id or name) says what a template needs."
)
NEEDS_A_TEMPLATE = "Name the template: template_id, or template_name, from platform_list_templates."
NO_SUCH_TEMPLATE = "No template {ref!r} in this workspace: platform_list_templates lists them."


def _text(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def template_row(template: Any) -> str:
    """One template as one line of the list, with what it makes (F348's ``supported_formats``)."""
    from modules.documents.template_formats import supported_formats

    return TEMPLATE_ROW.format(name=template.name, format=template.format, category=template.category,
                               makes=FORMAT_JOINER.join(supported_formats(template)), id=template.id)


def list_templates_answer(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Every template of the workspace that matches: document ones unless ``format`` names one."""
    from core.social_templates import is_social_format
    from modules.documents.template_service import DocumentTemplateService

    fmt, name = _text(params.get("format")), _text(params.get("name"))
    rows = DocumentTemplateService(db).list_templates(workspace_id, format=fmt, category=_text(params.get("category")))
    if not fmt:
        rows = [t for t in rows if not is_social_format(t.format)]
    if name:
        rows = [t for t in rows if name.casefold() in (t.name or "").casefold()]
    answer: Dict[str, Any] = {"success": True, "count": len(rows), "row_format": ROW_FORMAT}
    if not fmt:
        answer["note"] = SOCIAL_NOTE
    return {**answer, "templates": [template_row(t) for t in rows]}


def _as_id(value: Optional[str]) -> Optional[UUID]:
    try:
        return UUID(value) if value else None
    except ValueError:
        return None


def find_template(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """The template ``params`` names (by id or by name) and ``None``, or ``None`` and why not."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    ref = _text(params.get("template_id")) or _text(params.get("template_name"))
    if ref is None:
        return None, NEEDS_A_TEMPLATE
    template_id = _as_id(ref)
    if template_id is not None:
        template = service.get_template(template_id, workspace_id)
    else:
        template = service.get_template_by_name(workspace_id, ref) or next(
            (t for t in service.list_templates(workspace_id) if (t.name or "").casefold() == ref.casefold()), None
        )
    return (template, None) if template else (None, NO_SUCH_TEMPLATE.format(ref=ref))


def _block_fields(blocks: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A block template's auto-resolved chips and the ``data.*`` fields the caller supplies."""
    from modules.documents.blocks import collect_variable_paths, validate_blocks
    from modules.documents.variables import CATALOG_BY_PATH

    variables: List[Dict[str, Any]] = []
    data_fields: List[str] = []
    for path in sorted(collect_variable_paths(validate_blocks(blocks))):
        if path.startswith("data."):
            data_fields.append(path)
        elif path in CATALOG_BY_PATH:
            entry = CATALOG_BY_PATH[path]
            variables.append({"path": path, "label": entry["label"], "category": entry["category"]})
    return variables, data_fields


def template_schema_answer(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The data one template needs, its tables' columns, and what fills itself."""
    from core.social_templates import is_social_format
    from modules.documents.field_requirements import requirements_of
    from modules.documents.template_summary import social_variable_names

    template, problem = find_template(db, workspace_id, params)
    if problem:
        return {"success": False, "error": problem}
    social = is_social_format(template.format)
    variables: List[Dict[str, Any]] = []
    data_fields: List[str] = []
    if social:
        data_fields = [f"data.{name}" for name in social_variable_names(template.blocks)]
    elif template.blocks:
        variables, data_fields = _block_fields(template.blocks)
    schema = {
        "success": True,
        "id": str(template.id),
        "name": template.name,
        "format": template.format,
        "description": template.description,
        "uses_blocks": bool(template.blocks) and not social,
        "variables": variables,           # auto-resolved chips (user/company/brand/date)
        "data_fields": data_fields,       # data.* fields you must supply at generation
        **requirements_of(template),      # required_fields, fallback_fields, tables (F345/F346)
        "data_schema": template.data_schema or {},  # legacy templates
        "sample_data": template.sample_data or {},
    }
    if social:
        blocks = template.blocks if isinstance(template.blocks, dict) else {}
        schema["variables_schema"] = blocks.get("variables_schema") or {}
        schema["sizes"] = blocks.get("sizes") or []
    return schema


__all__ = ["find_template", "list_templates_answer", "template_row", "template_schema_answer"]
