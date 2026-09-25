"""API-facing shape of a document template (PRD-242 S2).

The Template Studio needs to tell a non-technical author three things a raw
``DocumentTemplate`` row does not say directly:

* **can it be edited in the block editor?** (``has_blocks`` — a legacy Jinja /
  uploaded-DOCX template can only be copied or rendered, not block-edited),
* **is it a platform starter?** (``is_starter`` — seeded rows are marked
  ``created_by='system'``; the gallery shows a badge and the copy-on-customise
  hint instead of pretending the user wrote them),
* **what must an agent (or a person) supply at generation time?**
  (``data_fields`` — every ``data.*`` chip the template references, which is
  exactly the contract ``generate_document(template_id, data)`` has to fill).

Pure — no DB, no IO — so the list endpoint can call it per row and tests can
drive it with plain objects.

PRD-251 S1.2: a social template (``social_image`` / ``social_video``) is a
composition, not a block tree: it is never block-editable, and its data fields
are its ``variables_schema`` (``core/social_templates.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.social_templates import is_social_format
from modules.documents.blocks import (
    BlockValidationError,
    collect_list_fields,
    collect_variable_paths,
    validate_blocks,
)
from modules.documents.variables.catalog import DYNAMIC_PREFIX

STARTER_CREATOR = "system"


def social_variable_names(blocks: Optional[dict]) -> List[str]:
    """A social template's own variables (what ``generate_document`` fills), in declared order."""
    schema = blocks.get("variables_schema") if isinstance(blocks, dict) else None
    return [str(name) for name in schema] if isinstance(schema, dict) else []


def variable_paths_of(blocks: Optional[dict]) -> List[str]:
    """Sorted, de-duplicated variable paths referenced by a block body ([] when none)."""
    if not blocks:
        return []
    try:
        doc = validate_blocks(blocks)
    except BlockValidationError:
        return []
    return sorted(set(collect_variable_paths(doc)))


def list_fields_of(blocks: Optional[dict]) -> List[Dict[str, Any]]:
    """The ``data.*`` LIST fields (``data_table`` blocks) with their column keys ([] when none)."""
    if not blocks:
        return []
    try:
        return collect_list_fields(validate_blocks(blocks))
    except BlockValidationError:
        return []


def data_fields_of(paths: List[str]) -> List[str]:
    """The ``data.*`` field names (without the prefix) an author/agent must supply."""
    return [p[len(DYNAMIC_PREFIX):] for p in paths if p.startswith(DYNAMIC_PREFIX)]


def summarize_template(t: Any) -> Dict[str, Any]:
    """The gallery/list entry for one template row (or row-shaped object)."""
    blocks = getattr(t, "blocks", None)
    social = is_social_format(getattr(t, "format", None))
    if social:
        names = social_variable_names(blocks)
        paths = [f"{DYNAMIC_PREFIX}{name}" for name in names]
    else:
        paths = variable_paths_of(blocks)
    created_at = getattr(t, "created_at", None)
    updated_at = getattr(t, "updated_at", None)
    return {
        "id": str(getattr(t, "id", "")),
        "name": getattr(t, "name", ""),
        "description": getattr(t, "description", None),
        "format": getattr(t, "format", "pdf"),
        "category": getattr(t, "category", "general"),
        "tags": list(getattr(t, "tags", None) or []),
        "version": getattr(t, "version", 1),
        "data_schema": getattr(t, "data_schema", None),
        "sample_data": getattr(t, "sample_data", None),
        # A social composition is not a block tree: the block editor never opens it.
        "has_blocks": bool(blocks) and not social,
        "is_starter": (getattr(t, "created_by", None) or "") == STARTER_CREATOR,
        "variable_paths": paths,
        "data_fields": data_fields_of(paths),
        "list_fields": [] if social else list_fields_of(blocks),
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }


__all__ = [
    "STARTER_CREATOR",
    "data_fields_of",
    "list_fields_of",
    "social_variable_names",
    "summarize_template",
    "variable_paths_of",
]
