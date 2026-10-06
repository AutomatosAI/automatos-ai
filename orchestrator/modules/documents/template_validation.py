"""The one check a template's body passes before it is saved (PRD-255 US-013).

The studio's routes (``POST`` / ``PUT /api/documents/templates``) and the agents'
template tools (``platform_create_template`` / ``platform_update_template``) save
through the same rules, so a template an agent makes is one the studio would have
accepted:

* a document template's ``blocks`` must be a valid PRD-167 block tree (its blocks,
  its chips and the brand board's once-per-document parts), normalised as the
  schema writes it; malformed blocks are refused with field-level errors;
* a social template's composition is checked by the service
  (``template_service.checked_blocks``) on save, with every problem named.

Each caller turns the errors into its own answer: the routes a 422, the tools a
failure the agent reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.social_templates import SocialTemplateError, is_social_format
from modules.documents.template_service import UnknownTemplateFormat

# What the service's save can fail with: a social composition that breaks its
# contract, or a format none of the template formats (PRD-251 S1.2).
TEMPLATE_SAVE_ERRORS = (SocialTemplateError, UnknownTemplateFormat)
INVALID_BLOCKS = "Invalid blocks"
INVALID_TEMPLATE = "Invalid template"


def validated_blocks(format: str, blocks: Optional[Any]) -> Optional[Any]:
    """``blocks`` as a template of ``format`` saves them.

    A document template's block tree is validated and normalised; a social
    composition passes as sent (the service checks it); ``None`` stays ``None``.
    Raises :class:`~modules.documents.blocks.BlockValidationError` with field-level
    errors on a malformed tree.
    """
    if blocks is None or is_social_format(format):
        return blocks
    from modules.documents.blocks import validate_blocks

    return validate_blocks(blocks).model_dump()


def save_errors(e: ValueError) -> List[Dict[str, Any]]:
    """The field-level errors of a refused save (one of :data:`TEMPLATE_SAVE_ERRORS`)."""
    if isinstance(e, SocialTemplateError):
        return list(e.errors)
    return [{"field": "format", "message": str(e)}]


__all__ = ["INVALID_BLOCKS", "INVALID_TEMPLATE", "TEMPLATE_SAVE_ERRORS", "save_errors", "validated_blocks"]
