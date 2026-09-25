"""
Template CRUD service for document generation (PRD-63).

PRD-251 S1.2 (D4): a ``social_image`` / ``social_video`` template is checked on
create and update here (the API's ``format`` field is untyped, so the service is
where the rule lives): its ``blocks`` must be a composition
``{html, css, variables_schema, sizes, audio_plan}`` whose variables_schema
declares every variable it uses, and it may not hardcode a colour, a font or a
logo (``core/social_templates.py``). A template that breaks the contract raises
:class:`~core.social_templates.SocialTemplateError` with every problem named.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import DOCUMENT_TEMPLATE_FORMATS, DocumentTemplate
from core.social_templates import is_social_format, validate_social_blocks

logger = logging.getLogger(__name__)


class UnknownTemplateFormat(ValueError):
    """The format is none of DOCUMENT_TEMPLATE_FORMATS."""


def checked_blocks(format: str, blocks: Any) -> Any:
    """``blocks`` as a template of ``format`` stores them: a social template's
    composition checked (variables_schema, sizes, the brand rule), any other
    format's unchanged (the API checks PRD-167 block trees)."""
    if format not in DOCUMENT_TEMPLATE_FORMATS:
        raise UnknownTemplateFormat(f"format must be one of {list(DOCUMENT_TEMPLATE_FORMATS)}")
    return validate_social_blocks(blocks, format) if is_social_format(format) else blocks


class DocumentTemplateService:
    """CRUD operations for document templates."""

    def __init__(self, db: Session):
        self.db = db

    def create_template(
        self,
        workspace_id: UUID,
        name: str,
        format: str,
        description: str = None,
        template_content: str = None,
        template_file_path: str = None,
        data_schema: dict = None,
        sample_data: dict = None,
        category: str = "general",
        tags: list = None,
        created_by: str = None,
        blocks: dict = None,
    ) -> DocumentTemplate:
        """Create a new document template.

        ``blocks`` (PRD-167 S2) is the canonical block-tree body; when present it is the
        render source of truth. For a social format it is the composition, checked
        (``checked_blocks``: html, css, variables_schema, sizes, audio_plan) before
        anything is written.
        """
        template = DocumentTemplate(
            workspace_id=workspace_id,
            name=name,
            description=description,
            format=format,
            template_content=template_content,
            template_file_path=template_file_path,
            data_schema=data_schema or {},
            sample_data=sample_data or {},
            category=category,
            tags=tags or [],
            created_by=created_by,
            blocks=checked_blocks(format, blocks),
        )
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        logger.info(f"Created template '{name}' (format={format}) for workspace {workspace_id}")
        return template

    def get_template(self, template_id: UUID, workspace_id: UUID) -> Optional[DocumentTemplate]:
        """Get a template by ID, scoped to its workspace.

        PRD-156 S4: confirmed cross-workspace IDOR — callers MUST pass the
        caller's workspace_id; another workspace's template returns None (the
        endpoints then 404), so it can't be read/updated/deleted across tenants.
        """
        return self.db.query(DocumentTemplate).filter(
            DocumentTemplate.id == template_id,
            DocumentTemplate.workspace_id == workspace_id,
            DocumentTemplate.is_active == True,
        ).first()

    def get_template_by_name(self, workspace_id: UUID, name: str) -> Optional[DocumentTemplate]:
        """Get the latest active version of a named template."""
        return (
            self.db.query(DocumentTemplate)
            .filter(
                DocumentTemplate.workspace_id == workspace_id,
                DocumentTemplate.name == name,
                DocumentTemplate.is_active == True,
            )
            .order_by(DocumentTemplate.version.desc())
            .first()
        )

    def list_templates(
        self,
        workspace_id: UUID,
        format: str = None,
        category: str = None,
    ) -> List[DocumentTemplate]:
        """List active templates for a workspace, optionally filtered."""
        query = self.db.query(DocumentTemplate).filter(
            DocumentTemplate.workspace_id == workspace_id,
            DocumentTemplate.is_active == True,
        )
        if format:
            query = query.filter(DocumentTemplate.format == format)
        if category:
            query = query.filter(DocumentTemplate.category == category)
        return query.order_by(DocumentTemplate.name, DocumentTemplate.version.desc()).all()

    def update_template(self, template_id: UUID, workspace_id: UUID, **kwargs) -> Optional[DocumentTemplate]:
        """Update specified fields on a template (PRD-156 S4: workspace-scoped).

        A social template's new ``blocks`` (or a change of format) is checked like
        a create; nothing is written when it breaks the contract.
        """
        template = self.get_template(template_id, workspace_id)
        if not template:
            return None
        if "blocks" in kwargs or "format" in kwargs:
            kwargs = {
                **kwargs,
                "blocks": checked_blocks(kwargs.get("format", template.format), kwargs.get("blocks", template.blocks)),
            }
        for key, value in kwargs.items():
            if hasattr(template, key) and key not in ("id", "workspace_id", "created_at"):
                setattr(template, key, value)
        template.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(template)
        return template

    def delete_template(self, template_id: UUID, workspace_id: UUID) -> bool:
        """Soft-delete a template by setting is_active=False (PRD-156 S4: workspace-scoped)."""
        template = self.get_template(template_id, workspace_id)
        if not template:
            return False
        template.is_active = False
        template.updated_at = datetime.utcnow()
        self.db.commit()
        return True
