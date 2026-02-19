"""
Template CRUD service for document generation (PRD-63).
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import DocumentTemplate

logger = logging.getLogger(__name__)


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
    ) -> DocumentTemplate:
        """Create a new document template."""
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
        )
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        logger.info(f"Created template '{name}' (format={format}) for workspace {workspace_id}")
        return template

    def get_template(self, template_id: UUID) -> Optional[DocumentTemplate]:
        """Get a template by ID."""
        return self.db.query(DocumentTemplate).filter(
            DocumentTemplate.id == template_id,
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

    def update_template(self, template_id: UUID, **kwargs) -> Optional[DocumentTemplate]:
        """Update specified fields on a template."""
        template = self.get_template(template_id)
        if not template:
            return None
        for key, value in kwargs.items():
            if hasattr(template, key) and key not in ("id", "workspace_id", "created_at"):
                setattr(template, key, value)
        template.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(template)
        return template

    def delete_template(self, template_id: UUID) -> bool:
        """Soft-delete a template by setting is_active=False."""
        template = self.get_template(template_id)
        if not template:
            return False
        template.is_active = False
        template.updated_at = datetime.utcnow()
        self.db.commit()
        return True
