"""
Workflow Templates API
=====================

CRUD operations for workflow templates that users can browse,
customize, and use to create workflows.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.database.database import get_db
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflow-templates", tags=["workflow-templates"])

# Import the model from main models file
from core.models import WorkflowTemplate
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext


@router.get("")
async def list_workflow_templates(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    is_featured: Optional[bool] = None,
    is_public: Optional[bool] = True,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    sort_by: str = Query('popularity', regex='^(popularity|created_at|use_count|average_rating|name)$'),
    db: Session = Depends(get_db)
):
    """
    List workflow templates with filtering and pagination.

    Query Parameters:
    - is_featured: Show only featured templates
    - is_public: Show only public templates (default: true)
    - search: Search in name and description
    - skip: Number of records to skip for pagination
    - limit: Maximum number of records to return (1-100)
    - sort_by: Sort field (popularity, created_at, use_count, average_rating, name)
    """
    try:
        query = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id)

        # Apply filters
        
        if is_featured is not None:
            query = query.filter(WorkflowTemplate.is_featured == is_featured)
        
        if is_public is not None:
            query = query.filter(WorkflowTemplate.is_public == is_public)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    WorkflowTemplate.name.ilike(search_pattern),
                    WorkflowTemplate.description.ilike(search_pattern)
                )
            )
        
        # Get total count before pagination
        total = query.count()
        
        # Apply sorting
        if sort_by == 'popularity':
            query = query.order_by(WorkflowTemplate.popularity.desc())
        elif sort_by == 'created_at':
            query = query.order_by(WorkflowTemplate.created_at.desc())
        elif sort_by == 'use_count':
            query = query.order_by(WorkflowTemplate.use_count.desc())
        elif sort_by == 'average_rating':
            query = query.order_by(WorkflowTemplate.average_rating.desc())
        elif sort_by == 'name':
            query = query.order_by(WorkflowTemplate.name.asc())
        
        # Apply pagination
        templates = query.offset(skip).limit(limit).all()
        
        return {
            "items": [template.to_dict() for template in templates],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing workflow templates: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing templates: {str(e)}")


@router.get("/{template_id}")
async def get_workflow_template(
    template_id: str,
    db: Session = Depends(get_db)
):
    """Get a single workflow template by its template_id"""
    try:
        template = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            WorkflowTemplate.template_id == template_id
        ).first()
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        return template.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting template: {str(e)}")


@router.post("")
async def create_workflow_template(
    template_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create a new workflow template.

    Required fields:
    - template_id: Unique identifier (e.g., "my-custom-template")
    - name: Display name
    - description: Description of what the template does
    - template_definition: JSON structure with steps, agents, config

    Optional fields:
    - tags: Array of tags
    - recommended_agents: Array of agent type names
    - required_tools: Array of tool names
    - is_public: Boolean (default: true)
    - is_featured: Boolean (default: false)
    """
    try:
        # Validate required fields
        required_fields = ['template_id', 'name', 'description', 'template_definition']
        for field in required_fields:
            if field not in template_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Check if template_id already exists
        existing = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            WorkflowTemplate.template_id == template_data['template_id']
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Template with ID '{template_data['template_id']}' already exists"
            )
        
        # Create template
        template = WorkflowTemplate(
            template_id=template_data['template_id'],
            name=template_data['name'],
            description=template_data['description'],
            template_definition=template_data['template_definition'],
            tags=template_data.get('tags', []),
            recommended_agents=template_data.get('recommended_agents', []),
            required_tools=template_data.get('required_tools', []),
            is_public=template_data.get('is_public', True),
            is_featured=template_data.get('is_featured', False),
            is_system=False,  # User-created templates are never system templates
            preview_image=template_data.get('preview_image'),
            documentation_url=template_data.get('documentation_url'),
            version=template_data.get('version', '1.0'),
            created_by=template_data.get('created_by', 'user')
        )
        
        db.add(template)
        db.commit()
        db.refresh(template)
        
        logger.info(f"Created workflow template: {template.template_id}")
        
        return {
            "message": "Template created successfully",
            "template": template.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow template: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")


@router.put("/{template_id}")
async def update_workflow_template(
    template_id: str,
    template_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update an existing workflow template.
    System templates cannot be modified.
    """
    try:
        template = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            WorkflowTemplate.template_id == template_id
        ).first()
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        if template.is_system:
            raise HTTPException(
                status_code=403,
                detail="System templates cannot be modified"
            )
        
        # Update fields if provided
        updatable_fields = [
            'name', 'description', 'tags',
            'template_definition', 'recommended_agents',
            'required_tools', 'is_public', 'is_featured',
            'preview_image', 'documentation_url', 'version', 'changelog'
        ]
        
        for field in updatable_fields:
            if field in template_data:
                setattr(template, field, template_data[field])
        
        template.updated_at = datetime.now()
        db.commit()
        db.refresh(template)
        
        logger.info(f"Updated workflow template: {template_id}")
        
        return {
            "message": "Template updated successfully",
            "template": template.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")


@router.delete("/{template_id}")
async def delete_workflow_template(
    template_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a workflow template.
    System templates cannot be deleted.
    """
    try:
        template = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            WorkflowTemplate.template_id == template_id
        ).first()
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        if template.is_system:
            raise HTTPException(
                status_code=403,
                detail="System templates cannot be deleted"
            )
        
        db.delete(template)
        db.commit()
        
        logger.info(f"Deleted workflow template: {template_id}")
        
        return {
            "message": "Template deleted successfully",
            "template_id": template_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")


@router.post("/{template_id}/use")
async def record_template_usage(
    template_id: str,
    db: Session = Depends(get_db)
):
    """
    Record that a template was used to create a workflow.
    Updates use_count and last_used_at.
    """
    try:
        template = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            WorkflowTemplate.template_id == template_id
        ).first()
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        template.use_count += 1
        template.last_used_at = datetime.now()
        db.commit()
        
        return {
            "message": "Template usage recorded",
            "template_id": template_id,
            "use_count": template.use_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording template usage for {template_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error recording usage: {str(e)}")


@router.get("/categories/list")
async def list_template_categories(db: Session = Depends(get_db)):
    """Get list of all unique tags used across templates"""
    try:
        templates = db.query(WorkflowTemplate.tags).filter(
            WorkflowTemplate.is_public == True
        ).all()

        # Aggregate tags across all templates
        tag_counts: dict = {}
        for (tags,) in templates:
            if tags:
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "categories": [
                {"name": name, "count": count}
                for name, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }

    except Exception as e:
        logger.error(f"Error listing template categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")


@router.get("/featured/list")
async def list_featured_templates(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get featured workflow templates"""
    try:
        templates = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ctx.workspace_id).filter(
            and_(
                WorkflowTemplate.is_featured == True,
                WorkflowTemplate.is_public == True
            )
        ).order_by(
            WorkflowTemplate.popularity.desc()
        ).limit(limit).all()
        
        return {
            "items": [template.to_dict() for template in templates],
            "total": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Error listing featured templates: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing featured templates: {str(e)}")

