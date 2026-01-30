"""
Workflow Recipes API
====================

CRUD operations for workflow recipes that users can browse,
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
router = APIRouter(prefix="/api/workflow-recipes", tags=["workflow-recipes"])

# Import the model from main models file
from core.models import WorkflowTemplate as WorkflowRecipe  # Aliased for transition
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext


@router.get("")
async def list_workflow_recipes(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    is_featured: Optional[bool] = None,
    is_public: Optional[bool] = True,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    sort_by: str = Query('popularity', regex='^(popularity|created_at|use_count|average_rating|name)$'),
    db: Session = Depends(get_db)
):
    """
    List workflow recipes with filtering and pagination.

    Query Parameters:
    - category: Filter by category (e.g., "Support", "Data Processing")
    - difficulty: Filter by difficulty (beginner, intermediate, advanced)
    - is_featured: Show only featured recipes
    - is_public: Show only public recipes (default: true)
    - search: Search in name and description
    - skip: Number of records to skip for pagination
    - limit: Maximum number of records to return (1-100)
    - sort_by: Sort field (popularity, created_at, use_count, average_rating, name)
    """
    try:
        query = db.query(WorkflowRecipe).filter(WorkflowRecipe.workspace_id == ctx.workspace_id)

        # Apply filters
        if category:
            query = query.filter(WorkflowRecipe.category == category)

        if difficulty:
            query = query.filter(WorkflowRecipe.difficulty == difficulty)

        if is_featured is not None:
            query = query.filter(WorkflowRecipe.is_featured == is_featured)

        if is_public is not None:
            query = query.filter(WorkflowRecipe.is_public == is_public)

        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    WorkflowRecipe.name.ilike(search_pattern),
                    WorkflowRecipe.description.ilike(search_pattern)
                )
            )

        # Get total count before pagination
        total = query.count()

        # Apply sorting
        if sort_by == 'popularity':
            query = query.order_by(WorkflowRecipe.popularity.desc())
        elif sort_by == 'created_at':
            query = query.order_by(WorkflowRecipe.created_at.desc())
        elif sort_by == 'use_count':
            query = query.order_by(WorkflowRecipe.use_count.desc())
        elif sort_by == 'average_rating':
            query = query.order_by(WorkflowRecipe.average_rating.desc())
        elif sort_by == 'name':
            query = query.order_by(WorkflowRecipe.name.asc())

        # Apply pagination
        recipes = query.offset(skip).limit(limit).all()

        return {
            "items": [recipe.to_dict() for recipe in recipes],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Error listing workflow recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing recipes: {str(e)}")


@router.get("/{recipe_id}")
async def get_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get a single workflow recipe by its template_id"""
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        return recipe.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recipe: {str(e)}")


@router.post("")
async def create_workflow_recipe(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create a new workflow recipe.

    Required fields:
    - template_id: Unique identifier (e.g., "my-custom-recipe")
    - name: Display name
    - description: Description of what the recipe does
    - category: Category (e.g., "Development", "Data Processing")
    - template_definition: JSON structure with steps, agents, config

    Optional fields:
    - tags: Array of tags
    - difficulty: beginner, intermediate, advanced (default: intermediate)
    - recommended_agents: Array of agent type names
    - estimated_time: e.g., "5-10 minutes"
    - required_tools: Array of tool names
    - is_public: Boolean (default: true)
    - is_featured: Boolean (default: false)
    - icon: Emoji or icon identifier
    """
    try:
        # Validate required fields
        required_fields = ['template_id', 'name', 'description', 'category', 'template_definition']
        for field in required_fields:
            if field not in recipe_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Check if template_id already exists
        existing = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_data['template_id']
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Recipe with ID '{recipe_data['template_id']}' already exists"
            )

        # Create recipe
        recipe = WorkflowRecipe(
            workspace_id=ctx.workspace_id,
            template_id=recipe_data['template_id'],
            name=recipe_data['name'],
            description=recipe_data['description'],
            category=recipe_data['category'],
            template_definition=recipe_data['template_definition'],
            tags=recipe_data.get('tags', []),
            difficulty=recipe_data.get('difficulty', 'intermediate'),
            recommended_agents=recipe_data.get('recommended_agents', []),
            estimated_time=recipe_data.get('estimated_time'),
            required_tools=recipe_data.get('required_tools', []),
            is_public=recipe_data.get('is_public', True),
            is_featured=recipe_data.get('is_featured', False),
            is_system=False,  # User-created recipes are never system recipes
            icon=recipe_data.get('icon'),
            preview_image=recipe_data.get('preview_image'),
            documentation_url=recipe_data.get('documentation_url'),
            version=recipe_data.get('version', '1.0'),
            created_by=recipe_data.get('created_by', ctx.user_email or f"user-{ctx.user_id}")
        )

        db.add(recipe)
        db.commit()
        db.refresh(recipe)

        logger.info(f"Created workflow recipe: {recipe.template_id}")

        return {
            "message": "Recipe created successfully",
            "recipe": recipe.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow recipe: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating recipe: {str(e)}")


@router.put("/{recipe_id}")
async def update_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update an existing workflow recipe.
    System recipes cannot be modified.
    """
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        if recipe.is_system:
            raise HTTPException(
                status_code=403,
                detail="System recipes cannot be modified"
            )

        # Update fields if provided
        updatable_fields = [
            'name', 'description', 'category', 'tags', 'difficulty',
            'template_definition', 'recommended_agents', 'estimated_time',
            'required_tools', 'is_public', 'is_featured', 'icon',
            'preview_image', 'documentation_url', 'version', 'changelog'
        ]

        for field in updatable_fields:
            if field in recipe_data:
                setattr(recipe, field, recipe_data[field])

        recipe.updated_at = datetime.now()
        db.commit()
        db.refresh(recipe)

        logger.info(f"Updated workflow recipe: {recipe_id}")

        return {
            "message": "Recipe updated successfully",
            "recipe": recipe.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating recipe {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating recipe: {str(e)}")


@router.delete("/{recipe_id}")
async def delete_workflow_recipe(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Delete a workflow recipe.
    System recipes cannot be deleted.
    """
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        if recipe.is_system:
            raise HTTPException(
                status_code=403,
                detail="System recipes cannot be deleted"
            )

        db.delete(recipe)
        db.commit()

        logger.info(f"Deleted workflow recipe: {recipe_id}")

        return {
            "message": "Recipe deleted successfully",
            "recipe_id": recipe_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recipe {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting recipe: {str(e)}")


@router.post("/{recipe_id}/use")
async def record_recipe_usage(
    recipe_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Record that a recipe was used to create a workflow.
    Updates use_count and last_used_at.
    """
    try:
        recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.template_id == recipe_id
        ).first()

        if not recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

        recipe.use_count += 1
        recipe.last_used_at = datetime.now()
        db.commit()

        return {
            "message": "Recipe usage recorded",
            "recipe_id": recipe_id,
            "use_count": recipe.use_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording recipe usage for {recipe_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error recording usage: {str(e)}")


@router.get("/categories/list")
async def list_recipe_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get list of all recipe categories with counts"""
    try:
        from sqlalchemy import func

        categories = db.query(
            WorkflowRecipe.category,
            func.count(WorkflowRecipe.id).label('count')
        ).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.is_public == True
        ).group_by(
            WorkflowRecipe.category
        ).all()

        return {
            "categories": [
                {"name": cat[0], "count": cat[1]}
                for cat in categories
            ]
        }

    except Exception as e:
        logger.error(f"Error listing recipe categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")


@router.get("/featured/list")
async def list_featured_recipes(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get featured workflow recipes"""
    try:
        recipes = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.is_featured == True,
            WorkflowRecipe.is_public == True
        ).order_by(
            WorkflowRecipe.popularity.desc()
        ).limit(limit).all()

        return {
            "items": [recipe.to_dict() for recipe in recipes],
            "total": len(recipes)
        }

    except Exception as e:
        logger.error(f"Error listing featured recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing featured recipes: {str(e)}")
