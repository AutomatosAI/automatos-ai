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
        query = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.owner_type == 'workspace',
            WorkflowRecipe.workspace_id == ctx.workspace_id
        )

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
            WorkflowRecipe.owner_type == 'workspace',
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
    Create a new workflow recipe (simple or complex).

    Required fields:
    - template_id: Unique identifier (e.g., "my-custom-recipe")
    - name: Display name
    - description: Description of what the recipe does
    - category: Category (e.g., "Development", "Data Processing")
    - recipe_type: 'simple' or 'complex' (default: 'complex')

    For simple recipes (recipe_type='simple'):
    - agent_id: ID of agent to run the recipe
    - prompt: Prompt text for the agent
    - inputs: Array of {name, type, required, default}
    - schedule: Optional cron string

    For complex recipes (recipe_type='complex'):
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
        recipe_type = recipe_data.get('recipe_type', 'complex')
        required_fields = ['template_id', 'name', 'description', 'category']

        # Add type-specific required fields
        if recipe_type == 'simple':
            required_fields.extend(['agent_id', 'prompt'])
        else:  # complex
            required_fields.append('template_definition')

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

        # Create recipe - template_definition only required for complex recipes
        template_def = recipe_data.get('template_definition') if recipe_type != 'simple' else recipe_data.get('template_definition')

        recipe = WorkflowRecipe(
            workspace_id=ctx.workspace_id,
            template_id=recipe_data['template_id'],
            name=recipe_data['name'],
            description=recipe_data['description'],
            category=recipe_data['category'],
            recipe_type=recipe_type,
            template_definition=template_def,
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
            created_by=recipe_data.get('created_by', ctx.user.email if ctx.user else "anonymous")
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
            WorkflowRecipe.owner_type == 'workspace',
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
            WorkflowRecipe.owner_type == 'workspace',
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
            WorkflowRecipe.owner_type == 'workspace',
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
            WorkflowRecipe.owner_type == 'workspace',
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
            WorkflowRecipe.owner_type == 'workspace',
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


# ===================================================================
# MARKETPLACE ENDPOINTS
# ===================================================================

@router.post("/submit")
async def submit_recipe_to_marketplace(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    recipe_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Submit a workspace recipe to the marketplace for approval.
    Trusted users (5+ approved items) auto-publish; others go to approval queue.

    Required fields:
    - recipe_id: ID of the workspace recipe to submit (template_id)
    - category: Optional marketplace category (uses recipe category if not provided)
    - icon: Optional marketplace icon URL
    """
    try:
        recipe_id = recipe_data.get('recipe_id')
        if not recipe_id:
            raise HTTPException(status_code=400, detail="recipe_id is required")

        # Get workspace recipe
        workspace_recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.template_id == recipe_id,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first()

        if not workspace_recipe:
            raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found in workspace")

        # Look up database user ID
        from core.models.core import User as UserModel
        user_id_int = None
        if ctx.user and ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Check if user is trusted (5+ approved marketplace items - agents OR recipes)
        from core.models.core import Agent
        approved_agent_count = db.query(Agent).filter(
            Agent.original_creator_id == user_id_int,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).count() if user_id_int else 0

        approved_recipe_count = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.original_creator_id == user_id_int,
            WorkflowRecipe.owner_type == 'marketplace',
            WorkflowRecipe.is_approved == True
        ).count() if user_id_int else 0

        total_approved = approved_agent_count + approved_recipe_count
        is_trusted = total_approved >= 5

        logger.info(f"User approval status - User ID: {user_id_int}, Approved items: {total_approved}, Is trusted: {is_trusted}")

        # Check if recipe already exists in marketplace
        existing = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.name == workspace_recipe.name,
            WorkflowRecipe.owner_type == 'marketplace'
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"A marketplace recipe with name '{workspace_recipe.name}' already exists"
            )

        # Clone to marketplace
        marketplace_recipe = WorkflowRecipe(
            template_id=f"marketplace-{workspace_recipe.template_id}-{datetime.now().timestamp()}",
            name=workspace_recipe.name,
            description=workspace_recipe.description,
            category=workspace_recipe.category,
            recipe_type=workspace_recipe.recipe_type,
            template_definition=workspace_recipe.template_definition,
            tags=workspace_recipe.tags,
            difficulty=workspace_recipe.difficulty,
            recommended_agents=workspace_recipe.recommended_agents,
            estimated_time=workspace_recipe.estimated_time,
            required_tools=workspace_recipe.required_tools,
            icon=workspace_recipe.icon,
            preview_image=workspace_recipe.preview_image,
            documentation_url=workspace_recipe.documentation_url,
            version=workspace_recipe.version or '1.0',

            # Marketplace ownership
            owner_type='marketplace',
            owner_id='marketplace',
            workspace_id=None,

            # Creator tracking
            original_creator_id=user_id_int,
            created_by_user_id=user_id_int,
            cloned_from_id=workspace_recipe.id,

            # Approval
            is_approved=is_trusted,
            marketplace_category=recipe_data.get('category') or workspace_recipe.category,
            marketplace_icon=recipe_data.get('icon') or workspace_recipe.icon,

            # Visibility
            is_public=True,
            is_featured=False,
            is_system=False,

            # Stats
            install_count=0,
            use_count=0,

            created_by=(ctx.user.email if ctx.user and ctx.user.email else "system")
        )

        db.add(marketplace_recipe)
        db.commit()
        db.refresh(marketplace_recipe)

        logger.info(f"Marketplace recipe created - ID: {marketplace_recipe.id}, Name: {marketplace_recipe.name}, Approved: {marketplace_recipe.is_approved}")

        message = "Recipe published to marketplace successfully" if is_trusted else "Recipe submitted for marketplace approval"

        return {
            "success": True,
            "message": message,
            "item_id": marketplace_recipe.id,
            "auto_approved": is_trusted
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting recipe to marketplace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting recipe: {str(e)}")


@router.post("/install/{recipe_id}")
async def install_recipe_from_marketplace(
    recipe_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Install a marketplace recipe to the user's workspace.
    Automatically clones the recipe and handles name collisions.
    Optionally auto-clones referenced agents if available in marketplace.
    """
    try:
        # Get the marketplace recipe
        marketplace_recipe = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.id == recipe_id,
            WorkflowRecipe.owner_type == 'marketplace',
            WorkflowRecipe.is_approved == True
        ).first()

        if not marketplace_recipe:
            raise HTTPException(status_code=404, detail="Marketplace recipe not found")

        cloned_items = []
        warnings = []

        # Look up database user ID
        from core.models.core import User as UserModel
        user_id_int = None
        if ctx.user and ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Check if recipe name already exists in workspace
        name_exists = db.query(WorkflowRecipe).filter(
            WorkflowRecipe.name == marketplace_recipe.name,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first() is not None

        recipe_name = f"{marketplace_recipe.name} (Copy)" if name_exists else marketplace_recipe.name

        # Generate unique template_id
        base_template_id = marketplace_recipe.template_id.replace('marketplace-', '')
        template_id = base_template_id
        counter = 1
        while db.query(WorkflowRecipe).filter(
            WorkflowRecipe.template_id == template_id,
            WorkflowRecipe.workspace_id == ctx.workspace_id,
            WorkflowRecipe.owner_type == 'workspace'
        ).first():
            template_id = f"{base_template_id}-{counter}"
            counter += 1

        # Clone recipe to workspace
        cloned_recipe = WorkflowRecipe(
            template_id=template_id,
            name=recipe_name,
            description=marketplace_recipe.description,
            category=marketplace_recipe.category,
            recipe_type=marketplace_recipe.recipe_type,
            template_definition=marketplace_recipe.template_definition,
            tags=marketplace_recipe.tags,
            difficulty=marketplace_recipe.difficulty,
            recommended_agents=marketplace_recipe.recommended_agents,
            estimated_time=marketplace_recipe.estimated_time,
            required_tools=marketplace_recipe.required_tools,
            icon=marketplace_recipe.icon,
            preview_image=marketplace_recipe.preview_image,
            documentation_url=marketplace_recipe.documentation_url,
            version=marketplace_recipe.version,

            # Ownership swap
            owner_type='workspace',
            owner_id=str(ctx.workspace_id),
            workspace_id=ctx.workspace_id,
            created_by_user_id=user_id_int,

            # Tracking
            cloned_from_id=marketplace_recipe.id,
            original_creator_id=marketplace_recipe.original_creator_id,

            # Visibility
            is_public=True,
            is_featured=False,
            is_system=False,
            is_approved=True,

            # Stats
            install_count=0,
            use_count=0,

            created_by=(ctx.user.email if ctx.user and ctx.user.email else "system")
        )

        db.add(cloned_recipe)
        db.flush()

        cloned_items.append({
            "type": "recipe",
            "name": recipe_name,
            "id": cloned_recipe.id,
            "template_id": cloned_recipe.template_id
        })

        # TODO: Auto-clone referenced agents if available in marketplace
        # This would require parsing template_definition and checking for agent references

        # Increment marketplace recipe install count
        marketplace_recipe.install_count += 1

        # Record installation in marketplace_installs using a savepoint so
        # failures don't roll back the main recipe install.
        from sqlalchemy import text
        install_query = text("""
            INSERT INTO marketplace_installs (user_id, marketplace_recipe_id, cloned_recipe_id, version, installed_at)
            VALUES (:user_id, :marketplace_recipe_id, :cloned_recipe_id, :version, NOW())
            ON CONFLICT DO NOTHING
        """)

        try:
            with db.begin_nested():
                db.execute(install_query, {
                    "user_id": user_id_int,
                    "marketplace_recipe_id": marketplace_recipe.id,
                    "cloned_recipe_id": cloned_recipe.id,
                    "version": marketplace_recipe.version
                })
        except Exception as e:
            # If marketplace_installs table doesn't have recipe columns yet, log warning
            logger.warning(f"Could not record recipe install in marketplace_installs: {e}")
            warnings.append("Install tracking not available for recipes yet")

        db.commit()

        return {
            "success": True,
            "message": f"{marketplace_recipe.name} installed successfully",
            "cloned_items": cloned_items,
            "warnings": warnings
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error installing marketplace recipe {recipe_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error installing recipe: {str(e)}")
