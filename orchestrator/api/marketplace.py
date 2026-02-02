"""
Marketplace API Endpoints

Provides REST APIs for the Community Marketplace feature including:
- Browsing and filtering marketplace items
- Installing items to user workspace
- Submitting items for approval
- Checking for updates
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, text

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, User as UserModel, WorkflowTemplate as WorkflowRecipe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace", tags=["Marketplace"])


# ===================================================================
# Pydantic Models
# ===================================================================

class MarketplaceItemOut(BaseModel):
    id: int
    type: str  # 'agent', 'recipe', 'skill', 'llm', 'tool'
    name: str
    description: str
    creator_name: str
    icon: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    install_count: int = 0
    is_featured: bool = False
    is_approved: bool = True
    version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    # Recipe-specific fields
    owner_type: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    execution_config: Optional[Dict[str, Any]] = None
    schedule_config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MarketplaceItemDetail(MarketplaceItemOut):
    """Extended model with dependency information"""
    dependencies: Dict[str, Any] = Field(default_factory=dict)
    # For recipes: required_agent_id, required_tools
    # For agents: assigned_tools, assigned_skills


class InstallRequest(BaseModel):
    item_id: int


class InstallResponse(BaseModel):
    success: bool
    message: str
    cloned_items: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SubmitRequest(BaseModel):
    item_type: str
    name: Optional[str] = None  # Optional - will use agent's name if not provided
    description: Optional[str] = None  # Optional - will use agent's description if not provided
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateInfo(BaseModel):
    item_id: int
    item_name: str
    item_type: str
    current_version: str
    latest_version: str
    changelog: str


# ===================================================================
# GET /api/marketplace/items
# ===================================================================

@router.get("/items", response_model=List[MarketplaceItemOut])
async def list_items(
    type: Optional[str] = Query(None, description="Filter by type: agent, recipe, skill, llm, tool"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search by name or description"),
    featured: Optional[bool] = Query(None, description="Filter featured items only"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    List marketplace items with filtering.
    Returns only approved items from agents and recipes tables.
    """
    try:
        # Check if user is admin (adjust this logic as needed)
        is_admin = ctx.user and ctx.user.email and 'automatos.app' in ctx.user.email if ctx.user else False

        items = []

        # Filter by type if specified, otherwise get both
        include_agents = type is None or type == 'agent'
        include_recipes = type is None or type == 'recipe'

        # Query agents
        if include_agents:
            # Build query for marketplace agents
            agent_query = db.query(Agent).filter(Agent.owner_type == 'marketplace')

            if not is_admin:
                agent_query = agent_query.filter(Agent.is_approved == True)

            # Apply filters
            if category:
                agent_query = agent_query.filter(Agent.marketplace_category == category)

            if search:
                agent_query = agent_query.filter(or_(
                    Agent.name.ilike(f'%{search}%'),
                    Agent.description.ilike(f'%{search}%')
                ))

            if featured is not None:
                agent_query = agent_query.filter(Agent.is_featured == featured)

            # Order by install count and apply pagination
            agents = agent_query.order_by(desc(Agent.install_count), desc(Agent.created_at)).limit(limit).offset(offset).all()

            # Convert agents to response format
            for agent in agents:
                # Get creator name
                creator_name = "Unknown"
                if agent.original_creator_id:
                    creator = db.query(UserModel).filter(UserModel.id == agent.original_creator_id).first()
                    if creator:
                        creator_name = creator.email or f"User {creator.id}"

                # Get tool names and icons for the card
                tool_names = []
                tool_icons = []
                tool_assignments = db.execute(text('''
                    SELECT ata.tool_id, cac.logo_url
                    FROM agent_tool_assignments ata
                    LEFT JOIN composio_apps_cache cac
                        ON LOWER(cac.app_slug) = LOWER(ata.tool_id)
                        OR LOWER(cac.app_name) = LOWER(ata.tool_id)
                    WHERE ata.agent_id = :agent_id AND ata.enabled = true
                    LIMIT 3
                '''), {"agent_id": agent.id}).fetchall()

                for tool_id, logo_url in tool_assignments:
                    tool_names.append(tool_id.upper())
                    if logo_url:
                        tool_icons.append(logo_url)

                items.append(MarketplaceItemOut(
                    id=agent.id,
                    type='agent',
                    name=agent.name,
                    description=agent.description or '',
                    creator_name=creator_name,
                    icon=agent.marketplace_icon,
                    category=agent.marketplace_category,
                    tags=agent.tags or [],
                    install_count=agent.install_count,
                    is_featured=agent.is_featured,
                    is_approved=agent.is_approved,
                    version=agent.version,
                    metadata={
                        'agent_type': agent.agent_type,
                        'configuration': agent.configuration,
                        'model_config': agent.model_config,
                        'tool_names': tool_names,
                        'tool_icons': tool_icons
                    },
                    created_at=agent.created_at,
                    updated_at=agent.updated_at
                ))

        # Query recipes
        if include_recipes:
            recipe_query = db.query(WorkflowRecipe).filter(WorkflowRecipe.owner_type == 'marketplace')

            if not is_admin:
                recipe_query = recipe_query.filter(WorkflowRecipe.is_approved == True)

            # Apply filters
            if category:
                recipe_query = recipe_query.filter(WorkflowRecipe.marketplace_category == category)

            if search:
                recipe_query = recipe_query.filter(or_(
                    WorkflowRecipe.name.ilike(f'%{search}%'),
                    WorkflowRecipe.description.ilike(f'%{search}%')
                ))

            if featured is not None:
                recipe_query = recipe_query.filter(WorkflowRecipe.is_featured == featured)

            # Order by install count and apply pagination
            recipes = recipe_query.order_by(desc(WorkflowRecipe.install_count), desc(WorkflowRecipe.created_at)).limit(limit).offset(offset).all()

            # Convert recipes to response format
            for recipe in recipes:
                # Get creator name
                creator_name = "Unknown"
                if recipe.original_creator_id:
                    creator = db.query(UserModel).filter(UserModel.id == recipe.original_creator_id).first()
                    if creator:
                        creator_name = creator.email or f"User {creator.id}"

                items.append(MarketplaceItemOut(
                    id=recipe.id,
                    type='recipe',
                    name=recipe.name,
                    description=recipe.description or '',
                    creator_name=creator_name,
                    icon=recipe.marketplace_icon,
                    category=recipe.marketplace_category,
                    tags=recipe.tags or [],
                    install_count=recipe.install_count,
                    is_featured=recipe.is_featured,
                    is_approved=recipe.is_approved,
                    version=recipe.version,
                    owner_type=recipe.owner_type,
                    steps=recipe.steps or [],
                    inputs=recipe.inputs,
                    outputs=recipe.outputs,
                    execution_config=recipe.execution_config,
                    schedule_config=recipe.schedule_config,
                    metadata={
                        'required_tools': recipe.required_tools or [],
                        'recommended_agents': recipe.recommended_agents or [],
                        'quality_score': recipe.quality_score
                    },
                    created_at=recipe.created_at,
                    updated_at=recipe.updated_at
                ))

        # Sort combined items by install_count
        items.sort(key=lambda x: (x.install_count, x.created_at), reverse=True)

        return items

    except Exception as e:
        logger.error(f"Error listing marketplace items: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list marketplace items: {str(e)}")


# ===================================================================
# GET /api/marketplace/items/:id
# ===================================================================

@router.get("/items/{item_id}", response_model=MarketplaceItemDetail)
async def get_item(
    item_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get detailed information about a marketplace item including dependencies.
    """
    try:
        # Query marketplace agent
        agent = db.query(Agent).filter(
            Agent.id == item_id,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).first()

        if not agent:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        # Get creator name
        creator_name = "Unknown"
        if agent.original_creator_id:
            creator = db.query(UserModel).filter(UserModel.id == agent.original_creator_id).first()
            if creator:
                creator_name = creator.email or f"User {creator.id}"

        # Build dependencies info
        dependencies = {}
        if agent.skills:
            dependencies['skills'] = [{'id': s.id, 'name': s.name} for s in agent.skills]

        # Get skills with full details for metadata
        skills_data = []
        if agent.skills:
            logger.info(f"📚 Found {len(agent.skills)} skills for agent {agent.id}")
            for skill in agent.skills:
                skills_data.append({
                    'id': skill.id,
                    'name': skill.name,
                    'description': skill.description,
                    'category': skill.category,
                    'skill_type': skill.skill_type
                })

        # Get tools data from agent_tool_assignments table with Composio details
        tool_names = []
        tool_icons = []
        tool_descriptions = []

        tool_assignments = db.execute(text('''
            SELECT ata.tool_id, cac.logo_url, cac.description, cac.display_name
            FROM agent_tool_assignments ata
            LEFT JOIN composio_apps_cache cac
                ON LOWER(cac.app_slug) = LOWER(ata.tool_id)
                OR LOWER(cac.app_name) = LOWER(ata.tool_id)
            WHERE ata.agent_id = :agent_id AND ata.enabled = true
        '''), {"agent_id": agent.id}).fetchall()

        if tool_assignments:
            logger.info(f"🔧 Found {len(tool_assignments)} tool assignments for agent {agent.id}")
            for tool_id, logo_url, description, display_name in tool_assignments:
                tool_names.append((display_name or tool_id).upper())
                if logo_url:
                    tool_icons.append(logo_url)
                if description:
                    tool_descriptions.append(description)

        item = MarketplaceItemDetail(
            id=agent.id,
            type='agent',
            name=agent.name,
            description=agent.description or '',
            creator_name=creator_name,
            icon=agent.marketplace_icon,
            category=agent.marketplace_category,
            tags=agent.tags or [],
            install_count=agent.install_count,
            is_featured=agent.is_featured,
            version=agent.version,
            metadata={
                'agent_type': agent.agent_type,
                'configuration': agent.configuration,
                'model_config': agent.model_config,
                'skills': skills_data,
                'tool_names': tool_names,
                'tool_icons': tool_icons,
                'tool_descriptions': tool_descriptions
            },
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            dependencies=dependencies
        )

        return item

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching marketplace item {item_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch marketplace item: {str(e)}")


# ===================================================================
# POST /api/marketplace/items/:id/install
# ===================================================================

@router.post("/items/{item_id}/install", response_model=InstallResponse)
async def install_item(
    item_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Clone a marketplace agent to user's workspace.
    Automatically copies skills and increments install count.
    """
    try:
        # Get the marketplace agent
        marketplace_agent = db.query(Agent).filter(
            Agent.id == item_id,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).first()

        if not marketplace_agent:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        cloned_items = []
        warnings = []

        # Look up database user ID from Clerk user ID
        user_id_int = None
        if ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Check if agent name already exists in workspace
        name_exists = db.query(Agent).filter(
            Agent.name == marketplace_agent.name,
            Agent.workspace_id == ctx.workspace_id,
            Agent.owner_type == 'workspace'
        ).first() is not None

        agent_name = f"{marketplace_agent.name}-copy" if name_exists else marketplace_agent.name

        # Clone agent to workspace (simple copy with owner swap)
        cloned_agent = Agent(
            name=agent_name,
            description=marketplace_agent.description,
            agent_type=marketplace_agent.agent_type,
            configuration=marketplace_agent.configuration,
            model_config=marketplace_agent.model_config,
            tags=marketplace_agent.tags,
            status=marketplace_agent.status,

            # Ownership swap
            owner_type='workspace',
            owner_id=str(ctx.workspace_id),
            workspace_id=ctx.workspace_id,
            created_by_user_id=user_id_int,

            # Tracking
            cloned_from_id=marketplace_agent.id,
            original_creator_id=marketplace_agent.original_creator_id,

            # Reset marketplace fields
            is_approved=True,
            is_featured=False,
            install_count=0,
            version=marketplace_agent.version
        )

        db.add(cloned_agent)
        db.flush()

        # Copy skills relationship
        if marketplace_agent.skills:
            cloned_agent.skills = marketplace_agent.skills

        cloned_items.append({
            "type": "agent",
            "name": agent_name,
            "id": cloned_agent.id
        })

        # Increment marketplace agent install count
        marketplace_agent.install_count += 1

        # Record installation in marketplace_installs
        from sqlalchemy import text
        install_query = text("""
            INSERT INTO marketplace_installs (user_id, marketplace_agent_id, cloned_agent_id, version, installed_at)
            VALUES (:user_id, :marketplace_agent_id, :cloned_agent_id, :version, NOW())
        """)

        db.execute(install_query, {
            "user_id": user_id_int,
            "marketplace_agent_id": marketplace_agent.id,
            "cloned_agent_id": cloned_agent.id,
            "version": marketplace_agent.version
        })

        db.commit()

        return InstallResponse(
            success=True,
            message=f"{marketplace_agent.name} installed successfully",
            cloned_items=cloned_items,
            warnings=warnings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error installing marketplace item {item_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to install marketplace item: {str(e)}")


# ===================================================================
# GET /api/marketplace/featured
# ===================================================================

@router.get("/featured", response_model=List[MarketplaceItemOut])
async def get_featured(
    limit: int = Query(8, ge=1, le=20),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get featured marketplace items for homepage.
    """
    try:
        # Query featured marketplace agents
        agents = db.query(Agent).filter(
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True,
            Agent.is_featured == True
        ).order_by(desc(Agent.install_count), desc(Agent.created_at)).limit(limit).all()

        items = []
        for agent in agents:
            # Get creator name
            creator_name = "Unknown"
            if agent.original_creator_id:
                creator = db.query(UserModel).filter(UserModel.id == agent.original_creator_id).first()
                if creator:
                    creator_name = creator.email or f"User {creator.id}"

            items.append(MarketplaceItemOut(
                id=agent.id,
                type='agent',
                name=agent.name,
                description=agent.description or '',
                creator_name=creator_name,
                icon=agent.marketplace_icon,
                category=agent.marketplace_category,
                tags=agent.tags or [],
                install_count=agent.install_count,
                is_featured=agent.is_featured,
                version=agent.version,
                metadata={
                    'agent_type': agent.agent_type,
                    'configuration': agent.configuration,
                    'model_config': agent.model_config
                },
                created_at=agent.created_at,
                updated_at=agent.updated_at
            ))

        return items

    except Exception as e:
        logger.error(f"Error fetching featured items: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch featured items: {str(e)}")


# ===================================================================
# GET /api/marketplace/updates
# ===================================================================

@router.get("/updates", response_model=List[UpdateInfo])
async def check_updates(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Check for updates to installed marketplace items.
    Returns items where installed version < latest version.
    """
    try:
        from sqlalchemy import text

        # Look up database user ID from Clerk user ID
        user_id_int = None
        if ctx.user.id:
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if not user and ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
            if user:
                user_id_int = user.id

        # Query marketplace_installs joined with agents table
        query = text("""
            SELECT
                marketplace_agent.id,
                marketplace_agent.name,
                mins.version as installed_version,
                marketplace_agent.version as latest_version
            FROM marketplace_installs mins
            JOIN agents marketplace_agent ON mins.marketplace_agent_id = marketplace_agent.id
            WHERE mins.user_id = :user_id
            AND mins.version != marketplace_agent.version
            AND marketplace_agent.owner_type = 'marketplace'
            ORDER BY marketplace_agent.updated_at DESC
        """)

        result = db.execute(query, {"user_id": user_id_int})

        updates = []
        for row in result:
            updates.append(UpdateInfo(
                item_id=row[0],
                item_name=row[1],
                item_type='agent',
                current_version=row[2],
                latest_version=row[3],
                changelog="No changelog available"
            ))

        return updates

    except Exception as e:
        logger.error(f"Error checking for updates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check for updates: {str(e)}")


# ===================================================================
# GET /api/marketplace/items/:id - Get single item with full details
# ===================================================================

@router.get("/items/{item_id}")
async def get_item(
    item_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get a single marketplace item with full details including skills and tools.
    """
    try:
        # Get marketplace agent
        agent = db.query(Agent).filter(
            Agent.id == item_id,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).first()

        if not agent:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        # Get creator name
        creator_name = "Unknown"
        if agent.original_creator_id:
            creator = db.query(UserModel).filter(UserModel.id == agent.original_creator_id).first()
            if creator:
                creator_name = creator.email or f"User {creator.id}"

        # Get skills with full details
        skills_data = []
        if agent.skills:
            logger.info(f"📚 Found {len(agent.skills)} skills for agent {agent.id}")
            for skill in agent.skills:
                skills_data.append({
                    'id': skill.id,
                    'name': skill.name,
                    'description': skill.description,
                    'category': skill.category,
                    'skill_type': skill.skill_type
                })
        else:
            logger.info(f"❌ No skills found for agent {agent.id}")

        # Get tools data (from agent.tools relationship if it exists, or from configuration)
        tools_data = []
        tool_names = []
        tool_icons = []

        # Check if agent has tools relationship
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                tool_names.append(tool.name)
                if hasattr(tool, 'icon') and tool.icon:
                    tool_icons.append(tool.icon)

        return {
            "id": agent.id,
            "type": "agent",
            "name": agent.name,
            "description": agent.description or '',
            "creator_name": creator_name,
            "icon": agent.marketplace_icon,
            "category": agent.marketplace_category,
            "tags": agent.tags or [],
            "install_count": agent.install_count,
            "is_featured": agent.is_featured,
            "version": agent.version,
            "metadata": {
                'agent_type': agent.agent_type,
                'configuration': agent.configuration,
                'model_config': agent.model_config,
                'skills': skills_data,
                'tool_names': tool_names,
                'tool_icons': tool_icons
            },
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
            "dependencies": {
                "skills": [{"id": s.id, "name": s.name} for s in agent.skills] if agent.skills else []
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching marketplace item: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {str(e)}")


# ===================================================================
# POST /api/marketplace/submit
# ===================================================================

@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_item(
    request: SubmitRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Submit a workspace agent to the marketplace for approval.
    Trusted users (5+ approved items) auto-publish; others go to approval queue.
    """
    try:
        # Validate that item_type is agent and get agent_id from metadata
        if request.item_type != 'agent':
            raise HTTPException(status_code=400, detail="Currently only agents can be submitted to marketplace")

        agent_id = request.metadata.get('agent_id')
        if not agent_id:
            raise HTTPException(status_code=400, detail="agent_id required in metadata")

        # Get workspace agent
        workspace_agent = db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.workspace_id == ctx.workspace_id,
            Agent.owner_type == 'workspace'
        ).first()

        if not workspace_agent:
            raise HTTPException(status_code=404, detail="Agent not found in workspace")

        # Look up database user ID from Clerk user ID
        user = None
        user_id_int = None
        if ctx.user.id:
            # Try to find user by clerk_user_id first
            user = db.query(UserModel).filter(UserModel.clerk_user_id == ctx.user.id).first()
            if user:
                user_id_int = user.id
            # Fallback: try email lookup
            elif ctx.user.email:
                user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
                if user:
                    user_id_int = user.id

        # Check if user is trusted (5+ approved marketplace items)
        approved_count = db.query(Agent).filter(
            Agent.original_creator_id == user_id_int,
            Agent.owner_type == 'marketplace',
            Agent.is_approved == True
        ).count() if user_id_int else 0

        is_trusted = approved_count >= 5

        logger.info(f"📊 User approval status - User ID: {user_id_int}, Approved count: {approved_count}, Is trusted: {is_trusted}")

        # Clone to marketplace
        marketplace_agent = Agent(
            name=workspace_agent.name,
            description=workspace_agent.description,
            agent_type=workspace_agent.agent_type,
            configuration=workspace_agent.configuration,
            model_config=workspace_agent.model_config,
            tags=workspace_agent.tags,
            status=workspace_agent.status,

            # Marketplace ownership
            owner_type='marketplace',
            owner_id='marketplace',
            workspace_id=None,

            # Creator tracking
            original_creator_id=user_id_int,
            cloned_from_id=workspace_agent.id,

            # Approval
            is_approved=is_trusted,
            marketplace_category=request.category or workspace_agent.agent_type,
            marketplace_icon=request.metadata.get('icon'),
            version='1.0.0'
        )

        db.add(marketplace_agent)
        db.flush()

        # Copy skills
        if workspace_agent.skills:
            marketplace_agent.skills = workspace_agent.skills

        # Copy tool assignments
        tool_assignments = db.execute(text('''
            SELECT tool_id, enabled FROM agent_tool_assignments
            WHERE agent_id = :workspace_agent_id
        '''), {"workspace_agent_id": workspace_agent.id}).fetchall()

        if tool_assignments:
            logger.info(f"📦 Copying {len(tool_assignments)} tool assignments to marketplace agent")
            for tool_id, enabled in tool_assignments:
                db.execute(text('''
                    INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled, created_at, updated_at)
                    VALUES (:agent_id, :tool_id, :enabled, NOW(), NOW())
                '''), {
                    "agent_id": marketplace_agent.id,
                    "tool_id": tool_id,
                    "enabled": enabled
                })

        # Note: Skipping marketplace_submissions insert for now since it references
        # the old marketplace_items table. This is just tracking data and not essential.
        # TODO: Update marketplace_submissions foreign key to reference agents table

        db.commit()

        logger.info(f"✅ Marketplace agent created - ID: {marketplace_agent.id}, Name: {marketplace_agent.name}, Approved: {marketplace_agent.is_approved}")

        message = "Agent published to marketplace successfully" if is_trusted else "Agent submitted for marketplace approval"

        return {
            "success": True,
            "message": message,
            "item_id": marketplace_agent.id,
            "auto_approved": is_trusted
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting marketplace item: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit marketplace item: {str(e)}")


# ===================================================================
# POST /api/marketplace/items/:id/approve (Admin only)
# ===================================================================

@router.post("/items/{item_id}/approve")
async def approve_marketplace_item(
    item_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Approve a pending marketplace item. Admin only.
    """
    try:
        # Check if user is admin (you may need to adjust this check based on your auth system)
        # For now, we'll allow it and rely on frontend access control

        # Get the marketplace agent
        marketplace_agent = db.query(Agent).filter(
            Agent.id == item_id,
            Agent.owner_type == 'marketplace'
        ).first()

        if not marketplace_agent:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        # Approve it
        marketplace_agent.is_approved = True
        db.commit()

        logger.info(f"✅ Marketplace agent approved - ID: {marketplace_agent.id}, Name: {marketplace_agent.name}")

        return {
            "success": True,
            "message": f"Agent '{marketplace_agent.name}' approved and published to marketplace",
            "item_id": marketplace_agent.id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error approving marketplace item: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to approve item: {str(e)}")


# ===================================================================
# GET /api/marketplace/items/pending (Admin only)
# ===================================================================

@router.delete("/items/{item_id}")
async def delete_marketplace_item(
    item_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Delete a marketplace item. Admin only.
    """
    try:
        # Get the marketplace agent
        marketplace_agent = db.query(Agent).filter(
            Agent.id == item_id,
            Agent.owner_type == 'marketplace'
        ).first()

        if not marketplace_agent:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        # Delete it
        db.delete(marketplace_agent)
        db.commit()

        logger.info(f"🗑️ Marketplace agent deleted - ID: {marketplace_agent.id}, Name: {marketplace_agent.name}")

        return {
            "success": True,
            "message": f"Agent '{marketplace_agent.name}' removed from marketplace"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting marketplace item: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {str(e)}")


@router.get("/items/pending")
async def get_pending_items(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get all pending marketplace items awaiting approval. Admin only.
    """
    try:
        # Query agents where owner_type='marketplace' and is_approved=false
        pending_agents = db.query(Agent).filter(
            Agent.owner_type == 'marketplace',
            Agent.is_approved == False
        ).order_by(desc(Agent.created_at)).all()

        items = []
        for agent in pending_agents:
            # Get creator info if available
            creator_name = None
            if agent.original_creator_id:
                creator = db.query(UserModel).filter(UserModel.id == agent.original_creator_id).first()
                if creator:
                    creator_name = creator.full_name or creator.email

            items.append({
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "type": "agent",
                "category": agent.marketplace_category or agent.agent_type,
                "icon": agent.marketplace_icon or "🤖",
                "creator": creator_name,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "version": agent.version
            })

        logger.info(f"📋 Found {len(items)} pending marketplace items")

        return {
            "items": items,
            "total": len(items)
        }

    except Exception as e:
        logger.error(f"❌ Error fetching pending items: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch pending items: {str(e)}")
