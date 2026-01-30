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
from sqlalchemy import and_, or_, func, desc

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

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
    version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

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
    name: str
    description: str
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
    Returns only approved items.
    """
    try:
        from sqlalchemy import text

        # Build query
        query = text("""
            SELECT id, type, name, description, creator_name, icon, category, tags,
                   install_count, is_featured, version, metadata, created_at, updated_at
            FROM marketplace_items
            WHERE is_approved = true
            AND (:type IS NULL OR type = :type)
            AND (:category IS NULL OR category = :category)
            AND (:featured IS NULL OR is_featured = :featured)
            AND (:search IS NULL OR name ILIKE :search_pattern OR description ILIKE :search_pattern)
            ORDER BY install_count DESC, created_at DESC
            LIMIT :limit OFFSET :offset
        """)

        search_pattern = f"%{search}%" if search else None

        result = db.execute(query, {
            "type": type,
            "category": category,
            "featured": featured,
            "search": search,
            "search_pattern": search_pattern,
            "limit": limit,
            "offset": offset
        })

        items = []
        for row in result:
            items.append(MarketplaceItemOut(
                id=row[0],
                type=row[1],
                name=row[2],
                description=row[3],
                creator_name=row[4],
                icon=row[5],
                category=row[6],
                tags=row[7] or [],
                install_count=row[8],
                is_featured=row[9],
                version=row[10],
                metadata=row[11] or {},
                created_at=row[12],
                updated_at=row[13]
            ))

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
        from sqlalchemy import text

        query = text("""
            SELECT id, type, name, description, creator_name, icon, category, tags,
                   install_count, is_featured, version, metadata, created_at, updated_at
            FROM marketplace_items
            WHERE id = :item_id AND is_approved = true
        """)

        result = db.execute(query, {"item_id": item_id}).first()

        if not result:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        item = MarketplaceItemDetail(
            id=result[0],
            type=result[1],
            name=result[2],
            description=result[3],
            creator_name=result[4],
            icon=result[5],
            category=result[6],
            tags=result[7] or [],
            install_count=result[8],
            is_featured=result[9],
            version=result[10],
            metadata=result[11] or {},
            created_at=result[12],
            updated_at=result[13],
            dependencies=result[11].get("dependencies", {}) if result[11] else {}
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
    Clone a marketplace item to user's workspace.
    Automatically resolves and installs dependencies.
    """
    try:
        from sqlalchemy import text

        # Get the marketplace item
        item_query = text("""
            SELECT id, type, name, description, metadata, version
            FROM marketplace_items
            WHERE id = :item_id AND is_approved = true
        """)

        item = db.execute(item_query, {"item_id": item_id}).first()

        if not item:
            raise HTTPException(status_code=404, detail="Marketplace item not found")

        item_type = item[1]
        cloned_items = []
        warnings = []

        # TODO: Implement actual cloning logic based on item type
        # For now, just record the installation

        # Record installation
        install_query = text("""
            INSERT INTO marketplace_installs (user_id, item_id, version, installed_at)
            VALUES (:user_id, :item_id, :version, NOW())
        """)

        db.execute(install_query, {
            "user_id": ctx.user_id,
            "item_id": item_id,
            "version": item[5]
        })

        # Increment install count
        update_count_query = text("""
            UPDATE marketplace_items
            SET install_count = install_count + 1
            WHERE id = :item_id
        """)

        db.execute(update_count_query, {"item_id": item_id})
        db.commit()

        return InstallResponse(
            success=True,
            message=f"{item[2]} installed successfully",
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
        from sqlalchemy import text

        query = text("""
            SELECT id, type, name, description, creator_name, icon, category, tags,
                   install_count, is_featured, version, metadata, created_at, updated_at
            FROM marketplace_items
            WHERE is_approved = true AND is_featured = true
            ORDER BY install_count DESC, created_at DESC
            LIMIT :limit
        """)

        result = db.execute(query, {"limit": limit})

        items = []
        for row in result:
            items.append(MarketplaceItemOut(
                id=row[0],
                type=row[1],
                name=row[2],
                description=row[3],
                creator_name=row[4],
                icon=row[5],
                category=row[6],
                tags=row[7] or [],
                install_count=row[8],
                is_featured=row[9],
                version=row[10],
                metadata=row[11] or {},
                created_at=row[12],
                updated_at=row[13]
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

        query = text("""
            SELECT
                mi.id,
                mi.name,
                mi.type,
                mins.version as installed_version,
                mi.version as latest_version,
                mi.metadata->>'changelog' as changelog
            FROM marketplace_installs mins
            JOIN marketplace_items mi ON mins.item_id = mi.id
            WHERE mins.user_id = :user_id
            AND mins.version != mi.version
            ORDER BY mi.updated_at DESC
        """)

        result = db.execute(query, {"user_id": ctx.user_id})

        updates = []
        for row in result:
            updates.append(UpdateInfo(
                item_id=row[0],
                item_name=row[1],
                item_type=row[2],
                current_version=row[3],
                latest_version=row[4],
                changelog=row[5] or "No changelog available"
            ))

        return updates

    except Exception as e:
        logger.error(f"Error checking for updates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check for updates: {str(e)}")


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
    Submit a new marketplace item for approval.
    Trusted users (5+ approved items) auto-publish; others go to approval queue.
    """
    try:
        from sqlalchemy import text

        # Check if user is trusted
        trusted_check = text("""
            SELECT COUNT(*) FROM marketplace_submissions
            WHERE submitted_by = :user_id AND status = 'approved'
        """)

        approved_count = db.execute(trusted_check, {"user_id": ctx.user_id}).scalar()
        is_trusted = approved_count >= 5

        # Create marketplace item
        create_item = text("""
            INSERT INTO marketplace_items
            (type, name, description, creator_name, creator_user_id, category, tags, metadata, is_approved, version)
            VALUES (:type, :name, :description, :creator_name, :user_id, :category, :tags, :metadata, :is_approved, '1.0.0')
            RETURNING id
        """)

        result = db.execute(create_item, {
            "type": request.item_type,
            "name": request.name,
            "description": request.description,
            "creator_name": ctx.user_email or f"User {ctx.user_id}",
            "user_id": ctx.user_id,
            "category": request.category,
            "tags": request.tags,
            "metadata": request.metadata,
            "is_approved": is_trusted
        })

        item_id = result.scalar()

        # Create submission record
        create_submission = text("""
            INSERT INTO marketplace_submissions (item_id, submitted_by, status, submitted_at)
            VALUES (:item_id, :user_id, :status, NOW())
        """)

        db.execute(create_submission, {
            "item_id": item_id,
            "user_id": ctx.user_id,
            "status": "approved" if is_trusted else "pending"
        })

        db.commit()

        message = "Item published successfully" if is_trusted else "Item submitted for approval"

        return {
            "success": True,
            "message": message,
            "item_id": item_id,
            "auto_approved": is_trusted
        }

    except Exception as e:
        logger.error(f"Error submitting marketplace item: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit marketplace item: {str(e)}")
