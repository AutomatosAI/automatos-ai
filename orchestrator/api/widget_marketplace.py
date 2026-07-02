"""
Widget Marketplace API (PRD-38-5)
==================================

Browsing, search, filtering, install/uninstall, reviews, and developer
endpoints for the widget marketplace.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, desc, func as sqlfunc, or_, text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.marketplace_widget import MarketplaceWidget
from core.models.widget_installation import WidgetInstallation
from core.models.widget_review import WidgetReview

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/widget-marketplace", tags=["Widget Marketplace"])


# ===================================================================
# Helpers
# ===================================================================

def _get_user_db_id(db: Session, ctx: RequestContext):
    """Resolve clerk_user_id -> users.id UUID."""
    clerk_id = getattr(ctx.user, "clerk_user_id", None) or ctx.user.id
    if not clerk_id:
        return None
    row = db.execute(
        text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
        {"cid": clerk_id},
    ).fetchone()
    return row[0] if row else None


def _require_user_db_id(db: Session, ctx: RequestContext):
    uid = _get_user_db_id(db, ctx)
    if uid is None:
        raise HTTPException(status_code=401, detail="User not found")
    return uid


def _is_admin(ctx: RequestContext) -> bool:
    # PRD-174 F043: shared admin check — super_admin ⊇ admin when the plane is on.
    from core.auth.roles import caller_is_admin
    return caller_is_admin(ctx.user)


def _assert_admin(ctx: RequestContext) -> None:
    if not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="Admin access required")


def _recalculate_rating(db: Session, widget_id) -> None:
    result = db.query(
        sqlfunc.avg(WidgetReview.rating),
        sqlfunc.count(WidgetReview.id),
    ).filter(
        WidgetReview.widget_id == widget_id,
        WidgetReview.status == "published",
    ).first()
    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if widget:
        widget.rating_average = float(result[0] or 0)
        widget.rating_count = result[1] or 0
        db.commit()


# ===================================================================
# Pydantic schemas
# ===================================================================

class WidgetSummaryOut(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str] = None
    developer_id: Optional[str] = None
    developer_name: Optional[str] = None
    version: Optional[str] = None
    pricing_type: Optional[str] = "free"
    price_cents: Optional[int] = None
    currency: Optional[str] = "USD"
    icon_url: Optional[str] = None
    categories: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    install_count: int = 0
    rating_average: float = 0.0
    rating_count: int = 0
    status: str = "draft"
    published_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WidgetDetailOut(WidgetSummaryOut):
    long_description: Optional[str] = None
    readme: Optional[str] = None
    changelog: Optional[str] = None
    screenshots: Optional[List[Any]] = None
    bundle_url: Optional[str] = None
    bundle_size: Optional[int] = None
    permissions: Optional[List[str]] = None
    min_plan: Optional[str] = None

    class Config:
        from_attributes = True


class WidgetListResponse(BaseModel):
    widgets: List[WidgetSummaryOut]
    total: int
    page: int
    page_size: int


class CategoryOut(BaseModel):
    name: str
    count: int


class InstallationOut(BaseModel):
    id: str
    widget_id: str
    workspace_id: str
    user_id: Optional[str] = None
    installed_at: datetime
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    widget: Optional[WidgetSummaryOut] = None

    class Config:
        from_attributes = True


class ReviewOut(BaseModel):
    id: str
    widget_id: str
    user_id: str
    rating: int
    title: Optional[str] = None
    body: Optional[str] = None
    is_verified_purchase: bool = False
    status: str = "published"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReviewListResponse(BaseModel):
    reviews: List[ReviewOut]
    total: int


class CreateReviewRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    body: Optional[str] = None


class UpdateReviewRequest(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    body: Optional[str] = None


class CreateWidgetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    long_description: Optional[str] = None
    version: Optional[str] = None
    pricing_type: Optional[str] = "free"
    price_cents: Optional[int] = None
    icon_url: Optional[str] = None
    screenshots: Optional[List[Any]] = None
    readme: Optional[str] = None
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    bundle_url: Optional[str] = None
    permissions: Optional[List[str]] = None
    min_plan: Optional[str] = None


class UpdateWidgetRequest(BaseModel):
    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    long_description: Optional[str] = None
    version: Optional[str] = None
    pricing_type: Optional[str] = None
    price_cents: Optional[int] = None
    icon_url: Optional[str] = None
    screenshots: Optional[List[Any]] = None
    readme: Optional[str] = None
    changelog: Optional[str] = None
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    bundle_url: Optional[str] = None
    bundle_size: Optional[int] = None
    permissions: Optional[List[str]] = None
    min_plan: Optional[str] = None


class DeveloperAnalyticsOut(BaseModel):
    total_widgets: int = 0
    total_installs: int = 0
    total_reviews: int = 0
    average_rating: float = 0.0


# ===================================================================
# 1. GET /widgets — Browse / search published widgets
# ===================================================================

@router.get("/widgets", response_model=WidgetListResponse)
async def browse_widgets(
    q: Optional[str] = Query(None, description="Search name, display_name, description"),
    category: Optional[str] = Query(None, description="Filter by category"),
    pricing: Optional[str] = Query(None, description="free | one_time | subscription"),
    sort: str = Query("popular", description="popular | newest | rating"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Browse and search published marketplace widgets."""
    query = db.query(MarketplaceWidget).filter(MarketplaceWidget.status == "published")

    # Full-text-ish search
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                MarketplaceWidget.name.ilike(like),
                MarketplaceWidget.display_name.ilike(like),
                MarketplaceWidget.description.ilike(like),
            )
        )

    # Category filter (ANY on postgres array)
    if category:
        query = query.filter(MarketplaceWidget.categories.any(category))

    # Pricing filter
    if pricing:
        query = query.filter(MarketplaceWidget.pricing_type == pricing)

    # Sorting
    if sort == "newest":
        query = query.order_by(desc(MarketplaceWidget.published_at))
    elif sort == "rating":
        query = query.order_by(desc(MarketplaceWidget.rating_average))
    else:  # popular (default)
        query = query.order_by(desc(MarketplaceWidget.install_count))

    total = query.count()
    offset = (page - 1) * page_size
    widgets = query.offset(offset).limit(page_size).all()

    return WidgetListResponse(
        widgets=[_widget_to_summary(w) for w in widgets],
        total=total,
        page=page,
        page_size=page_size,
    )


# ===================================================================
# 2. GET /widgets/{widget_id} — Full widget detail
# ===================================================================

@router.get("/widgets/{widget_id}", response_model=WidgetDetailOut)
async def get_widget_detail(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Full widget detail including readme, screenshots, changelog."""
    widget = db.query(MarketplaceWidget).filter(
        MarketplaceWidget.id == widget_id
    ).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")

    # Must be published, or owned by current user
    if widget.status != "published":
        user_db_id = _get_user_db_id(db, ctx)
        if str(widget.developer_id) != str(user_db_id) and not _is_admin(ctx):
            raise HTTPException(status_code=404, detail="Widget not found")

    return _widget_to_detail(widget)


# ===================================================================
# 3. GET /categories — Categories with counts
# ===================================================================

@router.get("/categories", response_model=List[CategoryOut])
async def list_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List distinct categories from published widgets with counts."""
    rows = db.execute(
        text("""
            SELECT cat, COUNT(*) AS cnt
            FROM marketplace_widgets, unnest(categories) AS cat
            WHERE status = 'published'
            GROUP BY cat
            ORDER BY cnt DESC, cat
        """)
    ).fetchall()
    return [CategoryOut(name=r[0], count=r[1]) for r in rows]


# ===================================================================
# 4. GET /featured — Featured widgets
# ===================================================================

@router.get("/featured", response_model=List[WidgetSummaryOut])
async def featured_widgets(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Top 8 published widgets by install count."""
    widgets = (
        db.query(MarketplaceWidget)
        .filter(MarketplaceWidget.status == "published")
        .order_by(desc(MarketplaceWidget.install_count))
        .limit(8)
        .all()
    )
    return [_widget_to_summary(w) for w in widgets]


# ===================================================================
# 5. POST /widgets/{widget_id}/install — Install widget (US-005)
# ===================================================================

@router.post("/widgets/{widget_id}/install", status_code=201)
async def install_widget(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Install a published widget into the current workspace."""
    user_db_id = _require_user_db_id(db, ctx)

    widget = db.query(MarketplaceWidget).filter(
        MarketplaceWidget.id == widget_id,
        MarketplaceWidget.status == "published",
    ).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")

    # Check duplicate (same widget + workspace, not uninstalled)
    existing = db.query(WidgetInstallation).filter(
        WidgetInstallation.widget_id == widget_id,
        WidgetInstallation.workspace_id == ctx.workspace_id,
        WidgetInstallation.uninstalled_at.is_(None),
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Widget already installed in this workspace")

    installation = WidgetInstallation(
        id=uuid.uuid4(),
        widget_id=widget.id,
        workspace_id=ctx.workspace_id,
        user_id=user_db_id,
    )
    db.add(installation)

    widget.install_count = (widget.install_count or 0) + 1
    db.commit()
    db.refresh(installation)

    return {
        "id": str(installation.id),
        "widget_id": str(installation.widget_id),
        "workspace_id": str(installation.workspace_id),
        "installed_at": installation.installed_at.isoformat() if installation.installed_at else None,
    }


# ===================================================================
# 6. DELETE /widgets/{widget_id}/install — Uninstall widget (US-005)
# ===================================================================

@router.delete("/widgets/{widget_id}/install")
async def uninstall_widget(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Uninstall a widget from the current workspace."""
    installation = db.query(WidgetInstallation).filter(
        WidgetInstallation.widget_id == widget_id,
        WidgetInstallation.workspace_id == ctx.workspace_id,
        WidgetInstallation.uninstalled_at.is_(None),
    ).first()
    if not installation:
        raise HTTPException(status_code=404, detail="Widget not installed in this workspace")

    installation.uninstalled_at = sqlfunc.now()

    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if widget and (widget.install_count or 0) > 0:
        widget.install_count = widget.install_count - 1

    db.commit()
    return {"detail": "Widget uninstalled"}


# ===================================================================
# 7. GET /installed — User's installed widgets (US-005)
# ===================================================================

@router.get("/installed", response_model=List[InstallationOut])
async def list_installed_widgets(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List widgets installed in the current workspace."""
    installations = (
        db.query(WidgetInstallation)
        .filter(
            WidgetInstallation.workspace_id == ctx.workspace_id,
            WidgetInstallation.uninstalled_at.is_(None),
        )
        .order_by(desc(WidgetInstallation.installed_at))
        .all()
    )
    results = []
    for inst in installations:
        widget = db.query(MarketplaceWidget).filter(
            MarketplaceWidget.id == inst.widget_id
        ).first()
        results.append(InstallationOut(
            id=str(inst.id),
            widget_id=str(inst.widget_id),
            workspace_id=str(inst.workspace_id),
            user_id=str(inst.user_id) if inst.user_id else None,
            installed_at=inst.installed_at,
            last_used_at=inst.last_used_at,
            use_count=inst.use_count or 0,
            widget=_widget_to_summary(widget) if widget else None,
        ))
    return results


# ===================================================================
# 8. GET /widgets/{widget_id}/reviews — List reviews (US-006)
# ===================================================================

@router.get("/widgets/{widget_id}/reviews", response_model=ReviewListResponse)
async def list_reviews(
    widget_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List reviews for a widget, newest first."""
    query = db.query(WidgetReview).filter(
        WidgetReview.widget_id == widget_id,
        WidgetReview.status == "published",
    )
    total = query.count()
    offset = (page - 1) * page_size
    reviews = (
        query.order_by(desc(WidgetReview.created_at))
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return ReviewListResponse(
        reviews=[_review_to_out(r) for r in reviews],
        total=total,
    )


# ===================================================================
# 9. POST /widgets/{widget_id}/reviews — Create review (US-006)
# ===================================================================

@router.post("/widgets/{widget_id}/reviews", response_model=ReviewOut, status_code=201)
async def create_review(
    widget_id: str,
    body: CreateReviewRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a review for a widget. One review per user per widget."""
    user_db_id = _require_user_db_id(db, ctx)

    # Widget must exist and be published
    widget = db.query(MarketplaceWidget).filter(
        MarketplaceWidget.id == widget_id,
        MarketplaceWidget.status == "published",
    ).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")

    # Check existing review (unique constraint)
    existing = db.query(WidgetReview).filter(
        WidgetReview.widget_id == widget_id,
        WidgetReview.user_id == user_db_id,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="You have already reviewed this widget")

    # Verified purchase check
    active_install = db.query(WidgetInstallation).filter(
        WidgetInstallation.widget_id == widget_id,
        WidgetInstallation.user_id == user_db_id,
        WidgetInstallation.uninstalled_at.is_(None),
    ).first()

    review = WidgetReview(
        id=uuid.uuid4(),
        widget_id=widget.id,
        user_id=user_db_id,
        rating=body.rating,
        title=body.title,
        body=body.body,
        is_verified_purchase=active_install is not None,
        status="published",
    )
    db.add(review)
    db.commit()

    _recalculate_rating(db, widget.id)
    db.refresh(review)

    return _review_to_out(review)


# ===================================================================
# 10. PUT /reviews/{review_id} — Update own review (US-006)
# ===================================================================

@router.put("/reviews/{review_id}", response_model=ReviewOut)
async def update_review(
    review_id: str,
    body: UpdateReviewRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update your own review."""
    user_db_id = _require_user_db_id(db, ctx)

    review = db.query(WidgetReview).filter(WidgetReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.user_id) != str(user_db_id):
        raise HTTPException(status_code=403, detail="You can only edit your own reviews")

    if body.rating is not None:
        review.rating = body.rating
    if body.title is not None:
        review.title = body.title
    if body.body is not None:
        review.body = body.body

    db.commit()
    _recalculate_rating(db, review.widget_id)
    db.refresh(review)

    return _review_to_out(review)


# ===================================================================
# 11. DELETE /reviews/{review_id} — Delete own review (US-006)
# ===================================================================

@router.delete("/reviews/{review_id}")
async def delete_review(
    review_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete your own review."""
    user_db_id = _require_user_db_id(db, ctx)

    review = db.query(WidgetReview).filter(WidgetReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.user_id) != str(user_db_id):
        raise HTTPException(status_code=403, detail="You can only delete your own reviews")

    widget_id = review.widget_id
    db.delete(review)
    db.commit()

    _recalculate_rating(db, widget_id)
    return {"detail": "Review deleted"}


# ===================================================================
# 12. POST /widgets — Create widget (US-007, developer)
# ===================================================================

@router.post("/widgets", response_model=WidgetDetailOut, status_code=201)
async def create_widget(
    body: CreateWidgetRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new widget in draft status."""
    user_db_id = _require_user_db_id(db, ctx)

    # Check name uniqueness
    existing = db.query(MarketplaceWidget).filter(
        MarketplaceWidget.name == body.name
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="A widget with this name already exists")

    # Resolve developer display name
    dev_row = db.execute(
        text("SELECT email FROM users WHERE id = :uid LIMIT 1"),
        {"uid": str(user_db_id)},
    ).fetchone()
    developer_name = dev_row[0] if dev_row else None

    widget = MarketplaceWidget(
        id=uuid.uuid4(),
        name=body.name,
        display_name=body.display_name,
        description=body.description,
        long_description=body.long_description,
        developer_id=user_db_id,
        developer_name=developer_name,
        version=body.version,
        pricing_type=body.pricing_type or "free",
        price_cents=body.price_cents,
        icon_url=body.icon_url,
        screenshots=body.screenshots or [],
        readme=body.readme,
        keywords=body.keywords,
        categories=body.categories,
        bundle_url=body.bundle_url,
        permissions=body.permissions,
        min_plan=body.min_plan,
        status="draft",
    )
    db.add(widget)
    db.commit()
    db.refresh(widget)

    return _widget_to_detail(widget)


# ===================================================================
# 13. PUT /widgets/{widget_id} — Update widget (US-007)
# ===================================================================

@router.put("/widgets/{widget_id}", response_model=WidgetDetailOut)
async def update_widget(
    widget_id: str,
    body: UpdateWidgetRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a widget. Must be owner. Only allowed in draft or published status."""
    user_db_id = _require_user_db_id(db, ctx)

    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    if str(widget.developer_id) != str(user_db_id) and not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="You can only edit your own widgets")
    if widget.status not in ("draft", "published"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update widget in '{widget.status}' status",
        )

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(widget, key, value)

    db.commit()
    db.refresh(widget)

    return _widget_to_detail(widget)


# ===================================================================
# 14. POST /widgets/{widget_id}/submit — Submit for review (US-007)
# ===================================================================

@router.post("/widgets/{widget_id}/submit")
async def submit_widget_for_review(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Submit a draft widget for admin review."""
    user_db_id = _require_user_db_id(db, ctx)

    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    if str(widget.developer_id) != str(user_db_id) and not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="You can only submit your own widgets")
    if widget.status != "draft":
        raise HTTPException(
            status_code=400,
            detail=f"Widget must be in draft status to submit (current: {widget.status})",
        )

    # Validate required fields
    missing = []
    if not widget.name:
        missing.append("name")
    if not widget.display_name:
        missing.append("display_name")
    if not widget.description:
        missing.append("description")
    if not widget.version:
        missing.append("version")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    widget.status = "review"
    db.commit()

    return {"detail": "Widget submitted for review", "status": "review"}


# ===================================================================
# 15. GET /developer/widgets — Developer's widgets (US-007)
# ===================================================================

@router.get("/developer/widgets", response_model=List[WidgetSummaryOut])
async def developer_widgets(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List widgets created by the current user."""
    user_db_id = _require_user_db_id(db, ctx)

    widgets = (
        db.query(MarketplaceWidget)
        .filter(MarketplaceWidget.developer_id == user_db_id)
        .order_by(desc(MarketplaceWidget.updated_at))
        .all()
    )
    return [_widget_to_summary(w) for w in widgets]


# ===================================================================
# 16. GET /developer/analytics — Developer analytics (US-007)
# ===================================================================

@router.get("/developer/analytics", response_model=DeveloperAnalyticsOut)
async def developer_analytics(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Aggregate analytics for the current developer's widgets."""
    user_db_id = _require_user_db_id(db, ctx)

    result = db.query(
        sqlfunc.count(MarketplaceWidget.id),
        sqlfunc.coalesce(sqlfunc.sum(MarketplaceWidget.install_count), 0),
        sqlfunc.coalesce(sqlfunc.sum(MarketplaceWidget.rating_count), 0),
        sqlfunc.avg(MarketplaceWidget.rating_average),
    ).filter(
        MarketplaceWidget.developer_id == user_db_id,
    ).first()

    return DeveloperAnalyticsOut(
        total_widgets=result[0] or 0,
        total_installs=int(result[1] or 0),
        total_reviews=int(result[2] or 0),
        average_rating=float(result[3] or 0),
    )


# ===================================================================
# 17. PUT /widgets/{widget_id}/approve — Admin approve (US-007)
# ===================================================================

@router.put("/widgets/{widget_id}/approve")
async def approve_widget(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Admin: approve a widget from review -> published."""
    _assert_admin(ctx)

    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    if widget.status != "review":
        raise HTTPException(
            status_code=400,
            detail=f"Widget must be in review status to approve (current: {widget.status})",
        )

    widget.status = "published"
    widget.published_at = sqlfunc.now()
    db.commit()

    return {"detail": "Widget approved and published", "status": "published"}


# ===================================================================
# 18. PUT /widgets/{widget_id}/suspend — Admin suspend (US-007)
# ===================================================================

@router.put("/widgets/{widget_id}/suspend")
async def suspend_widget(
    widget_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Admin: suspend a widget."""
    _assert_admin(ctx)

    widget = db.query(MarketplaceWidget).filter(MarketplaceWidget.id == widget_id).first()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")

    widget.status = "suspended"
    db.commit()

    return {"detail": "Widget suspended", "status": "suspended"}


# ===================================================================
# Serialization helpers
# ===================================================================

def _widget_to_summary(w: MarketplaceWidget) -> WidgetSummaryOut:
    return WidgetSummaryOut(
        id=str(w.id),
        name=w.name,
        display_name=w.display_name,
        description=w.description,
        developer_id=str(w.developer_id) if w.developer_id else None,
        developer_name=w.developer_name,
        version=w.version,
        pricing_type=w.pricing_type,
        price_cents=w.price_cents,
        currency=w.currency,
        icon_url=w.icon_url,
        categories=w.categories,
        keywords=w.keywords,
        install_count=w.install_count or 0,
        rating_average=float(w.rating_average or 0),
        rating_count=w.rating_count or 0,
        status=w.status,
        published_at=w.published_at,
        created_at=w.created_at,
        updated_at=w.updated_at,
    )


def _widget_to_detail(w: MarketplaceWidget) -> WidgetDetailOut:
    return WidgetDetailOut(
        id=str(w.id),
        name=w.name,
        display_name=w.display_name,
        description=w.description,
        long_description=w.long_description,
        developer_id=str(w.developer_id) if w.developer_id else None,
        developer_name=w.developer_name,
        version=w.version,
        pricing_type=w.pricing_type,
        price_cents=w.price_cents,
        currency=w.currency,
        icon_url=w.icon_url,
        screenshots=w.screenshots,
        readme=w.readme,
        changelog=w.changelog,
        categories=w.categories,
        keywords=w.keywords,
        bundle_url=w.bundle_url,
        bundle_size=w.bundle_size,
        permissions=w.permissions,
        min_plan=w.min_plan,
        install_count=w.install_count or 0,
        rating_average=float(w.rating_average or 0),
        rating_count=w.rating_count or 0,
        status=w.status,
        published_at=w.published_at,
        created_at=w.created_at,
        updated_at=w.updated_at,
    )


def _review_to_out(r: WidgetReview) -> ReviewOut:
    return ReviewOut(
        id=str(r.id),
        widget_id=str(r.widget_id),
        user_id=str(r.user_id),
        rating=r.rating,
        title=r.title,
        body=r.body,
        is_verified_purchase=r.is_verified_purchase or False,
        status=r.status,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )
