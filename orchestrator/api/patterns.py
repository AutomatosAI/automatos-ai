
"""
Patterns Management API
======================

API endpoints for managing agent coordination and decision patterns.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from database.database import get_db
from models import Pattern, PatternCreate, PatternUpdate, PatternResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/patterns", tags=["patterns"])

@router.get("/", response_model=List[PatternResponse])
async def get_patterns(
    pattern_type: Optional[str] = None,
    is_active: Optional[bool] = True,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all patterns with optional filtering"""
    try:
        query = db.query(Pattern)
        
        if pattern_type:
            query = query.filter(Pattern.pattern_type == pattern_type)
        
        if is_active is not None:
            query = query.filter(Pattern.is_active == is_active)
        
        patterns = query.offset(skip).limit(limit).all()
        
        return [
            PatternResponse(
                id=pattern.id,
                name=pattern.name,
                description=pattern.description,
                pattern_type=pattern.pattern_type,
                pattern_data=pattern.pattern_data,
                usage_count=pattern.usage_count,
                effectiveness_score=pattern.effectiveness_score,
                is_active=pattern.is_active,
                created_at=pattern.created_at,
                updated_at=pattern.updated_at,
                created_by=pattern.created_by
            ) for pattern in patterns
        ]
        
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=PatternResponse)
async def create_pattern(
    pattern: PatternCreate,
    db: Session = Depends(get_db)
):
    """Create a new pattern"""
    try:
        # Check if pattern name already exists
        existing = db.query(Pattern).filter(Pattern.name == pattern.name).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail="Pattern with this name already exists"
            )
        
        db_pattern = Pattern(
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            pattern_data=pattern.pattern_data,
            created_by="api"
        )
        
        db.add(db_pattern)
        db.commit()
        db.refresh(db_pattern)
        
        return PatternResponse(
            id=db_pattern.id,
            name=db_pattern.name,
            description=db_pattern.description,
            pattern_type=db_pattern.pattern_type,
            pattern_data=db_pattern.pattern_data,
            usage_count=db_pattern.usage_count,
            effectiveness_score=db_pattern.effectiveness_score,
            is_active=db_pattern.is_active,
            created_at=db_pattern.created_at,
            updated_at=db_pattern.updated_at,
            created_by=db_pattern.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating pattern: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pattern_id}", response_model=PatternResponse)
async def get_pattern(
    pattern_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific pattern by ID"""
    try:
        pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        return PatternResponse(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            pattern_data=pattern.pattern_data,
            usage_count=pattern.usage_count,
            effectiveness_score=pattern.effectiveness_score,
            is_active=pattern.is_active,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at,
            created_by=pattern.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{pattern_id}", response_model=PatternResponse)
async def update_pattern(
    pattern_id: int,
    pattern_update: PatternUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing pattern"""
    try:
        db_pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not db_pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Update fields if provided
        update_data = pattern_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_pattern, field, value)
        
        db.commit()
        db.refresh(db_pattern)
        
        return PatternResponse(
            id=db_pattern.id,
            name=db_pattern.name,
            description=db_pattern.description,
            pattern_type=db_pattern.pattern_type,
            pattern_data=db_pattern.pattern_data,
            usage_count=db_pattern.usage_count,
            effectiveness_score=db_pattern.effectiveness_score,
            is_active=db_pattern.is_active,
            created_at=db_pattern.created_at,
            updated_at=db_pattern.updated_at,
            created_by=db_pattern.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating pattern: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{pattern_id}")
async def delete_pattern(
    pattern_id: int,
    db: Session = Depends(get_db)
):
    """Delete a pattern (soft delete by setting is_active=False)"""
    try:
        db_pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not db_pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Soft delete
        db_pattern.is_active = False
        db.commit()
        
        return {"message": "Pattern deleted successfully", "pattern_id": pattern_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pattern: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pattern_id}/use")
async def use_pattern(
    pattern_id: int,
    effectiveness_score: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Record pattern usage and update effectiveness score"""
    try:
        db_pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not db_pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Increment usage count
        db_pattern.usage_count += 1
        
        # Update effectiveness score if provided
        if effectiveness_score is not None:
            if not (0.0 <= effectiveness_score <= 1.0):
                raise HTTPException(
                    status_code=400, 
                    detail="Effectiveness score must be between 0.0 and 1.0"
                )
            
            # Calculate weighted average of effectiveness scores
            current_score = db_pattern.effectiveness_score
            usage_count = db_pattern.usage_count
            
            # Weighted average: give more weight to recent scores
            weight = min(0.3, 1.0 / usage_count)  # Adaptive weight
            db_pattern.effectiveness_score = (
                current_score * (1 - weight) + effectiveness_score * weight
            )
        
        db.commit()
        
        return {
            "message": "Pattern usage recorded",
            "pattern_id": pattern_id,
            "usage_count": db_pattern.usage_count,
            "effectiveness_score": db_pattern.effectiveness_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording pattern usage: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
