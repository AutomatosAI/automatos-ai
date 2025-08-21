from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import logging

from database.database import get_db
from models import Pattern, PatternCreate, PatternResponse
from ..main import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/patterns", tags=["patterns"])

@router.get("/", dependencies=[Depends(require_api_key)])
async def list_patterns(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all patterns"""
    try:
        patterns = db.query(Pattern).offset(skip).limit(limit).all()
        
        pattern_responses = [PatternResponse(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            configuration=pattern.configuration or {},
            is_active=pattern.is_active,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at
        ) for pattern in patterns]
        
        return {"data": pattern_responses}
        
    except Exception as e:
        logger.error(f"Error listing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", dependencies=[Depends(require_api_key)])
async def create_pattern(pattern_data: PatternCreate, db: Session = Depends(get_db)):
    """Create a new pattern"""
    try:
        # Check if pattern name already exists
        existing = db.query(Pattern).filter(Pattern.name == pattern_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Pattern with this name already exists")
        
        pattern = Pattern(
            name=pattern_data.name,
            description=pattern_data.description,
            pattern_type=pattern_data.pattern_type,
            configuration=pattern_data.configuration or {}
        )
        
        db.add(pattern)
        db.commit()
        db.refresh(pattern)
        
        pattern_response = PatternResponse(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            configuration=pattern.configuration or {},
            is_active=pattern.is_active,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at
        )
        
        return {"data": pattern_response}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating pattern: {str(e)}")

@router.get("/{pattern_id}", dependencies=[Depends(require_api_key)])
async def get_pattern(pattern_id: int, db: Session = Depends(get_db)):
    """Get a specific pattern by ID"""
    try:
        pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        pattern_response = PatternResponse(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            configuration=pattern.configuration or {},
            is_active=pattern.is_active,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at
        )
        
        return {"data": pattern_response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{pattern_id}", dependencies=[Depends(require_api_key)])
async def delete_pattern(pattern_id: int, db: Session = Depends(get_db)):
    """Delete a pattern"""
    try:
        pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        db.delete(pattern)
        db.commit()
        
        return {"data": {"message": "Pattern deleted successfully", "pattern_id": pattern_id}}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))