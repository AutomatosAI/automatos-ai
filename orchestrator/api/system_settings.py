"""
System Settings API
==================

FastAPI routes for managing system-wide configuration settings.
Replaces .env file with database-backed settings management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import uuid
import json

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.models.system_settings import (
    SystemSetting, SystemSettingResponse, SystemSettingUpdate, 
    SystemSettingCreate, SystemSettingsByCategory, SystemSettingsStats,
    SettingCategory
)
from pydantic import BaseModel

class BulkUpdateItem(BaseModel):
    """Single item in bulk update request"""
    id: int
    value: str

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system-settings", tags=["system-settings"])


@router.get("/", response_model=List[SystemSettingResponse])
async def list_system_settings(
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """List all system settings, optionally filtered by category"""
    try:
        query = db.query(SystemSetting)
        if category:
            query = query.filter(SystemSetting.category == category)
        
        settings = query.order_by(SystemSetting.category, SystemSetting.key).all()
        return settings
    except Exception as e:
        logger.error(f"Failed to list system settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/categories", response_model=List[str])
async def list_setting_categories(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)):
    """List all available setting categories"""
    
    try:
        categories = db.query(SystemSetting.category).distinct().all()
        return [cat[0] for cat in categories]
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/by-category", response_model=List[SystemSettingsByCategory])
async def get_settings_by_category(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Get all settings grouped by category"""
    
    try:
        settings = db.query(SystemSetting).order_by(SystemSetting.category, SystemSetting.key).all()
        
        # Group by category
        categories = {}
        for setting in settings:
            if setting.category not in categories:
                categories[setting.category] = []
            categories[setting.category].append(setting)
        
        result = []
        for category, category_settings in categories.items():
            result.append(SystemSettingsByCategory(
                category=category,
                settings=category_settings,
                total_count=len(category_settings)
            ))
        
        return result
    except Exception as e:
        logger.error(f"Failed to get settings by category: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=SystemSettingsStats)
async def get_settings_stats(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Get system settings statistics"""
    
    try:
        total_settings = db.query(SystemSetting).count()
        
        # Count by category
        categories = {}
        category_counts = db.query(
            SystemSetting.category, 
            func.count(SystemSetting.id)
        ).group_by(SystemSetting.category).all()
        
        for category, count in category_counts:
            categories[category] = count
        
        sensitive_settings = db.query(SystemSetting).filter(SystemSetting.is_sensitive == True).count()
        required_settings = db.query(SystemSetting).filter(SystemSetting.is_required == True).count()
        
        return SystemSettingsStats(
            total_settings=total_settings,
            categories=categories,
            sensitive_settings=sensitive_settings,
            required_settings=required_settings
        )
    except Exception as e:
        logger.error(f"Failed to get settings stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{setting_id}", response_model=SystemSettingResponse)
async def get_system_setting(
    setting_id: int,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Get a specific system setting"""
    
    try:
        setting = db.query(SystemSetting).filter(SystemSetting.id == setting_id).first()
        if not setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        return setting
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system setting {setting_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{setting_id}", response_model=SystemSettingResponse)
async def update_system_setting(
    setting_id: int,
    update_data: SystemSettingUpdate,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Update a system setting"""
    
    try:
        setting = db.query(SystemSetting).filter(SystemSetting.id == setting_id).first()
        if not setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        setting.value = update_data.value
        if update_data.description is not None:
            setting.description = update_data.description
        
        db.commit()
        db.refresh(setting)
        
        logger.info(f"Updated system setting {setting.key} = {setting.value}")
        return setting
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update system setting {setting_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=SystemSettingResponse)
async def create_system_setting(
    setting_data: SystemSettingCreate,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Create a new system setting"""
    
    try:
        # Check if setting already exists
        existing = db.query(SystemSetting).filter(
            SystemSetting.category == setting_data.category,
            SystemSetting.key == setting_data.key
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Setting {setting_data.category}.{setting_data.key} already exists"
            )
        
        setting = SystemSetting(**setting_data.dict())
        db.add(setting)
        db.commit()
        db.refresh(setting)
        
        logger.info(f"Created system setting {setting.category}.{setting.key}")
        return setting
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create system setting: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{setting_id}")
async def delete_system_setting(
    setting_id: int,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Delete a system setting"""
    
    try:
        setting = db.query(SystemSetting).filter(SystemSetting.id == setting_id).first()
        if not setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        # Don't allow deletion of required settings
        if setting.is_required:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete required setting"
            )
        
        db.delete(setting)
        db.commit()
        
        logger.info(f"Deleted system setting {setting.category}.{setting.key}")
        return {"message": "Setting deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete system setting {setting_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-update")
async def bulk_update_settings(
    request: Request,
    updates: List[BulkUpdateItem] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Bulk update multiple settings"""
    
    try:
        # Log the raw body for debugging
        body = await request.body()
        logger.info(f"Raw request body: {body.decode('utf-8')}")
        
        if updates is None:
            raise HTTPException(status_code=422, detail="No updates provided")
        
        logger.info(f"Received bulk update request with {len(updates)} items")
        updated_count = 0
        updated_settings = []
        
        for update in updates:
            setting = db.query(SystemSetting).filter(SystemSetting.id == update.id).first()
            if setting:
                old_value = setting.value
                new_value = update.value
                
                # Log comparison details for debugging
                logger.info(
                    f"Comparing setting {setting.category}.{setting.key} (id={update.id}): "
                    f"old='{old_value}' (type={type(old_value).__name__}) vs "
                    f"new='{new_value}' (type={type(new_value).__name__})"
                )
                
                # Always update when explicitly requested (user clicked save)
                # The frontend should handle preventing unnecessary saves
                setting.value = new_value
                # Explicitly update timestamp (onupdate may not always trigger)
                setting.updated_at = datetime.utcnow()
                updated_settings.append(setting)
                updated_count += 1
                logger.info(f"Updated setting {setting.category}.{setting.key}: '{old_value}' → '{new_value}'")
            else:
                logger.warning(f"Setting {update.id} not found")
        
        if updated_count > 0:
            db.commit()
            # Refresh all updated settings to get latest timestamps
            for setting in updated_settings:
                db.refresh(setting)
            logger.info(f"Bulk updated {updated_count} system settings")
        else:
            logger.info("No settings were updated (all values unchanged)")
        
        return {"message": f"Updated {updated_count} settings successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk update settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset-to-defaults")
async def reset_settings_to_defaults(
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Reset settings to their default values"""
    
    try:
        query = db.query(SystemSetting)
        if category:
            query = query.filter(SystemSetting.category == category)
        
        settings = query.filter(SystemSetting.default_value.isnot(None)).all()
        
        reset_count = 0
        for setting in settings:
            setting.value = setting.default_value
            reset_count += 1
        
        db.commit()
        
        logger.info(f"Reset {reset_count} settings to defaults")
        return {"message": f"Reset {reset_count} settings to default values"}
    except Exception as e:
        logger.error(f"Failed to reset settings to defaults: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
