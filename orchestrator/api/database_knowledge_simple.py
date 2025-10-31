"""
Database Knowledge API Routes - Simplified
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/knowledge/sources/database", tags=["Database Knowledge"])

class DatabaseSourceCreate(BaseModel):
    name: str
    credential_id: int
    description: Optional[str] = None

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify routes are working"""
    return {"status": "Database Knowledge API is active", "version": "1.0"}

@router.get("/templates/list")
async def list_templates(dialect: Optional[str] = None):
    """List query templates"""
    templates = [
        {
            "id": 1,
            "name": "Top N by Revenue",
            "natural_language": "Show top {n} customers by revenue",
            "category": "analytics",
            "dialect": "postgresql",
            "visualization_type": "bar"
        },
        {
            "id": 2,
            "name": "Revenue Trend",
            "natural_language": "Show revenue trend over time",
            "category": "analytics",
            "dialect": "postgresql",
            "visualization_type": "line"
        },
        {
            "id": 3,
            "name": "Daily Summary",
            "natural_language": "Daily business summary",
            "category": "reporting",
            "dialect": "postgresql",
            "visualization_type": "table"
        }
    ]
    
    if dialect:
        templates = [t for t in templates if t["dialect"] == dialect]
    
    return templates

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "schema_cache_hits": 0,
        "schema_cache_misses": 0,
        "query_cache_hits": 0,
        "query_cache_misses": 0,
        "schema_hit_rate": 0,
        "query_hit_rate": 0
    }

@router.post("/")
async def create_database_source(source: DatabaseSourceCreate):
    """Create a new database knowledge source"""
    return {
        "success": True,
        "message": f"Database source '{source.name}' created (simulation)",
        "source_id": 1
    }
