"""
Database Query Analytics API
Get real analytics from database_query_audit table
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/database/analytics", tags=["Database Analytics"])

@router.get("/stats")
async def get_database_query_stats(db: Session = Depends(get_db)):
    """Get real-time database query statistics"""
    try:
        # Query the database_query_audit table for real stats
        result = db.execute(text("""
            SELECT 
                COUNT(*) as total_queries,
                COUNT(CASE WHEN success = true THEN 1 END) as successful_queries,
                AVG(execution_time_ms) as avg_execution_time,
                MAX(execution_time_ms) as max_execution_time,
                SUM(row_count) as total_rows_queried,
                COUNT(DISTINCT source_id) as active_sources
            FROM database_query_audit
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)).fetchone()
        
        if not result:
            return {
                "total_queries": 0,
                "success_rate": 100.0,
                "avg_execution_time_ms": 0,
                "max_execution_time_ms": 0,
                "total_rows": 0,
                "active_sources": 0
            }
        
        total = result[0] or 0
        successful = result[1] or 0
        
        return {
            "total_queries": total,
            "success_rate": (successful / total * 100) if total > 0 else 100.0,
            "avg_execution_time_ms": round(result[2] or 0, 2),
            "max_execution_time_ms": round(result[3] or 0, 2),
            "total_rows": result[4] or 0,
            "active_sources": result[5] or 0
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        # Return zeros instead of failing
        return {
            "total_queries": 0,
            "success_rate": 100.0,
            "avg_execution_time_ms": 0,
            "max_execution_time_ms": 0,
            "total_rows": 0,
            "active_sources": 0
        }

@router.get("/performance")
async def get_database_performance(db: Session = Depends(get_db)):
    """Get database query performance over time"""
    try:
        result = db.execute(text("""
            SELECT 
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as queries,
                AVG(execution_time_ms) as avg_time,
                COUNT(CASE WHEN success = true THEN 1 END)::float / COUNT(*) * 100 as success_rate
            FROM database_query_audit
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', created_at)
            ORDER BY hour
        """)).fetchall()
        
        performance_data = [
            {
                "time": row[0].strftime("%H:%00") if row[0] else "00:00",
                "queries": row[1] or 0,
                "avg_time_ms": round(row[2] or 0, 2),
                "success_rate": round(row[3] or 100, 1)
            }
            for row in result
        ]
        
        return performance_data if performance_data else []
        
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        return []

@router.get("/top-queries")
async def get_top_queries(limit: int = 10, db: Session = Depends(get_db)):
    """Get most frequently executed queries"""
    try:
        result = db.execute(text("""
            SELECT 
                natural_language_query,
                generated_sql,
                COUNT(*) as execution_count,
                AVG(execution_time_ms) as avg_time,
                MAX(created_at) as last_executed
            FROM database_query_audit
            WHERE created_at > NOW() - INTERVAL '7 days'
            GROUP BY natural_language_query, generated_sql
            ORDER BY execution_count DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()
        
        return [
            {
                "query": row[0],
                "sql": row[1],
                "count": row[2],
                "avg_time_ms": round(row[3] or 0, 2),
                "last_executed": row[4].isoformat() if row[4] else None
            }
            for row in result
        ]
        
    except Exception as e:
        logger.error(f"Error getting top queries: {e}")
        return []


