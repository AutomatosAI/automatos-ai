"""
Multimodal Knowledge Tools - Executor for PRD-19 Multimodal Knowledge Base
===========================================================================

Tool executor that wraps the multimodal knowledge API for use by agents.
Integrates with PRD-17 Tool Registry.

Provides:
- search_tables: Find structured data tables
- search_images: Find images and diagrams with AI descriptions
- search_formulas: Find mathematical formulas
- search_multimodal: Unified search across all types
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class MultimodalKnowledgeTools:
    """
    Executor for multimodal knowledge base tools.
    
    Wraps the knowledge_multimodal API to provide tool interface
    compatible with PRD-17 Tool Registry.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize multimodal tools with database session"""
        self.db = db_session
        self.logger = logging.getLogger(f"{__name__}.MultimodalKnowledgeTools")
    
    async def search_tables(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for tables in the knowledge base.
        
        Args:
            query: Search query for tables
            limit: Maximum number of results
            
        Returns:
            {
                "success": bool,
                "results": List[table items],
                "count": int
            }
        """
        try:
            self.logger.info(f"Searching tables: query='{query}', limit={limit}")
            
            # Query knowledge base for tables
            if not self.db:
                from database.database import SessionLocal
                self.db = SessionLocal()
            
            search_query = text("""
                SELECT 
                    ki.id,
                    ki.title,
                    ki.content,
                    ki.summary,
                    kt.markdown_representation,
                    kt.csv_data,
                    kt.row_count,
                    kt.column_count,
                    ki.quality_score,
                    ki.created_at,
                    1 - (ki.embedding <=> (
                        SELECT embedding FROM knowledge_items 
                        WHERE content = :query 
                        LIMIT 1
                    )) as similarity
                FROM knowledge_items ki
                JOIN kb_tables kt ON kt.knowledge_item_id = ki.id
                JOIN kb_types kbt ON ki.kb_type_id = kbt.id
                WHERE kbt.type_name = 'table'
                    AND ki.status = 'active'
                    AND ki.embedding IS NOT NULL
                ORDER BY ki.embedding <=> (
                    SELECT embedding FROM knowledge_items 
                    WHERE content = :query 
                    LIMIT 1
                )
                LIMIT :limit
            """)
            
            results = self.db.execute(
                search_query,
                {"query": query, "limit": limit}
            ).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row.id,
                    "title": row.title,
                    "summary": row.summary or row.content[:200],
                    "markdown": row.markdown_representation,
                    "csv": row.csv_data,
                    "rows": row.row_count,
                    "columns": row.column_count,
                    "quality": row.quality_score,
                    "similarity": float(row.similarity) if row.similarity else 0.0,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                })
            
            self.logger.info(f"Found {len(formatted_results)} tables")
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error searching tables: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    async def search_images(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for images and visual content.
        
        Args:
            query: Search query for images
            limit: Maximum number of results
            
        Returns:
            {
                "success": bool,
                "results": List[image items with descriptions],
                "count": int
            }
        """
        try:
            self.logger.info(f"Searching images: query='{query}', limit={limit}")
            
            if not self.db:
                from database.database import SessionLocal
                self.db = SessionLocal()
            
            search_query = text("""
                SELECT 
                    ki.id,
                    ki.title,
                    ki.content,
                    kimg.description,
                    kimg.detected_text,
                    kimg.width,
                    kimg.height,
                    kimg.format,
                    ki.quality_score,
                    ki.created_at,
                    1 - (ki.embedding <=> (
                        SELECT embedding FROM knowledge_items 
                        WHERE content = :query 
                        LIMIT 1
                    )) as similarity
                FROM knowledge_items ki
                JOIN kb_images kimg ON kimg.knowledge_item_id = ki.id
                JOIN kb_types kbt ON ki.kb_type_id = kbt.id
                WHERE kbt.type_name = 'image'
                    AND ki.status = 'active'
                    AND ki.embedding IS NOT NULL
                ORDER BY ki.embedding <=> (
                    SELECT embedding FROM knowledge_items 
                    WHERE content = :query 
                    LIMIT 1
                )
                LIMIT :limit
            """)
            
            results = self.db.execute(
                search_query,
                {"query": query, "limit": limit}
            ).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "ocr_text": row.detected_text,
                    "width": row.width,
                    "height": row.height,
                    "format": row.format,
                    "quality": row.quality_score,
                    "similarity": float(row.similarity) if row.similarity else 0.0,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                })
            
            self.logger.info(f"Found {len(formatted_results)} images")
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error searching images: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    async def search_formulas(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for mathematical formulas.
        
        Args:
            query: Search query for formulas
            limit: Maximum number of results
            
        Returns:
            {
                "success": bool,
                "results": List[formula items with LaTeX],
                "count": int
            }
        """
        try:
            self.logger.info(f"Searching formulas: query='{query}', limit={limit}")
            
            if not self.db:
                from database.database import SessionLocal
                self.db = SessionLocal()
            
            search_query = text("""
                SELECT 
                    ki.id,
                    ki.title,
                    ki.content,
                    kf.latex,
                    kf.ascii_math,
                    kf.variables,
                    kf.operators,
                    kf.formula_type,
                    kf.domain,
                    kf.complexity_level,
                    ki.quality_score,
                    ki.created_at,
                    1 - (ki.embedding <=> (
                        SELECT embedding FROM knowledge_items 
                        WHERE content = :query 
                        LIMIT 1
                    )) as similarity
                FROM knowledge_items ki
                JOIN kb_formulas kf ON kf.knowledge_item_id = ki.id
                JOIN kb_types kbt ON ki.kb_type_id = kbt.id
                WHERE kbt.type_name = 'formula'
                    AND ki.status = 'active'
                    AND ki.embedding IS NOT NULL
                ORDER BY ki.embedding <=> (
                    SELECT embedding FROM knowledge_items 
                    WHERE content = :query 
                    LIMIT 1
                )
                LIMIT :limit
            """)
            
            results = self.db.execute(
                search_query,
                {"query": query, "limit": limit}
            ).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row.id,
                    "title": row.title,
                    "latex": row.latex,
                    "ascii": row.ascii_math,
                    "variables": row.variables,
                    "operators": row.operators,
                    "formula_type": row.formula_type,
                    "domain": row.domain,
                    "complexity": row.complexity_level,
                    "quality": row.quality_score,
                    "similarity": float(row.similarity) if row.similarity else 0.0,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                })
            
            self.logger.info(f"Found {len(formatted_results)} formulas")
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error searching formulas: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    async def search_multimodal(
        self,
        query: str,
        kb_types: List[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified search across all knowledge types.
        
        Args:
            query: Search query
            kb_types: List of knowledge types to search (default: all)
            limit: Maximum total results
            
        Returns:
            {
                "success": bool,
                "results": List[items from all types],
                "count": int,
                "breakdown": Dict[type, count]
            }
        """
        try:
            self.logger.info(f"Multimodal search: query='{query}', types={kb_types}, limit={limit}")
            
            if kb_types is None:
                kb_types = ["document", "table", "image", "formula", "codegraph"]
            
            if not self.db:
                from database.database import SessionLocal
                self.db = SessionLocal()
            
            # Search across all specified types
            search_query = text("""
                SELECT 
                    ki.id,
                    ki.title,
                    ki.content,
                    ki.summary,
                    kbt.type_name as kb_type,
                    kbt.display_name as kb_type_display,
                    ki.quality_score,
                    ki.importance_score,
                    ki.created_at,
                    1 - (ki.embedding <=> (
                        SELECT embedding FROM knowledge_items 
                        WHERE content = :query 
                        LIMIT 1
                    )) as similarity
                FROM knowledge_items ki
                JOIN kb_types kbt ON ki.kb_type_id = kbt.id
                WHERE kbt.type_name = ANY(:kb_types)
                    AND ki.status = 'active'
                    AND ki.embedding IS NOT NULL
                ORDER BY ki.embedding <=> (
                    SELECT embedding FROM knowledge_items 
                    WHERE content = :query 
                    LIMIT 1
                )
                LIMIT :limit
            """)
            
            results = self.db.execute(
                search_query,
                {
                    "query": query,
                    "kb_types": kb_types,
                    "limit": limit
                }
            ).fetchall()
            
            # Format results
            formatted_results = []
            breakdown = {}
            
            for row in results:
                kb_type = row.kb_type
                if kb_type not in breakdown:
                    breakdown[kb_type] = 0
                breakdown[kb_type] += 1
                
                formatted_results.append({
                    "id": row.id,
                    "kb_type": kb_type,
                    "kb_type_display": row.kb_type_display,
                    "title": row.title,
                    "summary": row.summary or row.content[:200],
                    "content": row.content[:500],  # Truncate content
                    "quality": row.quality_score,
                    "importance": row.importance_score,
                    "similarity": float(row.similarity) if row.similarity else 0.0,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                })
            
            self.logger.info(f"Found {len(formatted_results)} items across {len(breakdown)} types")
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "breakdown": breakdown,
                "query": query,
                "kb_types_searched": kb_types
            }
            
        except Exception as e:
            self.logger.error(f"Error in multimodal search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0,
                "breakdown": {}
            }


# Singleton instance
_instance: Optional[MultimodalKnowledgeTools] = None


def get_multimodal_knowledge_tools(db_session: Optional[Session] = None) -> MultimodalKnowledgeTools:
    """Get singleton instance of MultimodalKnowledgeTools"""
    global _instance
    if _instance is None:
        _instance = MultimodalKnowledgeTools(db_session)
    return _instance

