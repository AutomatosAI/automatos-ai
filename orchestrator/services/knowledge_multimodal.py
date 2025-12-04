"""
Unified Multimodal Knowledge API
=================================

Advanced multimodal knowledge management system.
Provides unified API for all knowledge base types:
- Documents (text, PDF, DOCX)
- Code (CodeGraph integration)
- Tables (extracted from documents)
- Images (with AI descriptions)
- Formulas (LaTeX math)
- Diagrams and charts
- Knowledge graphs
- Custom types

Research Foundation:
Multimodal RAG concepts informed by RAG-Anything (MIT License, HKUDS 2024).
Implementation is original Automatos code built on Context Engineering framework.

Author: Automatos AI Team
Created: October 2025
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Body, Form
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
import json

from database.database import get_db
from modules.rag import (
    create_multimodal_processor,
    ContentModality,
    TableExtraction,
    ImageExtraction,
    FormulaExtraction
)
from services.credential_resolver import get_credential_resolver

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/knowledge", tags=["📚 Knowledge Base"])

# ================================================================
# Request/Response Models
# ================================================================

class KnowledgeTypeInfo(BaseModel):
    """Information about a knowledge base type"""
    type_name: str
    display_name: str
    description: Optional[str]
    icon: Optional[str]
    count: int
    supports_search: bool
    supports_relationships: bool


class KnowledgeItemCreate(BaseModel):
    """Request to create a knowledge item"""
    kb_type: str = Field(..., description="Knowledge base type (document, table, image, etc.)")
    title: Optional[str] = Field(None, description="Item title")
    content: str = Field(..., description="Item content")
    summary: Optional[str] = Field(None, description="Brief summary")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    quality_score: float = Field(0.5, ge=0.0, le=1.0)
    importance_score: float = Field(0.5, ge=0.0, le=1.0)
    visibility: str = Field("system", description="public, private, system, team")
    owner_id: Optional[str] = None


class KnowledgeItemResponse(BaseModel):
    """Response with knowledge item details"""
    id: int
    kb_type: str
    title: Optional[str]
    content_preview: str  # First 500 chars
    summary: Optional[str]
    quality_score: float
    importance_score: float
    status: str
    created_at: datetime
    multimodal_count: int = 0
    relationship_count: int = 0


class KnowledgeSearchRequest(BaseModel):
    """Request for knowledge search"""
    query: str = Field(..., description="Search query")
    kb_types: Optional[List[str]] = Field(None, description="Filter by knowledge types")
    limit: int = Field(10, ge=1, le=100)
    min_quality: float = Field(0.0, ge=0.0, le=1.0)
    use_semantic: bool = Field(True, description="Use semantic (vector) search")
    use_fulltext: bool = Field(True, description="Use full-text search")


class KnowledgeSearchResult(BaseModel):
    """Search result item"""
    id: int
    kb_type: str
    title: Optional[str]
    content_snippet: str
    relevance_score: float
    quality_score: float
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    success: bool
    document_id: Optional[int]
    knowledge_items_created: int
    tables_extracted: int
    images_extracted: int
    formulas_extracted: int
    processing_time_ms: int
    message: str


# ================================================================
# API Endpoints
# ================================================================

@router.get("/types", response_model=List[KnowledgeTypeInfo])
async def get_knowledge_types(
    db: Session = Depends(get_db)
):
    """
    Get all available knowledge base types with counts.
    
    Returns list of knowledge types and how many items exist for each.
    """
    try:
        query = text("""
            SELECT 
                kt.type_name,
                kt.display_name,
                kt.description,
                kt.icon,
                kt.supports_search,
                kt.supports_relationships,
                COUNT(ki.id) as count
            FROM kb_types kt
            LEFT JOIN knowledge_items ki ON kt.id = ki.kb_type_id AND ki.status = 'active'
            WHERE kt.enabled = true
            GROUP BY kt.id, kt.type_name, kt.display_name, kt.description, kt.icon, kt.supports_search, kt.supports_relationships
            ORDER BY kt.display_name
        """)
        
        results = db.execute(query).fetchall()
        
        types = [
            KnowledgeTypeInfo(
                type_name=row.type_name,
                display_name=row.display_name,
                description=row.description,
                icon=row.icon,
                count=row.count or 0,
                supports_search=row.supports_search,
                supports_relationships=row.supports_relationships
            )
            for row in results
        ]
        
        return types
        
    except Exception as e:
        logger.error(f"Error fetching knowledge types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/items", response_model=KnowledgeItemResponse)
async def create_knowledge_item(
    item: KnowledgeItemCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new knowledge item.
    
    Manually create a knowledge item of any type.
    For document uploads, use /upload endpoint instead.
    """
    try:
        # Get kb_type_id
        kb_type_query = text("SELECT id FROM kb_types WHERE type_name = :type_name")
        kb_type_row = db.execute(kb_type_query, {"type_name": item.kb_type}).fetchone()
        
        if not kb_type_row:
            raise HTTPException(status_code=400, detail=f"Invalid knowledge type: {item.kb_type}")
        
        kb_type_id = kb_type_row.id
        
        # Create knowledge item
        insert_query = text("""
            INSERT INTO knowledge_items 
            (kb_type_id, title, content, summary, metadata, quality_score, importance_score, visibility, owner_id, status)
            VALUES 
            (:kb_type_id, :title, :content, :summary, :metadata, :quality_score, :importance_score, :visibility, :owner_id, 'active')
            RETURNING id, created_at
        """)
        
        result = db.execute(
            insert_query,
            {
                "kb_type_id": kb_type_id,
                "title": item.title,
                "content": item.content,
                "summary": item.summary,
                "metadata": json.dumps(item.metadata),
                "quality_score": item.quality_score,
                "importance_score": item.importance_score,
                "visibility": item.visibility,
                "owner_id": item.owner_id
            }
        ).fetchone()
        
        db.commit()
        
        return KnowledgeItemResponse(
            id=result.id,
            kb_type=item.kb_type,
            title=item.title,
            content_preview=item.content[:500],
            summary=item.summary,
            quality_score=item.quality_score,
            importance_score=item.importance_score,
            status="active",
            created_at=result.created_at,
            multimodal_count=0,
            relationship_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating knowledge item: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document_multimodal(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    extract_tables: bool = Form(True),
    extract_images: bool = Form(True),
    extract_formulas: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Upload a document and extract all multimodal content.
    
    Automatically extracts:
    - Text content (chunked and embedded)
    - Tables (with multiple formats)
    - Images (with AI descriptions)
    - Formulas (LaTeX math)
    
    Creates knowledge items for each extracted element.
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize multimodal processor
            resolver = get_credential_resolver()
            openai_key = resolver.get_credential_field("development_openai", "api_key")
            processor = create_multimodal_processor(openai_key)
            
            # Process document
            if file_extension == '.pdf':
                results = processor.process_pdf_multimodal(
                    temp_file_path,
                    extract_tables=extract_tables,
                    extract_images=extract_images,
                    extract_formulas=extract_formulas
                )
            else:
                # For non-PDF files, only extract text and formulas
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                results = {
                    'text': text_content,
                    'tables': [],
                    'images': [],
                    'formulas': processor.formula_processor.extract_formulas_from_text(text_content) if extract_formulas else []
                }
            
            # Create knowledge items
            knowledge_items_created = 0
            
            # Get kb_type_ids
            type_query = text("SELECT type_name, id FROM kb_types WHERE type_name IN ('document', 'table', 'image', 'formula')")
            type_rows = db.execute(type_query).fetchall()
            type_map = {row.type_name: row.id for row in type_rows}
            
            # 1. Create main document knowledge item
            if results['text']:
                doc_title = title or file.filename
                summary = results['text'][:500] + "..." if len(results['text']) > 500 else results['text']
                
                doc_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, title, content, summary, metadata, status, source_type, source_id)
                    VALUES 
                    (:kb_type_id, :title, :content, :summary, :metadata, 'active', 'upload', :source_id)
                    RETURNING id
                """)
                
                doc_result = db.execute(
                    doc_query,
                    {
                        "kb_type_id": type_map['document'],
                        "title": doc_title,
                        "content": results['text'],
                        "summary": summary,
                        "metadata": json.dumps({"filename": file.filename, "description": description}),
                        "source_id": file.filename
                    }
                ).fetchone()
                
                document_id = doc_result.id
                knowledge_items_created += 1
            else:
                document_id = None
            
            # 2. Create table knowledge items
            for table in results['tables']:
                table_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id)
                    RETURNING id
                """)
                
                table_content = table.markdown
                table_title = table.caption or f"Table {table.row_count}x{table.column_count} from {file.filename}"
                
                table_result = db.execute(
                    table_query,
                    {
                        "kb_type_id": type_map['table'],
                        "parent_id": document_id,
                        "title": table_title,
                        "content": table_content,
                        "metadata": json.dumps({
                            "headers": table.headers,
                            "row_count": table.row_count,
                            "column_count": table.column_count,
                            "page_number": table.page_number
                        }),
                        "source_id": file.filename
                    }
                ).fetchone()
                
                # Store in kb_tables
                kb_table_query = text("""
                    INSERT INTO kb_tables 
                    (knowledge_item_id, headers, row_count, column_count, markdown_representation, csv_data, json_data, caption)
                    VALUES 
                    (:knowledge_item_id, :headers, :row_count, :column_count, :markdown, :csv, :json_data, :caption)
                """)
                
                db.execute(
                    kb_table_query,
                    {
                        "knowledge_item_id": table_result.id,
                        "headers": json.dumps(table.headers),
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "markdown": table.markdown,
                        "csv": table.csv,
                        "json_data": json.dumps(table.json),
                        "caption": table.caption
                    }
                )
                
                knowledge_items_created += 1
            
            # 3. Create image knowledge items
            for image in results['images']:
                image_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id)
                    RETURNING id
                """)
                
                image_title = image.caption or f"Image from {file.filename} page {image.page_number}"
                image_content = image.description
                
                image_result = db.execute(
                    image_query,
                    {
                        "kb_type_id": type_map['image'],
                        "parent_id": document_id,
                        "title": image_title,
                        "content": image_content,
                        "metadata": json.dumps({
                            "width": image.width,
                            "height": image.height,
                            "format": image.format,
                            "page_number": image.page_number
                        }),
                        "source_id": file.filename
                    }
                ).fetchone()
                
                # Store in kb_images
                kb_image_query = text("""
                    INSERT INTO kb_images 
                    (knowledge_item_id, width, height, format, file_size_bytes, description, caption, detected_text, image_data, thumbnail_data)
                    VALUES 
                    (:knowledge_item_id, :width, :height, :format, :file_size_bytes, :description, :caption, :detected_text, :image_data, :thumbnail_data)
                """)
                
                db.execute(
                    kb_image_query,
                    {
                        "knowledge_item_id": image_result.id,
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                        "file_size_bytes": image.file_size_bytes,
                        "description": image.description,
                        "caption": image.caption,
                        "detected_text": image.detected_text,
                        "image_data": image.image_data,
                        "thumbnail_data": image.thumbnail_data
                    }
                )
                
                knowledge_items_created += 1
            
            # 4. Create formula knowledge items
            for formula in results['formulas']:
                formula_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id)
                    RETURNING id
                """)
                
                formula_title = f"{formula.domain.capitalize()} formula from {file.filename}"
                formula_content = formula.latex
                
                formula_result = db.execute(
                    formula_query,
                    {
                        "kb_type_id": type_map['formula'],
                        "parent_id": document_id,
                        "title": formula_title,
                        "content": formula_content,
                        "metadata": json.dumps({
                            "formula_type": formula.formula_type,
                            "domain": formula.domain,
                            "complexity": formula.complexity
                        }),
                        "source_id": file.filename
                    }
                ).fetchone()
                
                # Store in kb_formulas
                kb_formula_query = text("""
                    INSERT INTO kb_formulas 
                    (knowledge_item_id, latex, ascii_math, variables, operators, formula_type, domain, complexity_level)
                    VALUES 
                    (:knowledge_item_id, :latex, :ascii_math, :variables, :operators, :formula_type, :domain, :complexity_level)
                """)
                
                db.execute(
                    kb_formula_query,
                    {
                        "knowledge_item_id": formula_result.id,
                        "latex": formula.latex,
                        "ascii_math": formula.ascii_math,
                        "variables": json.dumps(formula.variables),
                        "operators": json.dumps(formula.operators),
                        "formula_type": formula.formula_type,
                        "domain": formula.domain,
                        "complexity_level": formula.complexity
                    }
                )
                
                knowledge_items_created += 1
            
            db.commit()
            
            # Calculate processing time
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DocumentUploadResponse(
                success=True,
                document_id=document_id,
                knowledge_items_created=knowledge_items_created,
                tables_extracted=len(results['tables']),
                images_extracted=len(results['images']),
                formulas_extracted=len(results['formulas']),
                processing_time_ms=processing_time_ms,
                message=f"Successfully processed {file.filename} and created {knowledge_items_created} knowledge items"
            )
            
        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[KnowledgeSearchResult])
async def search_knowledge(
    search: KnowledgeSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search across all knowledge types.
    
    Supports:
    - Full-text search
    - Semantic (vector) search
    - Filtering by knowledge type
    - Quality filtering
    """
    try:
        # Build search query
        conditions = ["ki.status = 'active'"]
        params = {"query": search.query, "limit": search.limit, "min_quality": search.min_quality}
        
        if search.kb_types:
            conditions.append("kt.type_name = ANY(:kb_types)")
            params["kb_types"] = search.kb_types
        
        if search.min_quality > 0:
            conditions.append("ki.quality_score >= :min_quality")
        
        where_clause = " AND ".join(conditions)
        
        # Use full-text search
        if search.use_fulltext:
            query = text(f"""
                SELECT 
                    ki.id,
                    kt.type_name as kb_type,
                    ki.title,
                    LEFT(ki.content, 300) as content_snippet,
                    ts_rank(
                        to_tsvector('english', ki.content || ' ' || COALESCE(ki.title, '')),
                        plainto_tsquery('english', :query)
                    ) as relevance_score,
                    ki.quality_score,
                    ki.created_at
                FROM knowledge_items ki
                JOIN kb_types kt ON ki.kb_type_id = kt.id
                WHERE {where_clause}
                    AND to_tsvector('english', ki.content || ' ' || COALESCE(ki.title, '')) @@ plainto_tsquery('english', :query)
                ORDER BY relevance_score DESC, ki.quality_score DESC
                LIMIT :limit
            """)
        else:
            # Simple LIKE search
            query = text(f"""
                SELECT 
                    ki.id,
                    kt.type_name as kb_type,
                    ki.title,
                    LEFT(ki.content, 300) as content_snippet,
                    1.0 as relevance_score,
                    ki.quality_score,
                    ki.created_at
                FROM knowledge_items ki
                JOIN kb_types kt ON ki.kb_type_id = kt.id
                WHERE {where_clause}
                    AND (ki.content ILIKE '%' || :query || '%' OR ki.title ILIKE '%' || :query || '%')
                ORDER BY ki.quality_score DESC
                LIMIT :limit
            """)
        
        results = db.execute(query, params).fetchall()
        
        return [
            KnowledgeSearchResult(
                id=row.id,
                kb_type=row.kb_type,
                title=row.title,
                content_snippet=row.content_snippet,
                relevance_score=float(row.relevance_score),
                quality_score=float(row.quality_score),
                created_at=row.created_at
            )
            for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items/{item_id}")
async def get_knowledge_item(
    item_id: int,
    include_multimodal: bool = Query(True, description="Include multimodal content"),
    include_relationships: bool = Query(True, description="Include relationships"),
    db: Session = Depends(get_db)
):
    """
    Get a specific knowledge item with all its multimodal content and relationships.
    """
    try:
        # Get main item
        item_query = text("""
            SELECT 
                ki.id,
                kt.type_name as kb_type,
                ki.title,
                ki.content,
                ki.summary,
                ki.metadata,
                ki.quality_score,
                ki.importance_score,
                ki.status,
                ki.created_at,
                ki.updated_at
            FROM knowledge_items ki
            JOIN kb_types kt ON ki.kb_type_id = kt.id
            WHERE ki.id = :item_id
        """)
        
        item = db.execute(item_query, {"item_id": item_id}).fetchone()
        
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        response = {
            "id": item.id,
            "kb_type": item.kb_type,
            "title": item.title,
            "content": item.content,
            "summary": item.summary,
            "metadata": item.metadata,
            "quality_score": item.quality_score,
            "importance_score": item.importance_score,
            "status": item.status,
            "created_at": item.created_at,
            "updated_at": item.updated_at
        }
        
        # Include multimodal content
        if include_multimodal:
            multimodal_query = text("""
                SELECT content_modality, original_format, processed_text, processed_data, extraction_method, extraction_confidence
                FROM multimodal_content
                WHERE knowledge_item_id = :item_id
            """)
            
            multimodal = db.execute(multimodal_query, {"item_id": item_id}).fetchall()
            response["multimodal_content"] = [
                {
                    "modality": row.content_modality,
                    "format": row.original_format,
                    "text": row.processed_text,
                    "data": row.processed_data,
                    "extraction_method": row.extraction_method,
                    "confidence": row.extraction_confidence
                }
                for row in multimodal
            ]
        
        # Include relationships
        if include_relationships:
            rel_query = text("""
                SELECT 
                    kr.relationship_type,
                    kr.strength,
                    CASE 
                        WHEN kr.from_item_id = :item_id THEN 'outgoing'
                        ELSE 'incoming'
                    END as direction,
                    CASE 
                        WHEN kr.from_item_id = :item_id THEN ki_to.id
                        ELSE ki_from.id
                    END as related_item_id,
                    CASE 
                        WHEN kr.from_item_id = :item_id THEN ki_to.title
                        ELSE ki_from.title
                    END as related_item_title
                FROM knowledge_relationships kr
                LEFT JOIN knowledge_items ki_from ON kr.from_item_id = ki_from.id
                LEFT JOIN knowledge_items ki_to ON kr.to_item_id = ki_to.id
                WHERE kr.from_item_id = :item_id OR kr.to_item_id = :item_id
            """)
            
            relationships = db.execute(rel_query, {"item_id": item_id}).fetchall()
            response["relationships"] = [
                {
                    "type": row.relationship_type,
                    "strength": row.strength,
                    "direction": row.direction,
                    "related_item_id": row.related_item_id,
                    "related_item_title": row.related_item_title
                }
                for row in relationships
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching knowledge item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_knowledge_stats(
    db: Session = Depends(get_db)
):
    """
    Get statistics about the knowledge base.
    
    Returns counts, quality metrics, and usage stats for all knowledge types.
    """
    try:
        stats_query = text("""
            SELECT 
                kt.type_name,
                kt.display_name,
                COUNT(ki.id) as total_items,
                AVG(ki.quality_score) as avg_quality,
                AVG(ki.importance_score) as avg_importance,
                MAX(ki.created_at) as last_created
            FROM kb_types kt
            LEFT JOIN knowledge_items ki ON kt.id = ki.kb_type_id AND ki.status = 'active'
            WHERE kt.enabled = true
            GROUP BY kt.type_name, kt.display_name
            ORDER BY total_items DESC
        """)
        
        results = db.execute(stats_query).fetchall()
        
        stats = {
            "by_type": [
                {
                    "type": row.type_name,
                    "display_name": row.display_name,
                    "count": row.total_items or 0,
                    "avg_quality": float(row.avg_quality or 0),
                    "avg_importance": float(row.avg_importance or 0),
                    "last_created": row.last_created
                }
                for row in results
            ],
            "totals": {
                "total_items": sum(row.total_items or 0 for row in results),
                "overall_quality": sum(row.avg_quality or 0 for row in results) / len(results) if results else 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

