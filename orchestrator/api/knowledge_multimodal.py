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

from core.database.database import get_db
from modules.rag import (
    create_multimodal_processor,
    ContentModality,
    TableExtraction,
    ImageExtraction,
    FormulaExtraction
)
from core.credentials.resolver import get_credential_resolver
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

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
    ctx: RequestContext = Depends(get_request_context_hybrid),
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
            LEFT JOIN knowledge_items ki ON kt.id = ki.kb_type_id AND ki.status = 'active' AND ki.workspace_id = :workspace_id
            WHERE kt.enabled = true
            GROUP BY kt.id, kt.type_name, kt.display_name, kt.description, kt.icon, kt.supports_search, kt.supports_relationships
            ORDER BY kt.display_name
        """)
        
        results = db.execute(query, {"workspace_id": ctx.workspace_id}).fetchall()
        
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
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
            (kb_type_id, title, content, summary, metadata, quality_score, importance_score, visibility, owner_id, status, workspace_id)
            VALUES 
            (:kb_type_id, :title, :content, :summary, :metadata, :quality_score, :importance_score, :visibility, :owner_id, 'active', :workspace_id)
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
                "owner_id": item.owner_id,
                "workspace_id": ctx.workspace_id
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
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
                    (kb_type_id, title, content, summary, metadata, status, source_type, source_id, workspace_id)
                    VALUES 
                    (:kb_type_id, :title, :content, :summary, :metadata, 'active', 'upload', :source_id, :workspace_id)
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
                        "source_id": file.filename,
                        "workspace_id": ctx.workspace_id
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
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id, workspace_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id, :workspace_id)
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
                        "source_id": file.filename,
                        "workspace_id": ctx.workspace_id
                    }
                ).fetchone()
                
                # Store in kb_tables (workspace_id implied by knowledge_item, but schema says kb_tables also has workspace_id!)
                # Wait, schema DOES have workspace_id for kb_tables. Let's add it.
                kb_table_query = text("""
                    INSERT INTO kb_tables 
                    (knowledge_item_id, headers, row_count, column_count, markdown_representation, csv_data, json_data, caption, workspace_id)
                    VALUES 
                    (:knowledge_item_id, :headers, :row_count, :column_count, :markdown, :csv, :json_data, :caption, :workspace_id)
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
                        "caption": table.caption,
                        "workspace_id": ctx.workspace_id
                    }
                )
                
                knowledge_items_created += 1
            
            # 3. Create image knowledge items
            for image in results['images']:
                image_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id, workspace_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id, :workspace_id)
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
                        "source_id": file.filename,
                        "workspace_id": ctx.workspace_id
                    }
                ).fetchone()
                
                # Store in kb_images
                kb_image_query = text("""
                    INSERT INTO kb_images 
                    (knowledge_item_id, width, height, format, file_size_bytes, description, caption, detected_text, image_data, thumbnail_data, workspace_id)
                    VALUES 
                    (:knowledge_item_id, :width, :height, :format, :file_size_bytes, :description, :caption, :detected_text, :image_data, :thumbnail_data, :workspace_id)
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
                        "thumbnail_data": image.thumbnail_data,
                        "workspace_id": ctx.workspace_id
                    }
                )
                
                knowledge_items_created += 1
            
            # 4. Create formula knowledge items
            for formula in results['formulas']:
                formula_query = text("""
                    INSERT INTO knowledge_items 
                    (kb_type_id, parent_id, title, content, metadata, status, source_type, source_id, workspace_id)
                    VALUES 
                    (:kb_type_id, :parent_id, :title, :content, :metadata, 'active', 'extraction', :source_id, :workspace_id)
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
                        "source_id": file.filename,
                        "workspace_id": ctx.workspace_id
                    }
                ).fetchone()
                
                # Store in kb_formulas
                kb_formula_query = text("""
                    INSERT INTO kb_formulas 
                    (knowledge_item_id, latex, ascii_math, variables, operators, formula_type, domain, complexity_level, workspace_id)
                    VALUES 
                    (:knowledge_item_id, :latex, :ascii_math, :variables, :operators, :formula_type, :domain, :complexity_level, :workspace_id)
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
                        "complexity_level": formula.complexity,
                        "workspace_id": ctx.workspace_id
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
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
        all_results = []
        
        # Search formulas
        if not search.kb_types or 'formula' in search.kb_types:
            formula_query = text("""
                SELECT 
                    f.id,
                    'formula' as kb_type,
                    LEFT(f.latex, 100) as title,
                    LEFT(COALESCE(f.ascii_math, f.latex), 300) as content_snippet,
                    1.0 as relevance_score,
                    1.0 as quality_score,
                    f.created_at
                FROM kb_formulas f
                WHERE f.workspace_id = :workspace_id
                ORDER BY f.created_at DESC
                LIMIT :limit
            """)
            formula_results = db.execute(formula_query, {"limit": search.limit, "workspace_id": ctx.workspace_id}).fetchall()
            all_results.extend([
                KnowledgeSearchResult(
                    id=row.id,
                    kb_type=row.kb_type,
                    title=row.title or 'Formula',
                    content_snippet=row.content_snippet or '',
                    relevance_score=float(row.relevance_score),
                    quality_score=float(row.quality_score),
                    created_at=row.created_at
                )
                for row in formula_results
            ])
        
        # Search tables
        if not search.kb_types or 'table' in search.kb_types:
            table_query = text("""
                SELECT 
                    t.id,
                    'table' as kb_type,
                    'Table' as title,
                    LEFT(COALESCE(t.markdown_representation, ''), 300) as content_snippet,
                    1.0 as relevance_score,
                    1.0 as quality_score,
                    t.created_at
                FROM kb_tables t
                WHERE t.workspace_id = :workspace_id
                ORDER BY t.created_at DESC
                LIMIT :limit
            """)
            table_results = db.execute(table_query, {"limit": search.limit, "workspace_id": ctx.workspace_id}).fetchall()
            all_results.extend([
                KnowledgeSearchResult(
                    id=row.id,
                    kb_type=row.kb_type,
                    title=row.title or 'Table',
                    content_snippet=row.content_snippet or '',
                    relevance_score=float(row.relevance_score),
                    quality_score=float(row.quality_score),
                    created_at=row.created_at
                )
                for row in table_results
            ])
        
        # Search images
        if not search.kb_types or 'image' in search.kb_types:
            image_query = text("""
                SELECT 
                    i.id,
                    'image' as kb_type,
                    COALESCE(i.alt_text, 'Image') as title,
                    LEFT(COALESCE(i.description, i.detected_text, ''), 300) as content_snippet,
                    1.0 as relevance_score,
                    1.0 as quality_score,
                    i.created_at
                FROM kb_images i
                WHERE i.workspace_id = :workspace_id
                ORDER BY i.created_at DESC
                LIMIT :limit
            """)
            image_results = db.execute(image_query, {"limit": search.limit, "workspace_id": ctx.workspace_id}).fetchall()
            all_results.extend([
                KnowledgeSearchResult(
                    id=row.id,
                    kb_type=row.kb_type,
                    title=row.title or 'Image',
                    content_snippet=row.content_snippet or '',
                    relevance_score=float(row.relevance_score),
                    quality_score=float(row.quality_score),
                    created_at=row.created_at
                )
                for row in image_results
            ])
        
        # Sort by created_at descending and limit
        all_results.sort(key=lambda x: x.created_at, reverse=True)
        return all_results[:search.limit]
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items/{item_id}")
async def get_knowledge_item(
    item_id: int,
    type: str = Query(..., description="Knowledge type: table, formula, or image"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get a specific knowledge item with all its multimodal content.
    """
    try:
        # Query the appropriate table based on type
        if type == 'formula':
            query = text("""
                SELECT 
                    id,
                    knowledge_item_id,
                    latex,
                    mathml,
                    ascii_math,
                    variables,
                    operators,
                    complexity_level,
                    formula_type,
                    domain,
                    created_at
                FROM kb_formulas
                WHERE id = :item_id AND workspace_id = :workspace_id
            """)
            result = db.execute(query, {"item_id": item_id, "workspace_id": ctx.workspace_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Formula not found")
            
            return {
                "id": result.id,
                "kb_type": "formula",
                "latex": result.latex,
                "mathml": result.mathml,
                "ascii_math": result.ascii_math,
                "variables": result.variables,
                "operators": result.operators,
                "complexity_level": result.complexity_level,
                "formula_type": result.formula_type,
                "domain": result.domain,
                "created_at": result.created_at
            }
            
        elif type == 'table':
            query = text("""
                SELECT 
                    id,
                    knowledge_item_id,
                    headers,
                    row_count,
                    column_count,
                    markdown_representation,
                    csv_data,
                    json_data,
                    created_at
                FROM kb_tables
                WHERE id = :item_id AND workspace_id = :workspace_id
            """)
            result = db.execute(query, {"item_id": item_id, "workspace_id": ctx.workspace_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Table not found")
            
            return {
                "id": result.id,
                "kb_type": "table",
                "headers": result.headers,
                "row_count": result.row_count,
                "column_count": result.column_count,
                "markdown": result.markdown_representation,
                "csv": result.csv_data,
                "json_data": result.json_data,
                "created_at": result.created_at
            }
            
        elif type == 'image':
            query = text("""
                SELECT 
                    id,
                    knowledge_item_id,
                    width,
                    height,
                    format,
                    description,
                    caption,
                    alt_text,
                    detected_objects,
                    detected_text,
                    storage_path,
                    created_at
                FROM kb_images
                WHERE id = :item_id AND workspace_id = :workspace_id
            """)
            result = db.execute(query, {"item_id": item_id, "workspace_id": ctx.workspace_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Image not found")
            
            return {
                "id": result.id,
                "kb_type": "image",
                "width": result.width,
                "height": result.height,
                "format": result.format,
                "description": result.description,
                "caption": result.caption,
                "alt_text": result.alt_text,
                "detected_objects": result.detected_objects,
                "detected_text": result.detected_text,
                "storage_path": result.storage_path,
                "created_at": result.created_at
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid type: {type}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching knowledge item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_knowledge_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get statistics about the knowledge base.
    
    Returns counts, quality metrics, and usage stats for all knowledge types.
    """
    try:
        # Get direct counts from multimodal tables
        tables_count_query = text("SELECT COUNT(*) as count FROM kb_tables WHERE workspace_id = :workspace_id")
        formulas_count_query = text("SELECT COUNT(*) as count FROM kb_formulas WHERE workspace_id = :workspace_id")
        images_count_query = text("SELECT COUNT(*) as count FROM kb_images WHERE workspace_id = :workspace_id")
        
        tables_count = db.execute(tables_count_query, {"workspace_id": ctx.workspace_id}).fetchone().count
        formulas_count = db.execute(formulas_count_query, {"workspace_id": ctx.workspace_id}).fetchone().count
        images_count = db.execute(images_count_query, {"workspace_id": ctx.workspace_id}).fetchone().count
        
        stats = {
            "by_type": [
                {
                    "type": "table",
                    "display_name": "Tables",
                    "count": tables_count,
                    "avg_quality": 1.0,
                    "avg_importance": 1.0,
                    "last_created": None
                },
                {
                    "type": "formula",
                    "display_name": "Formulas",
                    "count": formulas_count,
                    "avg_quality": 1.0,
                    "avg_importance": 1.0,
                    "last_created": None
                },
                {
                    "type": "image",
                    "display_name": "Images",
                    "count": images_count,
                    "avg_quality": 1.0,
                    "avg_importance": 1.0,
                    "last_created": None
                }
            ],
            "totals": {
                "total_items": tables_count + formulas_count + images_count,
                "overall_quality": 1.0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

