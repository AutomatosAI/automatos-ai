"""
Database Knowledge API Routes
==============================
PRD-21: API endpoints for database knowledge source management
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.database.database import get_db
from core.models.database_knowledge import (
    DatabaseKnowledgeSourceCreate,
    SemanticMetricCreate,
    SemanticDimensionCreate,
    DatabaseQueryRequest,
    QueryTemplateExecute
)
from modules.nl2sql import DatabaseKnowledgeService, NaturalLanguageToSQLService
from core.database.database_cache_service import get_database_cache_service
from modules.tools.services.database_tool_integration import get_database_tool_integration
from core.credentials.resolver import get_credential_resolver

import logging
logger = logging.getLogger(__name__)

# NEW imports for introspection wiring
from core.models.database_knowledge import DatabaseKnowledgeSource
from core.credentials.service import CredentialStore
from modules.nl2sql import DatabaseIntrospectionService
# NEW import for auditing
from core.models.database_knowledge import DatabaseQueryAudit
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge/sources/database", tags=["Database Knowledge"])

# Initialize services
db_service = None
cache_service = None
tool_integration = None

def get_services():
    global db_service, cache_service, tool_integration
    if not db_service:
        from core.llm import create_llm_manager
        from modules.rag import RAGService
        from modules.search.services.context_engineering_service import ContextEngineeringService
        from core.services.audit_service import AuditService
        
        db_service = DatabaseKnowledgeService(
            credential_resolver=get_credential_resolver(),
            llm_provider=create_llm_manager(service_name="orchestrator"),
            rag_service=RAGService(),
            context_engineering=ContextEngineeringService(),
            audit_service=AuditService()
        )
        cache_service = get_database_cache_service()
        tool_integration = get_database_tool_integration()
    
    return db_service, cache_service, tool_integration


def _map_dialect_to_introspector(dialect_value: Optional[str]) -> str:
    """Map stored dialect to DatabaseIntrospectionService dialects."""
    if not dialect_value:
        return "postgres"  # default safe assumption
    dv = dialect_value.lower()
    if dv.startswith("postgres"):
        return "postgres"
    if dv.startswith("mysql"):
        return "mysql"
    raise HTTPException(status_code=400, detail=f"Unsupported dialect for introspection: {dialect_value}")


@router.get("/", response_model=List[Dict[str, Any]])
async def get_item(
    ctx: RequestContext = Depends(get_request_context_hybrid), 
    db: Session = Depends(get_db),
    active_only: bool = False  # Show all sources by default
):
    """
    List all database knowledge sources for the current tenant.
    """
    try:
        query = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
            DatabaseKnowledgeSource.tenant_id == str(ctx.workspace_id)
        )
        
        if active_only:
            query = query.filter(DatabaseKnowledgeSource.is_active == True)
        
        sources = query.order_by(DatabaseKnowledgeSource.created_at.desc()).all()
        
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "credential_id": s.credential_id,
                "dialect": s.dialect,
                "is_active": s.is_active,
                "status": s.status,
                "total_queries_executed": s.total_queries_executed or 0,
                "avg_query_time_ms": s.avg_query_time_ms,
                "last_introspected": s.last_introspected.isoformat() if s.last_introspected else None,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "schema_tables_count": len(s.schema_metadata.get('tables', {})) if s.schema_metadata else 0,
            }
            for s in sources
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=Dict[str, Any])
async def create_database_source(
    source: DatabaseKnowledgeSourceCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Add a new database as a knowledge source.
    Introspects schema and creates agent tools.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating database source: name={source.name}, credential_id={source.credential_id}, dialect={source.dialect}")
    
    service, cache, tools = get_services()
    
    try:
        # Create database source
        result = await service.add_database_source(
            name=source.name,
            credential_id=source.credential_id,
            tenant_id=str(ctx.workspace_id),
            description=source.description
        )
        
        # Create tools for agents
        created_tools = tools.create_database_tools(result)
        
        return {
            "success": True,
            "source_id": result.id,
            "message": f"Database source '{source.name}' created successfully",
            "tools_created": len(created_tools),
            "schema_tables": len(result.schema_metadata.get('tables', {})) if result.schema_metadata else 0
        }
    
    except Exception as e:
        logger.error(f"Failed to create database source: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to create database source")


@router.post("/{source_id}/query", response_model=Dict[str, Any])
async def query_database(
    source_id: int,
    request: DatabaseQueryRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Execute natural language query against database.
    Returns results with visualization hints.
    """
    service, cache, _ = get_services()
    
    try:
        result = await service.smart_query(
            source_id=str(source_id),
            text=request.query,
            user_id="1",  # TODO: Get from auth
            agent_id=None
        )
        
        return result
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"API query_database failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Query execution failed")


@router.post("/{source_id}/introspect")
async def introspect_schema(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Re-introspect database schema and update metadata.
    """
    # Fetch source
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")
    
    # Resolve DB credentials
    cred_store = CredentialStore(db)
    try:
        creds = cred_store.get_decrypted_credential(
            credential_id=source.credential_id,
            service_name="database_introspection"
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to resolve credentials for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to resolve database credentials")

    # Introspect
    dialect = _map_dialect_to_introspector(source.dialect)
    try:
        inspector = DatabaseIntrospectionService(credential=creds, dialect=dialect)
        metadata = inspector.introspect(include_samples=True, sample_limit=5)
    except Exception as e:
        logging.getLogger(__name__).error(f"Introspection failed for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Schema introspection failed")
    
    # Persist
    source.schema_metadata = metadata
    source.last_introspected = datetime.utcnow()
    db.commit()
    
    # Return summary
    tables = metadata.get("tables", []) or []
    relationships = metadata.get("relationships", []) or []
    return {
        "success": True,
        "message": "Schema introspection completed",
        "tables": len(tables),
        "relationships": len(relationships),
        "introspection_ms": metadata.get("stats", {}).get("introspection_ms")
    }


@router.get("/{source_id}/schema")
async def get_schema(
    source_id: int,
    use_cache: bool = True,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get database schema metadata.
    """
    _, cache, _ = get_services()
    
    # Try cache first
    if use_cache:
        try:
            cached = cache.get_cached_schema(source_id, tenant_id=str(ctx.workspace_id))
            if cached:
                return cached
        except Exception:
            # Cache service optional; ignore on failure
            pass
    
    # Load from DB
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")
    
    if not source.schema_metadata:
        raise HTTPException(status_code=404, detail="Schema metadata not available; run introspection")
    
    return source.schema_metadata


@router.post("/{source_id}/semantic")
async def update_semantic_layer(
    source_id: int,
    metrics: List[SemanticMetricCreate] = Body(default=[]),
    dimensions: List[SemanticDimensionCreate] = Body(default=[]),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Update semantic layer (metrics and dimensions).
    """
    service, cache, _ = get_services()
    
    try:
        # Convert to proper types
        from modules.nl2sql import SemanticMetric, SemanticDimension
        
        metric_objects = [
            SemanticMetric(**m.dict()) for m in metrics
        ]
        dimension_objects = [
            SemanticDimension(**d.dict()) for d in dimensions
        ]
        
        await service.update_semantic_layer(
            source_id=str(source_id),
            metrics=metric_objects,
            dimensions=dimension_objects
        )
        
        return {
            "success": True,
            "metrics_updated": len(metrics),
            "dimensions_updated": len(dimensions)
        }
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to update semantic layer for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to update semantic layer")


@router.get("/{source_id}")
async def get_database_source(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get database source details.
    """
    # Query database for source details
    source = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")
    
    return {
        "id": source.id,
        "name": source.name,
        "description": source.description,
        "dialect": source.dialect,
        "status": source.status,
        "is_active": source.is_active,
        "total_queries_executed": source.total_queries_executed,
        "avg_query_time_ms": source.avg_query_time_ms,
        "last_introspected": source.last_introspected,
        "created_at": source.created_at
    }


@router.delete("/{source_id}")
async def delete_database_source(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Delete a database source and its associated schema/semantic data.
    """
    # Query database for source details
    source = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")
    
    source_name = source.name
    
    try:
        # Delete the source (cascade will handle related records)
        db.delete(source)
        db.commit()
        
        return {
            "success": True,
            "message": f"Database source '{source_name}' deleted successfully",
            "deleted_id": source_id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates/list")
async def list_query_templates(
    dialect: Optional[str] = None,
    category: Optional[str] = None,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List available query templates.
    """
    from core.models.database_knowledge import DatabaseQueryTemplate
    
    query = db.query(DatabaseQueryTemplate)
    
    if dialect:
        query = query.filter(DatabaseQueryTemplate.dialect == dialect)
    if category:
        query = query.filter(DatabaseQueryTemplate.category == category)
    
    templates = query.filter(
        DatabaseQueryTemplate.is_featured == True
    ).limit(20).all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "natural_language": t.natural_language,
            "category": t.category,
            "visualization_type": t.visualization_type,
            "parameters": t.parameters
        }
        for t in templates
    ]


@router.post("/{source_id}/template/execute")
async def execute_template(
    source_id: int,
    request: QueryTemplateExecute,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Execute a query template with parameters.
    """
    service, _, _ = get_services()
    
    try:
        # This would execute the template
        # Implementation would be in the service
        return {
            "success": True,
            "template_id": request.template_id,
            "data": [],
            "visualization_type": "table"
        }
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to execute template for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Template execution failed")


@router.post("/{source_id}/query/sql")
async def execute_validated_sql(
    source_id: int,
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    DEV-ONLY: Execute a validated SELECT statement against the source.
    Body: { sql: string, max_rows?: number }
    """
    sql = (payload.get("sql") or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="Missing sql")

    # Fetch source
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(DatabaseKnowledgeSource.workspace_id == ctx.workspace_id).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")

    # Validate SQL
    from modules.nl2sql import SQLValidator, SQLValidationError
    validator = SQLValidator(max_limit=min(int(payload.get("max_rows", 1000)), source.max_rows_limit or 1000))
    try:
        validated_sql, reasons = validator.validate_and_rewrite(sql, schema_metadata=source.schema_metadata)
    except SQLValidationError as e:
        # audit failure
        db.add(DatabaseQueryAudit(
            tenant_id=str(ctx.workspace_id),
            source_id=source.id,
            user_id=None,
            agent_id=None,
            session_id=None,
            natural_language_query=None,
            generated_sql=sql,
            validated_sql=None,
            execution_time_ms=0,
            row_count=0,
            bytes_processed=None,
            success=False,
            error_message=str(e),
            validation_errors={"error": str(e)},
            was_cached=False,
            cache_key=None,
            visualization_type=None,
            confidence_score=None
        ))
        db.commit()
        logging.getLogger(__name__).error(f"SQL validation failed for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="SQL validation failed")

    # Resolve creds and execute
    from sqlalchemy import create_engine, text
    from core.credentials.service import CredentialStore
    cred_store = CredentialStore(db)
    try:
        creds = cred_store.get_decrypted_credential(
            credential_id=source.credential_id,
            service_name="database_query"
        )
    except Exception as e:
        # audit failure
        db.add(DatabaseQueryAudit(
            tenant_id=str(ctx.workspace_id),
            source_id=source.id,
            user_id=None,
            agent_id=None,
            session_id=None,
            natural_language_query=None,
            generated_sql=sql,
            validated_sql=validated_sql,
            execution_time_ms=0,
            row_count=0,
            bytes_processed=None,
            success=False,
            error_message=f"Credential error: {e}",
            validation_errors=None,
            was_cached=False,
            cache_key=None,
            visualization_type=None,
            confidence_score=None
        ))
        db.commit()
        logging.getLogger(__name__).error(f"Failed to resolve credentials for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to resolve database credentials")

    # Build URL using SQLAlchemy URL.create() to safely escape credential values
    from sqlalchemy.engine import URL as SAURL
    from urllib.parse import quote_plus
    dialect = source.dialect.lower()
    if dialect.startswith("postgres"):
        url = SAURL.create(
            drivername="postgresql+psycopg2",
            username=creds.get('user'),
            password=creds.get('password'),
            host=creds.get('host'),
            port=int(creds.get('port', 5432)),
            database=creds.get('database'),
        )
    elif dialect.startswith("mysql"):
        url = SAURL.create(
            drivername="mysql+pymysql",
            username=creds.get('user'),
            password=creds.get('password'),
            host=creds.get('host'),
            port=int(creds.get('port', 3306)),
            database=creds.get('database'),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported dialect: {source.dialect}")

    engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)

    # Execute with timeout if supported (Postgres SET LOCAL statement_timeout)
    rows: List[Dict[str, Any]] = []
    columns: List[str] = []
    duration_ms = 0
    try:
        with engine.connect() as conn:
            # Set timeout for Postgres
            if dialect.startswith("postgres") and (source.query_timeout_seconds or 0) > 0:
                timeout_ms = int((source.query_timeout_seconds or 30) * 1000)
                conn.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
            import time
            start = time.time()
            result = conn.execute(text(validated_sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, r)) for r in result.fetchall()]
            duration_ms = int((time.time() - start) * 1000)
            # audit success
            db.add(DatabaseQueryAudit(
                tenant_id=str(ctx.workspace_id),
                source_id=source.id,
                user_id=None,
                agent_id=None,
                session_id=None,
                natural_language_query=None,
                generated_sql=sql,
                validated_sql=validated_sql,
                execution_time_ms=duration_ms,
                row_count=len(rows),
                bytes_processed=None,
                success=True,
                error_message=None,
                validation_errors=None,
                was_cached=False,
                cache_key=None,
                visualization_type=None,
                confidence_score=None
            ))
            db.commit()
    except Exception as e:
        # audit failure
        db.add(DatabaseQueryAudit(
            tenant_id=str(ctx.workspace_id),
            source_id=source.id,
            user_id=None,
            agent_id=None,
            session_id=None,
            natural_language_query=None,
            generated_sql=sql,
            validated_sql=validated_sql,
            execution_time_ms=duration_ms,
            row_count=0,
            bytes_processed=None,
            success=False,
            error_message=str(e),
            validation_errors=None,
            was_cached=False,
            cache_key=None,
            visualization_type=None,
            confidence_score=None
        ))
        db.commit()
        logging.getLogger(__name__).error(f"SQL query execution failed for source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Query execution failed")

    return {
        "sql": validated_sql,
        "reasons": reasons,
        "columns": columns,
        "rows": rows,
        "stats": {"duration_ms": duration_ms, "row_count": len(rows)}
    }


@router.get("/cache/stats")
async def get_cache_statistics(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    tenant_id: int = 1
):
    """
    Get cache statistics for monitoring.
    """
    _, cache, _ = get_services()

    stats = cache.get_cache_stats(tenant_id)

    return stats


# =============================================================================
# PRD-61: Training Examples API (US-004, US-013)
# =============================================================================

@router.get("/{source_id}/examples")
async def list_training_examples(
    source_id: int,
    verified_only: bool = False,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """List training examples for a database source."""
    from core.models.database_knowledge import NL2SQLTrainingExample

    query = db.query(NL2SQLTrainingExample).filter(
        NL2SQLTrainingExample.workspace_id == ctx.workspace_id,
        NL2SQLTrainingExample.database_source_id == source_id
    )
    if verified_only:
        query = query.filter(NL2SQLTrainingExample.is_verified == True)

    examples = query.order_by(NL2SQLTrainingExample.created_at.desc()).all()

    return [
        {
            "id": ex.id,
            "question": ex.question,
            "sql": ex.sql,
            "tables_used": ex.tables_used,
            "is_verified": ex.is_verified,
            "verification_source": ex.verification_source,
            "usage_count": ex.usage_count or 0,
            "last_used_at": ex.last_used_at.isoformat() if ex.last_used_at else None,
            "created_at": ex.created_at.isoformat() if ex.created_at else None,
            "metadata": ex.extra_metadata or {},
        }
        for ex in examples
    ]


@router.post("/{source_id}/examples")
async def add_training_example(
    source_id: int,
    body: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Add a new training example (question/SQL pair)."""
    from modules.nl2sql.training.example_store import SQLExampleStore

    store = SQLExampleStore(db_session=db)
    example_id = await store.add_example(
        question=body.get("question", ""),
        sql=body.get("sql", ""),
        database_source_id=str(source_id),
        workspace_id=str(ctx.workspace_id),
        tables_used=body.get("tables_used", []),
        is_verified=body.get("is_verified", False),
        verification_source="manual",
        created_by=str(ctx.user_id) if ctx.user_id else None,
        metadata=body.get("metadata", {})
    )

    return {"success": True, "example_id": example_id}


@router.put("/{source_id}/examples/{example_id}")
async def update_training_example(
    source_id: int,
    example_id: int,
    body: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Update a training example."""
    from core.models.database_knowledge import NL2SQLTrainingExample

    example = db.query(NL2SQLTrainingExample).filter(
        NL2SQLTrainingExample.id == example_id,
        NL2SQLTrainingExample.workspace_id == ctx.workspace_id,
        NL2SQLTrainingExample.database_source_id == source_id
    ).first()

    if not example:
        raise HTTPException(status_code=404, detail="Training example not found")

    if "question" in body:
        example.question = body["question"]
    if "sql" in body:
        example.sql = body["sql"]
    if "tables_used" in body:
        example.tables_used = body["tables_used"]
    if "is_verified" in body:
        example.is_verified = body["is_verified"]
    if "metadata" in body:
        example.extra_metadata = body["metadata"]

    from datetime import datetime
    example.updated_at = datetime.utcnow()
    db.commit()

    return {"success": True, "example_id": example_id}


@router.put("/{source_id}/examples/{example_id}/verify")
async def verify_training_example(
    source_id: int,
    example_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Mark an example as verified (Golden SQL)."""
    from modules.nl2sql.training.example_store import SQLExampleStore

    store = SQLExampleStore(db_session=db)
    await store.mark_verified(example_id, str(ctx.workspace_id))

    return {"success": True, "example_id": example_id, "verified": True}


@router.delete("/{source_id}/examples/{example_id}")
async def delete_training_example(
    source_id: int,
    example_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Delete a training example."""
    from modules.nl2sql.training.example_store import SQLExampleStore

    store = SQLExampleStore(db_session=db)
    await store.delete_example(example_id, str(ctx.workspace_id))

    return {"success": True, "deleted_id": example_id}


@router.post("/{source_id}/examples/import")
async def import_training_examples(
    source_id: int,
    body: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Bulk import examples from JSON array."""
    from modules.nl2sql.training.example_store import SQLExampleStore

    examples = body.get("examples", [])
    if not examples:
        raise HTTPException(status_code=400, detail="No examples provided")

    store = SQLExampleStore(db_session=db)
    imported = 0

    for ex in examples:
        try:
            await store.add_example(
                question=ex.get("question", ""),
                sql=ex.get("sql", ""),
                database_source_id=str(source_id),
                workspace_id=str(ctx.workspace_id),
                tables_used=ex.get("tables_used", []),
                is_verified=ex.get("is_verified", False),
                verification_source="import",
                created_by=str(ctx.user_id) if ctx.user_id else None,
            )
            imported += 1
        except Exception as e:
            logger.warning(f"Failed to import example: {e}")

    return {"success": True, "imported": imported, "total": len(examples)}


@router.get("/{source_id}/examples/stats")
async def get_training_example_stats(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get training example statistics."""
    from modules.nl2sql.training.example_store import SQLExampleStore

    store = SQLExampleStore(db_session=db)
    stats = await store.get_stats(str(source_id), str(ctx.workspace_id))

    return stats


# =============================================================================
# PRD-61: Schema Refresh API (US-010)
# =============================================================================

@router.post("/{source_id}/schema/refresh")
async def refresh_schema(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Re-introspect schema and invalidate cache.
    PRD-61 Bug Fix: Explicit cache invalidation on schema changes.
    """
    source = db.query(DatabaseKnowledgeSource).filter(
        DatabaseKnowledgeSource.id == source_id,
        DatabaseKnowledgeSource.workspace_id == ctx.workspace_id
    ).first()

    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")

    # Resolve credentials and introspect
    cred_store = CredentialStore(db)
    try:
        creds = cred_store.get_decrypted_credential(
            credential_id=source.credential_id,
            service_name="schema_refresh"
        )
    except Exception as e:
        logger.error("Failed to resolve credentials for schema refresh on source %s: %s", source_id, e)
        raise HTTPException(status_code=400, detail="Failed to resolve database credentials")

    dialect = _map_dialect_to_introspector(source.dialect)
    try:
        inspector = DatabaseIntrospectionService(credential=creds, dialect=dialect)
        metadata = inspector.introspect(include_samples=True, sample_limit=5)
    except Exception as e:
        logger.error("Schema introspection failed for source %s: %s", source_id, e)
        raise HTTPException(status_code=400, detail="Schema introspection failed")

    # Update source
    source.schema_metadata = metadata
    source.last_introspected = datetime.utcnow()
    db.commit()

    # Invalidate schema cache
    from modules.nl2sql.schema.provider import get_schema_provider
    try:
        provider = get_schema_provider(db)
        provider.invalidate_cache(str(source_id))
    except Exception:
        pass

    tables = metadata.get("tables", []) or []
    return {
        "success": True,
        "tables": len(tables),
        "message": "Schema refreshed and cache invalidated"
    }


# =============================================================================
# PRD-61: Benchmark API (US-018)
# =============================================================================

@router.post("/{source_id}/benchmark/run")
async def run_benchmark(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Run NL2SQL benchmark against verified training examples."""
    from modules.nl2sql.benchmarks.runner import NL2SQLBenchmarkRunner
    from modules.nl2sql.training.example_store import SQLExampleStore
    from core.models.database_knowledge import NL2SQLBenchmarkRun, NL2SQLBenchmarkResult

    # Get source
    source = db.query(DatabaseKnowledgeSource).filter(
        DatabaseKnowledgeSource.id == source_id,
        DatabaseKnowledgeSource.workspace_id == ctx.workspace_id
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")

    if not source.schema_metadata:
        raise HTTPException(status_code=400, detail="No schema metadata. Run introspection first.")

    service, _, _ = get_services()

    runner = NL2SQLBenchmarkRunner(
        nl2sql_service=NaturalLanguageToSQLService(llm_provider=service.llm_provider),
        example_store=SQLExampleStore(db_session=db)
    )

    result = await runner.run_benchmark(
        database_source_id=str(source_id),
        workspace_id=str(ctx.workspace_id),
        schema_metadata=source.schema_metadata,
        dialect=source.dialect or "postgresql"
    )

    # Persist benchmark run
    run = NL2SQLBenchmarkRun(
        workspace_id=ctx.workspace_id,
        database_source_id=source_id,
        total_examples=result.total,
        exact_match_rate=result.exact_match_rate,
        execution_match_rate=result.execution_match_rate,
    )
    db.add(run)
    db.flush()

    for detail in result.details:
        db.add(NL2SQLBenchmarkResult(
            benchmark_run_id=run.id,
            question=detail.get('question'),
            expected_sql=detail.get('expected_sql'),
            generated_sql=detail.get('generated_sql'),
            exact_match=detail.get('exact_match', False),
            execution_match=detail.get('execution_match', False),
        ))

    db.commit()

    return {
        "success": True,
        "benchmark_id": run.id,
        "total": result.total,
        "exact_match_rate": round(result.exact_match_rate, 3),
        "execution_match_rate": round(result.execution_match_rate, 3),
        "details": result.details[:20],  # Limit response size
    }


@router.get("/{source_id}/benchmark/history")
async def get_benchmark_history(
    source_id: int,
    limit: int = 50,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get historical benchmark results for accuracy trends."""
    from core.models.database_knowledge import NL2SQLBenchmarkRun

    runs = db.query(NL2SQLBenchmarkRun).filter(
        NL2SQLBenchmarkRun.workspace_id == ctx.workspace_id,
        NL2SQLBenchmarkRun.database_source_id == source_id
    ).order_by(NL2SQLBenchmarkRun.created_at.desc()).limit(min(limit, 100)).all()

    return [
        {
            "id": r.id,
            "total_examples": r.total_examples,
            "exact_match_rate": r.exact_match_rate,
            "execution_match_rate": r.execution_match_rate,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]
