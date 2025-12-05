"""
Database Knowledge API Routes
==============================
PRD-21: API endpoints for database knowledge source management
"""

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
from modules.nl_to_sql import DatabaseKnowledgeService
from core.database.database_cache_service import get_database_cache_service
from modules.tools.services.database_tool_integration import get_database_tool_integration
from core.credentials.resolver import get_credential_resolver

# NEW imports for introspection wiring
from core.models.database_knowledge import DatabaseKnowledgeSource
from core.credentials.service import CredentialStore
from modules.nl_to_sql import DatabaseIntrospectionService
# NEW import for auditing
from core.models.database_knowledge import DatabaseQueryAudit

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
        from services.audit_service import AuditService
        
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
async def list_database_sources(
    db: Session = Depends(get_db),
    active_only: bool = False  # Show all sources by default
):
    """
    List all database knowledge sources for the current tenant.
    """
    try:
        query = db.query(DatabaseKnowledgeSource).filter(
            DatabaseKnowledgeSource.tenant_id == 1  # TODO: Get from auth context
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
        raise HTTPException(status_code=500, detail=f"Failed to list database sources: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_database_source(
    source: DatabaseKnowledgeSourceCreate,
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
            tenant_id=1,  # TODO: Get from auth context
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
        import traceback
        logger.error(f"Failed to create database source: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{source_id}/query", response_model=Dict[str, Any])
async def query_database(
    source_id: int,
    request: DatabaseQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Execute natural language query against database.
    Returns results with visualization hints.
    """
    service, cache, _ = get_services()
    
    try:
        result = await service.query_database(
            source_id=str(source_id),
            natural_language_query=request.query,
            user_id="1",  # TODO: Get from auth
            agent_id=None
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{source_id}/introspect")
async def introspect_schema(
    source_id: int,
    db: Session = Depends(get_db)
):
    """
    Re-introspect database schema and update metadata.
    """
    # Fetch source
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(
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
        raise HTTPException(status_code=400, detail=f"Failed to resolve credentials: {e}")
    
    # Introspect
    dialect = _map_dialect_to_introspector(source.dialect)
    try:
        inspector = DatabaseIntrospectionService(credential=creds, dialect=dialect)
        metadata = inspector.introspect(include_samples=True, sample_limit=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Introspection failed: {e}")
    
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
    db: Session = Depends(get_db)
):
    """
    Get database schema metadata.
    """
    _, cache, _ = get_services()
    
    # Try cache first
    if use_cache:
        try:
            cached = cache.get_cached_schema(source_id, tenant_id=1)
            if cached:
                return cached
        except Exception:
            # Cache service optional; ignore on failure
            pass
    
    # Load from DB
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(
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
    db: Session = Depends(get_db)
):
    """
    Update semantic layer (metrics and dimensions).
    """
    service, cache, _ = get_services()
    
    try:
        # Convert to proper types
        from modules.nl_to_sql import SemanticMetric, SemanticDimension
        
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
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{source_id}")
async def get_database_source(
    source_id: int,
    db: Session = Depends(get_db)
):
    """
    Get database source details.
    """
    # Query database for source details
    source = db.query(DatabaseKnowledgeSource).filter(
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


@router.get("/templates/list")
async def list_query_templates(
    dialect: Optional[str] = None,
    category: Optional[str] = None,
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
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{source_id}/query/sql")
async def execute_validated_sql(
    source_id: int,
    payload: Dict[str, Any] = Body(...),
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
    source: Optional[DatabaseKnowledgeSource] = db.query(DatabaseKnowledgeSource).filter(
        DatabaseKnowledgeSource.id == source_id
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database source not found")

    # Validate SQL
    from modules.nl_to_sql import SQLValidator, SQLValidationError
    validator = SQLValidator(max_limit=min(int(payload.get("max_rows", 1000)), source.max_rows_limit or 1000))
    try:
        validated_sql, reasons = validator.validate_and_rewrite(sql, schema_metadata=source.schema_metadata)
    except SQLValidationError as e:
        # audit failure
        db.add(DatabaseQueryAudit(
            tenant_id=1,  # TODO from auth
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
        raise HTTPException(status_code=400, detail=str(e))

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
            tenant_id=1,
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
        raise HTTPException(status_code=400, detail=f"Failed to resolve credentials: {e}")

    # Build URL
    dialect = source.dialect.lower()
    if dialect.startswith("postgres"):
        url = f"postgresql+psycopg2://{creds.get('user')}:{creds.get('password')}@{creds.get('host')}:{creds.get('port')}/{creds.get('database')}"
    elif dialect.startswith("mysql"):
        url = f"mysql+pymysql://{creds.get('user')}:{creds.get('password')}@{creds.get('host')}:{creds.get('port')}/{creds.get('database')}"
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
                conn.execute(text(f"SET LOCAL statement_timeout = {(source.query_timeout_seconds or 30) * 1000}"))
            import time
            start = time.time()
            result = conn.execute(text(validated_sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, r)) for r in result.fetchall()]
            duration_ms = int((time.time() - start) * 1000)
            # audit success
            db.add(DatabaseQueryAudit(
                tenant_id=1,
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
            tenant_id=1,
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
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")

    return {
        "sql": validated_sql,
        "reasons": reasons,
        "columns": columns,
        "rows": rows,
        "stats": {"duration_ms": duration_ms, "row_count": len(rows)}
    }


@router.get("/cache/stats")
async def get_cache_statistics(
    tenant_id: int = 1
):
    """
    Get cache statistics for monitoring.
    """
    _, cache, _ = get_services()
    
    stats = cache.get_cache_stats(tenant_id)
    
    return stats
