"""
NL to SQL Module
================

Natural language to SQL query generation.

Usage:
    from modules.nl_to_sql import NLToSQLService, DatabaseKnowledgeService
    
    # Simple SQL generation
    service = NLToSQLService(llm_provider)
    sql, explanation, metadata = service.generate_sql(question, schema)
    
    # Full database knowledge service
    from modules.nl_to_sql import DatabaseKnowledgeService
    db_service = DatabaseKnowledgeService(...)
    result = await db_service.query_database(source_id, query, user_id)

Sellable as: automatos-nl2sql
"""

# Main service
from .service import (
    DatabaseKnowledgeService,
    DatabaseDialect,
    SemanticMetric,
    SemanticDimension,
)

# Query generation
from .query.nl_to_sql_service import NaturalLanguageToSQLService
from .query.validator import SQLValidator, SQLValidationError

# Schema management
from .schema.introspection import DatabaseIntrospectionService, make_json_serializable
from .schema.provider import SchemaProvider, get_schema_provider

# Alias for convenience
NLToSQLService = NaturalLanguageToSQLService

__all__ = [
    # Main service
    "DatabaseKnowledgeService",
    "DatabaseDialect",
    "SemanticMetric",
    "SemanticDimension",
    # Query
    "NaturalLanguageToSQLService",
    "NLToSQLService",
    "SQLValidator",
    "SQLValidationError",
    # Schema
    "DatabaseIntrospectionService",
    "make_json_serializable",
    "SchemaProvider",
    "get_schema_provider",
]
