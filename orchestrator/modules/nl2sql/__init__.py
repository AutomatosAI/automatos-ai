"""
NL2SQL Module
=============

Natural language to SQL query generation with intelligent features.

Usage:
    # Basic SQL generation
    from modules.nl2sql import NLToSQLService
    service = NLToSQLService(llm_provider)
    sql, explanation, metadata = service.generate_sql(question, schema)
    
    # Full database knowledge service
    from modules.nl2sql import DatabaseKnowledgeService
    db_service = DatabaseKnowledgeService(...)
    result = await db_service.query_database(source_id, query, user_id)
    
    # Smart Agent (PandasAI-inspired)
    from modules.nl2sql import SmartNL2SQLAgent
    agent = SmartNL2SQLAgent(llm_provider, schema_metadata)
    
    # Check if clarification needed
    if agent.needs_clarification("show me sales"):
        questions = agent.get_clarifications("show me sales")
    
    # Rephrase vague queries
    better_query, reason = agent.rephrase_query("show me stuff")
    
    # Full smart query with all features
    result = await agent.query("show me sales trends")

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
from .query.nl2sql_service import NaturalLanguageToSQLService
from .query.validator import SQLValidator, SQLValidationError

# Schema management
from .schema.introspection import DatabaseIntrospectionService, make_json_serializable
from .schema.provider import SchemaProvider, get_schema_provider

# Intelligence features (PandasAI-inspired)
from .intelligence import (
    SmartNL2SQLAgent,
    QueryClarifier,
    QueryRephraser,
    ResultExplainer,
    VisualizationSuggester,
)
from .intelligence.agent import create_smart_agent

# Aliases for convenience
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
    # Intelligence (Smart Agent)
    "SmartNL2SQLAgent",
    "create_smart_agent",
    "QueryClarifier",
    "QueryRephraser",
    "ResultExplainer",
    "VisualizationSuggester",
]
