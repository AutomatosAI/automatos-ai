"""
NL2SQL Module
=============

Natural language to SQL query generation with intelligent features.

PRD-61: Enhanced with RAG-based few-shot examples, error self-correction,
schema linking, confidence scoring, Golden SQL training, and benchmarking.

Usage:
    # Basic SQL generation
    from modules.nl2sql import NLToSQLService
    service = NLToSQLService(llm_provider)
    sql, explanation, metadata = service.generate_sql(question, schema)

    # Full database knowledge service
    from modules.nl2sql import DatabaseKnowledgeService
    db_service = DatabaseKnowledgeService(...)
    result = await db_service.query_database(source_id, query, user_id)

    # Training examples (Vanna-inspired)
    from modules.nl2sql import SQLExampleStore
    store = SQLExampleStore()
    await store.add_example("top 10 customers", "SELECT ...", source_id, ...)

    # Confidence scoring
    from modules.nl2sql import QueryConfidenceScorer
    scorer = QueryConfidenceScorer()
    confidence = scorer.score(context)

Sellable as: automatos-nl2sql
"""

# Main service
from .service import (
    DatabaseKnowledgeService,
    DatabaseDialect,
    get_database_knowledge_service,
)

# Query generation
from .query.nl2sql_service import NaturalLanguageToSQLService
from .query.validator import SQLValidator, SQLValidationError

# Schema management
from .schema.introspection import DatabaseIntrospectionService, make_json_serializable
from .schema.provider import SchemaProvider, get_schema_provider

# PRD-199 S5: the intelligence/ package (1,687 LOC, zero external callers,
# advertised by a tool card it never backed) and the SchemaLinker (0-caller,
# false "embedding" docstring — the assigned embedding_manager was never
# used) are DELETED, not kept on life support. A real embedding schema
# linker is a new bet against a working keyword design, not this dead code.

# PRD-61: Training (Vanna-inspired RAG for SQL)
from .training.example_store import SQLExampleStore

# PRD-61: Confidence Scoring
from .query.confidence import QueryConfidenceScorer, ConfidenceScore, ScoringContext

# PRD-61: Benchmarking
from .benchmarks.runner import NL2SQLBenchmarkRunner
from .benchmarks.comparator import SQLComparator

# Aliases for convenience
NLToSQLService = NaturalLanguageToSQLService

__all__ = [
    # Main service
    "DatabaseKnowledgeService",
    "get_database_knowledge_service",
    "DatabaseDialect",
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
    # PRD-61: Training
    "SQLExampleStore",
    # PRD-61: Schema Linking
    # PRD-61: Confidence
    "QueryConfidenceScorer",
    "ConfidenceScore",
    "ScoringContext",
    # PRD-61: Benchmarking
    "NL2SQLBenchmarkRunner",
    "SQLComparator",
]
