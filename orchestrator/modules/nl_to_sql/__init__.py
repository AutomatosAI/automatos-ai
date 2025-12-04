"""
NL to SQL Module
================

Natural language to SQL query generation.

Usage:
    from modules.nl_to_sql import NLToSQLService
    
    service = NLToSQLService(llm_provider)
    sql, explanation, metadata = service.generate_sql(question, schema)

Sellable as: automatos-nl2sql
"""

from .query.nl_to_sql_service import NaturalLanguageToSQLService

# Alias for convenience
NLToSQLService = NaturalLanguageToSQLService

__all__ = [
    "NaturalLanguageToSQLService",
    "NLToSQLService"
]
