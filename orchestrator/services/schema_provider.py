"""
Schema Provider Service
======================

Centralized schema management for database introspection and context.
Replaces all hardcoded schema strings across the codebase.

Provides:
- Database table schemas for NL-to-SQL
- Context snippets for different domains
- Schema metadata for tool configuration
"""

import logging
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


class SchemaProvider:
    """
    Centralized schema provider for database and context information.
    Used by chatbot, workflows, and NL-to-SQL tools.
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self._schema_cache = {}
        self._context_cache = {}

    @lru_cache(maxsize=1)
    def get_database_schema_overview(self) -> str:
        """
        Get comprehensive database schema overview for LLM context.
        Used by chatbot and NL-to-SQL tools.
        """
        try:
            inspector = inspect(self.db.bind)

            schema_parts = ["PostgreSQL Database Schema Overview:"]

            # Get all tables
            tables = inspector.get_table_names()

            for table in sorted(tables):
                schema_parts.append(f"\n- {table}")

                # Get columns
                columns = inspector.get_columns(table)
                if columns:
                    col_descriptions = []
                    for col in columns:
                        col_type = str(col['type'])
                        nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                        col_descriptions.append(f"{col['name']}({col_type}, {nullable})")

                    schema_parts.append(f"  Columns: {', '.join(col_descriptions[:5])}")  # Limit for brevity
                    if len(col_descriptions) > 5:
                        schema_parts.append(f"  ... and {len(col_descriptions) - 5} more columns")

                # Get foreign keys
                fks = inspector.get_foreign_keys(table)
                if fks:
                    fk_descriptions = []
                    for fk in fks[:3]:  # Limit FKs
                        fk_descriptions.append(f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
                    schema_parts.append(f"  Foreign Keys: {', '.join(fk_descriptions)}")
                    if len(fks) > 3:
                        schema_parts.append(f"  ... and {len(fks) - 3} more FKs")

            # Add common patterns and relationships
            schema_parts.extend([
                "\nCommon Patterns:",
                "- User sessions and authentication via user_sessions, auth_tokens",
                "- Workflow executions tracked in workflow_executions table",
                "- Agent operations logged in agent_performance table",
                "- Document storage and RAG via knowledge_documents, document_chunks",
                "- Tool usage tracked in tool_usage_logs"
            ])

            return "\n".join(schema_parts)

        except Exception as e:
            logger.error(f"Failed to generate schema overview: {e}")
            return self._get_fallback_schema()

    def get_table_schema_detailed(self, table_name: str) -> Optional[str]:
        """
        Get detailed schema for a specific table.
        Used when focusing on particular tables.
        """
        try:
            inspector = inspect(self.db.bind)

            if table_name not in inspector.get_table_names():
                return None

            schema_parts = [f"Detailed Schema for {table_name}:"]

            # Columns with full details
            columns = inspector.get_columns(table_name)
            schema_parts.append("Columns:")
            for col in columns:
                col_type = str(col['type'])
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                default = f", DEFAULT: {col.get('default')}" if col.get('default') else ""
                schema_parts.append(f"  - {col['name']}: {col_type} {nullable}{default}")

            # Primary keys
            pks = inspector.get_pk_constraint(table_name)
            if pks and pks.get('constrained_columns'):
                schema_parts.append(f"Primary Key: {', '.join(pks['constrained_columns'])}")

            # Foreign keys
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                schema_parts.append("Foreign Keys:")
                for fk in fks:
                    schema_parts.append(f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")

            # Indexes
            indexes = inspector.get_indexes(table_name)
            if indexes:
                schema_parts.append("Indexes:")
                for idx in indexes:
                    unique = "UNIQUE " if idx.get('unique') else ""
                    schema_parts.append(f"  - {unique}INDEX on {', '.join(idx['column_names'])}")

            return "\n".join(schema_parts)

        except Exception as e:
            logger.error(f"Failed to get detailed schema for {table_name}: {e}")
            return None

    @lru_cache(maxsize=32)
    def get_domain_context(self, domain: str) -> str:
        """
        Get context snippets for different domains.
        Used to provide domain-specific knowledge to LLMs.
        """
        domain_contexts = {
            'workflow': """
Workflow Domain Context:
- Workflows are automated processes with 9 stages: Task Decomposition, Agent Selection, Context Engineering, Execution, Result Aggregation, Learning, Quality Assessment, Memory Storage, Response Generation
- Each workflow has executions tracked in workflow_executions table
- Agents are assigned to subtasks with performance metrics in agent_performance
- Workflows can be scheduled, manual, or API-triggered
- Context engineering provides relevant information at each stage
- Memory systems learn from successful executions
""",

            'agent': """
Agent Domain Context:
- Agents are specialized AI workers with skills, performance metrics, and memory
- Agent lifecycle: creation -> skill assignment -> task execution -> performance tracking -> learning
- Skills include: coding, research, analysis, writing, coordination
- Performance tracked via success rate, execution time, quality scores
- Agents can communicate inter-agent via messaging system
- Memory systems maintain agent-specific knowledge and experiences
""",

            'knowledge': """
Knowledge Domain Context:
- Knowledge base consists of documents, code, and multimodal content
- RAG (Retrieval-Augmented Generation) provides semantic search
- Documents are chunked and embedded for efficient retrieval
- CodeGraph enables codebase navigation and search
- NL-to-SQL allows natural language database queries
- Multimodal search supports images, diagrams, and mixed content
""",

            'system': """
System Domain Context:
- Platform monitors: agents, workflows, tools, performance metrics
- Configuration managed via system_settings table
- Logging and auditing via audit_service
- Tool registry manages available capabilities
- MCP (Model Context Protocol) integrates external tools
- Credentials managed securely with encryption
""",

            'security': """
Security Domain Context:
- All database operations use parameterized queries
- Credentials encrypted with AES-256
- API endpoints validate authentication and authorization
- Audit logs track all sensitive operations
- Input validation prevents injection attacks
- HTTPS required for all external communications
"""
        }

        return domain_contexts.get(domain, "")

    def get_query_patterns(self, domain: str = "general") -> List[str]:
        """
        Get common query patterns for different domains.
        Helps LLMs understand typical queries and responses.
        """
        patterns = {
            'general': [
                "SELECT * FROM {table} WHERE {condition}",
                "SELECT COUNT(*) FROM {table} GROUP BY {column}",
                "SELECT * FROM {table} ORDER BY {column} DESC LIMIT {n}",
                "SELECT DISTINCT {column} FROM {table}",
            ],

            'workflow': [
                "SELECT * FROM workflow_executions WHERE status = 'running'",
                "SELECT COUNT(*) FROM workflows WHERE created_at > NOW() - INTERVAL '7 days'",
                "SELECT agent_id, AVG(performance_score) FROM agent_performance GROUP BY agent_id",
                "SELECT * FROM workflow_executions ORDER BY created_at DESC LIMIT 10",
            ],

            'agent': [
                "SELECT * FROM agents WHERE status = 'active'",
                "SELECT skill_name, COUNT(*) FROM agent_skills GROUP BY skill_name",
                "SELECT * FROM agent_performance ORDER BY performance_score DESC LIMIT 5",
                "SELECT agent_id, AVG(execution_time) FROM agent_executions GROUP BY agent_id",
            ],

            'knowledge': [
                "SELECT * FROM knowledge_documents WHERE content_type = 'pdf'",
                "SELECT COUNT(*) FROM document_chunks WHERE created_at > NOW() - INTERVAL '1 day'",
                "SELECT * FROM knowledge_documents ORDER BY access_count DESC LIMIT 10",
                "SELECT DISTINCT tags FROM knowledge_documents WHERE tags IS NOT NULL",
            ]
        }

        return patterns.get(domain, patterns['general'])

    def get_table_relationships(self) -> Dict[str, List[str]]:
        """
        Get table relationships for complex query understanding.
        """
        try:
            inspector = inspect(self.db.bind)
            relationships = {}

            for table in inspector.get_table_names():
                fks = inspector.get_foreign_keys(table)
                relationships[table] = []

                for fk in fks:
                    relationship = f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}"
                    relationships[table].append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Failed to get table relationships: {e}")
            return {}

    def _get_fallback_schema(self) -> str:
        """Fallback schema when database inspection fails"""
        return """
PostgreSQL Database Schema (Fallback):
- workflow_executions(id, workflow_id, status, created_at, updated_at, result)
- workflows(id, name, description, created_by, created_at, config)
- agents(id, name, skills, status, created_at, performance_score)
- agent_performance(agent_id, workflow_id, execution_time, success_rate, quality_score)
- knowledge_documents(id, title, content, content_type, tags, created_at)
- document_chunks(document_id, chunk_index, content, embedding, created_at)
- user_sessions(user_id, session_token, expires_at, created_at)
- tool_usage_logs(tool_name, agent_id, execution_time, success, created_at)
- system_settings(key, value, category, updated_at)

Common Relationships:
- workflow_executions.workflow_id -> workflows.id
- agent_performance.agent_id -> agents.id
- document_chunks.document_id -> knowledge_documents.id
- tool_usage_logs.agent_id -> agents.id
"""

    def invalidate_cache(self):
        """Invalidate all caches (useful after schema changes)"""
        self._schema_cache.clear()
        self._context_cache.clear()
        self.get_database_schema_overview.cache_clear()
        self.get_domain_context.cache_clear()


# Global instance
_schema_provider = None

def get_schema_provider(db_session: Session) -> SchemaProvider:
    """Get or create the global schema provider instance"""
    global _schema_provider
    if _schema_provider is None:
        _schema_provider = SchemaProvider(db_session)
        logger.info("✅ Schema Provider initialized")
    return _schema_provider
