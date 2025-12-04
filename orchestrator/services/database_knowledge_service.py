"""
Database Knowledge Source Implementation for Automatos AI
=========================================================
PRD-21: Native text-to-SQL with semantic layer support

This demonstrates how the Database Knowledge Source integrates with:
- Credential Management (PRD-18)
- RAG Service (PRD-08)
- Context Engineering (PRD-03)
- Agent Tools (PRD-17)
- Knowledge Base (PRD-19)
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import sqlparse

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
# import pandas as pd

# Automatos imports (from existing system)
from services.credential_resolver import CredentialResolver
from services.llm_provider import LLMProvider
from modules.rag import RAGService
from services.context_engineering_service import ContextEngineeringService
from services.audit_service import AuditService
# Placeholder models


class DatabaseDialect(Enum):
    """Supported database dialects"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"


@dataclass
class SemanticMetric:
    """Semantic layer metric definition"""
    name: str
    display_name: str
    sql_expression: str
    aggregation: str  # sum, avg, count, min, max
    format: str  # number, currency, percentage
    description: str
    tables: List[str]


@dataclass
class SemanticDimension:
    """Semantic layer dimension definition"""
    name: str
    display_name: str
    sql_expression: str
    type: str  # categorical, temporal, geographical
    hierarchy: Optional[List[str]] = None


class DatabaseKnowledgeService:
    """
    Main service for database knowledge source operations.
    Handles schema introspection, SQL generation, validation, and execution.
    """
    
    def __init__(
        self,
        credential_resolver: CredentialResolver,
        llm_provider: LLMProvider,
        rag_service: RAGService,
        context_engineering: ContextEngineeringService,
        audit_service: AuditService
    ):
        self.credential_resolver = credential_resolver
        self.llm_provider = llm_provider
        self.rag_service = rag_service
        self.context_engineering = context_engineering
        self.audit_service = audit_service
        self.schema_cache = {}
        self.query_cache = {}
    
    async def _get_source(self, source_id: str) -> 'DatabaseKnowledgeSource':
        """Fetch database source by ID"""
        from database.database import SessionLocal
        from models.database_knowledge import DatabaseKnowledgeSource as DBKSource
        
        db_session = SessionLocal()
        try:
            source = db_session.query(DBKSource).filter(DBKSource.id == int(source_id)).first()
            if not source:
                raise ValueError(f"Database source {source_id} not found")
            return source
        finally:
            db_session.close()
        
    async def add_database_source(
        self,
        name: str,
        credential_id: str,
        tenant_id: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new database as a knowledge source.
        """
        # Step 1: Resolve and test credentials
        from services.credential_service import CredentialStore
        from database.database import SessionLocal
        from services.encryption_service import EncryptionService
        
        db_session = SessionLocal()
        try:
            cred_store = CredentialStore(db_session)
            credential = cred_store.get_credential(credential_id)
            if not credential:
                raise ValueError(f"Credential {credential_id} not found")
            
            # Decrypt the credential data
            encryption = EncryptionService()
            credentials = encryption.decrypt_dict(credential.encrypted_data)
        finally:
            db_session.close()
        
        # Step 2: Create database source record
        from models.database_knowledge import DatabaseKnowledgeSource
        from database.database import SessionLocal
        from sqlalchemy.exc import IntegrityError
        
        db_session = SessionLocal()
        try:
            # Check if source already exists (trim whitespace for comparison)
            name_trimmed = name.strip()
            existing = db_session.query(DatabaseKnowledgeSource).filter(
                DatabaseKnowledgeSource.tenant_id == tenant_id,
                DatabaseKnowledgeSource.name == name_trimmed
            ).first()
            
            # Also check for name with trailing/leading spaces (case-insensitive)
            if not existing:
                existing = db_session.query(DatabaseKnowledgeSource).filter(
                    DatabaseKnowledgeSource.tenant_id == tenant_id,
                    DatabaseKnowledgeSource.name.ilike(name_trimmed)
                ).first()
            
            if existing:
                # Update existing source instead of creating duplicate
                existing.name = name_trimmed  # Normalize name (remove trailing spaces)
                existing.description = description
                existing.credential_id = credential_id
                existing.is_active = True
                db_session.commit()
                db_session.refresh(existing)
                logger.info(f"Updated existing database source '{name_trimmed}' (ID: {existing.id})")
                return existing
            
            db_source = DatabaseKnowledgeSource(
                tenant_id=tenant_id,
                name=name_trimmed,  # Use trimmed name
                description=description,
                credential_id=credential_id,  # Use credential_id (as per current schema)
                dialect='postgresql',  # TODO: detect from credentials
                schema_metadata={},  # Will be populated during introspection
                is_active=True
            )
            
            db_session.add(db_source)
            db_session.commit()
            db_session.refresh(db_source)
            
            result = db_source
        except IntegrityError as e:
            db_session.rollback()
            # Check again in case of race condition (with trimmed name)
            existing = db_session.query(DatabaseKnowledgeSource).filter(
                DatabaseKnowledgeSource.tenant_id == tenant_id,
                DatabaseKnowledgeSource.name.ilike(name_trimmed)
            ).first()
            if existing:
                # Update and activate the existing source
                existing.name = name_trimmed
                existing.description = description
                existing.credential_id = credential_id
                existing.is_active = True
                db_session.commit()
                db_session.refresh(existing)
                logger.info(f"Database source '{name_trimmed}' already exists, updated and activated (ID: {existing.id})")
                return existing
            raise
        finally:
            db_session.close()
        
        return result
    
    async def query_database(
        self,
        source_id: str,
        natural_language_query: str,
        user_id: str,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a natural language query against a database source.
        REAL IMPLEMENTATION - No mock data.
        """
        from sqlalchemy import create_engine, text
        from services.credential_service import CredentialStore
        from database.database import SessionLocal
        from services.encryption_service import EncryptionService
        from services.nl_to_sql_service import NaturalLanguageToSQLService
        from services.sql_validator import SQLValidator
        import json
        
        start_time = datetime.utcnow()
        
        # Step 1: Get source
        source = await self._get_source(source_id)
        
        # Step 2: Get and decrypt credentials
        db_session = SessionLocal()
        try:
            cred_store = CredentialStore(db_session)
            credential = cred_store.get_credential(source.credential_id)
            if not credential:
                raise ValueError(f"Credential {source.credential_id} not found")
            
            encryption = EncryptionService()
            credentials = encryption.decrypt_dict(credential.encrypted_data)
        finally:
            db_session.close()
        
        # Step 3: Get schema metadata
        schema_metadata = source.schema_metadata or {}
        if not schema_metadata or not schema_metadata.get('tables'):
            return {
                "success": False,
                "error": "No schema metadata available. Please run introspection first.",
                "sql": None,
                "data": [],
                "row_count": 0
            }
        
        # Step 4: Generate SQL using NL-to-SQL service
        nl_to_sql = NaturalLanguageToSQLService(llm_provider=self.llm_provider)
        sql, explanation, metadata = nl_to_sql.generate_sql(
            question=natural_language_query,
            schema_metadata=schema_metadata,
            dialect=source.dialect
        )
        
        sql_result = {
            'sql': sql,
            'explanation': explanation,
            'metadata': metadata
        }
        
        generated_sql = sql_result.get('sql', '')
        
        # Step 5: Validate SQL
        validator = SQLValidator()
        validated_sql, warnings = validator.validate_and_rewrite(
            sql=generated_sql,
            schema_metadata=schema_metadata
        )
        
        # Step 6: Create database connection and execute
        try:
            # Build connection string
            host = credentials.get("host")
            port = credentials.get("port")
            database = credentials.get("database")
            user = credentials.get("user") or credentials.get("username")
            password = credentials.get("password")
            
            if source.dialect.lower().startswith('postgres'):
                conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
            elif source.dialect.lower().startswith('mysql'):
                conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            else:
                raise ValueError(f"Unsupported dialect: {source.dialect}")
            
            # Execute query
            engine = create_engine(conn_str, pool_pre_ping=True)
            with engine.connect() as conn:
                result = conn.execute(text(validated_sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "sql": validated_sql,
                "data": rows,
                "columns": columns,
                "row_count": len(rows),
                "execution_time_ms": round(execution_time, 2),
                "explanation": sql_result.get('explanation', '')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
                "sql": validated_sql,
                "data": [],
                "row_count": 0
            }
    
    async def update_semantic_layer(
        self,
        source_id: str,
        metrics: List[SemanticMetric],
        dimensions: List[SemanticDimension]
    ) -> None:
        """
        Update the semantic layer for a database source.
        """
        source = await self._get_source(source_id)
        
        # Validate metrics and dimensions reference existing tables/columns
        schema_context = await self._get_schema_context(source)
        await self._validate_semantic_definitions(metrics, dimensions, schema_context)
        
        # Update source
        source.semantic_layer = {
            'metrics': [m.__dict__ for m in metrics],
            'dimensions': [d.__dict__ for d in dimensions],
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Clear cache to force refresh
        if source_id in self.schema_cache:
            del self.schema_cache[source_id]
    
    async def _generate_sql(
        self,
        query: str,
        schema_context: Dict,
        semantic_layer: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language using LLM.
        """
        # Build prompt with schema and semantic context
        system_prompt = self._build_sql_generation_prompt(
            schema_context,
            semantic_layer
        )
        
        response = await self.llm_provider.generate(
            system_prompt=system_prompt,
            user_prompt=query,
            temperature=0.1,
            response_format="json"
        )
        
        return json.loads(response)
    
    async def _validate_sql(
        self,
        sql: str,
        schema_context: Dict,
        dialect: DatabaseDialect
    ) -> str:
        """
        Three-tier SQL validation system.
        """
        # Tier 1: Syntax validation
        try:
            parsed = sqlparse.parse(sql)[0]
            if not parsed:
                raise ValueError("Invalid SQL syntax")
        except Exception as e:
            raise ValueError(f"SQL syntax error: {e}")
        
        # Tier 2: Security validation
        sql_upper = sql.upper()
        security_checks = [
            ('SELECT' in sql_upper and not any(
                word in sql_upper for word in 
                ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
            ), "Query must be SELECT-only"),
            ('LIMIT' in sql_upper or 'FETCH' in sql_upper, "Query must have LIMIT clause"),
            (';' not in sql or sql.strip().endswith(';'), "No multiple statements allowed"),
            ('INFORMATION_SCHEMA' not in sql_upper and 'PG_' not in sql_upper, 
             "System tables not allowed")
        ]
        
        for check, message in security_checks:
            if not check:
                raise ValueError(f"Security validation failed: {message}")
        
        # Tier 3: Semantic validation
        # Verify tables and columns exist in schema
        tables_in_query = self._extract_tables_from_sql(sql)
        valid_tables = set(schema_context.get('tables', {}).keys())
        
        for table in tables_in_query:
            if table.lower() not in valid_tables:
                raise ValueError(f"Table '{table}' not found in schema")
        
        # Add LIMIT if missing
        if 'LIMIT' not in sql_upper:
            sql = f"{sql.rstrip(';')} LIMIT 1000"
        
        return sql
    
    async def _execute_query(
        self,
        sql: str,
        credentials: Dict,
        dialect: DatabaseDialect
    ) -> List[Dict]:
        """
        Execute SQL query with timeout and row limits.
        """
        connection_string = self._build_connection_string(credentials, dialect)
        engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        
        try:
            with engine.connect() as connection:
                # Set timeout based on dialect
                if dialect == DatabaseDialect.POSTGRESQL:
                    connection.execute(text("SET statement_timeout = 30000"))
                elif dialect == DatabaseDialect.MYSQL:
                    connection.execute(text("SET max_execution_time = 30000"))
                
                # Execute query
                result = connection.execute(text(sql))
                rows = result.fetchall()
                
                # Convert to list of dicts
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            raise ValueError(f"Query execution failed: {e}")
        finally:
            engine.dispose()
    
    async def _create_agent_tools(self, source: Dict[str, Any]) -> None:
        """
        Auto-create agent tools for database source.
        """
        # Create a specific tool for this database
        tool_definition = {
            "name": f"query_{source.name.lower().replace(' ', '_')}_database",
            "display_name": f"Query {source.name} Database",
            "description": f"Query {source.name} database using natural language",
            "category": "database",
            "security_level": "read_only",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Natural language query",
                    "required": True
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return",
                    "default": 100
                }
            },
            "metadata": {
                "source_id": source.id,
                "dialect": source.dialect,
                "requires_credential": source.credential_id
            }
        }
        
        # Register with tool registry
        # This would integrate with your Tool Registry (PRD-17)
        pass
    
    def _build_sql_generation_prompt(
        self,
        schema_context: Dict,
        semantic_layer: Optional[Dict]
    ) -> str:
        """
        Build the system prompt for SQL generation.
        """
        prompt = """You are an expert SQL query generator. Generate safe, efficient SELECT-only queries.

Database Schema:
"""
        # Add tables and columns
        for table_name, table_info in schema_context.get('tables', {}).items():
            prompt += f"\nTable: {table_name}\n"
            for col in table_info.get('columns', []):
                prompt += f"  - {col['name']} ({col['type']})\n"
        
        # Add relationships
        if schema_context.get('relationships'):
            prompt += "\nRelationships:\n"
            for rel in schema_context['relationships']:
                prompt += f"  - {rel['from']} -> {rel['to']} ({rel['type']})\n"
        
        # Add semantic layer
        if semantic_layer:
            if semantic_layer.get('metrics'):
                prompt += "\nAvailable Metrics:\n"
                for metric in semantic_layer['metrics']:
                    prompt += f"  - {metric['name']}: {metric['description']} (SQL: {metric['sql_expression']})\n"
            
            if semantic_layer.get('dimensions'):
                prompt += "\nAvailable Dimensions:\n"
                for dim in semantic_layer['dimensions']:
                    prompt += f"  - {dim['name']}: {dim['description']} (SQL: {dim['sql_expression']})\n"
        
        prompt += """
Rules:
1. Generate SELECT-only queries
2. Always include LIMIT (max 1000)
3. Use proper JOINs based on relationships
4. Apply semantic metrics when relevant
5. Return JSON with: sql, explanation, visualization_hint, confidence_score
"""
        return prompt
    
    def _get_cache_key(self, sql: str, source_id: str) -> str:
        """Generate cache key for query results."""
        return hashlib.md5(f"{source_id}:{sql}".encode()).hexdigest()
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        # Simple extraction - would use proper SQL parser in production
        tables = []
        parsed = sqlparse.parse(sql)[0]
        for token in parsed.tokens:
            if token.ttype is None and isinstance(token, sqlparse.sql.Identifier):
                tables.append(str(token))
        return tables
