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
from core.credentials.resolver import CredentialResolver
from core.llm import LLMProvider
from modules.rag import RAGService
from modules.search.services.context_engineering_service import ContextEngineeringService
from core.services.audit_service import AuditService

# Module-internal imports
from .query.nl2sql_service import NaturalLanguageToSQLService
from .query.validator import SQLValidator
from .primitive_heartbeat import _emit_nl2sql_primitive
from modules.tools.services.pandas_ai_service import get_pandasai_service


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
        self.analytics_engine = get_pandasai_service()
    
    async def _get_source(
        self,
        source_id: str,
        workspace_id: Optional[str] = None,
    ) -> 'DatabaseKnowledgeSource':
        """Fetch database source by ID.

        When ``workspace_id`` is provided, the lookup is workspace-scoped:
        a source from another workspace will NOT come back (raises). The
        kwarg is opt-in so in-process callers that already enforce
        isolation upstream keep their narrower interface; the API route
        is the call site that MUST pass it (W3-S9 cross-tenant guard).
        """
        from core.database.database import SessionLocal
        from core.models.database_knowledge import DatabaseKnowledgeSource as DBKSource

        db_session = SessionLocal()
        try:
            query = db_session.query(DBKSource).filter(DBKSource.id == int(source_id))
            if workspace_id is not None:
                query = query.filter(DBKSource.workspace_id == str(workspace_id))
            source = query.first()
            if not source:
                # A None here on the workspace-scoped path is either a
                # genuinely missing source OR a cross-tenant attempt; either
                # way the caller cannot reach this source.
                raise ValueError(f"Database source {source_id} not found")
            return source
        finally:
            db_session.close()
        
    async def add_database_source(
        self,
        name: str,
        credential_id: str,
        workspace_id=None,
        description: Optional[str] = None,
        dialect: str = "postgresql",
        tenant_id: str = None,  # deprecated, use workspace_id
    ) -> Dict[str, Any]:
        """
        Add a new database as a knowledge source.
        """
        # Step 1: Resolve and test credentials
        from core.credentials.service import CredentialStore
        from core.database.database import SessionLocal
        from core.credentials.encryption import EncryptionService
        
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
        from core.models.database_knowledge import DatabaseKnowledgeSource
        from core.database.database import SessionLocal
        from sqlalchemy.exc import IntegrityError
        
        # Resolve workspace_id (prefer new param, fall back to legacy tenant_id)
        ws_id = workspace_id or tenant_id
        if not ws_id:
            raise ValueError("workspace_id is required to add a database source")

        db_session = SessionLocal()
        try:
            # Check if source already exists (trim whitespace for comparison)
            name_trimmed = name.strip()
            existing = db_session.query(DatabaseKnowledgeSource).filter(
                DatabaseKnowledgeSource.workspace_id == ws_id,
                DatabaseKnowledgeSource.name == name_trimmed
            ).first()

            # Also check for name with trailing/leading spaces (case-insensitive)
            if not existing:
                existing = db_session.query(DatabaseKnowledgeSource).filter(
                    DatabaseKnowledgeSource.workspace_id == ws_id,
                    DatabaseKnowledgeSource.name.ilike(name_trimmed)
                ).first()

            if existing:
                # Update existing source instead of creating duplicate
                existing.name = name_trimmed  # Normalize name (remove trailing spaces)
                existing.description = description
                existing.credential_id = credential_id
                existing.dialect = dialect
                existing.is_active = True
                db_session.commit()
                db_session.refresh(existing)
                logger.info(f"Updated existing database source '{name_trimmed}' (ID: {existing.id})")
                return existing

            db_source = DatabaseKnowledgeSource(
                workspace_id=ws_id,
                tenant_id=1,  # legacy column, kept for schema compat
                name=name_trimmed,  # Use trimmed name
                description=description,
                credential_id=credential_id,
                dialect=dialect,
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
                DatabaseKnowledgeSource.workspace_id == ws_id,
                DatabaseKnowledgeSource.name.ilike(name_trimmed)
            ).first()
            if existing:
                # Update and activate the existing source
                existing.name = name_trimmed
                existing.description = description
                existing.credential_id = credential_id
                existing.dialect = dialect
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
        agent_id: Optional[str] = None,
        auto_correct: bool = True,
        max_retries: int = 2,
        auto_train: bool = True,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a natural language query against a database source.

        PRD-61: Enhanced with error self-correction loop, few-shot examples
        from training store, and confidence scoring.

        Args:
            auto_correct: If True, retry with error context on failure
            max_retries: Number of correction attempts
            auto_train: If True, auto-save successful queries as training examples
        """
        from sqlalchemy import create_engine, text
        from core.credentials.service import CredentialStore
        from core.database.database import SessionLocal
        from core.credentials.encryption import EncryptionService

        start_time = datetime.utcnow()

        # Step 1: Get source (workspace-scoped when the caller — typically
        # the API route — passed the authenticated workspace_id).
        source = await self._get_source(source_id, workspace_id=workspace_id)

        # W3-S9: capture the source's workspace_id once — the local
        # ``workspace_id`` name is reused later in this function for
        # few-shot / prompt purposes, so we keep a stable handle for
        # the heartbeat emit at every return point. Falsy => helper
        # silently skips (honest 'unknown' on the tile).
        _emit_ws_id = str(getattr(source, 'workspace_id', '') or '') or None

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
            _emit_nl2sql_primitive(
                _emit_ws_id,
                success=False,
                detail="No schema metadata — introspect required",
            )
            return {
                "success": False,
                "error": "No schema metadata available. Please run introspection first.",
                "sql": None,
                "data": [],
                "row_count": 0
            }

        # PRD-61: Get few-shot examples from training store
        few_shot_examples = []
        example_store = self._get_example_store()
        if example_store:
            try:
                workspace_id = str(getattr(source, 'workspace_id', ''))
                few_shot_examples = await example_store.get_similar_examples(
                    question=natural_language_query,
                    database_source_id=str(source_id),
                    workspace_id=workspace_id,
                    limit=5
                )
            except Exception as e:
                logger.warning(f"Failed to get few-shot examples: {e}")

        # PRD-80 US-019: Build NL2SQL system prompt via ContextService
        nl2sql_system_prompt = None
        try:
            from types import SimpleNamespace
            from modules.context import ContextService, ContextMode

            workspace_id = str(getattr(source, 'workspace_id', 'system'))
            ctx_result = await ContextService().build_context(
                mode=ContextMode.NL2SQL,
                agent=SimpleNamespace(
                    id=None,
                    name="SQL Query Generator",
                    role=f"Natural language to {source.dialect} SQL converter",
                    description=None,
                ),
                workspace_id=workspace_id,
                task_description=natural_language_query,
            )
            nl2sql_system_prompt = ctx_result.system_prompt
        except Exception as e:
            logger.warning(f"ContextService unavailable for NL2SQL, proceeding without: {e}")

        # PRD-61 US-005: Error self-correction loop
        last_error = None
        attempted_sqls = []
        retries = max_retries if auto_correct else 0

        for attempt in range(retries + 1):
            # Step 4: Generate SQL
            nl2sql = NaturalLanguageToSQLService(llm_provider=self.llm_provider)
            sql, explanation, metadata = nl2sql.generate_sql(
                question=natural_language_query,
                schema_metadata=schema_metadata,
                dialect=source.dialect,
                examples=few_shot_examples if few_shot_examples else None,
                error_context=last_error,
                previous_attempts=attempted_sqls if attempted_sqls else None,
                system_prompt=nl2sql_system_prompt,
            )

            generated_sql = sql

            # Generation itself failed (LLM error): return a structured error
            # with NO executable SQL. The self-correction loop corrects
            # *validation* errors, not LLM failures, so retrying is pointless.
            if not generated_sql or metadata.get("success") is False:
                gen_error = metadata.get("error") or explanation or "SQL generation failed"
                _emit_nl2sql_primitive(_emit_ws_id, success=False, detail=gen_error)
                return {
                    "success": False,
                    "error": gen_error,
                    "sql": None,
                    "data": [],
                    "row_count": 0,
                    "attempts": attempt + 1,
                }

            # Step 5: Validate SQL
            validator = SQLValidator()
            try:
                validated_sql, warnings = validator.validate_and_rewrite(
                    sql=generated_sql,
                    schema_metadata=schema_metadata
                )
            except Exception as ve:
                last_error = f"Validation failed: {str(ve)}"
                attempted_sqls.append({"sql": generated_sql, "error": last_error})
                if attempt < retries:
                    logger.info(f"SQL validation failed (attempt {attempt + 1}), retrying: {ve}")
                    continue
                _emit_nl2sql_primitive(
                    _emit_ws_id,
                    success=False,
                    detail=last_error,
                )
                return {
                    "success": False,
                    "error": last_error,
                    "sql": generated_sql,
                    "data": [],
                    "row_count": 0,
                    "attempts": attempt + 1,
                    "corrections": attempted_sqls
                }

            # Step 6: Execute
            try:
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

                engine = create_engine(conn_str, pool_pre_ping=True)
                with engine.connect() as conn:
                    result = conn.execute(text(validated_sql))
                    columns = list(result.keys())
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]

                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                # PRD-61: Auto-train on first-try success
                if auto_train and attempt == 0 and example_store:
                    try:
                        workspace_id = str(getattr(source, 'workspace_id', ''))
                        await example_store.add_example(
                            question=natural_language_query,
                            sql=validated_sql,
                            database_source_id=str(source_id),
                            workspace_id=workspace_id,
                            is_verified=False,
                            verification_source='auto'
                        )
                    except Exception as e:
                        logger.warning(f"Auto-train failed: {e}")

                # PRD-61 US-014: Confidence scoring
                confidence_data = self._calculate_confidence(
                    few_shot_examples, validated_sql, warnings
                )

                _emit_nl2sql_primitive(
                    _emit_ws_id,
                    success=True,
                    detail=f"{len(rows)} rows in {round(execution_time)}ms",
                )
                return {
                    "success": True,
                    "sql": validated_sql,
                    "data": rows,
                    "columns": columns,
                    "row_count": len(rows),
                    "execution_time_ms": round(execution_time, 2),
                    "explanation": explanation,
                    "attempts": attempt + 1,
                    "corrections": attempted_sqls,
                    "confidence": confidence_data,
                }

            except Exception as e:
                last_error = f"Execution error: {str(e)}"
                attempted_sqls.append({"sql": validated_sql, "error": last_error})
                if attempt < retries:
                    logger.info(f"SQL execution failed (attempt {attempt + 1}), retrying: {e}")
                    continue

        _emit_nl2sql_primitive(
            _emit_ws_id,
            success=False,
            detail=last_error or "execution failed after retries",
        )
        return {
            "success": False,
            "error": last_error,
            "sql": attempted_sqls[-1]["sql"] if attempted_sqls else None,
            "data": [],
            "row_count": 0,
            "attempts": retries + 1,
            "corrections": attempted_sqls
        }

    def _get_example_store(self):
        """Lazy-load the SQL example store."""
        try:
            from .training.example_store import SQLExampleStore
            return SQLExampleStore()
        except Exception:
            return None

    def _calculate_confidence(
        self,
        similar_examples: list,
        sql: str,
        validation_warnings: list
    ) -> Dict[str, Any]:
        """Calculate confidence score for the generated query."""
        try:
            from .query.confidence import QueryConfidenceScorer, ScoringContext
            scorer = QueryConfidenceScorer()
            context = ScoringContext(
                similar_examples=similar_examples,
                sql=sql,
                validation_clean=len(validation_warnings) == 0
            )
            result = scorer.score(context)
            return {
                "score": result.score,
                "level": result.level,
                "factors": result.factors,
                "recommendation": result.recommendation,
            }
        except Exception as e:
            logger.warning(f"Confidence scoring failed: {e}")
            return {"score": 0, "level": "unknown", "factors": {}, "recommendation": "review_sql"}
    
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

    async def analyze_database(
        self,
        source_id: str,
        natural_language_query: str,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform advanced analytics/visualization on database data.
        1. Convert NL -> SQL to fetch raw data
        2. Pass data -> Pandas Engine for analysis/plotting
        """
        # 1. Fetch Data (using existing SQL pipeline)
        # We append "Return all relevant columns for analysis" to prompt implicitly
        # by asking for a broad SQL query first.

        # For analysis, we often need more data than a simple answer.
        # We'll ask the SQL generator to get the raw data first.

        # Heuristic: Ask LLM to generate a SQL query that fetches the DATA needed for the analysis
        fetch_query_prompt = f"Generate a SQL query to fetch the raw data needed to answer this analysis question: '{natural_language_query}'. Do not aggregate yet if the analysis requires raw data points (like scatter plots)."

        sql_result = await self.query_database(
            source_id, fetch_query_prompt, user_id,
            workspace_id=workspace_id,
        )
        
        if not sql_result['success']:
            return sql_result
            
        data = sql_result['data']
        
        # 2. Analyze with Pandas
        if not self.analytics_engine:
            return {
                "success": False,
                "error": "PandasAI service is not enabled or configured."
            }

        analysis_result = self.analytics_engine.generate_insight(
            question=natural_language_query,
            rows=data,
            columns=sql_result.get('columns')
        )
        
        if not analysis_result:
             return {
                "success": False,
                "error": "Failed to generate analysis."
            }

        # Extract first chart if available
        chart_base64 = None
        if analysis_result.get('charts'):
            chart_base64 = analysis_result['charts'][0].get('base64')

        return {
            "success": True,
            "type": "analysis",
            "sql": sql_result['sql'],
            "insight": analysis_result.get('summary', ''),
            "chart_base64": chart_base64,
            "code": None, # PandasAIService doesn't expose code by default
            "row_count": len(data)
        }

    async def smart_query(
        self,
        source_id: str,
        text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Intelligently route between SQL Query and Data Analysis.
        """
        # Simple heuristic router (could be LLM-based)
        analysis_keywords = ['plot', 'chart', 'graph', 'trend', 'correlation', 'visualize', 'compare', 'forecast']
        is_analysis = any(keyword in text.lower() for keyword in analysis_keywords)

        if is_analysis:
            return await self.analyze_database(
                source_id, text, user_id, workspace_id=workspace_id
            )
        else:
            return await self.query_database(
                source_id, text, user_id, agent_id,
                workspace_id=workspace_id,
            )
