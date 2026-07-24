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


# PRD-199 S1/S2: the SemanticMetric/SemanticDimension dataclasses are gone.
# They were the broken writer's intermediate format (list-of-__dict__ with
# sql_expression keys) — a shape the reader (nl2sql_service.py, dict-of-dicts
# keyed on 'sql') could never consume. The canonical semantic doc is the
# READER's shape, built at the API edge; no intermediate classes.


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
    
    # ------------------------------------------------------------------
    # PRD-160 S2 — accuracy stack: execution guards + value sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _quote_ident(dialect: str, ident: str) -> str:
        if (dialect or "").lower().startswith("mysql"):
            return "`" + str(ident).replace("`", "``") + "`"
        return '"' + str(ident).replace('"', '""') + '"'

    def _nl2sql_connection_string(self, credentials: Dict[str, Any], dialect: str) -> str:
        """Build a SQLAlchemy URL for a knowledge source's database."""
        host = credentials.get("host")
        port = credentials.get("port")
        database = credentials.get("database")
        user = credentials.get("user") or credentials.get("username")
        password = credentials.get("password")
        d = (dialect or "").lower()
        if d.startswith("postgres"):
            return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        if d.startswith("mysql"):
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        raise ValueError(f"Unsupported dialect: {dialect}")

    @staticmethod
    def _statement_timeout_sql(dialect: str, seconds: int) -> Optional[str]:
        ms = int(max(1, seconds) * 1000)
        d = (dialect or "").lower()
        if d.startswith("postgres"):
            return f"SET statement_timeout = {ms}"
        if d.startswith("mysql"):
            return f"SET max_execution_time = {ms}"
        return None

    def _run_sql_with_guards(
        self, source, credentials: Dict[str, Any], sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Execute a validated read-only query under a per-statement timeout,
        after an EXPLAIN dry-run.

        PRD-160 S2: the timeout bounds a runaway query; the EXPLAIN runs the
        planner over the statement first, so malformed / bad-column SQL fails
        cheaply and feeds the self-correction loop *before* a full table scan.
        Raises on any failure (the caller's retry loop catches it).

        PRD-160 S4: ``params`` carries bound parameters for Query Template
        execution (``:name`` placeholders), kept as binds so values are never
        string-interpolated into the SQL.
        """
        conn_str = self._nl2sql_connection_string(credentials, source.dialect)
        timeout_s = getattr(source, "query_timeout_seconds", None) or 30
        binds = params or {}
        engine = create_engine(conn_str, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                timeout_sql = self._statement_timeout_sql(source.dialect, timeout_s)
                if timeout_sql:
                    conn.execute(text(timeout_sql))
                conn.execute(text(f"EXPLAIN {sql}"), binds)  # dry-run validation
                result = conn.execute(text(sql), binds)
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return columns, rows
        finally:
            engine.dispose()

    async def run_validated_readonly_sql(self, source, sql: str) -> List[Dict[str, Any]]:
        """PRD-199 S6: the benchmark executor — validate the statement
        (read-only roots, table allowlist, LIMIT inject/cap via the S2 AST
        validator) then execute it against the connected source under the
        pipeline's existing EXPLAIN dry-run + statement-timeout guards.
        Returns rows as dicts; raises on validation or execution failure
        (the benchmark scores a raise as a non-match)."""
        credentials = self._decrypt_source_credentials(source)
        validated, _ = SQLValidator().validate_and_rewrite(
            sql, schema_metadata=source.schema_metadata or {}
        )
        _columns, rows = await asyncio.to_thread(
            self._run_sql_with_guards, source, credentials, validated
        )
        return rows

    def _augment_schema_with_samples(
        self,
        source,
        credentials: Dict[str, Any],
        schema_metadata: Dict[str, Any],
        max_distinct: int = 12,
        max_columns: int = 40,
    ) -> None:
        """Populate ``col['samples']`` for low-cardinality columns, in place.

        PRD-160 S2: grounding generation in real categorical values (e.g.
        status ∈ {active, churned, trial}) sharply cuts wrong-literal errors.
        Only text/enum/bool columns are probed, and a value set is kept only
        when small (≤ max_distinct) — high-cardinality columns (ids,
        timestamps, free text) never reach the prompt. Best-effort: any
        failure leaves the schema untouched. The generator already renders
        ``col['samples']`` into the prompt (nl2sql_service._build_prompt).
        """
        tables = schema_metadata.get("tables") or []
        if not tables:
            return
        try:
            conn_str = self._nl2sql_connection_string(credentials, source.dialect)
        except ValueError:
            return
        engine = create_engine(conn_str, pool_pre_ping=True)
        sampled = 0
        try:
            with engine.connect() as conn:
                to = self._statement_timeout_sql(source.dialect, 5)
                if to:
                    conn.execute(text(to))
                for table in tables:
                    tname = table.get("name")
                    if not tname:
                        continue
                    for col in (table.get("columns") or []):
                        if sampled >= max_columns:
                            return
                        cname = col.get("name")
                        ctype = (col.get("type") or "").lower()
                        if not cname or col.get("samples"):
                            continue
                        # Categorical-ish only: char/varchar/enum/bool. Skip
                        # numeric ids, timestamps, blobs, and free-form text.
                        if not any(t in ctype for t in ("char", "enum", "bool")):
                            continue
                        tq = self._quote_ident(source.dialect, tname)
                        cq = self._quote_ident(source.dialect, cname)
                        try:
                            rows = conn.execute(
                                text(
                                    f"SELECT DISTINCT {cq} AS v FROM {tq} "
                                    f"WHERE {cq} IS NOT NULL LIMIT :lim"
                                ),
                                {"lim": max_distinct + 1},
                            ).fetchall()
                            vals = [r[0] for r in rows]
                            if 0 < len(vals) <= max_distinct:
                                col["samples"] = [str(v) for v in vals]
                                sampled += 1
                        except Exception:
                            continue
        except Exception as e:  # noqa: BLE001 — sampling is best-effort
            logger.debug(f"value sampling skipped: {e}")
        finally:
            engine.dispose()

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

        # PRD-160 S2: ground generation in real low-cardinality column values
        # (status ∈ {active, churned}, …) so the LLM emits correct literals.
        try:
            self._augment_schema_with_samples(source, credentials, schema_metadata)
        except Exception as e:
            logger.debug(f"value sampling skipped: {e}")

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
                # PRD-160 S4: inject the per-connection semantic layer (business
                # metrics/dimensions + admin instructions) so definitions steer
                # generation. It was stored but never passed to the generator.
                semantic_layer=source.semantic_layer,
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

            # Step 6: Execute under PRD-160 S2 guards — per-statement timeout
            # bounds runaway queries; EXPLAIN dry-run catches bad SQL cheaply
            # and (on failure) feeds the self-correction loop below.
            try:
                columns, rows = self._run_sql_with_guards(source, credentials, validated_sql)

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
    
    async def get_semantic_layer(
        self,
        source_id: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the stored canonical semantic doc (PRD-199 S2 — the
        editor's load path; only POST existed before, so every load 405'd)."""
        source = await self._get_source(source_id, workspace_id=workspace_id)
        return source.semantic_layer or {
            "instructions": "",
            "metrics": {},
            "dimensions": {},
        }

    async def update_semantic_layer(
        self,
        source_id: str,
        semantic_doc: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist the canonical semantic doc — PRD-199 S1.

        The doc is the READER's shape (nl2sql_service.py renders it into the
        generation prompt): ``{instructions: str, metrics: {name: {sql,
        description}}, dimensions: {category: {name: sql}}}``.

        The pre-199 writer was broken end-to-end: it called two methods that
        were never defined anywhere (AttributeError on every save), wrote to
        the detached object ``_get_source`` returns (session closed — no
        commit), stored ``metrics`` as a LIST with ``sql_expression`` keys the
        dict-shaped reader could not consume, and had no ``instructions``
        write path at all. It also never workspace-scoped the lookup — the
        crash was the only thing stopping a cross-tenant write, so repairing
        the save without the scope (W3-S9) would have armed that hole.

        The SQL fragments are deliberately NOT validated here: they are
        prompt guidance the reader interpolates as text, never executed
        directly — the generated SQL they influence still passes the AST
        validator downstream. The phantom ``_validate_semantic_definitions``
        is deleted, not reimplemented as a stub.
        """
        from core.database.database import SessionLocal
        from core.models.database_knowledge import DatabaseKnowledgeSource as DBKSource

        doc = {
            "instructions": str(semantic_doc.get("instructions") or "").strip(),
            "metrics": dict(semantic_doc.get("metrics") or {}),
            "dimensions": dict(semantic_doc.get("dimensions") or {}),
            "updated_at": datetime.utcnow().isoformat(),
        }

        db_session = SessionLocal()
        try:
            query = db_session.query(DBKSource).filter(DBKSource.id == int(source_id))
            if workspace_id is not None:
                query = query.filter(DBKSource.workspace_id == str(workspace_id))
            source = query.first()
            if not source:
                raise ValueError(f"Database source {source_id} not found")
            source.semantic_layer = doc
            db_session.commit()
        finally:
            db_session.close()

        # Clear cache to force refresh
        if source_id in self.schema_cache:
            del self.schema_cache[source_id]
        return doc
    
    # PRD-199 S5: the legacy PRD-21 limb is deleted — _generate_sql /
    # _validate_sql / _execute_query / _create_agent_tools /
    # _build_sql_generation_prompt / _get_cache_key / _extract_tables_from_sql
    # were a shadow duplicate of the real pipeline (modules/nl2sql/query/*)
    # with zero callers, and _execute_query called a _build_connection_string
    # that was never defined anywhere (AttributeError if ever reached).

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

    async def resolve_source_id(
        self,
        workspace_id: str,
        database_name: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> Optional[str]:
        """Resolve a workspace's database source to a concrete ``source_id``.

        PRD-160 S1: the agent tool exposes an optional ``database_name`` but
        the query pipeline addresses a source by id. Resolution is ALWAYS
        scoped to the caller's workspace so an agent can never reach another
        workspace's source by guessing a name. When ``database_name`` is
        provided it is matched case-insensitively within the workspace;
        otherwise, a workspace with exactly one active source uses it. Zero
        matches and an ambiguous no-name pick (multiple active sources) both
        return ``None`` — the caller surfaces a helpful message rather than
        silently querying the wrong database.

        ``db_session`` is opt-in: the in-process executor passes its request
        session (saving a pooled connection); callers that omit it get a
        short-lived one.
        """
        from core.database.database import SessionLocal
        from core.models.database_knowledge import DatabaseKnowledgeSource as DBKSource

        if not workspace_id:
            return None
        own_session = db_session is None
        session = db_session or SessionLocal()
        try:
            q = session.query(DBKSource).filter(
                DBKSource.workspace_id == str(workspace_id),
                DBKSource.is_active.is_(True),
            )
            if database_name and database_name.strip():
                src = q.filter(DBKSource.name.ilike(database_name.strip())).first()
                return str(src.id) if src else None
            sources = q.order_by(DBKSource.created_at.desc()).all()
            if len(sources) == 1:
                return str(sources[0].id)
            # zero or ambiguous (multiple sources, none named) → caller decides
            return None
        finally:
            if own_session:
                session.close()

    def _decrypt_source_credentials(self, source) -> Dict[str, Any]:
        """Resolve + decrypt a source's DB credentials (shared by the NL path
        and PRD-160 S4 template execution)."""
        from core.credentials.service import CredentialStore
        from core.database.database import SessionLocal
        from core.credentials.encryption import EncryptionService

        db_session = SessionLocal()
        try:
            cred_store = CredentialStore(db_session)
            credential = cred_store.get_credential(source.credential_id)
            if not credential:
                raise ValueError(f"Credential {source.credential_id} not found")
            encryption = EncryptionService()
            return encryption.decrypt_dict(credential.encrypted_data)
        finally:
            db_session.close()

    async def execute_template(
        self,
        source_id,
        template_id,
        parameters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """Execute a saved Query Template (PRD-160 S4).

        The frontend's "Execute Query" button had no working backend route. This
        loads the workspace-scoped template, validates its SQL (SELECT-only +
        table allowlist via the AST validator, which preserves ``:name``
        placeholders), and runs it under the S2 guards with BOUND parameters so
        values are never interpolated into the SQL.
        """
        parameters = parameters or {}
        source = await self._get_source(str(source_id), workspace_id=workspace_id)

        from core.database.database import SessionLocal
        from core.models.database_knowledge import DatabaseQueryTemplate

        db = SessionLocal()
        try:
            tmpl = db.query(DatabaseQueryTemplate).filter(
                DatabaseQueryTemplate.id == int(template_id),
                DatabaseQueryTemplate.source_id == int(source_id),
            ).first()
            if not tmpl:
                return {"success": False, "error": "Template not found",
                        "data": [], "columns": [], "row_count": 0}
            sql_template = tmpl.sql_template
            viz = tmpl.visualization_type or "table"
            if sql_template:
                tmpl.usage_count = (tmpl.usage_count or 0) + 1
                db.commit()
        finally:
            db.close()

        if not sql_template:
            return {"success": False, "error": "Template has no SQL to execute",
                    "data": [], "columns": [], "row_count": 0}

        credentials = self._decrypt_source_credentials(source)

        validator = SQLValidator(
            max_limit=min(int(max_rows or 1000), source.max_rows_limit or 1000)
        )
        try:
            validated_sql, _ = validator.validate_and_rewrite(
                sql_template, schema_metadata=source.schema_metadata
            )
        except Exception as e:  # SQLValidationError or parse failure
            return {"success": False, "error": f"Template SQL invalid: {e}",
                    "data": [], "columns": [], "row_count": 0}

        try:
            columns, rows = self._run_sql_with_guards(
                source, credentials, validated_sql, params=parameters
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Template {template_id} execution failed: {e}")
            return {"success": False, "error": "Template execution failed",
                    "data": [], "columns": [], "row_count": 0}

        return {"success": True, "sql": validated_sql, "data": rows,
                "columns": columns, "row_count": len(rows),
                "visualization_type": viz}

    async def write_nl_audit(
        self,
        *,
        source_id,
        user_id=None,
        agent_id=None,
        nl_query: str,
        result: Dict[str, Any],
    ) -> None:
        """Write one ``DatabaseQueryAudit`` row for a natural-language query.

        PRD-160 S4: both NL entry points — the in-process agent tool and the
        ``/query`` API route — call this so every NL query lands exactly one
        audit row (workspace via the source FK, agent, SQL, outcome). Workspace
        is carried by the workspace-scoped source per the PRD-156 S3 source-join
        audit convention, so no separate column is needed. Best-effort: a failed
        audit write never breaks the query.
        """
        from core.database.database import SessionLocal
        from core.models.database_knowledge import DatabaseQueryAudit

        try:
            try:
                uid = int(user_id) if user_id not in (None, "") else None
            except (TypeError, ValueError):
                uid = None
            conf = result.get("confidence")
            conf_score = conf.get("score") if isinstance(conf, dict) else None
            db = SessionLocal()
            try:
                db.add(DatabaseQueryAudit(
                    tenant_id=1,  # legacy integer column
                    source_id=int(source_id),
                    user_id=uid,
                    agent_id=str(agent_id) if agent_id not in (None, "") else None,
                    natural_language_query=nl_query or "",
                    generated_sql=result.get("sql"),
                    validated_sql=result.get("sql"),
                    execution_time_ms=int(result.get("execution_time_ms") or 0),
                    row_count=int(result.get("row_count") or 0),
                    success=bool(result.get("success")),
                    error_message=result.get("error"),
                    confidence_score=conf_score,
                ))
                db.commit()
            finally:
                db.close()
        except Exception as e:  # noqa: BLE001 — audit is best-effort
            logger.warning(f"NL2SQL audit write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Canonical service accessor (PRD-160 S1)
# ---------------------------------------------------------------------------
_db_knowledge_service_singleton: Optional["DatabaseKnowledgeService"] = None


def get_database_knowledge_service() -> "DatabaseKnowledgeService":
    """Return the shared :class:`DatabaseKnowledgeService` singleton.

    PRD-160 S1: the in-process agent path (``exec_research``) and the API
    routes (``api/database_knowledge.get_services``) must construct the
    service from ONE place so its dependencies are wired identically
    everywhere — the API layer previously owned the only construction site,
    which an in-process tool executor cannot import without a modules→api
    layering inversion. Lazily builds the singleton with the same deps.
    """
    global _db_knowledge_service_singleton
    if _db_knowledge_service_singleton is None:
        from core.credentials.resolver import get_credential_resolver
        from core.llm import create_llm_manager

        _db_knowledge_service_singleton = DatabaseKnowledgeService(
            credential_resolver=get_credential_resolver(),
            llm_provider=create_llm_manager(service_name="orchestrator"),
            rag_service=RAGService(),
            context_engineering=ContextEngineeringService(),
            audit_service=AuditService(),
        )
    return _db_knowledge_service_singleton
