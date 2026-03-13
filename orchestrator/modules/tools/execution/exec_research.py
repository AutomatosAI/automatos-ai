"""
Database / research tool executors -- NL-to-SQL, smart queries, direct DB fallback.
Extracted from unified_executor.py.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def execute_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Execute database query using natural language.
    Routes to knowledge sources or main database fallback.
    """
    import httpx
    from config import config
    from modules.tools.services.pandas_ai_service import get_pandasai_service

    query = parameters.get('query', '')
    database_name = parameters.get('database_name')
    analysis_prompt = parameters.get('analysis_prompt', query)

    logger.info(f"Agent {agent_id} querying database: {query[:50]}...")

    try:
        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get available database sources
            sources = []
            try:
                resp = await client.get(f"{base_url}/api/knowledge/sources/database/", params={"active_only": True})
                if resp.status_code == 200:
                    sources = resp.json() or []
            except Exception as e:
                logger.warning(f"Failed to list database sources: {e}")

            # Select source
            selected = None
            if database_name and sources:
                selected = next((s for s in sources if str(s.get("name", "")).lower() == database_name.lower()), None)
            if not selected and sources:
                selected = sources[0]

            # Fallback to direct main DB query if no sources
            if not selected:
                logger.info("No knowledge sources - using direct DB fallback")
                return await query_main_database(executor, query, analysis_prompt)

            # Query via knowledge API
            source_id = int(selected.get("id"))
            resp = await client.post(
                f"{base_url}/api/knowledge/sources/database/{source_id}/query",
                json={"query": query, "source_id": source_id, "use_cache": True, "include_explanation": True}
            )

            if resp.status_code == 200:
                data = resp.json()
                result = {
                    "success": True,
                    "database": data.get('database') or selected.get('name', ''),
                    "sql": data.get('sql', ''),
                    "row_count": data.get('row_count', 0),
                    "data": data.get('data', []),
                    "columns": data.get('columns', []),
                    "execution_time_ms": data.get('execution_time_ms', 0)
                }

                # Add PandasAI insight if available
                pandasai = get_pandasai_service()
                if pandasai and result.get('data'):
                    insight = pandasai.generate_insight(analysis_prompt, result['data'], result['columns'])
                    if insight:
                        result["pandas_ai"] = insight

                return result
            else:
                # Knowledge API failed - fallback to direct main database query
                logger.warning(f"Knowledge API returned {resp.status_code}, falling back to direct DB")
                return await query_main_database(executor, query, analysis_prompt)

    except Exception as e:
        logger.error(f"Database tool error: {e}")
        return {"success": False, "error": str(e)}


async def query_main_database(executor, query: str, analysis_prompt: str = None) -> Dict[str, Any]:
    """Direct query to main Automatos database using NL-to-SQL."""
    from core.llm import create_llm_manager
    from modules.tools.services.pandas_ai_service import get_pandasai_service
    from core.database.database import get_db_session
    from sqlalchemy import text

    try:
        start_time = time.time()

        # Get schema from centralized provider
        from modules.nl2sql import get_schema_provider
        schema_provider = get_schema_provider(executor.db)
        schema = schema_provider.get_database_schema_overview()

        # Add query guidance
        schema += "\n\nUse DATE_TRUNC('day', col) for daily grouping. Use NOW() - INTERVAL 'N days' for date ranges.\nAlways use explicit JOIN syntax. Aggregate by date for time-based queries."

        # Get LLM from orchestrator service settings
        llm_manager = create_llm_manager(service_name="orchestrator")

        response = await llm_manager.generate_response([
            {"role": "system", "content": f"You are a PostgreSQL expert. Generate ONLY the SQL query, nothing else. Only SELECT allowed. Always include proper date handling and grouping.\n\n{schema}"},
            {"role": "user", "content": f"Generate SQL for: {query}"}
        ])

        sql = response.content.strip()
        # Clean markdown code blocks
        if "```" in sql:
            parts = sql.split("```")
            for part in parts:
                if "SELECT" in part.upper():
                    sql = part.strip()
                    break
            sql = sql.replace("sql", "").strip()

        sql = sql.strip()

        if not sql.upper().startswith("SELECT"):
            return {"success": False, "error": "Only SELECT queries allowed"}

        logger.info(f"[Database Tool] Generated SQL: {sql[:200]}...")

        # Execute SQL
        with get_db_session() as session:
            result = session.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()

            data = []
            for row in rows[:1000]:  # Limit to 1000 rows
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    # Handle datetime serialization
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    # Handle Decimal
                    elif hasattr(val, '__float__'):
                        val = float(val)
                    # Handle JSON/dict
                    elif isinstance(val, dict):
                        val = val
                    row_dict[col] = val
                data.append(row_dict)

        tool_result = {
            "success": True,
            "database": "automatos_main",
            "sql": sql,
            "row_count": len(data),
            "data": data,
            "columns": columns,
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }

        # Generate insight if analysis prompt provided
        if analysis_prompt and data:
            pandasai = get_pandasai_service()
            if pandasai:
                insight = pandasai.generate_insight(analysis_prompt, data, columns)
                if insight:
                    tool_result["pandas_ai"] = insight

        return tool_result
    except Exception as e:
        logger.error(f"Direct DB query error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def execute_smart_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Execute smart database query using SmartNL2SQLAgent.

    Features:
    - Query clarification (returns questions if query is ambiguous)
    - Query rephrasing (improves vague queries)
    - Result explanation (explains what the data means)
    - Visualization suggestions (recommends chart types)
    - Multi-turn conversation support
    """
    from core.llm import create_llm_manager
    from modules.nl2sql import SmartNL2SQLAgent, get_schema_provider
    from modules.tools.services.pandas_ai_service import get_pandasai_service
    from sqlalchemy import text

    query = parameters.get('query', '')
    skip_clarification = parameters.get('skip_clarification', False)
    clarification_answers = parameters.get('clarification_answers')
    database_name = parameters.get('database_name')
    include_visualization = parameters.get('include_visualization', True)

    logger.info(f"Smart DB Query: {query[:50]}...")

    try:
        # Get schema
        schema_provider = get_schema_provider(executor.db)
        schema_metadata = schema_provider.get_schema_metadata()

        # Get LLM
        llm_manager = create_llm_manager(service_name="nl2sql")

        # Create smart agent
        agent = SmartNL2SQLAgent(
            llm_provider=llm_manager,
            schema_metadata=schema_metadata,
            auto_clarify=not skip_clarification,
            auto_rephrase=True,
            auto_explain=True,
            auto_visualize=include_visualization,
        )

        # Define SQL executor - use fresh session to avoid transaction conflicts
        async def execute_sql(sql: str):
            from core.database.database import SessionLocal
            session = SessionLocal()
            try:
                result = session.execute(text(sql))
                columns = list(result.keys())
                rows = result.fetchall()
                data = []
                for row in rows[:1000]:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        val = row[i]
                        if hasattr(val, 'isoformat'):
                            val = val.isoformat()
                        elif hasattr(val, '__float__'):
                            val = float(val)
                        row_dict[col] = val
                    data.append(row_dict)
                return data
            finally:
                session.close()

        # Execute smart query
        result = await agent.query(
            natural_language_query=query,
            clarification_answers=clarification_answers,
            skip_clarification=skip_clarification,
            execute_sql=True,
            db_executor=execute_sql,
        )

        # Handle clarification needed
        if result.get('status') == 'needs_clarification':
            return {
                "success": True,
                "status": "needs_clarification",
                "clarifications": result.get('clarifications', []),
                "message": "Please provide more details to complete the query.",
                "original_query": query,
            }

        # Handle error from SmartNL2SQLAgent (e.g., SQL validation failed)
        if result.get('status') == 'error':
            error_msg = result.get('error') or result.get('message') or 'Query failed'
            logger.warning(f"Smart query failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "sql": result.get('sql', ''),
                "original_query": query,
            }

        # Add PandasAI insight if not already present
        if result.get('data') and not result.get('pandas_ai'):
            pandasai = get_pandasai_service()
            if pandasai:
                insight = pandasai.generate_insight(query, result['data'], list(result['data'][0].keys()) if result['data'] else [])
                if insight:
                    result['pandas_ai'] = insight

        # Debug: Log what we're returning
        return_data = result.get('data', [])
        logger.info(f"Smart query returning {len(return_data)} rows")
        if return_data:
            logger.info(f"Sample row: {return_data[0] if return_data else 'EMPTY'}")

        return {
            "success": True,
            "database": database_name or "automatos_main",
            "sql": result.get('sql', ''),
            "row_count": len(return_data),
            "data": return_data,
            "columns": list(return_data[0].keys()) if return_data else [],
            "execution_time_ms": 0,
            "explanation": result.get('explanation', ''),
            "rephrased_query": result.get('rephrased_query'),
            "visualization": result.get('visualization'),
            "follow_up_questions": result.get('follow_up_questions', []),
            "pandas_ai": result.get('pandas_ai'),
        }

    except Exception as e:
        logger.error(f"Smart DB query error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
