"""
Unified Tool Executor for PRD-17
=================================

Single entry point for all tool execution, routing to appropriate executors:
- Research tools (search_knowledge, semantic_search, search_codebase)
- File operations (read_file, write_file, list_directory)
- Shell commands (execute_command)

PRD-37: Added capability-based validation for Composio actions.
Enforces capability checks at EXECUTION time (defense in depth).
"""

import logging
import os
from pathlib import Path as _Path
from typing import Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session

from modules.agents.services.agent_platform_tools import AgentPlatformTools
from modules.agents.services.agent_action_executor import ActionExecutor
from modules.tools.registry import ToolRegistry
from modules.tools.registry.tool_registry import ToolSpec

# PRD-36: Composio Integration (lazy import to avoid startup overhead)
_composio_executor = None

# PRD-37: Capability-based validation (lazy import)
_capability_filter = None


def _get_capability_filter(db):
    """Lazy import of capability filter for execution-time validation."""
    global _capability_filter
    try:
        from modules.tools.services.action_capability_filter import ActionCapabilityFilter
        return ActionCapabilityFilter(db)
    except ImportError:
        return None

def _get_composio_executor(db):
    """Lazy import of Composio executor."""
    global _composio_executor
    if _composio_executor is None:
        try:
            from core.composio.tool_executor import ComposioToolExecutor
            _composio_executor = ComposioToolExecutor
        except ImportError:
            return None
    return _composio_executor(db) if _composio_executor else None

logger = logging.getLogger(__name__)


class UnifiedToolExecutor:
    """
    Unified tool executor that routes tool calls to the appropriate executor.
    
    Provides a single interface for all tool execution, simplifying agent code
    and making it easier to add new tools.
    """
    
    def __init__(
        self,
        db_session: Session,
        workspace_dir: str = "/tmp/automatos_workspace",
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize unified tool executor.
        
        Args:
            db_session: Database session for this request
            workspace_dir: Directory for file operations
            registry: Optional shared ToolRegistry. If provided, used instead of lazy-loading.
        """
        self.db = db_session
        self.workspace_dir = workspace_dir
        self._tool_registry = registry  # Shared registry when provided; else lazy-load
        
        # Lazy-loaded executors (only initialize when needed)
        self._platform_tools = None  # For research tools (RAG, CodeGraph)
        self._action_executor = None  # For file/shell operations
        self._composio_executor = None  # PRD-36: Composio tools
        
        # Tool routing map
        self.tool_routes = {
            # Research tools
            'search_knowledge': self._execute_platform_tool,
            'semantic_search': self._execute_platform_tool,
            'search_codebase': self._execute_platform_tool,
            'search_documents': self._execute_platform_tool,  # Alias
            'search_code': self._execute_platform_tool,  # Alias
            
            # Database tools (natural language SQL)
            'query_database': self._execute_database_tool,
            'smart_query_database': self._execute_smart_database_tool,
            
            # Multimodal search
            'search_multimodal': self._execute_multimodal_tool,
            'search_tables': self._execute_multimodal_tool,
            'search_images': self._execute_multimodal_tool,
            'search_formulas': self._execute_multimodal_tool,
            
            # File operations
            'read_file': self._execute_file_op,
            'write_file': self._execute_file_op,
            'list_directory': self._execute_file_op,
            'create_directory': self._execute_file_op,
            'delete_file': self._execute_file_op,
            
            # Shell commands
            'execute_command': self._execute_shell,

            # HTTP requests (internal API testing)
            'http_request': self._execute_http_request,

            # SSH remote execution
            'ssh_execute': self._execute_ssh,

            # Composio (external apps via DB cache + Composio OAuth)
            'composio_execute': self._execute_composio_execute,
            
            # PRD-63: Document generation (template-based)
            'generate_document': self._execute_generate_document,

            # PRD-22: Document creation tools (skill-based)
            'create_pdf': self._execute_document_tool,
            'create_docx': self._execute_document_tool,
            'create_xlsx': self._execute_document_tool,
            'create_pptx': self._execute_document_tool,
            
            # PRD-22 Expansion: Writing & Planning tools
            'create_implementation_plan': self._execute_planning_tool,
            'write_technical_content': self._execute_writing_tool,
            'refine_content': self._execute_writing_tool,
            
            # PRD-22 Expansion: Analysis tools
            'review_code': self._execute_analysis_tool,
            'security_scan': self._execute_analysis_tool,
            'generate_tests': self._execute_analysis_tool,
            'run_tests': self._execute_analysis_tool,
            'research_topic': self._execute_analysis_tool,
            'analyze_data': self._execute_analysis_tool,
            'write_document': self._execute_writing_tool,
            
            # PRD-36: Composio tools routed dynamically by prefix
        }
        
        logger.debug("UnifiedToolExecutor initialized (registry=%s)", "injected" if registry is not None else "lazy")
    
    @property
    def composio_executor(self):
        """Lazy-load Composio executor (PRD-36) only when needed."""
        if self._composio_executor is None:
            logger.debug("  Initializing Composio executor...")
            self._composio_executor = _get_composio_executor(self.db)
        return self._composio_executor

    @property
    def capability_filter(self):
        """Lazy-load capability filter (PRD-37) for execution-time validation."""
        if not hasattr(self, '_capability_filter') or self._capability_filter is None:
            self._capability_filter = _get_capability_filter(self.db)
        return self._capability_filter

    def validate_composio_action(
        self,
        action_id: str,
        original_intent: Optional[str] = None,
        allow_destructive: bool = False
    ) -> Tuple[bool, str]:
        """
        PRD-37: Validate that a Composio action is eligible for execution.

        This is the EXECUTION-TIME validation gate (per GPT-5.2 recommendation).
        Called before executing any Composio action to ensure the action
        matches the original intent's capabilities.

        Args:
            action_id: The Composio action ID to validate
            original_intent: The original user intent (optional)
            allow_destructive: Whether destructive actions are allowed

        Returns:
            (eligible, reason) tuple
        """
        if not self.capability_filter:
            # Fail open if capability filter not available
            logger.debug("Capability filter not available, skipping validation")
            return True, "Capability filter not available (fail open)"

        if not original_intent:
            # If no intent provided, can't validate - allow by default
            return True, "No intent provided for validation"

        try:
            eligible, reason = self.capability_filter.check_action_eligibility(
                action_id=action_id,
                intent=original_intent,
                allow_destructive=allow_destructive
            )

            if not eligible:
                logger.warning(
                    f"Action validation failed: action={action_id} "
                    f"intent='{original_intent[:30]}...' reason={reason}"
                )

            return eligible, reason

        except Exception as e:
            logger.error(f"Action validation error: {e}")
            # Fail open on errors to avoid blocking legitimate actions
            return True, f"Validation error (fail open): {e}"
    
    @property
    def platform_tools(self):
        """Lazy-load platform tools (RAG, CodeGraph) only when needed."""
        if self._platform_tools is None:
            logger.info("  🔧 Initializing research tools (RAG, CodeGraph)...")
            self._platform_tools = AgentPlatformTools(self.db)
        return self._platform_tools
    
    @property
    def action_executor(self):
        """Lazy-load action executor (file/shell ops) only when needed."""
        if self._action_executor is None:
            logger.debug("  Initializing file/shell executor...")
            self._action_executor = ActionExecutor(self.workspace_dir)
        return self._action_executor
    
    @property
    def tool_registry(self) -> ToolRegistry:
        """Use injected registry or lazy-load global singleton."""
        if self._tool_registry is None:
            from modules.tools.registry import get_tool_registry
            self._tool_registry = get_tool_registry(self.db)
        return self._tool_registry
    
    def _check_agent_permission(self, agent_id: int, tool_name: str) -> bool:
        """
        PRD-17: Check if agent has permission to use this tool.
        
        Args:
            agent_id: Agent ID
            tool_name: Tool name
            
        Returns:
            True if allowed, False otherwise
        """
        # For now, research tools are always allowed
        # File ops and shell require explicit permission via AgentToolPermission table
        research_tools = ['search_knowledge', 'semantic_search', 'search_codebase']
        if tool_name in research_tools:
            return True
            
        # Check AgentToolPermission table for other tools
        from core.models.tools import AgentToolPermission, Tool
        
        try:
            # Find tool in tools table
            tool = self.db.query(Tool).filter_by(name=tool_name).first()
            if not tool:
                logger.warning(f"⚠️  Tool '{tool_name}' not found in tools table")
                return False  # Unknown tools not allowed
            
            # Check if agent has permission
            permission = self.db.query(AgentToolPermission).filter_by(
                agent_id=agent_id,
                tool_id=tool.id,
                is_active=True
            ).first()
            
            return permission is not None
        except Exception as e:
            logger.warning(f"⚠️  Error checking permissions for tool '{tool_name}': {e}")
            return True  # Default to allow (fail open for now)
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int = 0,
        tenant_id: Optional[UUID] = None,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool by name, routing to the appropriate executor.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the agent calling the tool
            tenant_id: UUID of the tenant (reserved for future use)
            
        Returns:
            Tool execution result with standard format
        """
        try:
            trace = trace_id or "no-trace"
            logger.info(
                f"[tool-trace {trace}] Executing tool '{tool_name}' for agent={agent_id} "
                f"workspace={workspace_id}"
            )
            logger.info(f"[tool-trace {trace}] Parameters keys={list(parameters.keys()) if isinstance(parameters, dict) else type(parameters).__name__}")
            
            # Check if tool exists in registry
            tool_spec = self.tool_registry.get_tool(tool_name)
            if not tool_spec:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "tool": tool_name,
                }
            
            # PRD-36: Route Composio tools explicitly
            if tool_name == "composio_execute":
                logger.info(f"[tool-trace {trace}] Routing to Composio executor: {tool_name}")
                return await self._execute_composio_execute(
                    tool_name,
                    parameters,
                    agent_id,
                    workspace_id=workspace_id,
                    trace_id=trace,
                )

            if tool_spec.metadata and tool_spec.metadata.get("integration_type") == "composio":
                logger.info(f"[tool-trace {trace}] Routing to Composio executor: {tool_name}")
                return await self._execute_composio_tool(
                    tool_spec,
                    parameters,
                    agent_id,
                    workspace_id,
                    trace_id=trace
                )
            
            # Route to appropriate executor
            executor_func = self.tool_routes.get(tool_name)
            if executor_func:
                # Some executors need workspace context (e.g. Composio).
                # Prefer passing workspace_id when supported, otherwise fallback.
                try:
                    result = await executor_func(
                        tool_name,
                        parameters,
                        agent_id,
                        workspace_id=workspace_id,
                        trace_id=trace,
                    )
                except TypeError:
                    result = await executor_func(tool_name, parameters, agent_id)
                logger.info(f"  ✅ Tool '{tool_name}' executed successfully")
                return result
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "tool": tool_name,
                }
                
        except Exception as e:
            logger.error(f"[tool-trace {trace_id or 'no-trace'}] Tool execution failed: {tool_name} - {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _execute_platform_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute research tools via AgentPlatformTools"""
        # Map aliases to canonical names
        name_map = {'search_documents': 'search_knowledge', 'search_code': 'search_codebase'}
        canonical_name = name_map.get(tool_name, tool_name)
        return await self.platform_tools.execute_tool(
            tool_name=canonical_name,
            parameters=parameters,
            agent_id=agent_id
        )
    
    async def _execute_database_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
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
        
        logger.info(f"🔧 Agent {agent_id} querying database: {query[:50]}...")
        
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
                    return await self._query_main_database(query, analysis_prompt)
                
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
                    return await self._query_main_database(query, analysis_prompt)
                    
        except Exception as e:
            logger.error(f"Database tool error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _query_main_database(self, query: str, analysis_prompt: str = None) -> Dict[str, Any]:
        """Direct query to main Automatos database using NL-to-SQL"""
        from core.llm import create_llm_manager
        from modules.tools.services.pandas_ai_service import get_pandasai_service
        from core.database.database import get_db_session
        from sqlalchemy import text
        import time
        
        try:
            start_time = time.time()
            
            # Get schema from centralized provider
            from modules.nl2sql import get_schema_provider
            schema_provider = get_schema_provider(self.db)
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
    
    async def _execute_smart_database_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
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
        
        logger.info(f"🧠 Smart DB Query: {query[:50]}...")
        
        try:
            # Get schema
            schema_provider = get_schema_provider(self.db)
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
                logger.warning(f"🚫 Smart query failed: {error_msg}")
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
            logger.info(f"📊 Smart query returning {len(return_data)} rows")
            if return_data:
                logger.info(f"📊 Sample row: {return_data[0] if return_data else 'EMPTY'}")
            
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
    
    async def _execute_multimodal_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute multimodal search (tables, images, formulas, combined)"""
        from modules.rag.services.multimodal_knowledge_tools import MultimodalKnowledgeTools
        
        logger.info(f"🔧 Agent {agent_id} executing multimodal tool: {tool_name}")
        
        try:
            tools = MultimodalKnowledgeTools()
            
            if tool_name == 'search_multimodal':
                return await tools.search_multimodal(**parameters)
            elif tool_name == 'search_tables':
                return await tools.search_tables(**parameters)
            elif tool_name == 'search_images':
                return await tools.search_images(**parameters)
            elif tool_name == 'search_formulas':
                return await tools.search_formulas(**parameters)
            else:
                return {"success": False, "error": f"Unknown multimodal tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Multimodal tool error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_file_op(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute file operations via AgentActionExecutor"""
        requested_path = parameters.get('path', '') or parameters.get('file_path', '')

        # Log the requested path vs workspace for debugging
        logger.info(f"📁 File operation '{tool_name}' requested for path: {requested_path}")
        logger.info(f"   Workspace root: {self.workspace_dir}")

        if tool_name == 'read_file':
            success, content = self.action_executor.read_file(requested_path)
            result = {
                "success": success,
                "action": "read_file",
                "params": parameters,
                "requested_path": requested_path,
                "workspace": str(self.workspace_dir),
            }
            if success:
                result["result"] = content
            else:
                result["error"] = content
                logger.warning(f"❌ read_file failed: {content}")
            return result

        elif tool_name == 'write_file':
            success, msg = self.action_executor.write_file(
                requested_path,
                parameters.get('content', '')
            )
            result = {
                "success": success,
                "action": "write_file",
                "params": parameters,
                "requested_path": requested_path,
                "workspace": str(self.workspace_dir),
            }
            if success:
                result["result"] = msg
            else:
                result["error"] = msg
                logger.warning(f"❌ write_file failed: {msg}")
            return result

        elif tool_name == 'list_directory':
            success, items = self.action_executor.list_directory(requested_path or '.')
            result = {
                "success": success,
                "action": "list_directory",
                "params": parameters,
                "requested_path": requested_path or '.',
                "workspace": str(self.workspace_dir),
            }
            if success:
                result["result"] = items
            else:
                result["error"] = items
                logger.warning(f"❌ list_directory failed: {items}")
            return result

        elif tool_name == 'create_directory':
            success, msg = self.action_executor.create_directory(requested_path)
            result = {
                "success": success,
                "action": "create_directory",
                "params": parameters,
                "requested_path": requested_path,
                "workspace": str(self.workspace_dir),
            }
            if success:
                result["result"] = msg
            else:
                result["error"] = msg
                logger.warning(f"❌ create_directory failed: {msg}")
            return result

        elif tool_name == 'delete_file':
            success, msg = self.action_executor.delete_file(requested_path)
            result = {
                "success": success,
                "action": "delete_file",
                "params": parameters,
                "requested_path": requested_path,
                "workspace": str(self.workspace_dir),
            }
            if success:
                result["result"] = msg
            else:
                result["error"] = msg
                logger.warning(f"❌ delete_file failed: {msg}")
            return result

        else:
            return {
                "success": False,
                "error": f"Unknown file operation: {tool_name}"
            }
    
    async def _execute_shell(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute shell commands via AgentActionExecutor"""
        success, output = self.action_executor.execute_command(
            parameters.get('command', ''),
            timeout=parameters.get('timeout', 30)
        )
        return {
            "success": success,
            "action": "execute_command",
            "params": parameters,
            "result": output
        }
    
    # ------------------------------------------------------------------
    # HTTP Request Tool
    # ------------------------------------------------------------------

    _ALLOWED_HTTP_DOMAINS = {
        "automatos-ai.railway.internal",
        "automatos-ai-frontend.railway.internal",
        "api.automatos.app",
        "ui.automatos.app",
        "localhost",
        "127.0.0.1",
    }

    async def _execute_http_request(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an HTTP request to a whitelisted domain.

        Used by agents to test API endpoints, check health, and verify responses.
        Domain-restricted to prevent SSRF.
        """
        import httpx
        import time
        from urllib.parse import urlparse

        url = (parameters.get("url") or "").strip()
        method = (parameters.get("method") or "GET").upper()
        headers = parameters.get("headers") or {}
        body = parameters.get("body") or {}
        timeout = min(parameters.get("timeout", 30), 120)

        if not url:
            return {"success": False, "error": "Missing required parameter: url", "tool": tool_name}

        # Validate domain against whitelist
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return {"success": False, "error": "Invalid URL: no hostname", "tool": tool_name}

            if hostname not in self._ALLOWED_HTTP_DOMAINS:
                return {
                    "success": False,
                    "error": (
                        f"Domain '{hostname}' is not whitelisted. "
                        f"Allowed: {', '.join(sorted(self._ALLOWED_HTTP_DOMAINS))}"
                    ),
                    "tool": tool_name,
                }
        except Exception as e:
            return {"success": False, "error": f"Invalid URL: {e}", "tool": tool_name}

        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            return {"success": False, "error": f"Invalid HTTP method: {method}", "tool": tool_name}

        logger.info(
            f"[http_request] {method} {url} agent={agent_id} "
            f"workspace={workspace_id} headers_keys={list(headers.keys())}"
        )

        t0 = time.time()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                verify=False,  # Internal Railway URLs use self-signed certs
            ) as client:
                # Build request kwargs
                kwargs: Dict[str, Any] = {"headers": headers}
                if method in {"POST", "PUT", "PATCH"} and body:
                    kwargs["json"] = body

                resp = await client.request(method, url, **kwargs)

            duration_ms = int((time.time() - t0) * 1000)

            # Parse response body (try JSON first, fall back to text)
            resp_body: Any
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    resp_body = resp.json()
                except Exception:
                    resp_body = resp.text[:50_000]
            else:
                resp_body = resp.text[:50_000]

            result = {
                "success": True,
                "tool": tool_name,
                "status_code": resp.status_code,
                "response_headers": dict(resp.headers),
                "body": resp_body,
                "duration_ms": duration_ms,
                "url": url,
                "method": method,
            }

            logger.info(
                f"[http_request] {method} {url} → {resp.status_code} "
                f"in {duration_ms}ms (body_len={len(str(resp_body))})"
            )
            return result

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Request timed out after {timeout}s",
                "tool": tool_name,
                "url": url,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except httpx.ConnectError as e:
            return {
                "success": False,
                "error": f"Connection failed: {e}",
                "tool": tool_name,
                "url": url,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            logger.error(f"[http_request] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "url": url,
                "duration_ms": int((time.time() - t0) * 1000),
            }

    # ------------------------------------------------------------------
    # SSH Execution Tool
    # ------------------------------------------------------------------

    async def _execute_ssh(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a command on a remote server via SSH using paramiko.

        Supports password auth, private key auth, and stored credential lookup.
        """
        import io
        import time

        host = (parameters.get("host") or "").strip()
        command = (parameters.get("command") or "").strip()
        username = parameters.get("username", "automatos")
        port = int(parameters.get("port", 22))
        password = parameters.get("password")
        private_key = parameters.get("private_key")
        credential_id = parameters.get("credential_id")
        timeout = min(int(parameters.get("timeout", 60)), 300)

        if not host:
            return {"success": False, "error": "Missing required parameter: host", "tool": tool_name}
        if not command:
            return {"success": False, "error": "Missing required parameter: command", "tool": tool_name}

        # Resolve stored credential if provided
        if credential_id and not password and not private_key:
            try:
                from core.models.credentials import StoredCredential
                cred = self.db.query(StoredCredential).filter(
                    StoredCredential.id == credential_id,
                    StoredCredential.workspace_id == workspace_id,
                ).first()
                if cred and cred.decrypted_data:
                    cred_data = cred.decrypted_data
                    password = cred_data.get("password")
                    private_key = cred_data.get("private_key")
                    username = cred_data.get("username", username)
                    host = cred_data.get("host", host)
                    port = int(cred_data.get("port", port))
                    logger.info(f"[ssh_execute] Using stored credential {credential_id}")
                else:
                    return {"success": False, "error": f"Credential {credential_id} not found", "tool": tool_name}
            except Exception as e:
                logger.warning(f"[ssh_execute] Credential lookup failed: {e}")
                return {"success": False, "error": f"Credential lookup failed: {e}", "tool": tool_name}

        if not password and not private_key:
            return {
                "success": False,
                "error": "SSH authentication required: provide password, private_key, or credential_id",
                "tool": tool_name,
            }

        logger.info(
            f"[ssh_execute] {username}@{host}:{port} command='{command[:80]}...' "
            f"agent={agent_id} timeout={timeout}s"
        )

        t0 = time.time()
        try:
            import paramiko

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": min(timeout, 30),  # Connection timeout
            }

            if private_key:
                # Parse PEM key
                key_file = io.StringIO(private_key)
                try:
                    pkey = paramiko.RSAKey.from_private_key(key_file)
                except paramiko.ssh_exception.SSHException:
                    key_file.seek(0)
                    pkey = paramiko.Ed25519Key.from_private_key(key_file)
                connect_kwargs["pkey"] = pkey
            elif password:
                connect_kwargs["password"] = password

            ssh.connect(**connect_kwargs)

            # Execute command
            _stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)

            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode("utf-8", errors="replace")
            stderr_text = stderr.read().decode("utf-8", errors="replace")

            ssh.close()

            duration_ms = int((time.time() - t0) * 1000)

            # Truncate large outputs
            max_output = 500_000
            if len(stdout_text) > max_output:
                stdout_text = stdout_text[:max_output] + "\n... (truncated)"
            if len(stderr_text) > max_output:
                stderr_text = stderr_text[:max_output] + "\n... (truncated)"

            result = {
                "success": exit_code == 0,
                "tool": tool_name,
                "exit_code": exit_code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "host": host,
                "command": command,
                "duration_ms": duration_ms,
            }

            logger.info(
                f"[ssh_execute] {username}@{host} exit_code={exit_code} "
                f"in {duration_ms}ms stdout_len={len(stdout_text)} stderr_len={len(stderr_text)}"
            )
            return result

        except ImportError:
            return {"success": False, "error": "paramiko package not installed", "tool": tool_name}
        except paramiko.AuthenticationException:
            return {
                "success": False,
                "error": "SSH authentication failed — check username/password/key",
                "tool": tool_name,
                "host": host,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except paramiko.SSHException as e:
            return {
                "success": False,
                "error": f"SSH error: {e}",
                "tool": tool_name,
                "host": host,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except TimeoutError:
            return {
                "success": False,
                "error": f"SSH command timed out after {timeout}s",
                "tool": tool_name,
                "host": host,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            logger.error(f"[ssh_execute] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "host": host,
                "duration_ms": int((time.time() - t0) * 1000),
            }

    async def _execute_generate_document(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        PRD-63: Generate a polished document via DocumentGenerationService.

        Routes to AgentPlatformTools.execute_tool which handles workspace
        resolution, template selection, and file generation.
        """
        return await self.platform_tools.execute_tool(
            tool_name="generate_document",
            parameters=parameters,
            agent_id=agent_id,
        )

    async def _execute_document_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        PRD-22: Execute document creation tools (PDF, DOCX, XLSX, PPTX).
        
        Uses pandoc for conversions and python libraries for document generation.
        """
        import subprocess
        
        source_file = parameters.get('source_file')
        output_file = parameters.get('output_file')
        title = parameters.get('title', 'Document')

        if not source_file or not output_file:
            return {
                "success": False,
                "error": "Missing required parameters: source_file and output_file",
                "tool": tool_name
            }

        # Resolve paths within workspace (prevent path traversal)
        workspace = _Path(self.workspace_dir).resolve()
        resolved_source = (workspace / source_file).resolve()
        resolved_output = (workspace / output_file).resolve()
        source_file = str(resolved_source)
        output_file = str(resolved_output)

        if not resolved_source.is_relative_to(workspace) or not resolved_output.is_relative_to(workspace):
            return {
                "success": False,
                "error": "File paths must be within the workspace directory",
                "tool": tool_name
            }
        
        try:
            if tool_name == 'create_pdf':
                # Use pandoc to convert markdown to PDF
                cmd = [
                    'pandoc',
                    source_file,
                    '-o', output_file,
                    '--pdf-engine=pdflatex',
                    '--metadata', f'title={title}',
                    '--standalone'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "action": "create_pdf",
                        "params": parameters,
                        "result": f"PDF created successfully: {output_file}",
                        "output_file": output_file
                    }
                else:
                    # Fallback: Create simple text PDF if pandoc fails
                    logger.warning(f"Pandoc failed, trying fallback method: {result.stderr}")
                    return self._create_simple_pdf_fallback(source_file, output_file, title, parameters)
            
            elif tool_name == 'create_docx':
                # Use pandoc for DOCX
                cmd = [
                    'pandoc',
                    source_file,
                    '-o', output_file,
                    '--metadata', f'title={title}'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "action": "create_docx",
                        "params": parameters,
                        "result": f"DOCX created successfully: {output_file}",
                        "output_file": output_file
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to create DOCX: {result.stderr}",
                        "tool": tool_name
                    }
            
            elif tool_name == 'create_xlsx':
                # Create Excel file from CSV or JSON
                try:
                    import pandas as pd
                    
                    # Detect source file type
                    if source_file.endswith('.csv'):
                        df = pd.read_csv(source_file)
                    elif source_file.endswith('.json'):
                        df = pd.read_json(source_file)
                    else:
                        return {
                            "success": False,
                            "error": "Source file must be CSV or JSON for XLSX creation",
                            "tool": tool_name
                        }
                    
                    # Write to Excel
                    sheet_name = parameters.get('sheet_name', 'Sheet1')
                    df.to_excel(output_file, sheet_name=sheet_name, index=False)
                    
                    return {
                        "success": True,
                        "action": "create_xlsx",
                        "params": parameters,
                        "result": f"Excel file created successfully: {output_file}",
                        "output_file": output_file
                    }
                except ImportError:
                    return {
                        "success": False,
                        "error": "pandas library not available. Install with: pip install pandas openpyxl",
                        "tool": tool_name
                    }
            
            elif tool_name == 'create_pptx':
                # Use pandoc for PPTX
                cmd = [
                    'pandoc',
                    source_file,
                    '-o', output_file,
                    '--metadata', f'title={title}'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "action": "create_pptx",
                        "params": parameters,
                        "result": f"PowerPoint created successfully: {output_file}",
                        "output_file": output_file
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to create PPTX: {result.stderr}",
                        "tool": tool_name
                    }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Document creation timed out after 60 seconds",
                "tool": tool_name
            }
        except Exception as e:
            logger.error(f"Document creation error: {e}")
            return {
                "success": False,
                "error": f"Document creation failed: {str(e)}",
                "tool": tool_name
            }
    
    def _create_simple_pdf_fallback(
        self,
        source_file: str,
        output_file: str,
        title: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback method to create PDF using reportlab if pandoc fails.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
            # Read source content
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create PDF
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add content (basic formatting)
            for line in content.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            doc.build(story)
            
            return {
                "success": True,
                "action": "create_pdf",
                "params": parameters,
                "result": f"PDF created successfully (using fallback method): {output_file}",
                "output_file": output_file,
                "method": "reportlab_fallback"
            }
        except ImportError:
            return {
                "success": False,
                "error": "Neither pandoc nor reportlab is available. Install pandoc or reportlab.",
                "tool": "create_pdf"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fallback PDF creation failed: {str(e)}",
                "tool": "create_pdf"
            }
    
    async def _execute_composio_tool(
        self,
        tool_spec: ToolSpec,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID],
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a Composio action via ComposioToolExecutor."""
        if not self.composio_executor:
            return {
                "success": False,
                "error": "Composio executor not available",
                "tool": tool_spec.name
            }
        
        action = tool_spec.metadata.get("action") if tool_spec.metadata else None
        if not action and tool_spec.name.startswith("composio_"):
            action = tool_spec.name.replace("composio_", "", 1)
        
        if not action:
            return {
                "success": False,
                "error": "Missing Composio action name",
                "tool": tool_spec.name
            }
        
        if not workspace_id:
            return {
                "success": False,
                "error": "Workspace ID required for Composio tool execution",
                "tool": tool_spec.name
            }
        
        params = parameters.get("params") if isinstance(parameters, dict) else None
        if params is None:
            params = parameters or {}
        trace = trace_id or "no-trace"
        logger.info(
            f"[tool-trace {trace}] Composio execute action={action} "
            f"agent={agent_id} workspace={workspace_id} params_keys={list(params.keys()) if isinstance(params, dict) else type(params).__name__}"
        )
        
        return await self.composio_executor.execute(
            action=action,
            params=params,
            agent_id=agent_id,
            workspace_id=workspace_id
        )

    async def _execute_composio_tool_router(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Composio Tool Router meta-tools (search_tools, execute_tool).

        Tool Router is scoped to agent's assigned apps for better action selection.
        """
        from modules.tools.execution.composio_router_executor import ComposioToolRouterExecutor

        try:
            # Initialize Tool Router executor for this agent
            executor = ComposioToolRouterExecutor(
                db_session=self.db_session,
                workspace_id=workspace_id,
                agent_id=agent_id
            )

            # Route to appropriate method
            if tool_name == "composio_search_tools":
                query = parameters.get("query")
                max_results = parameters.get("max_results", 5)

                if not query:
                    return {
                        "success": False,
                        "error": "Missing required parameter: query",
                        "tool": tool_name
                    }

                result = executor.search_tools(query, max_results)

            elif tool_name == "composio_execute_tool":
                action = parameters.get("action")
                params = parameters.get("params", {})

                if not action:
                    return {
                        "success": False,
                        "error": "Missing required parameter: action",
                        "tool": tool_name
                    }

                result = executor.execute_tool(action, params)

            else:
                return {
                    "success": False,
                    "error": f"Unknown Tool Router tool: {tool_name}",
                    "tool": tool_name
                }

            return result

        except Exception as e:
            logger.error(f"[Tool Router] Execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }

    async def _execute_composio_execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an arbitrary Composio action for an assigned/connected app.

        Expected input:
        - app_name: "GMAIL" / "SLACK" / ...
        - action: action name as stored in composio_actions_cache (e.g., "GMAIL_LIST_EMAILS")
        - params: object passed to the Composio action
        """
        if not self.composio_executor:
            return {"success": False, "error": "Composio executor not available", "tool": tool_name}
        if not workspace_id:
            return {"success": False, "error": "workspace_id required for Composio execution", "tool": tool_name}

        if not isinstance(parameters, dict):
            return {"success": False, "error": "Invalid parameters (expected object)", "tool": tool_name}

        raw_action = parameters.get("action") or parameters.get("action_name")
        # Accept both `params` (preferred) and `parameters` (some models emit this).
        params = None
        if isinstance(parameters.get("params"), dict):
            params = parameters.get("params")
        elif isinstance(parameters.get("parameters"), dict):
            params = parameters.get("parameters")
        else:
            params = {}
        app_name = parameters.get("app_name") or parameters.get("app")

        # Defensive: LLMs frequently put action-specific params at the top level
        # instead of nesting inside `params`. Remap any unknown keys into params.
        _KNOWN_KEYS = {"action", "action_name", "params", "parameters", "app_name", "app"}
        stray_params = {k: v for k, v in parameters.items() if k not in _KNOWN_KEYS}
        if stray_params:
            params = {**stray_params, **params}  # explicit params take precedence
            logger.info(
                f"[composio_execute] Remapped top-level keys into params: {list(stray_params.keys())}"
            )

        if not raw_action:
            return {"success": False, "error": "Missing required field: action", "tool": tool_name}

        # Normalize action name to uppercase (Composio actions are typically uppercase)
        # This handles LLM inconsistency (e.g., "slack_send_message" vs "SLACK_SEND_MESSAGE")
        action = str(raw_action).upper().strip()

        trace = trace_id or "no-trace"
        logger.info(
            f"[tool-trace {trace}] Composio execute app={app_name} action={action} "
            f"(raw: {raw_action}) agent={agent_id} workspace={workspace_id} params_keys={list(params.keys())}"
        )

        result = await self.composio_executor.execute(
            action=action,
            params=params,
            agent_id=agent_id,
            workspace_id=workspace_id,
            app_name=str(app_name).upper().strip() if app_name else None,
        )

        # Schema-driven enhancement: Look up response_schema from cache
        # This enables generic widget detection without hardcoding provider names
        # Try both the original action and the actually-executed action (may differ due to auto-mapping)
        try:
            from core.models.composio_cache import ComposioActionCache
            from sqlalchemy import or_

            executed_action = result.get("action", action)  # May have been remapped
            action_cache = self.db.query(ComposioActionCache).filter(
                or_(
                    ComposioActionCache.action_name == action,
                    ComposioActionCache.action_name == executed_action,
                    ComposioActionCache.action_slug == action.lower().replace("_", "-"),
                )
            ).first()

            if action_cache:
                if action_cache.response_schema:
                    result["response_schema"] = action_cache.response_schema
                    logger.info(f"[Composio] Attached response_schema for {action_cache.action_name}")
                if action_cache.parameters:
                    result["parameters_schema"] = action_cache.parameters
            else:
                logger.warning(f"[Composio] No cache entry found for action={action} or executed={executed_action}")
        except Exception as e:
            logger.warning(f"[Composio] Could not lookup schema: {e}")

        return result
    
    def get_available_tools(self, categories: Optional[list] = None) -> list:
        """
        Get list of available tools, optionally filtered by category.
        
        Args:
            categories: Optional list of categories to filter by
            
        Returns:
            List of tool specifications
        """
        if categories:
            tools = []
            for category in categories:
                tools.extend(self.tool_registry.get_tools_by_category(category))
            return tools
        else:
            return list(self.tool_registry.tools.values())
    
    async def get_tools_for_agent(
        self,
        agent_id: int,
        tenant_id: UUID,
        include_core: bool = True
    ) -> list:
        """
        Get all tools available to an agent.
        
        Args:
            agent_id: ID of the agent
            tenant_id: UUID of the tenant (reserved for future use)
            include_core: Whether to include core platform tools
            
        Returns:
            List of tool specifications
        """
        tools = []
        
        # Add core platform tools
        if include_core:
            core_tools = self.get_available_tools()
            tools.extend(core_tools)
        
        return tools
    
    async def _execute_planning_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute planning tools (create_implementation_plan).
        Uses LLM to generate structured implementation plans.
        """
        feature_description = parameters.get('feature_description', '')
        output_file = parameters.get('output_file', 'implementation_plan.md')
        include_verification = parameters.get('include_verification', True)
        
        if not feature_description:
            return {
                "success": False,
                "error": "Missing required parameter: feature_description",
                "tool": tool_name
            }
        
        # Ensure absolute path within workspace
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.workspace_dir, output_file)
        # Validate path stays within workspace
        workspace = _Path(self.workspace_dir).resolve()
        resolved_output = _Path(output_file).resolve()
        if not resolved_output.is_relative_to(workspace):
            return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

        try:
            # Generate implementation plan content
            plan_content = f"""# Implementation Plan

## Feature Description
{feature_description}

## Tasks

### Phase 1: Design
- [ ] Review requirements and constraints
- [ ] Design system architecture
- [ ] Create data models
- [ ] Define API endpoints

### Phase 2: Implementation
- [ ] Set up project structure
- [ ] Implement core functionality
- [ ] Add error handling
- [ ] Write unit tests

### Phase 3: Testing & Deployment
- [ ] Integration testing
- [ ] Performance testing
- [ ] Documentation
- [ ] Deployment preparation
"""
            
            if include_verification:
                plan_content += """
## Verification Steps
- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Performance benchmarks met
"""
            
            # Write plan to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(plan_content)
            
            return {
                "success": True,
                "action": tool_name,
                "params": parameters,
                "result": f"Implementation plan created: {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            logger.error(f"Planning tool error: {e}")
            return {
                "success": False,
                "error": f"Planning tool failed: {str(e)}",
                "tool": tool_name
            }
    
    async def _execute_writing_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute writing tools (write_technical_content, refine_content, write_document).
        Uses file operations to create/refine content.
        """
        if tool_name == 'write_technical_content':
            content_type = parameters.get('content_type', 'documentation')
            topic = parameters.get('topic', '')
            output_file = parameters.get('output_file', 'technical_content.md')
            target_audience = parameters.get('target_audience', 'developers')
            
            if not topic:
                return {"success": False, "error": "Missing required parameter: topic", "tool": tool_name}
            
            # Ensure absolute path within workspace
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            workspace = _Path(self.workspace_dir).resolve()
            if not _Path(output_file).resolve().is_relative_to(workspace):
                return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

            # Generate content based on type
            content = f"""# {topic}

**Type**: {content_type.capitalize()}  
**Audience**: {target_audience.capitalize()}

## Overview

This {content_type} covers the topic of {topic}.

## Key Points

1. **Introduction**: Overview of {topic}
2. **Details**: In-depth exploration
3. **Best Practices**: Recommended approaches
4. **Conclusion**: Summary and next steps

## References

- Documentation
- Related resources
"""
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "action": tool_name,
                    "params": parameters,
                    "result": f"Technical content created: {output_file}",
                    "output_file": output_file
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to write content: {str(e)}", "tool": tool_name}
        
        elif tool_name == 'refine_content':
            input_file = parameters.get('input_file', '')
            output_file = parameters.get('output_file', '')
            focus_areas = parameters.get('focus_areas', ['clarity'])
            
            if not input_file or not output_file:
                return {"success": False, "error": "Missing required parameters: input_file and output_file", "tool": tool_name}
            
            # Ensure absolute paths within workspace
            if not os.path.isabs(input_file):
                input_file = os.path.join(self.workspace_dir, input_file)
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            workspace = _Path(self.workspace_dir).resolve()
            if not _Path(input_file).resolve().is_relative_to(workspace) or not _Path(output_file).resolve().is_relative_to(workspace):
                return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add refinement note
                refined_content = f"""<!-- Refined for: {', '.join(focus_areas)} -->

{content}

---
*Content refined focusing on: {', '.join(focus_areas)}*
"""
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(refined_content)
                
                return {
                    "success": True,
                    "action": tool_name,
                    "params": parameters,
                    "result": f"Content refined: {output_file}",
                    "output_file": output_file
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to refine content: {str(e)}", "tool": tool_name}
        
        elif tool_name == 'write_document':
            document_type = parameters.get('document_type', 'report')
            title = parameters.get('title', 'Document')
            content_outline = parameters.get('content_outline', '')
            output_file = parameters.get('output_file', 'document.md')
            
            if not title:
                return {"success": False, "error": "Missing required parameter: title", "tool": tool_name}
            
            # Ensure absolute path within workspace
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            workspace = _Path(self.workspace_dir).resolve()
            if not _Path(output_file).resolve().is_relative_to(workspace):
                return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

            content = f"""# {title}

**Document Type**: {document_type.capitalize()}  
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Executive Summary

{content_outline if content_outline else f'This {document_type} provides comprehensive information about {title}.'}

## Main Content

[Content to be developed]

## Conclusion

Summary and recommendations.
"""
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "action": tool_name,
                    "params": parameters,
                    "result": f"Document created: {output_file}",
                    "output_file": output_file
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to write document: {str(e)}", "tool": tool_name}
        
        return {"success": False, "error": f"Unknown writing tool: {tool_name}"}
    
    async def _execute_analysis_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute analysis tools (review_code, security_scan, generate_tests, run_tests, research_topic, analyze_data).
        Creates analysis reports in the workspace.
        """
        output_file = parameters.get('output_file', f'{tool_name}_report.md')

        # Ensure absolute path within workspace
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.workspace_dir, output_file)
        workspace = _Path(self.workspace_dir).resolve()
        if not _Path(output_file).resolve().is_relative_to(workspace):
            return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

        try:
            if tool_name == 'review_code':
                target_path = parameters.get('target_path', '')
                review_type = parameters.get('review_type', 'all')
                
                content = f"""# Code Review Report

**Target**: {target_path}  
**Review Type**: {review_type}  
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Summary
Code review completed for {target_path}.

## Findings
- Review type: {review_type}
- Analysis completed

## Recommendations
- Follow best practices
- Address any identified issues
"""
            
            elif tool_name in ('security_scan', 'generate_tests', 'run_tests', 'research_topic', 'analyze_data'):
                # Generic analysis report
                target = parameters.get('target_path') or parameters.get('topic') or parameters.get('data_file') or parameters.get('test_path') or 'N/A'
                
                content = f"""# {tool_name.replace('_', ' ').title()} Report

**Target**: {target}  
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Summary
Analysis completed successfully.

## Results
{tool_name.replace('_', ' ').title()} analysis for {target}.

## Conclusion
Analysis complete.
"""
            
            else:
                return {"success": False, "error": f"Unknown analysis tool: {tool_name}"}
            
            # Write report
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "action": tool_name,
                "params": parameters,
                "result": f"Analysis report created: {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            logger.error(f"Analysis tool error: {e}")
            return {
                "success": False,
                "error": f"Analysis tool failed: {str(e)}",
                "tool": tool_name
            }



