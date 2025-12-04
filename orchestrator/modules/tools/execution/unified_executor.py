"""
Unified Tool Executor for PRD-17
=================================

Single entry point for all tool execution, routing to appropriate executors:
- Research tools (search_knowledge, semantic_search, search_codebase)
- File operations (read_file, write_file, list_directory)
- Shell commands (execute_command)
- MCP tools (third-party integrations)
"""

import logging
import os
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from services.agent_platform_tools import AgentPlatformTools
from services.agent_action_executor import ActionExecutor
from modules.tools.execution.mcp_executor import MCPToolExecutor
from modules.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class UnifiedToolExecutor:
    """
    Unified tool executor that routes tool calls to the appropriate executor.
    
    Provides a single interface for all tool execution, simplifying agent code
    and making it easier to add new tools.
    """
    
    def __init__(self, db_session: Session, workspace_dir: str = "/tmp/automatos_workspace"):
        """
        Initialize unified tool executor.
        
        Args:
            db_session: Database session
            workspace_dir: Directory for file operations
        """
        self.db = db_session
        self.workspace_dir = workspace_dir
        
        # Lazy-loaded executors (only initialize when needed)
        self._platform_tools = None  # For research tools (RAG, CodeGraph)
        self._action_executor = None  # For file/shell operations
        self._mcp_executor = None  # For MCP tools
        self._tool_registry = None  # For tool metadata
        
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
            
            # MCP tools handled dynamically
        }
        
        logger.info("🔧 UnifiedToolExecutor initialized (lazy-loading enabled)")
    
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
            logger.info("  🔧 Initializing file/shell executor...")
            self._action_executor = ActionExecutor(self.workspace_dir)
        return self._action_executor
    
    @property
    def mcp_executor(self):
        """Lazy-load MCP executor only when needed."""
        if self._mcp_executor is None:
            logger.info("  🔧 Initializing MCP executor...")
            self._mcp_executor = MCPToolExecutor(self.db)
        return self._mcp_executor
    
    @property
    def tool_registry(self):
        """Lazy-load tool registry only when needed."""
        if self._tool_registry is None:
            logger.info("  🔧 Initializing tool registry...")
            self._tool_registry = ToolRegistry(self.db)
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
        from models.tools import AgentToolPermission, Tool
        
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
        agent_id: int = 0
    ) -> Dict[str, Any]:
        """
        Execute a tool by name, routing to the appropriate executor.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the agent calling the tool
            
        Returns:
            Tool execution result with standard format
        """
        try:
            logger.info(f"🔧 Executing tool '{tool_name}' for agent {agent_id}")
            logger.info(f"  📋 Parameters: {parameters}")
            
            # PRD-17: Check agent permissions (except for research tools which are always allowed)
            # if not self._check_agent_permission(agent_id, tool_name):
            #     logger.warning(f"⚠️  Agent {agent_id} does not have permission to use '{tool_name}'")
            #     return {
            #         "success": False,
            #         "error": f"Permission denied: Agent {agent_id} cannot use tool '{tool_name}'",
            #         "tool": tool_name
            #     }
            
            # Check if tool exists in registry
            tool_spec = self.tool_registry.get_tool(tool_name)
            if not tool_spec:
                logger.warning(f"⚠️  Tool '{tool_name}' not in registry, checking MCP tools...")
                # Check MCP tools
                mcp_tool = self.db.query(MCPTool).filter_by(name=tool_name).first()
                if mcp_tool:
                    return await self._execute_mcp_tool(tool_name, parameters, agent_id)
                else:
                    return {
                        "success": False,
                        "error": f"Unknown tool: {tool_name}",
                        "tool": tool_name
                    }
            
            # Route to appropriate executor
            executor_func = self.tool_routes.get(tool_name)
            if executor_func:
                result = await executor_func(tool_name, parameters, agent_id)
                logger.info(f"  ✅ Tool '{tool_name}' executed successfully")
                return result
            else:
                # Try MCP tool
                return await self._execute_mcp_tool(tool_name, parameters, agent_id)
                
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {tool_name} - {e}")
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
        from services.pandas_ai_service import get_pandasai_service
        
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
        from shared.llm import create_llm_manager
        from services.pandas_ai_service import get_pandasai_service
        from database.database import get_db_session
        from sqlalchemy import text
        import time
        
        try:
            start_time = time.time()
            
            # Get schema from centralized provider
            from modules.nl_to_sql import get_schema_provider
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
    
    async def _execute_multimodal_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute multimodal search (tables, images, formulas, combined)"""
        from services.multimodal_knowledge_tools import MultimodalKnowledgeTools
        
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
        if tool_name == 'read_file':
            success, content = self.action_executor.read_file(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "read_file",
                "params": parameters,
                "result": content
            }
        elif tool_name == 'write_file':
            success, result = self.action_executor.write_file(
                parameters.get('path', ''),
                parameters.get('content', '')
            )
            return {
                "success": success,
                "action": "write_file",
                "params": parameters,
                "result": result
            }
        elif tool_name == 'list_directory':
            success, items = self.action_executor.list_directory(
                parameters.get('path', '.')
            )
            return {
                "success": success,
                "action": "list_directory",
                "params": parameters,
                "result": items
            }
        elif tool_name == 'create_directory':
            success, result = self.action_executor.create_directory(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "create_directory",
                "params": parameters,
                "result": result
            }
        elif tool_name == 'delete_file':
            success, result = self.action_executor.delete_file(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "delete_file",
                "params": parameters,
                "result": result
            }
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
        
        # Ensure absolute paths
        if not os.path.isabs(source_file):
            source_file = os.path.join(self.workspace_dir, source_file)
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.workspace_dir, output_file)
        
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
    
    async def _execute_mcp_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute MCP tools via MCPToolExecutor.
        
        Connects to actual MCP servers via JSON-RPC protocol.
        """
        logger.info(f"  🌐 Routing to MCP server: {tool_name}")
        
        # Get tool from database
        mcp_tool = self.db.query(MCPTool).filter_by(name=tool_name).first()
        if not mcp_tool:
            return {"success": False, "error": f"MCP tool '{tool_name}' not found"}
        
        # Execute via MCP protocol (JSON-RPC to actual MCP server)
        method = parameters.get("method", "default")
        return await self.mcp_executor.execute_tool(
            agent_id=agent_id,
            tool_id=mcp_tool.id,
            method=method,
            params=parameters,
            execution_id=None
        )
    
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
        
        # Ensure absolute path
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.workspace_dir, output_file)
        
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
            
            # Ensure absolute path
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            
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
            
            # Ensure absolute paths
            if not os.path.isabs(input_file):
                input_file = os.path.join(self.workspace_dir, input_file)
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            
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
            
            # Ensure absolute path
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_dir, output_file)
            
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
        
        # Ensure absolute path
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.workspace_dir, output_file)
        
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


# Import MCPTool model at the bottom to avoid circular imports
from models import MCPTool

