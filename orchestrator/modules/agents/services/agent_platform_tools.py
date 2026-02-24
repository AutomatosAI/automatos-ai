"""
Agent Platform Tools - Enable agents to use Automatos platform capabilities
===========================================================================

Gives agents access to:
1. RAG Service - Search knowledge base
2. Semantic Search - Find similar content
3. CodeGraph - Query codebase structure

NO web search - keep it within platform knowledge bases.

Uses ToolResultFormatter for consistent result formatting across all tools.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from modules.rag import RAGService
from modules.codegraph import CodeGraphService
from config import config

logger = logging.getLogger(__name__)


class AgentPlatformTools:
    """
    Provides platform research tools to agents during execution.
    Restricted to internal knowledge bases only.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.rag_service = RAGService()  # auto-detects S3 vs pgvector from config
        # RAG config (min_similarity etc.) from modules.rag (DB-backed)
        try:
            from modules.rag.config import RAGModuleConfig
            self.rag_config = RAGModuleConfig()
        except Exception:
            self.rag_config = None
        # CodeGraphService uses centralized embedding manager
        self.code_graph = CodeGraphService(db_session)
        self.logger = logger
    
    def _resolve_workspace_id(self, agent_id: int) -> Optional[str]:
        """Resolve workspace_id from an agent ID."""
        try:
            from core.models import Agent as AgentModel
            agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent_row and getattr(agent_row, "workspace_id", None):
                return str(agent_row.workspace_id)
        except Exception:
            pass
        return None

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for function calling"""
        return [
            {
                "name": "search_knowledge",
                "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - describe what you want to find"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "semantic_search",
                "description": "Find semantically similar content across all platform documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concept or topic to find similar content for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_codebase",
                "description": "Search indexed codebase for symbols (functions, classes, methods) by name or semantic similarity. Results are ranked by structural importance (PageRank).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — symbol name or natural language description"
                        },
                        "project_name": {
                            "type": "string",
                            "description": "CodeGraph project name to search in"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["fuzzy", "semantic"],
                            "default": "fuzzy",
                            "description": "Search type: fuzzy (name matching) or semantic (meaning-based)"
                        },
                        "symbol_type": {
                            "type": "string",
                            "enum": ["function", "class", "method", "interface", "all"],
                            "default": "all",
                            "description": "Filter by symbol type"
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Filter by file extension (e.g., 'py', 'ts', 'js')"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max results to return"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_call_graph",
                "description": "Get the call graph for a symbol, showing what it calls and what calls it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol name to get call graph for"
                        },
                        "project_name": {
                            "type": "string",
                            "description": "CodeGraph project name"
                        },
                        "depth": {
                            "type": "integer",
                            "default": 2,
                            "description": "How many levels deep to traverse (max 5)"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming", "both"],
                            "default": "both",
                            "description": "Direction to traverse"
                        }
                    },
                    "required": ["symbol", "project_name"]
                }
            },
            {
                "name": "analyze_architecture",
                "description": "Get high-level architecture overview of an indexed codebase: modules, key classes, dependency patterns, top-referenced symbols.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "CodeGraph project name"
                        },
                        "focus_path": {
                            "type": "string",
                            "description": "Optional directory path to focus analysis on"
                        }
                    },
                    "required": ["project_name"]
                }
            },
            {
                "name": "find_dependencies",
                "description": "Find all symbols that depend on a given symbol, or that a symbol depends on.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol name to find dependencies for"
                        },
                        "project_name": {
                            "type": "string",
                            "description": "CodeGraph project name"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["dependents", "dependencies", "both"],
                            "default": "both",
                            "description": "dependents = who uses this, dependencies = what this uses"
                        }
                    },
                    "required": ["symbol", "project_name"]
                }
            },

            {
                "name": "generate_document",
                "description": "Generate a polished PDF, DOCX, or XLSX document from data. Use when the user asks for a report, invoice, export, or any formatted document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Document title"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["pdf", "docx", "xlsx"],
                            "description": "Output format"
                        },
                        "template_name": {
                            "type": "string",
                            "description": "Template to use (e.g. 'Basic Report', 'Invoice'). Omit for auto-selection."
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to populate the template — must match the template's expected schema"
                        }
                    },
                    "required": ["title", "format", "data"]
                }
            },
        ]



    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute a platform tool on behalf of an agent.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the requesting agent
            
        Returns:
            Tool execution results
        """
        # Import once at function start to avoid UnboundLocalError when exceptions happen
        from modules.tools.formatting.result_formatter import ToolResultFormatter

        self.logger.info(f"🔧 Agent {agent_id} calling tool: {tool_name}")
        self.logger.info(f"  Parameters: {parameters}")
        
        try:
            if tool_name == "search_knowledge":
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)

                self.logger.info(f"  🔍 Searching knowledge base: '{query}' (limit: {limit})")

                # Resolve workspace_id for multi-tenant isolation
                workspace_id = None
                try:
                    from core.models import Agent as AgentModel
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and getattr(agent_row, "workspace_id", None):
                        workspace_id = str(agent_row.workspace_id)
                except Exception:
                    workspace_id = None

                # Call RAG service retrieve_context method
                min_similarity = 0.65
                try:
                    if self.rag_config is not None:
                        min_similarity = float(self.rag_config.retrieval.min_similarity)
                except Exception:
                    pass

                rag_result = await self.rag_service.retrieve_context(
                    query=query,
                    top_k=limit,
                    min_similarity=min_similarity,
                    workspace_id=workspace_id
                )
                
                # RAGResult has .chunks (list of dicts with content, source_file, similarity)
                chunks = rag_result.chunks if hasattr(rag_result, 'chunks') else []
                self.logger.info(f"  📊 RAG returned {len(chunks)} results")
                
                # Convert chunks to raw dicts for formatter
                raw_results = []
                for chunk in chunks[:limit]:
                    raw_results.append({
                        "content": chunk.get("content", ""),
                        "source": chunk.get("source_file", "knowledge-base"),
                        "similarity": float(chunk.get("similarity", 0.0))
                    })
                
                # Use unified formatter - NO MORE DUPLICATE LOGIC
                formatted = ToolResultFormatter.format_documents(raw_results)
                
                self.logger.info(f"  ✅ Returning {len(formatted)} formatted results")
                if formatted:
                    self.logger.info(f"  📄 Sample: {formatted[0].get('excerpt', '')[:100]}...")
                
                # Return standardized format (empty results are still success)
                return ToolResultFormatter.standardize_result(
                    {"success": True, "results": formatted},
                    tool_name
                )
            
            elif tool_name == "semantic_search":
                # Use RAG service for semantic search as well
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)

                self.logger.info(f"  🔍 Semantic search via RAG: '{query}'")

                # Resolve workspace_id for multi-tenant isolation
                workspace_id = None
                try:
                    from core.models import Agent as AgentModel
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and getattr(agent_row, "workspace_id", None):
                        workspace_id = str(agent_row.workspace_id)
                except Exception:
                    workspace_id = None

                min_similarity = 0.65
                try:
                    if self.rag_config is not None:
                        min_similarity = float(self.rag_config.retrieval.min_similarity)
                except Exception:
                    pass

                rag_result = await self.rag_service.retrieve_context(
                    query=query,
                    top_k=limit,
                    min_similarity=min_similarity,
                    workspace_id=workspace_id
                )
                
                # RAGResult has .chunks (list of dicts with content, source_file, similarity)
                chunks = rag_result.chunks if hasattr(rag_result, 'chunks') else []
                
                # Convert to raw format for formatter
                raw_results = []
                for chunk in chunks[:limit]:
                    raw_results.append({
                        "content": chunk.get("content", ""),
                        "source": chunk.get("source_file", "knowledge-base"),
                        "similarity": float(chunk.get("similarity", 0.0))
                    })
                
                # Use unified formatter
                formatted = ToolResultFormatter.format_documents(raw_results)
                
                self.logger.info(f"  ✅ Found {len(formatted)} semantic results")
                return ToolResultFormatter.standardize_result(
                    {"success": True, "results": formatted},
                    tool_name
                )
            
            elif tool_name == "search_codebase":
                query = parameters.get("query", "")
                file_type = parameters.get("file_type")
                project_name = parameters.get("project_name")

                # Resolve workspace_id from agent (CodeGraph projects are workspace-scoped)
                workspace_id = None
                try:
                    from core.models import Agent as AgentModel
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and getattr(agent_row, "workspace_id", None):
                        workspace_id = str(agent_row.workspace_id)
                except Exception:
                    workspace_id = None

                # If project_name not provided, pick the most recently indexed active project
                if not project_name and workspace_id:
                    try:
                        from sqlalchemy import text as _text
                        row = self.db.execute(
                            _text(
                                """
                                SELECT name
                                FROM codegraph_projects
                                WHERE workspace_id = :workspace_id AND status = 'active'
                                ORDER BY last_indexed DESC NULLS LAST, updated_at DESC NULLS LAST
                                LIMIT 1
                                """
                            ),
                            {"workspace_id": workspace_id},
                        ).fetchone()
                        if row and row[0]:
                            project_name = row[0]
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ Failed to look up CodeGraph project: {e}")
                        project_name = None

                # No hardcoded fallback — fail gracefully if no project found
                if not project_name:
                    return ToolResultFormatter.standardize_result(
                        {
                            "success": False,
                            "error": (
                                "No CodeGraph project found for this workspace. "
                                "Please index a codebase first via the CodeGraph UI, "
                                "or specify a project_name in your query."
                            ),
                        },
                        tool_name
                    )
                
                if not self.code_graph:
                    self.logger.warning(f"  ⚠️ CodeGraphService not available (missing API key)")
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": "Codebase search unavailable - missing API key"},
                        tool_name
                    )
                
                self.logger.info(f"  🔍 Searching codebase: '{query}' in project '{project_name}'")
                try:
                    result_dict = await self.code_graph.search_symbols(
                        project_name=project_name,
                        query=query,
                        limit=20,  # Fetch more, will filter down to 10
                        workspace_id=workspace_id,
                    )
                    # search_symbols returns a dict with 'results' key
                    results = result_dict.get("results", []) if isinstance(result_dict, dict) else []
                    
                except Exception as e:
                    # Project might not exist or name mismatch - return helpful message
                    self.logger.warning(f"  ⚠️ CodeGraph search failed: {str(e)}")
                    return ToolResultFormatter.standardize_result(
                        {
                            "success": False,
                            "error": (
                                f"Codebase search unavailable - no indexed CodeGraph project found for this workspace "
                                f"(requested '{project_name}'). Please index the codebase first."
                            ),
                        },
                        tool_name
                    )
                
                # PRD-62: Apply PageRank ranking for structurally important results
                try:
                    from modules.codegraph.ranking import PageRankRanker
                    from sqlalchemy import text as sa_text

                    # Fetch relationships for the project to build the graph
                    proj_row = self.db.execute(
                        sa_text("SELECT id FROM codegraph_projects WHERE name = :n AND workspace_id = :w"),
                        {"n": project_name, "w": workspace_id}
                    ).fetchone()

                    if proj_row:
                        rels = self.db.execute(
                            sa_text("""
                                SELECT from_symbol_id, to_symbol_id, relationship_type
                                FROM codegraph_relationships
                                WHERE project_id = :pid AND relationship_type != 'external_reference'
                            """),
                            {"pid": proj_row.id}
                        ).fetchall()
                        rel_dicts = [
                            {"from_symbol_id": r.from_symbol_id, "to_symbol_id": r.to_symbol_id,
                             "relationship_type": r.relationship_type}
                            for r in rels
                        ]
                        ranker = PageRankRanker()
                        results = ranker.rank_symbols(results, rel_dicts, token_budget=2048)
                        self.logger.info(f"  PageRank ranked {len(results)} results")
                except Exception as e:
                    self.logger.debug(f"  PageRank ranking skipped: {e}")

                # Convert to raw format for formatter
                raw_results = []
                for r in results:
                    code_snippet = r.get("code_snippet", "")
                    if len(code_snippet.strip()) < 50:
                        continue
                    raw_results.append({
                        "symbol_name": r.get("name", "Unknown"),
                        "file_path": r.get("file_path", "Unknown"),
                        "symbol_type": r.get("symbol_type", "symbol"),
                        "line_number": r.get("line_number", 0),
                        "code": code_snippet,
                        "docstring": r.get("docstring", ""),
                        "signature": r.get("signature", ""),
                        "score": r.get("score", 0.8),
                        "importance_rank": r.get("importance_rank", 0.0),
                    })
                    if len(raw_results) >= 10:
                        break
                
                # Use unified formatter
                formatted = ToolResultFormatter.format_code(raw_results)
                
                self.logger.info(f"  ✅ Found {len(formatted)} code results")
                return ToolResultFormatter.standardize_result(
                    {"success": True, "results": formatted},
                    tool_name
                )
            
            elif tool_name == "get_call_graph":
                symbol = parameters.get("symbol", "")
                project_name = parameters.get("project_name", "")
                depth = min(parameters.get("depth", 2), 5)
                direction = parameters.get("direction", "both")

                workspace_id = self._resolve_workspace_id(agent_id)
                self.logger.info(f"  Getting call graph for '{symbol}' in '{project_name}'")
                try:
                    result = await self.code_graph.get_call_graph(
                        project_name=project_name,
                        symbol=symbol,
                        depth=depth,
                        direction=direction,
                        workspace_id=workspace_id,
                    )
                    return ToolResultFormatter.standardize_result(
                        {"success": True, "results": [result]}, tool_name
                    )
                except Exception as e:
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": str(e)}, tool_name
                    )

            elif tool_name == "analyze_architecture":
                project_name = parameters.get("project_name", "")
                focus_path = parameters.get("focus_path")

                workspace_id = self._resolve_workspace_id(agent_id)
                self.logger.info(f"  Analyzing architecture of '{project_name}'")
                try:
                    # Look up project ID from name
                    from sqlalchemy import text
                    row = self.db.execute(
                        text("SELECT id FROM codegraph_projects WHERE name = :n AND workspace_id = :w"),
                        {"n": project_name, "w": workspace_id}
                    ).fetchone()
                    if not row:
                        return ToolResultFormatter.standardize_result(
                            {"success": False, "error": f"Project '{project_name}' not found"}, tool_name
                        )
                    result = await self.code_graph.analyze_architecture(
                        project_id=row.id,
                        workspace_id=workspace_id,
                        focus_path=focus_path,
                    )
                    return ToolResultFormatter.standardize_result(
                        {"success": True, "results": [result]}, tool_name
                    )
                except Exception as e:
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": str(e)}, tool_name
                    )

            elif tool_name == "find_dependencies":
                symbol = parameters.get("symbol", "")
                project_name = parameters.get("project_name", "")
                direction = parameters.get("direction", "both")

                workspace_id = self._resolve_workspace_id(agent_id)
                self.logger.info(f"  Finding dependencies for '{symbol}' in '{project_name}'")
                try:
                    from sqlalchemy import text
                    row = self.db.execute(
                        text("SELECT id FROM codegraph_projects WHERE name = :n AND workspace_id = :w"),
                        {"n": project_name, "w": workspace_id}
                    ).fetchone()
                    if not row:
                        return ToolResultFormatter.standardize_result(
                            {"success": False, "error": f"Project '{project_name}' not found"}, tool_name
                        )
                    result = await self.code_graph.find_dependencies(
                        project_id=row.id,
                        symbol_name=symbol,
                        direction=direction,
                        workspace_id=workspace_id,
                    )
                    return ToolResultFormatter.standardize_result(
                        {"success": True, "results": [result]}, tool_name
                    )
                except Exception as e:
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": str(e)}, tool_name
                    )

            elif tool_name == "generate_document":
                title = parameters.get("title", "Document")
                fmt = parameters.get("format", "pdf")
                data = parameters.get("data", {})
                template_name = parameters.get("template_name")

                self.logger.info(f"  📄 Generating {fmt.upper()} document: '{title}'")

                # Resolve workspace_id from agent
                workspace_id = None
                try:
                    from core.models import Agent as AgentModel
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and getattr(agent_row, "workspace_id", None):
                        workspace_id = agent_row.workspace_id
                except Exception:
                    pass

                if not workspace_id:
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": "Cannot resolve workspace for document generation"},
                        tool_name
                    )

                from modules.documents.generation_service import DocumentGenerationService
                gen_service = DocumentGenerationService(self.db, workspace_id)
                result = await gen_service.generate(
                    title=title,
                    format=fmt,
                    data=data,
                    workspace_id=workspace_id,
                    template_name=template_name,
                )

                self.logger.info(f"  ✅ Document generated: {result.filename} ({result.size // 1024}KB)")
                return ToolResultFormatter.standardize_result(
                    {
                        "success": True,
                        "results": [{
                            "status": "success",
                            "filename": result.filename,
                            "format": result.format,
                            "download_url": result.download_url,
                            "size_kb": result.size // 1024,
                        }],
                    },
                    tool_name
                )

            else:
                self.logger.error(f"  ❌ Unknown tool: {tool_name}")
                return ToolResultFormatter.standardize_result(
                    {"success": False, "error": f"Unknown tool: {tool_name}"},
                    tool_name
                )
                
        except Exception as e:
            self.logger.error(f"  ❌ Tool execution failed: {e}")
            return ToolResultFormatter.standardize_result(
                {"success": False, "error": str(e)},
                tool_name
            )
    
    def format_tool_results_for_prompt(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format tool results into a context section for agent prompt.
        Uses ToolResultFormatter for consistent formatting.
        
        Args:
            tool_results: Results from tool executions
            
        Returns:
            Formatted context string
        """
        if not tool_results:
            return ""

        # Lazy import to avoid circular dependency at module import time
        from modules.tools.formatting.result_formatter import ToolResultFormatter
        
        context_parts = ["## Research Results from Your Tool Calls:", ""]
        
        for idx, result in enumerate(tool_results, 1):
            tool_name = result.get("metadata", {}).get("tool") or result.get("tool", "Unknown")
            
            if not result.get("success"):
                context_parts.append(f"### Tool Call {idx}: {tool_name} - FAILED")
                context_parts.append(f"Error: {result.get('error', 'Unknown error')}")
                context_parts.append("")
                continue
            
            # Use unified formatter for LLM context
            llm_context = ToolResultFormatter.format_for_llm(result, tool_name, max_chars=1500)
            context_parts.append(f"### Tool Call {idx}:")
            context_parts.append(llm_context)
            context_parts.append("")
        
        context_parts.append("---")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def create_tool_enabled_prompt(
        self,
        base_task: str,
        initial_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt that enables tool use for research.
        
        Args:
            base_task: The task description
            initial_context: Optional initial context from RAG
            
        Returns:
            Tool-enabled prompt
        """
        
        tools = self.get_available_tools()
        
        prompt_parts = [
            f"# Task: {base_task}",
            "",
            "## Available Research Tools:",
            "",
            "You have access to these platform tools to help you research:",
            ""
        ]
        
        for tool in tools:
            prompt_parts.append(f"### `{tool['name']}`")
            prompt_parts.append(f"{tool['description']}")
            prompt_parts.append(f"Parameters: {', '.join(tool['parameters']['properties'].keys())}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## How to Use Tools:",
            "",
            "Call tools using this JSON format:",
            '```',
            '{"action": "search_knowledge", "params": {"query": "your search query"}}',
            '```',
            "",
            "You can call multiple tools to gather comprehensive information.",
            ""
        ])
        
        if initial_context:
            prompt_parts.extend([
                "## Initial Context Provided:",
                "",
                initial_context,
                "",
                "Use the tools above to find additional information if needed.",
                ""
            ])
        
        prompt_parts.extend([
            "## Instructions:",
            "1. If you need more information, call the research tools",
            "2. Use the results to provide accurate, detailed responses",
            "3. Cite your sources (which tool and what you found)",
            "4. Do NOT say 'I don't have information' - use the tools to find it!",
            "",
            "Begin working on the task now."
        ])
        
        return "\n".join(prompt_parts)

