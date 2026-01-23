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
        self.rag_service = RAGService()
        # RAG config (min_similarity etc.) from modules.rag (DB-backed)
        try:
            from modules.rag.config import RAGModuleConfig
            self.rag_config = RAGModuleConfig()
        except Exception:
            self.rag_config = None
        # CodeGraphService uses centralized embedding manager
        self.code_graph = CodeGraphService(db_session)
        self.logger = logger
    
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
                "description": "Search the codebase for functions, classes, and implementations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Code pattern, function name, or concept to search for"
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Filter by file extension (e.g., 'py', 'ts', 'js')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "switch_context",
                "description": "Switch your specialized toolset (e.g., from 'general' to 'coding' or 'ops'). Use this when you need tools not currently visible.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "enum": ["general", "coding", "ops", "research", "communication"],
                            "description": "The new context/mode to switch to"
                        }
                    },
                    "required": ["context"]
                }
            }
        ]
    
    async def switch_context(
        self,
        context: str,
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Switch the agent's active toolset context.
        
        Args:
            context: The new context (coding, ops, communication, research, all)
            agent_id: The agent ID
            
        Returns:
            Result message
        """
        from core.models import Agent
        from modules.tools.formatting.result_formatter import ToolResultFormatter
        
        self.logger.info(f"🔄 Agent {agent_id} switching context to '{context}'")
        
        try:
            # Update agent's active context in DB
            agent = self.db.query(Agent).get(agent_id)
            if not agent:
                return ToolResultFormatter.standardize_result(
                    {"success": False, "error": f"Agent {agent_id} not found"},
                    "switch_context"
                )
            
            # Map context to tool groups (simple for now)
            # This 'active_group' field might need to be added to Agent model
            # For now, we'll assume it exists or use a metadata field if available
            previous_context = getattr(agent, "active_context", "all")
            
            # Save new context to configuration (standard config field)
            if not agent.configuration:
                agent.configuration = {}
                
            # Make a copy to trigger SQLAlchemy detection of change for JSON field
            new_config = dict(agent.configuration)
            new_config["active_context"] = context
            agent.configuration = new_config
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(agent, "configuration")
            
            self.db.commit()
            
            return ToolResultFormatter.standardize_result(
                {
                    "success": True, 
                    "message": f"Context switched from '{previous_context}' to '{context}'. Your available tools have been updated.",
                    "previous_context": previous_context,
                    "new_context": context
                },
                "switch_context"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Context switch failed: {e}")
            return ToolResultFormatter.standardize_result(
                {"success": False, "error": str(e)},
                "switch_context"
            )

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
            if tool_name == "switch_context":
                return await self.switch_context(
                    context=parameters.get("context", "all"),
                    agent_id=agent_id
                )
            
            elif tool_name == "search_knowledge":
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)
                
                self.logger.info(f"  🔍 Searching knowledge base: '{query}' (limit: {limit})")
                
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
                    min_similarity=min_similarity
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
                min_similarity = 0.65
                try:
                    if self.rag_config is not None:
                        min_similarity = float(self.rag_config.retrieval.min_similarity)
                except Exception:
                    pass

                rag_result = await self.rag_service.retrieve_context(
                    query=query,
                    top_k=limit,
                    min_similarity=min_similarity
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
                project_name = parameters.get("project_name", "Automatos-ai")  # Default project (case-sensitive!)
                
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
                        limit=5
                    )
                    # search_symbols returns a dict with 'results' key
                    results = result_dict.get("results", []) if isinstance(result_dict, dict) else []
                    
                except Exception as e:
                    # Project might not exist or name mismatch - return helpful message
                    self.logger.warning(f"  ⚠️ CodeGraph search failed: {str(e)}")
                    return ToolResultFormatter.standardize_result(
                        {"success": False, "error": f"Codebase search unavailable - project '{project_name}' not found. Please index the codebase first."},
                        tool_name
                    )
                
                # Convert to raw format for formatter
                raw_results = []
                for r in results[:5]:
                    raw_results.append({
                        "symbol_name": r.get("name", "Unknown"),
                        "file_path": r.get("file_path", "Unknown"),
                        "symbol_type": r.get("symbol_type", "symbol"),
                        "code": r.get("code_snippet", "")[:600],
                        "score": r.get("score", 0.8)
                    })
                
                # Use unified formatter
                formatted = ToolResultFormatter.format_code(raw_results)
                
                self.logger.info(f"  ✅ Found {len(formatted)} code results")
                return ToolResultFormatter.standardize_result(
                    {"success": True, "results": formatted},
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

