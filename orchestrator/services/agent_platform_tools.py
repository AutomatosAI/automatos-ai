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

from services.rag_service import RAGService
from services.codegraph_service import CodeGraphService
from services.tool_result_formatter import ToolResultFormatter
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
            }
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
        
        self.logger.info(f"🔧 Agent {agent_id} calling tool: {tool_name}")
        self.logger.info(f"  Parameters: {parameters}")
        
        try:
            if tool_name == "search_knowledge":
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)
                
                self.logger.info(f"  🔍 Searching knowledge base: '{query}' (limit: {limit})")
                
                # Call RAG service retrieve_context method
                search_results = await self.rag_service.retrieve_context(
                    query=query,
                    top_k=limit,
                    min_similarity=config.RAG_MIN_SIMILARITY
                )
                
                self.logger.info(f"  📊 RAG returned {len(search_results)} results")
                
                # Convert SearchResult objects to raw dicts for formatter
                raw_results = []
                for r in search_results[:limit]:
                    doc_content = r.document.content if hasattr(r, 'document') else str(r)
                    doc_source = 'Unknown'
                    if hasattr(r, 'document') and r.document:
                        meta = r.document.metadata or {}
                        doc_source = (
                            meta.get('source_file') or 
                            meta.get('source') or 
                            meta.get('filename') or 
                            r.document.source or 
                            'knowledge-base'
                        )
                    raw_results.append({
                        "content": doc_content,
                        "source": doc_source,
                        "similarity": float(r.score) if hasattr(r, 'score') else 0.0
                    })
                
                # Use unified formatter - NO MORE DUPLICATE LOGIC
                formatted = ToolResultFormatter.format_documents(raw_results)
                
                self.logger.info(f"  ✅ Returning {len(formatted)} formatted results")
                if formatted:
                    self.logger.info(f"  📄 Sample: {formatted[0].get('excerpt', '')[:100]}...")
                
                # Return standardized format
                return ToolResultFormatter.standardize_result(
                    {"success": bool(formatted), "results": formatted},
                    tool_name
                )
            
            elif tool_name == "semantic_search":
                # Use RAG service for semantic search as well
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)
                
                self.logger.info(f"  🔍 Semantic search via RAG: '{query}'")
                search_results = await self.rag_service.retrieve_context(
                    query=query,
                    top_k=limit,
                    min_similarity=config.RAG_MIN_SIMILARITY
                )
                
                # Convert to raw format for formatter
                raw_results = []
                for r in search_results[:limit]:
                    doc_content = r.document.content if hasattr(r, 'document') else str(r)
                    doc_source = r.document.metadata.get('source', 'Unknown') if hasattr(r, 'document') and r.document.metadata else 'Unknown'
                    raw_results.append({
                        "content": doc_content,
                        "source": doc_source,
                        "similarity": float(r.score) if hasattr(r, 'score') else 0.0
                    })
                
                # Use unified formatter
                formatted = ToolResultFormatter.format_documents(raw_results)
                
                self.logger.info(f"  ✅ Found {len(formatted)} semantic results")
                return ToolResultFormatter.standardize_result(
                    {"success": bool(formatted), "results": formatted},
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
                    {"success": bool(formatted), "results": formatted},
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

