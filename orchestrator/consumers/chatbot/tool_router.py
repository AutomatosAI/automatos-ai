"""
Tool Router - Unified Tool Execution for Chat
==============================================

Routes tool execution through modules.tools (single source of truth).

Handles:
- Getting tools from modules.tools.ToolRegistry
- Executing tools via modules.tools.UnifiedToolExecutor
- Formatting results for frontend and LLM
- Building tool context injection messages
"""

import json
import logging
from typing import List, Dict, Any, Optional

# Use modules.tools directly - NO duplicate tool definitions
from modules.tools import ToolRegistry, ToolCategory, UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from database.database import get_db_session

logger = logging.getLogger(__name__)

# Global tool registry instance
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the ToolRegistry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        logger.info(f"✅ ToolRouter connected to modules.tools.ToolRegistry ({len(_tool_registry.tools)} tools)")
    return _tool_registry


def get_chatbot_tools() -> List[Dict[str, Any]]:
    """
    Get tools from modules.tools.ToolRegistry in OpenAI function format.
    SINGLE SOURCE OF TRUTH - no duplicate definitions.
    """
    registry = get_tool_registry()
    
    try:
        # Get RESEARCH and DATABASE tools for chatbot
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        database_tools = registry.get_tools_by_category(ToolCategory.DATABASE_TOOLS)
        all_tools = research_tools + database_tools
        
        # Convert to OpenAI function format
        openai_tools = []
        for tool in all_tools:
            schema = tool.to_openai_format()
            openai_tools.append({
                "type": "function",
                "function": schema
            })
        
        logger.debug(f"Loaded {len(openai_tools)} tools from modules.tools.ToolRegistry")
        return openai_tools
    except Exception as e:
        logger.error(f"Error loading tools from registry: {e}")
        return []


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: int = 1
) -> Dict[str, Any]:
    """
    Execute a tool via modules.tools.UnifiedToolExecutor.
    SINGLE ENTRY POINT for all tool execution in chat.
    """
    with get_db_session() as db_session:
        executor = UnifiedToolExecutor(db_session)
        return await executor.execute_tool(tool_name, tool_args, agent_id)


class ToolRouter:
    """
    Routes tool execution and formats results.
    Used by StreamingChatService for tool handling.
    """
    
    def __init__(self):
        self.formatter = ToolResultFormatter
    
    async def execute_and_format(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        agent_id: int = 1
    ) -> Dict[str, Any]:
        """
        Execute a tool and return formatted results.
        
        Returns:
            {
                'success': bool,
                'frontend_data': dict,  # For artifact viewer
                'llm_context': str,     # For LLM injection
                'raw_result': dict      # Original result
            }
        """
        logger.info(f"[ToolRouter] Executing {tool_name} with args: {tool_args}")
        
        try:
            result = await execute_tool(tool_name, tool_args, agent_id)
            
            # Check success
            success = (
                result.get('success') or
                result.get('status') == 'success' or
                bool(result.get('results'))
            )
            
            if success:
                frontend_data = self.formatter.format_for_frontend(result, tool_name)
                llm_context = self.formatter.format_for_llm(result, tool_name)
                
                logger.info(f"[ToolRouter] {tool_name} succeeded")
                return {
                    'success': True,
                    'frontend_data': frontend_data,
                    'llm_context': llm_context,
                    'raw_result': result
                }
            else:
                error = result.get('error', 'Unknown error')
                logger.warning(f"[ToolRouter] {tool_name} failed: {error}")
                return {
                    'success': False,
                    'frontend_data': {},
                    'llm_context': f"Tool {tool_name} failed: {error}",
                    'raw_result': result
                }
                
        except Exception as e:
            logger.error(f"[ToolRouter] {tool_name} exception: {e}")
            return {
                'success': False,
                'frontend_data': {},
                'llm_context': f"Tool {tool_name} error: {str(e)}",
                'raw_result': {'success': False, 'error': str(e)}
            }
    
    def build_tool_context_message(
        self,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Build a system message with tool results for LLM context.
        
        Returns None if results are empty or tool failed.
        """
        if not result.get('success'):
            return None
        
        raw = result.get('raw_result', {})
        standardized = self.formatter.standardize_result(raw, tool_name)
        
        if not standardized.get('results'):
            return None
        
        # Build context based on tool type
        if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
            docs = standardized['results'][:3]
            if not docs:
                return None
            
            doc_context = "\n\n".join([
                f"Document: {d.get('filename', d.get('source', 'Unknown'))}\n{d.get('excerpt', d.get('content', ''))[:500]}"
                for d in docs
            ])
            
            return {
                "role": "system",
                "content": f"📚 DOCUMENTS FROM USER'S KNOWLEDGE BASE:\n\n{doc_context}\n\nYou MUST use this information to answer."
            }
        
        elif tool_name in ['search_codebase', 'search_code']:
            code_results = standardized['results'][:5]
            if not code_results:
                return None
            
            code_context = "\n\n".join([
                f"📁 {r.get('symbol_name', 'Code')} ({r.get('file_path', 'unknown')})\n```python\n{r.get('code', '')[:800]}\n```"
                for r in code_results
            ])
            
            return {
                "role": "system",
                "content": f"💻 CODE FROM USER'S CODEBASE:\n\n{code_context}\n\nYou MUST show and explain this code when answering."
            }
        
        elif tool_name == 'query_database':
            data_preview = raw.get('data', [])[:5]
            data_str = json.dumps(data_preview, default=str)[:1000]
            db_context = f"Database query result: {raw.get('row_count', 0)} rows\nSQL: {raw.get('sql', 'N/A')[:200]}\nData preview: {data_str}"
            
            return {
                "role": "system",
                "content": f"🗄️ DATABASE QUERY RESULTS:\n\n{db_context}\n\nYou MUST use this data when answering."
            }
        
        return None
    
    def truncate_for_llm(
        self,
        result: Dict[str, Any],
        tool_name: str = "unknown",
        max_chars: int = 3000
    ) -> str:
        """Truncate tool results for LLM context."""
        return self.formatter.format_for_llm(result, tool_name, max_chars)


# Module-level instance
_tool_router = None

def get_tool_router() -> ToolRouter:
    """Get or create the global ToolRouter instance."""
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router


# Expose commonly used functions at module level
CHAT_TOOLS = None  # Lazy load to avoid import issues

def get_chat_tools() -> List[Dict[str, Any]]:
    """Get chatbot tools (lazy loaded)."""
    global CHAT_TOOLS
    if CHAT_TOOLS is None:
        CHAT_TOOLS = get_chatbot_tools()
    return CHAT_TOOLS

