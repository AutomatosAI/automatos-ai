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
from contextlib import contextmanager

# Use modules.tools directly - NO duplicate tool definitions
from modules.tools import ToolCategory, UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from modules.tools.registry import get_tool_registry as registry_get_tool_registry
from core.database.database import SessionLocal

logger = logging.getLogger(__name__)

@contextmanager
def _session_scope():
    """Lightweight context manager to ensure sessions are always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Global tool registry instance (backed by modules.tools.registry singleton)
_tool_registry = None


def _get_registry(db_session=None):
    """
    Get the shared ToolRegistry with a DB session so MCP tools are loaded.
    If the registry already exists without a DB, injecting a session later
    will populate MCP tools via registry_get_tool_registry.
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = registry_get_tool_registry(db_session=db_session)
        logger.info(f"✅ ToolRouter connected to ToolRegistry ({len(_tool_registry.tools)} tools)")
    elif db_session and not _tool_registry.db:
        _tool_registry = registry_get_tool_registry(db_session=db_session)
    return _tool_registry


def get_chatbot_tools(agent_id: Optional[int] = None, db_session=None) -> List[Dict[str, Any]]:
    """
    Get tools from modules.tools.ToolRegistry in OpenAI function format.
    SINGLE SOURCE OF TRUTH - no duplicate definitions.
    
    Note: We expose `smart_query_database` as the primary database tool,
    filtering out the basic `query_database` for better user experience.
    """
    # Ensure registry has DB so MCP tools are available
    registry = _get_registry(db_session=db_session)
    session_used = db_session
    if session_used is None:
        # Open a short-lived session for permission checks
        try:
            session_used = SessionLocal()
        except Exception:
            session_used = None
    
    try:
        # Get RESEARCH, DATABASE, and MCP tools for chatbot
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        database_tools = registry.get_tools_by_category(ToolCategory.DATABASE_TOOLS)
        mcp_tools = registry.get_tools_by_category(ToolCategory.MCP_TOOLS)
        all_tools = research_tools + database_tools + mcp_tools
        
        # Filter: Use smart_query_database as THE database tool
        # Remove basic query_database to avoid LLM choosing the dumber option
        filtered_tools = [
            tool for tool in all_tools 
            if tool.name != 'query_database'  # Prefer smart version
        ]

        # Agent-aware filtering (especially for MCP tools)
        if agent_id is not None and session_used is not None:
            filtered_tools = [
                tool for tool in filtered_tools
                if registry.validate_tool_access(agent_id=agent_id, tool_name=tool.name, db=session_used)[0]
            ]
        
        # Convert to OpenAI function format
        openai_tools = []
        for tool in filtered_tools:
            schema = tool.to_openai_format()
            openai_tools.append({
                "type": "function",
                "function": schema
            })
        
        logger.info(
            f"✅ Loaded {len(openai_tools)} tools for chatbot "
            f"(agent_id={agent_id}, smart_query_database enabled, includes MCP={len(mcp_tools) > 0})"
        )
        return openai_tools
    except Exception as e:
        logger.error(f"Error loading tools from registry: {e}")
        return []
    finally:
        if db_session is None and session_used is not None:
            # Only close if we opened it
            session_used.close()


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: int = 1
) -> Dict[str, Any]:
    """
    Execute a tool via modules.tools.UnifiedToolExecutor.
    SINGLE ENTRY POINT for all tool execution in chat.
    """
    from core.database.database import SessionLocal
    
    # Use SessionLocal directly - simpler and works
    db_session = SessionLocal()
    try:
        executor = UnifiedToolExecutor(db_session)
        result = await executor.execute_tool(tool_name, tool_args, agent_id)
        db_session.commit()
        return result
    except Exception as e:
        db_session.rollback()
        raise
    finally:
        db_session.close()


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
            docs = standardized['results']
            if not docs:
                return None
            
            # Group chunks by document
            docs_by_source = {}
            for doc in docs:
                source = doc.get('filename', doc.get('source', 'Unknown'))
                similarity = doc.get('similarity', doc.get('score', 0.0))
                content = doc.get('excerpt', doc.get('content', ''))
                
                # NEW: Extract file_path from metadata
                metadata = doc.get('metadata', {})
                file_path = metadata.get('file_path', f"/var/automatos/documents/{source}")
                
                if source not in docs_by_source:
                    docs_by_source[source] = {
                        'source': source,
                        'file_path': file_path,  # NEW: Store file path
                        'chunks': [],
                        'max_similarity': 0.0
                    }
                
                docs_by_source[source]['chunks'].append(content)
                docs_by_source[source]['max_similarity'] = max(
                    docs_by_source[source]['max_similarity'],
                    similarity
                )
            
            # Sort by relevance
            sorted_docs = sorted(
                docs_by_source.values(),
                key=lambda d: d['max_similarity'],
                reverse=True
            )[:5]  # Top 5 documents
            
            # Build user-friendly context
            doc_parts = [f"I found relevant information in {len(sorted_docs)} document(s):", ""]
            
            for i, doc_info in enumerate(sorted_docs, 1):
                source = doc_info['source']
                file_path = doc_info['file_path']
                relevance = int(doc_info['max_similarity'] * 100)
                chunk_count = len(doc_info['chunks'])
                
                # Extract title from filename
                title = source.replace('.md', '').replace('.pdf', '')
                title = title.replace('-', ' ').replace('_', ' ').title()
                
                # Remove numbering prefix (e.g., "02-" or "30-")
                import re
                title = re.sub(r'^\d+\s*', '', title)
                
                doc_parts.append(f"📄 **{title}** ({source})")
                doc_parts.append(f"   • {chunk_count} relevant section(s) found")
                doc_parts.append(f"   • Relevance: {relevance}%")
                
                # NEW: Add download link
                doc_parts.append(f"   • [Download Full Document](/api/documents/download?path={file_path})")
                doc_parts.append("")
            
            # Add full content from all chunks for LLM to use
            doc_parts.append("\n📚 **FULL CONTENT FROM DOCUMENTS** (use this to answer the question):\n")
            for i, doc_info in enumerate(sorted_docs, 1):
                source = doc_info['source']
                title = source.replace('.md', '').replace('.pdf', '')
                title = title.replace('-', ' ').replace('_', ' ').title()
                import re
                title = re.sub(r'^\d+\s*', '', title)
                
                doc_parts.append(f"\n--- {title} ---")
                # Include ALL relevant chunks, not just preview
                for chunk in doc_info['chunks']:
                    doc_parts.append(chunk)
                    doc_parts.append("")
            
            doc_context = "\n".join(doc_parts)
            
            return {
                "role": "system",
                "content": f"{doc_context}\n\n**CRITICAL INSTRUCTIONS**:\n1. READ ALL the full content provided above\n2. SYNTHESIZE information from multiple sources\n3. Write a COMPREHENSIVE answer using the actual content\n4. Do NOT just list documents or say 'search didn't yield results'\n5. The content IS THERE - use it to write detailed responses\n6. Cite document names when referencing specific information\n7. If asked for code examples, include the actual code shown above\n8. If asked for statistics, use the actual numbers provided"
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
        
        elif tool_name in ['query_database', 'smart_query_database']:
            # Handle clarification needed
            if raw.get('status') == 'needs_clarification':
                clarifications = raw.get('clarifications', [])
                return {
                    "role": "system",
                    "content": f"🤔 CLARIFICATION NEEDED:\n\nThe query is ambiguous. Ask the user:\n" + 
                              "\n".join(f"• {q}" for q in clarifications) +
                              "\n\nOnce they answer, you can query again with more specifics."
                }
            
            # Include ALL data - already limited at query level
            all_data = raw.get('data', [])
            data_str = json.dumps(all_data, default=str, indent=2)[:3000]
            
            db_context = f"Database query result: {raw.get('row_count', 0)} rows returned\n"
            db_context += f"SQL: {raw.get('sql', 'N/A')[:300]}\n"
            
            # Include rephrased query if different
            if raw.get('rephrased_query'):
                db_context += f"Interpreted as: {raw.get('rephrased_query')}\n"
            
            db_context += f"COMPLETE DATA:\n{data_str}"
            
            # Include explanation
            if raw.get('explanation'):
                db_context += f"\n\n📝 EXPLANATION: {raw.get('explanation')}"
            
            # Include PandasAI insight if available
            pandas_ai = raw.get('pandas_ai', {})
            if pandas_ai:
                db_context += f"\n\n📊 AI ANALYSIS: {pandas_ai.get('summary', '')}"
            
            # Include visualization suggestion
            if raw.get('visualization'):
                viz = raw.get('visualization', {})
                db_context += f"\n\n📈 VISUALIZATION: Recommended {viz.get('type', 'chart')} chart"
            
            # Include follow-up suggestions
            follow_ups = raw.get('follow_up_questions', [])
            if follow_ups:
                db_context += f"\n\n💡 FOLLOW-UP IDEAS:\n" + "\n".join(f"• {q}" for q in follow_ups[:3])
            
            return {
                "role": "system",
                "content": f"🗄️ DATABASE QUERY RESULTS:\n\n{db_context}\n\nYou MUST present ALL this data to the user, not just a summary. Show complete results. If there's a chart in the artifacts panel, tell the user to check it."
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
def get_chat_tools(agent_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get chatbot tools. Do NOT cache globally because availability can
    depend on agent permissions (especially MCP).
    """
    with _session_scope() as session:
        return get_chatbot_tools(agent_id=agent_id, db_session=session)

