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
import time
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from contextlib import contextmanager

# Use modules.tools directly - NO duplicate tool definitions
from modules.tools import ToolCategory, UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from modules.tools.registry import get_tool_registry as registry_get_tool_registry
from core.database.database import SessionLocal

logger = logging.getLogger(__name__)

def _new_trace_id() -> str:
    return uuid4().hex[:12]

def _summarize_args(args: Any) -> str:
    if isinstance(args, dict):
        keys = list(args.keys())
        return f"dict keys={keys[:12]}{'...' if len(keys) > 12 else ''}"
    if isinstance(args, list):
        return f"list len={len(args)}"
    return f"{type(args).__name__}"

def _is_fatal_dependency_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "composio openai sdk not available" in lowered or "composio-openai" in lowered

@contextmanager
def _session_scope():
    """Lightweight context manager to ensure sessions are always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_chatbot_tools(agent_id: Optional[int] = None, db_session=None) -> List[Dict[str, Any]]:
    """
    Get tools from modules.tools.ToolRegistry in OpenAI function format.
    SINGLE SOURCE OF TRUTH - no duplicate definitions.
    """
    session_used = db_session or SessionLocal()
    trace_id = _new_trace_id()
    start_time = time.time()
    try:
        registry = registry_get_tool_registry(db_session=session_used)
        all_candidates = registry.get_all_tools(active_only=True)
        filtered_tools = all_candidates
        denied: List[Dict[str, Any]] = []

        if agent_id is not None:
            filtered_tools = []
            for tool in all_candidates:
                allowed, reason = registry.validate_tool_access(
                    agent_id=agent_id,
                    tool_name=tool.name,
                    db=session_used
                )
                if allowed:
                    filtered_tools.append(tool)
                else:
                    denied.append({"tool": tool.name, "reason": reason})
        
        # Convert to OpenAI function format
        openai_tools = []
        for tool in filtered_tools:
            schema = tool.to_openai_format()
            openai_tools.append({
                "type": "function",
                "function": schema
            })
        
        has_mcp = any(t.category == ToolCategory.MCP_TOOLS for t in filtered_tools)
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[tool-trace {trace_id}] Loaded {len(openai_tools)} tools "
            f"(agent_id={agent_id}, includes MCP={has_mcp}, denied={len(denied)}, "
            f"candidates={len(all_candidates)}, {elapsed_ms}ms)"
        )
        if denied:
            sample = denied[:10]
            logger.info(f"[tool-trace {trace_id}] Denied sample: {sample}")
        return openai_tools
    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Error loading tools from registry: {e}")
        return []
    finally:
        if db_session is None:
            session_used.close()


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a tool via modules.tools.UnifiedToolExecutor.
    SINGLE ENTRY POINT for all tool execution in chat.
    """
    from core.database.database import SessionLocal
    
    trace = trace_id or _new_trace_id()
    # Use SessionLocal directly - simpler and works
    db_session = SessionLocal()
    try:
        if workspace_id is None and agent_id:
            try:
                from core.models import Agent as AgentModel
                agent_row = db_session.query(AgentModel).filter(AgentModel.id == agent_id).first()
                if agent_row and agent_row.workspace_id:
                    workspace_id = agent_row.workspace_id
                    logger.info(f"[tool-trace {trace}] Resolved workspace_id from agent {agent_id}")
                else:
                    logger.warning(f"[tool-trace {trace}] Agent workspace_id missing for agent={agent_id}")
            except Exception as exc:
                logger.warning(f"[tool-trace {trace}] Failed to resolve workspace_id: {exc}")
        executor = UnifiedToolExecutor(db_session)
        logger.info(
            f"[tool-trace {trace}] execute_tool start tool={tool_name} "
            f"agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )
        result = await executor.execute_tool(
            tool_name,
            tool_args,
            agent_id,
            workspace_id=workspace_id,
            trace_id=trace
        )
        db_session.commit()
        logger.info(
            f"[tool-trace {trace}] execute_tool done tool={tool_name} "
            f"success={bool(result.get('success'))}"
        )
        return result
    except Exception as e:
        db_session.rollback()
        logger.error(f"[tool-trace {trace}] execute_tool error tool={tool_name}: {e}")
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
        agent_id: int = 1,
        workspace_id: Optional[UUID] = None
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
        trace_id = _new_trace_id()
        logger.info(
            f"[tool-trace {trace_id}] ToolRouter execute_and_format "
            f"tool={tool_name} agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )
        
        try:
            result = await execute_tool(
                tool_name,
                tool_args,
                agent_id,
                workspace_id=workspace_id,
                trace_id=trace_id
            )
            
            # Check success
            success = (
                result.get('success') or
                result.get('status') == 'success' or
                bool(result.get('results'))
            )
            
            if success:
                frontend_data = self.formatter.format_for_frontend(result, tool_name)
                llm_context = self.formatter.format_for_llm(result, tool_name)
                
                logger.info(f"[tool-trace {trace_id}] {tool_name} succeeded")
                return {
                    'success': True,
                    'frontend_data': frontend_data,
                    'llm_context': llm_context,
                    'raw_result': result,
                    'fatal_error': False,
                    'error_type': None
                }
            else:
                error = result.get('error', 'Unknown error')
                error_type = result.get('error_type')
                fatal_error = bool(result.get('fatal')) or _is_fatal_dependency_error(error)
                llm_error = error
                if fatal_error:
                    llm_error = "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
                logger.warning(f"[tool-trace {trace_id}] {tool_name} failed: {error}")
                return {
                    'success': False,
                    'frontend_data': {},
                    'llm_context': f"Tool {tool_name} failed: {llm_error}",
                    'raw_result': result,
                    'fatal_error': fatal_error,
                    'error_type': error_type
                }
                
        except Exception as e:
            error_msg = str(e)
            fatal_error = _is_fatal_dependency_error(error_msg)
            logger.error(f"[tool-trace {trace_id}] {tool_name} exception: {error_msg}")
            llm_error = error_msg
            if fatal_error:
                llm_error = "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
            return {
                'success': False,
                'frontend_data': {},
                'llm_context': f"Tool {tool_name} error: {llm_error}",
                'raw_result': {'success': False, 'error': error_msg},
                'fatal_error': fatal_error,
                'error_type': 'dependency_missing' if fatal_error else None
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

