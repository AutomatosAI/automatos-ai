"""
Tool Router - Unified Tool Execution for Chat
==============================================

Routes tool execution through modules.tools (single source of truth).

Handles:
- Getting tools from modules.tools.ToolRegistry
- Executing tools via modules.tools.UnifiedToolExecutor
- Formatting results for frontend and LLM
- Building tool context injection messages
- Capability-based action filtering (PRD-37)
- Execution-time validation (defense in depth)
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
from contextlib import contextmanager
import re

# Use modules.tools directly - NO duplicate tool definitions
from modules.tools import ToolCategory, UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from modules.tools.registry import get_tool_registry as registry_get_tool_registry
from core.database.database import SessionLocal

# Capability-based filtering imports (PRD-37)
try:
    from modules.tools.services.action_capability_filter import (
        ActionCapabilityFilter,
        get_action_capability_filter,
        ActionFilterResult,
    )
    CAPABILITY_FILTER_AVAILABLE = True
except ImportError:
    CAPABILITY_FILTER_AVAILABLE = False

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


def get_chatbot_tools(agent_id: Optional[int] = None, db_session=None, workspace_id: Optional[Any] = None) -> List[Dict[str, Any]]:
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

        # Resolve workspace_id from agent if not provided (needed for Composio gating)
        if agent_id is not None and workspace_id is None:
            try:
                from core.models import Agent as AgentModel
                agent_row = session_used.query(AgentModel).filter(AgentModel.id == agent_id).first()
                if agent_row and getattr(agent_row, "workspace_id", None):
                    workspace_id = agent_row.workspace_id
                    logger.info(f"[tool-trace {trace_id}] Resolved workspace_id from agent {agent_id} for tool list gating")
            except Exception as exc:
                logger.warning(f"[tool-trace {trace_id}] Failed to resolve workspace_id for agent {agent_id}: {exc}")

        if agent_id is not None:
            filtered_tools = []
            for tool in all_candidates:
                allowed, reason = registry.validate_tool_access(
                    agent_id=agent_id,
                    tool_name=tool.name,
                    db=session_used,
                    workspace_id=workspace_id
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
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[tool-trace {trace_id}] Loaded {len(openai_tools)} tools "
            f"(agent_id={agent_id}, denied={len(denied)}, "
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
        # ------------------------------------------------------------------
        # Compatibility aliases (LLM sometimes emits "helper" tool names).
        # We do NOT register these as real tools; we transparently route them
        # to the correct ToolRegistry tool to avoid "Unknown tool" loops.
        #
        # IMPORTANT: This must stay generic for 800+ apps. We map patterns like:
        # - send_slack_message
        # - send_<app>_message
        # to composio_execute with a mapped action from composio_actions_cache.
        # ------------------------------------------------------------------
        normalized_name = (tool_name or "").strip()
        alias_app: Optional[str] = None

        if normalized_name in {"send_slack_message", "slack_send_message"}:
            alias_app = "SLACK"
        else:
            m = re.match(r"^send_(?P<app>[a-z0-9]+)_message$", normalized_name, flags=re.I)
            if m:
                alias_app = str(m.group("app") or "").upper().strip()

        if alias_app:
            channel = None
            text = None
            if isinstance(tool_args, dict):
                channel = tool_args.get("channel") or tool_args.get("channel_id") or tool_args.get("conversation")
                text = tool_args.get("message") or tool_args.get("text") or tool_args.get("content")

            # Pick a safe mapped action for this app that looks like "send message".
            # If not found, still route to composio_execute and let the executor return a mapped-action error.
            action_name: Optional[str] = None
            try:
                from core.models.composio_cache import ComposioActionCache

                with _session_scope() as s:
                    rows = (
                        s.query(ComposioActionCache.action_name)
                        .filter(ComposioActionCache.app_name == alias_app)
                        .filter(
                            (ComposioActionCache.action_name.ilike("%send%"))
                            | (ComposioActionCache.action_name.ilike("%message%"))
                            | (ComposioActionCache.description.ilike("%send%"))
                            | (ComposioActionCache.description.ilike("%message%"))
                        )
                        .limit(20)
                        .all()
                    )
                candidates = [str(r[0]) for r in rows if r and r[0]]
                dangerous = ("archive", "delete", "remove", "revoke", "clear", "close", "disable", "ban", "kick")
                for c in candidates:
                    cl = c.lower()
                    if any(tok in cl for tok in dangerous):
                        continue
                    action_name = c
                    break
            except Exception:
                action_name = None

            mapped_args: Dict[str, Any] = {
                "app_name": alias_app,
                "action": action_name or "",
                "params": {"channel": channel, "text": text},
            }
            mapped_args["params"] = {k: v for k, v in mapped_args["params"].items() if v is not None}
            tool_name = "composio_execute"
            tool_args = mapped_args

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
def get_chat_tools(agent_id: Optional[int] = None, workspace_id: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Get chatbot tools. Do NOT cache globally because availability can
    depend on agent permissions.
    """
    with _session_scope() as session:
        return get_chatbot_tools(agent_id=agent_id, db_session=session, workspace_id=workspace_id)


# =============================================================================
# CAPABILITY-BASED FILTERING (PRD-37)
# =============================================================================

async def get_filtered_composio_actions(
    intent: str,
    enabled_apps: List[str],
    db_session=None,
    include_destructive: bool = False,
    max_actions: int = 15
) -> Dict[str, Any]:
    """
    Get Composio actions filtered by capability for a given intent.

    This is the core of the capability-based filtering system:
    1. Extracts capabilities from intent text
    2. Filters actions from local metadata (no API calls)
    3. Excludes destructive actions unless explicitly allowed
    4. Returns ranked, relevant actions

    Args:
        intent: User's intent text (e.g., "send a message to #general")
        enabled_apps: List of Composio app IDs user has enabled
        db_session: Optional database session
        include_destructive: Whether to include destructive actions
        max_actions: Maximum number of actions to return

    Returns:
        Dict with filtered actions and metadata
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        logger.warning("Capability filter not available, returning empty result")
        return {
            "success": False,
            "error": "Capability filter not available",
            "actions": [],
            "capabilities": []
        }

    trace_id = _new_trace_id()
    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        result = await filter_service.get_actions_for_intent(
            intent=intent,
            enabled_apps=enabled_apps,
            include_destructive=include_destructive,
            max_actions=max_actions
        )

        logger.info(
            f"[tool-trace {trace_id}] Capability filter: "
            f"intent='{intent[:50]}...' caps={result.extracted_capabilities} "
            f"matched={result.filtered_count}/{result.total_available} "
            f"fallback={result.fallback_used}"
        )

        # Convert to dict for API response
        actions_data = [
            {
                "action_id": a.action_id,
                "app_id": a.app_id,
                "capabilities": a.capabilities,
                "relevance_score": a.relevance_score,
                "description": a.description,
                "destructive": a.destructive,
                "requires_confirmation": a.requires_confirmation
            }
            for a in result.actions
        ]

        return {
            "success": True,
            "intent": result.intent,
            "extracted_capabilities": result.extracted_capabilities,
            "total_available": result.total_available,
            "filtered_count": result.filtered_count,
            "fallback_used": result.fallback_used,
            "actions": actions_data
        }

    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Capability filter error: {e}")
        return {
            "success": False,
            "error": str(e),
            "actions": [],
            "capabilities": []
        }
    finally:
        if db_session is None:
            session_used.close()


def validate_action_for_intent(
    action_id: str,
    intent: str,
    db_session=None,
    allow_destructive: bool = False
) -> Tuple[bool, str]:
    """
    Validate that an action is eligible for execution given an intent.

    This is the EXECUTION-TIME validation gate (per GPT-5.2 recommendation).
    Called before executing any Composio action to ensure the action
    matches the original intent's capabilities.

    Args:
        action_id: The Composio action ID to validate
        intent: The original user intent
        db_session: Optional database session
        allow_destructive: Whether destructive actions are allowed

    Returns:
        (eligible, reason) tuple
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        # Fail open if capability filter not available
        logger.warning("Capability filter not available, allowing action")
        return True, "Capability filter not available (fail open)"

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        eligible, reason = filter_service.check_action_eligibility(
            action_id=action_id,
            intent=intent,
            allow_destructive=allow_destructive
        )

        if not eligible:
            logger.warning(
                f"Action validation failed: action={action_id} "
                f"intent='{intent[:30]}...' reason={reason}"
            )

        return eligible, reason

    except Exception as e:
        logger.error(f"Action validation error: {e}")
        # Fail open on errors to avoid blocking legitimate actions
        return True, f"Validation error (fail open): {e}"
    finally:
        if db_session is None:
            session_used.close()


async def execute_tool_with_validation(
    tool_name: str,
    tool_args: Dict[str, Any],
    original_intent: str,
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    allow_destructive: bool = False
) -> Dict[str, Any]:
    """
    Execute a tool with intent-based validation.

    This wraps execute_tool() with an additional validation layer
    that checks if the action matches the original intent's capabilities.

    Args:
        tool_name: Tool to execute
        tool_args: Tool arguments
        original_intent: The original user intent (for validation)
        agent_id: Agent ID
        workspace_id: Workspace ID
        allow_destructive: Whether to allow destructive actions

    Returns:
        Tool execution result (with validation info)
    """
    trace_id = _new_trace_id()

    # Only validate Composio actions
    is_composio = tool_name.startswith("composio_") or tool_name == "composio_execute"

    if is_composio and original_intent:
        # Extract action ID from tool name or args
        action_id = None
        if tool_name.startswith("composio_"):
            action_id = tool_name.replace("composio_", "")
        elif isinstance(tool_args, dict):
            action_id = tool_args.get("action") or tool_args.get("action_name")

        if action_id:
            eligible, reason = validate_action_for_intent(
                action_id=action_id,
                intent=original_intent,
                allow_destructive=allow_destructive
            )

            if not eligible:
                logger.warning(
                    f"[tool-trace {trace_id}] Execution blocked by validation: "
                    f"tool={tool_name} action={action_id} reason={reason}"
                )
                return {
                    "success": False,
                    "error": f"Action not allowed for this intent: {reason}",
                    "error_type": "validation_blocked",
                    "blocked_by": "capability_validation",
                    "action_id": action_id,
                    "intent": original_intent[:100]
                }

    # Proceed with normal execution
    return await execute_tool(
        tool_name=tool_name,
        tool_args=tool_args,
        agent_id=agent_id,
        workspace_id=workspace_id,
        trace_id=trace_id
    )


def get_capability_filter_stats(db_session=None) -> Dict[str, Any]:
    """
    Get statistics about the capability filter system.

    Returns info about classified actions, coverage, etc.
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        return {"available": False, "error": "Capability filter not available"}

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)
        stats = filter_service.get_statistics()
        stats["available"] = True
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}
    finally:
        if db_session is None:
            session_used.close()

