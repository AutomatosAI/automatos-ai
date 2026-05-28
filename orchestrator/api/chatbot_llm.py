"""
Chatbot LLM API
===============

FastAPI endpoints for chatbot functionality with tool calling.
PRD-25: Migrated to use LLM service with per-request model selection.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import httpx

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.llm import create_llm_manager
from modules.tools.services.pandas_ai_service import get_pandasai_service
from config import config

# Import tools from modules.tools (single source of truth)
from modules.tools.tool_router import get_tools_for_agent, execute_tool as consumer_execute_tool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])

# Lazy initialization of LLM Manager
_llm_manager = None


def get_llm_manager():
    """
    Lazy load LLM manager with orchestrator settings.
    
    PRD-25: Uses new LLM service with per-service configuration.
    Chatbot defaults to orchestrator LLM settings (can be overridden per-request).
    """
    global _llm_manager
    if _llm_manager is None:
        try:
            # Use orchestrator settings (default for chatbot)
            _llm_manager = create_llm_manager(service_name="orchestrator")
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return _llm_manager

# Legacy function for backward compatibility (if needed)
def get_anthropic_client():
    """
    Legacy function - now uses LLM service instead.
    Kept for backward compatibility only.
    """
    from core.llm import create_llm_manager
    llm = create_llm_manager(service_name="orchestrator", provider="anthropic")
    # Return a mock Anthropic client interface (not used, but maintains compatibility)
    # Actual API calls should use llm.generate_response() instead
    return None

# Models
class ChatContext(BaseModel):
    current_page: str = "dashboard"
    selected_items: List[Dict[str, Any]] = []
    user_role: str = "user"
    recent_actions: List[Dict[str, Any]] = []

class ChatQuery(BaseModel):
    query: str
    context: Optional[ChatContext] = None
    session_id: Optional[str] = None
    provider: Optional[str] = None  # Override provider (openai, anthropic, etc.)
    model: Optional[str] = None  # Override model (gpt-4, claude-3-5-sonnet-20241022, etc.)


# Tool implementations
async def search_code_tool(query: str) -> Dict[str, Any]:
    """Search codebase using CodeGraph semantic search"""
    try:
        logger.info(f"[CodeGraph] Searching for: {query}")
        
        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{base_url}/api/code-graph/search/semantic",
                params={
                    "project": "Automatos-ai",
                    "q": query,
                    "limit": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[CodeGraph] Found {data.get('count', 0)} results")
                
                # Format results
                results = []
                for result in data.get('results', [])[:5]:
                    results.append({
                        "symbol_name": result.get('name'),
                        "symbol_type": result.get('symbol_type'),
                        "file_path": result.get('file_path'),
                        "line_number": result.get('line_number'),
                        "code": result.get('code_snippet') or result.get('signature'),
                        "docstring": result.get('docstring')
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"[CodeGraph] Search error: {e}")
        return {"success": False, "error": str(e)}


async def search_documents_tool(query: str) -> Dict[str, Any]:
    """Search documents using RAG semantic search"""
    try:
        logger.info(f"[Documents] Searching for: {query}")
        
        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/documents/search",
                params={
                    "query": query,
                    "limit": 8,
                    "min_similarity": config.RAG_MIN_SIMILARITY
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[Documents] Found {data.get('total_results', 0)} results")
                
                # Format results
                results = []
                for doc in data.get('results', [])[:8]:
                    results.append({
                        "id": doc.get('document_id'),
                        "filename": (doc.get('source', {}) or {}).get('filename') if isinstance(doc.get('source'), dict) else doc.get('filename'),
                        "excerpt": doc.get('excerpt', ''),
                        "content": doc.get('preview', ''),
                        "preview": doc.get('preview', ''),
                        "chunk_count": doc.get('chunk_count'),
                        "chunk_index": doc.get('chunk_index'),
                        "preview_chunk_start": doc.get('preview_chunk_start'),
                        "preview_chunk_end": doc.get('preview_chunk_end'),
                        "similarity": doc.get('similarity', 0.0),
                        "has_full_content": doc.get('full_content_available', False)
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"[Documents] Search error: {e}")
        return {"success": False, "error": str(e)}


async def query_database_tool(
    query: str,
    database_name: Optional[str] = None,
    analysis_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Query database using natural language via knowledge source endpoints."""
    try:
        logger.info(f"[Database] Querying: {query}, DB: {database_name}")

        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Resolve database source
            selected_source: Optional[Dict[str, Any]] = None
            sources: List[Dict[str, Any]] = []

            try:
                sources_response = await client.get(
                    f"{base_url}/api/knowledge/sources/database/",
                    params={"active_only": True}
                )
                if sources_response.status_code == 200:
                    sources = sources_response.json() or []
            except Exception as source_err:
                logger.warning(f"[Database] Failed to list database sources: {source_err}")

            if database_name and sources:
                lowered = database_name.lower()
                selected_source = next(
                    (src for src in sources if str(src.get("name", "")).lower() == lowered),
                    None
                )

            if not selected_source and sources:
                selected_source = sources[0]

            if not selected_source:
                return {"success": False, "error": "No active database sources available"}

            source_id = selected_source.get("id")
            response = await client.post(
                f"{base_url}/api/knowledge/sources/database/{source_id}/query",
                json={
                    "query": query,
                    "source_id": source_id,
                    "use_cache": True,
                    "include_explanation": True
                }
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"[Database] Query successful (source={selected_source.get('name')})")

                return {
                    "success": True,
                    "database": data.get('database') or selected_source.get('name', ''),
                    "sql": data.get('sql', ''),
                    "row_count": data.get('row_count', 0),
                    "data": data.get('data', []),
                    "columns": data.get('columns', []),
                    "execution_time_ms": data.get('execution_time_ms', 0)
                }
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}

    except Exception as e:
        logger.error(f"[Database] Query error: {e}")
        return {"success": False, "error": str(e)}


# Tool definitions for LLM — single source of truth
TOOLS = get_tools_for_agent()


@router.get("/models")
async def list_available_models(
    provider: Optional[str] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    List available LLM models for chatbot selection.
    
    Returns models organized by provider for easy selection in UI.
    """
    try:
        from core.llm.model_registry import ModelRegistry
        
        registry = ModelRegistry(db)
        
        if provider:
            models = registry.get_all_models(provider=provider, status='active')
        else:
            models = registry.get_all_models(status='active')
        
        # Group by provider
        by_provider = {}
        for model in models:
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append({
                "model_id": model.model_id,
                "display_name": model.display_name,
                "description": model.description,
                "context_window": model.context_window,
                "max_output_tokens": model.max_output_tokens,
                "input_cost_per_1k_tokens": model.input_cost_per_1k_tokens,
                "output_cost_per_1k_tokens": model.output_cost_per_1k_tokens,
                "supports_functions": model.supports_functions,
                "recommended_for": model.recommended_for
            })
        
        # Get default model from settings
        from core.llm.manager import get_provider_and_model_from_settings
        default_provider, default_model = get_provider_and_model_from_settings("orchestrator")
        
        return {
            "providers": by_provider,
            "default_provider": default_provider,
            "default_model": default_model
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        # Fallback when ModelRegistry is unreachable — return platform defaults
        # rather than a stale hardcoded list.
        from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
        return {
            "providers": {},
            "default_provider": DEFAULT_LLM_PROVIDER,
            "default_model": DEFAULT_LLM_MODEL,
        }


@router.post("/query")
async def process_chatbot_query(
    chat_query: ChatQuery,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """
    Process chatbot query using LLM service with function calling.
    LLM decides whether to search code or documents.
    Uses orchestrator LLM settings by default, but can be overridden per-request.
    
    Query parameters:
    - provider: Optional override (openai, anthropic, google, azure, huggingface)
    - model: Optional override (model ID like gpt-4, claude-3-5-sonnet-20241022)
    """
    try:
        request_id = str(uuid.uuid4())[:12]
        logger.info(f"[req={request_id}] Chatbot query: '{chat_query.query}' (provider={chat_query.provider}, model={chat_query.model})")
        
        # Build initial prompt
        user_message = f"""User query: {chat_query.query}

Context: {chat_query.context.dict() if chat_query.context else 'None'}

You are the Automatos AI Assistant. Analyze the user's query and decide:
1. If they're asking about CODE (functions, classes, implementation), use the search_code tool
2. If they're asking about DOCUMENTS (guides, architecture, docs), use the search_documents tool
3. If they're asking about DATA (database queries, statistics), use the query_database tool
4. You can use MULTIPLE tools if the query would benefit from multiple sources

Then provide a helpful, technical response based on the search results."""

        # Call LLM with tool use
        logger.info(f"[req={request_id}] Calling LLM service with tools...")
        
        messages = [{"role": "user", "content": user_message}]
        code_results = []
        document_results = []
        database_results = []
        tool_calls_made = []
        
        # Agentic loop - let LLM call tools iteratively
        for iteration in range(3):  # Max 3 tool call iterations
            # Use LLM service - allow per-request provider/model override
            from core.llm import create_llm_manager
            
            # Use override if provided, otherwise use default orchestrator settings
            if chat_query.provider or chat_query.model:
                llm_manager = create_llm_manager(
                    service_name="orchestrator",
                    provider=chat_query.provider,
                    model=chat_query.model
                )
                logger.info(f"[req={request_id}] Using LLM override: provider={chat_query.provider}, model={chat_query.model}")
            else:
                llm_manager = get_llm_manager()
            
            # Convert messages to LLM service format
            llm_messages = []
            for msg in messages:
                # Handle both string content and dict content (tool results)
                if isinstance(msg.get("content"), str):
                    llm_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                elif isinstance(msg.get("content"), list):
                    # Handle Anthropic-style tool results format
                    content_str = ""
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                content_str += f"Tool Result: {block.get('content', '')}\n"
                        else:
                            content_str += str(block) + "\n"
                    llm_messages.append({
                        "role": msg["role"],
                        "content": content_str.strip() or json.dumps(msg.get("content"))
                    })
                else:
                    llm_messages.append({
                        "role": msg["role"],
                        "content": json.dumps(msg.get("content", ""))
                    })
            
            # Generate response using LLM service
            response = await llm_manager.generate_response(
                messages=llm_messages,
                tools=TOOLS
            )
            
            logger.info(f"[req={request_id}] LLM response finish_reason: {response.finish_reason}")
            
            # Check if LLM wants to use tools
            if response.finish_reason == "tool_use" or (response.tool_calls and len(response.tool_calls) > 0):
                # Collect all tool results first
                tool_results = []
                
                # Handle tool calls from LLMResponse
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        tool_input_str = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Parse tool arguments (handle both string and dict)
                        try:
                            if isinstance(tool_input_str, str):
                                tool_input = json.loads(tool_input_str)
                            else:
                                tool_input = tool_input_str
                        except Exception:
                            tool_input = {"query": str(tool_input_str)}
                        
                        tool_id = tool_call.get("id", "")
                        
                        logger.info(f"[req={request_id}] LLM called tool: {tool_name} with query: {tool_input.get('query')}")
                        tool_calls_made.append(tool_name)
                        
                        # Execute the tool
                        if tool_name == "search_code":
                            tool_result = await search_code_tool(tool_input.get("query", ""))
                            if tool_result.get("success"):
                                code_results.extend(tool_result.get("results", []))
                        elif tool_name == "search_documents":
                            tool_result = await search_documents_tool(tool_input.get("query", ""))
                            if tool_result.get("success"):
                                document_results.extend(tool_result.get("results", []))
                        elif tool_name == "query_database":
                            # Define analysis_prompt with explicit fallback to prevent NameError
                            analysis_prompt = chat_query.query or tool_input.get("query", "")
                            tool_result = await query_database_tool(
                                tool_input.get("query", ""),
                                tool_input.get("database_name"),
                                analysis_prompt=analysis_prompt,
                            )
                            if tool_result.get("success"):
                                pandasai = get_pandasai_service()
                                if pandasai:
                                    insight = pandasai.generate_insight(
                                        analysis_prompt,
                                        tool_result.get("data", []),
                                        tool_result.get("columns", []),
                                    )
                                    if insight:
                                        tool_result["pandas_ai"] = insight
                                database_results.append(tool_result)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        # Collect tool result (format depends on provider)
                        # OpenAI format: {type: "function", function: {name, arguments}}
                        # Anthropic format: {type: "tool_result", tool_use_id, content}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
                        })
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls
                })
                
                # Add user message with tool results
                # For Anthropic: content must be an array of content blocks
                # For OpenAI: content can be JSON string (will be converted by provider)
                messages.append({
                    "role": "user",
                    "content": tool_results if tool_results else ""  # Send as array directly for Anthropic
                })
                
                # Continue the loop to let LLM process tool results
                continue
                
            else:
                # LLM provided final response
                final_content = response.content or ""
                
                # Get model info from LLM manager
                llm_info = llm_manager.get_provider_info()
                model_name = llm_info.get("model", "unknown")
                
                logger.info(f"[req={request_id}] LLM finished. Used tools: {tool_calls_made}")
                logger.info(f"[req={request_id}] Code results: {len(code_results)}, Doc results: {len(document_results)}, DB results: {len(database_results)}")
                
                # Determine primary source
                primary_source = "llm"
                if database_results:
                    primary_source = "database"
                elif code_results:
                    primary_source = "codegraph"
                elif document_results:
                    primary_source = "rag"
                
                # Build response
                return {
                    "message": {
                        "id": f"bot-{datetime.now().timestamp()}",
                        "content": final_content,
                        "timestamp": datetime.now().isoformat(),
                        "code_snippets": code_results,
                        "documents": document_results,
                        "database_results": database_results,
                        "metadata": {
                            "tools_used": tool_calls_made,
                            "source": primary_source,
                            "llm_provider": llm_info.get("provider", "unknown"),
                            "llm_model": model_name,
                            "code_count": len(code_results),
                            "document_count": len(document_results),
                            "database_count": len(database_results)
                        }
                    },
                    "session_id": chat_query.session_id or str(uuid.uuid4())
                }
        
        # If we exit the loop without a final response, return what we have
        return {
            "message": {
                "id": f"bot-{datetime.now().timestamp()}",
                "content": "I gathered information but couldn't formulate a complete response.",
                "timestamp": datetime.now().isoformat(),
                "code_snippets": code_results,
                "documents": document_results,
                "database_results": database_results,
                "metadata": {
                    "tools_used": tool_calls_made,
                    "error": "Max tool iterations reached"
                }
            },
            "session_id": chat_query.session_id or str(uuid.uuid4())
        }
        
    except Exception as e:
        logger.error(f"Chatbot query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
