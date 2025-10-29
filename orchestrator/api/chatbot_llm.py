"""
Chatbot API with LLM Function Calling
======================================

Intelligent chatbot using Claude function calling to decide:
- When to search code (CodeGraph)
- When to search documents (RAG)
- How to synthesize results

Uses Anthropic's native tool use, not custom text parsing.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
import json
import uuid
from pydantic import BaseModel
from database.database import get_db
import httpx

# Use Anthropic directly for native tool use
from anthropic import Anthropic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])

# Lazy initialization of Anthropic client
_anthropic_client = None

def get_anthropic_client() -> Anthropic:
    """Lazy load Anthropic client with credentials"""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from services.credential_resolver import get_credential_resolver
            resolver = get_credential_resolver()
            anthropic_key = resolver.get_credential_field("development_anthropic", "api_key")
        except Exception:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        if not anthropic_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY not configured. Add it to Settings > Credentials or set ANTHROPIC_API_KEY environment variable."
            )
        _anthropic_client = Anthropic(api_key=anthropic_key)
    return _anthropic_client

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


# Tool implementations
async def search_code_tool(query: str) -> Dict[str, Any]:
    """Search codebase using CodeGraph semantic search"""
    try:
        logger.info(f"[CodeGraph] Searching for: {query}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "http://127.0.0.1:8000/api/code-graph/search/semantic",
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://127.0.0.1:8000/api/documents/search",
                params={
                    "query": query,
                    "limit": 8,
                    "min_similarity": 0.65
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[Documents] Found {data.get('total_results', 0)} results")
                
                # Format results
                results = []
                for result in data.get('results', [])[:8]:
                    content = result.get('content', '')
                    results.append({
                        "id": result.get('chunk_id', result.get('id', 0)),
                        "filename": result.get('source', {}).get('filename', 'Unknown'),
                        "excerpt": content,  # Frontend expects 'excerpt'
                        "content": content,   # Keep for backwards compatibility
                        "similarity": result.get('similarity', 0.0),
                        "file_type": result.get('source', {}).get('file_type', 'unknown')
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


async def query_database_tool(query: str, database_name: Optional[str] = None) -> Dict[str, Any]:
    """Query connected databases using natural language"""
    try:
        logger.info(f"[Database] Query: {query}, Database: {database_name or 'auto-detect'}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # First, get available database sources
            sources_response = await client.get("http://127.0.0.1:8000/api/knowledge/sources/database/")
            
            if sources_response.status_code != 200:
                return {"success": False, "error": "No database sources available"}
            
            sources = sources_response.json()
            if not sources:
                return {"success": False, "error": "No database sources configured"}
            
            # Select database source
            target_source = None
            if database_name:
                target_source = next((s for s in sources if database_name.lower() in s['name'].lower()), None)
            
            if not target_source:
                target_source = sources[0]  # Use first available database
            
            logger.info(f"[Database] Using source: {target_source['name']}")
            
            # Execute query
            query_response = await client.post(
                f"http://127.0.0.1:8000/api/knowledge/sources/database/{target_source['id']}/query",
                json={
                    "query": query,
                    "source_id": target_source['id']
                }
            )
            
            if query_response.status_code == 200:
                data = query_response.json()
                logger.info(f"[Database] Query successful: {data.get('row_count', 0)} rows")
                
                return {
                    "success": data.get('success', True),
                    "database": target_source['name'],
                    "sql": data.get('sql', ''),
                    "row_count": data.get('row_count', 0),
                    "data": data.get('data', [])[:10],  # Limit to 10 rows for chat
                    "columns": data.get('columns', []),
                    "execution_time_ms": data.get('execution_time_ms', 0),
                    "explanation": data.get('explanation', '')
                }
            else:
                error_data = query_response.json()
                return {"success": False, "error": error_data.get('detail', 'Query failed')}
                
    except Exception as e:
        logger.error(f"[Database] Query error: {e}")
        return {"success": False, "error": str(e)}


# Define tools for Claude
TOOLS = [
    {
        "name": "search_code",
        "description": "Search the codebase for functions, classes, methods, and code implementations using CodeGraph semantic search. Use this when the user asks about code, how something works, implementation details, or specific code symbols like AgentFactory, functions, classes, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for code (e.g., 'AgentFactory', 'authentication flow', 'how agents are created')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_documents",
        "description": "Search documentation, guides, PDFs, and uploaded documents using RAG semantic search. Use this when the user asks about documentation, guides, architecture, deployment, setup, specifications, or general information about the platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for documents (e.g., 'deployment guide', 'architecture documentation', 'how to setup')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_database",
        "description": "Query connected databases using natural language. Converts questions like 'show me top customers' into SQL and returns actual data. Use this when users ask about data, analytics, statistics, customer information, orders, revenue, or any database queries. Works with PostgreSQL and MySQL databases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language database query (e.g., 'show me top 10 customers by revenue', 'how many orders in the last month', 'what is the average order value')"
                },
                "database_name": {
                    "type": "string",
                    "description": "Optional: specific database name to query. If not provided, uses the first available database."
                }
            },
            "required": ["query"]
        }
    }
]


@router.post("/query")
async def process_chatbot_query(
    chat_query: ChatQuery,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process chatbot query using Claude with function calling.
    Claude decides whether to search code or documents.
    """
    try:
        request_id = str(uuid.uuid4())[:12]
        logger.info(f"[req={request_id}] Chatbot query: '{chat_query.query}'")
        
        # Build initial prompt
        user_message = f"""User query: {chat_query.query}

Context: {chat_query.context.dict() if chat_query.context else 'None'}

You are the Automatos AI Assistant. Analyze the user's query and decide:
1. If they're asking about CODE (functions, classes, implementation), use the search_code tool
2. If they're asking about DOCUMENTS (guides, architecture, docs), use the search_documents tool
3. You can use BOTH tools if the query would benefit from both code and documentation

Then provide a helpful, technical response based on the search results."""

        # Call Claude with tool use
        logger.info(f"[req={request_id}] Calling Claude with tools...")
        
        messages = [{"role": "user", "content": user_message}]
        code_results = []
        document_results = []
        database_results = []
        tool_calls_made = []
        
        # Agentic loop - let Claude call tools iteratively
        for iteration in range(3):  # Max 3 tool call iterations
            response = get_anthropic_client().messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                tools=TOOLS,
                messages=messages
            )
            
            logger.info(f"[req={request_id}] Claude response stop_reason: {response.stop_reason}")
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Collect all tool results first
                tool_results = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id
                        
                        logger.info(f"[req={request_id}] Claude called tool: {tool_name} with query: {tool_input.get('query')}")
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
                            tool_result = await query_database_tool(
                                tool_input.get("query", ""),
                                tool_input.get("database_name")
                            )
                            if tool_result.get("success"):
                                database_results.append(tool_result)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        # Collect tool result
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(tool_result)
                        })
                
                # Add assistant message with ALL tool_use blocks
                messages.append({"role": "assistant", "content": response.content})
                
                # Add ONE user message with ALL tool_result blocks
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Continue the loop to let Claude process tool results
                continue
                
            else:
                # Claude provided final response
                final_content = ""
                for content_block in response.content:
                    if hasattr(content_block, "text"):
                        final_content += content_block.text
                
                logger.info(f"[req={request_id}] Claude finished. Used tools: {tool_calls_made}")
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
                            "llm_model": "claude-3-5-sonnet-20241022",
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
        raise HTTPException(status_code=500, detail=str(e))

