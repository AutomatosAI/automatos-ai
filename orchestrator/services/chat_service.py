"""
Chat Service for PRD-27
========================

Service layer for managing chat sessions, messages, and streaming responses.
Integrates with existing LLM service, tool ecosystem, AND memory system.
"""

import json
import logging
import os
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import httpx

from services.pandas_ai_service import get_pandasai_service
from services.tool_result_formatter import ToolResultFormatter
from config import config

# Import models from models package
from models import Chat, Message, Vote, Artifact

logger = logging.getLogger(__name__)

# Memory system integration
_memory_system = None
_context_retrieval_engine = None
_tool_registry = None

def get_tool_registry():
    """Get or create the ToolRegistry instance (dynamic tool discovery)"""
    global _tool_registry
    if _tool_registry is None:
        try:
            from services.tool_registry import ToolRegistry, ToolCategory
            _tool_registry = ToolRegistry()
            logger.info(f"✅ Chat service connected to ToolRegistry ({len(_tool_registry.tools)} tools available)")
        except Exception as e:
            logger.warning(f"ToolRegistry not available: {e}")
    return _tool_registry

def get_chatbot_tools_from_registry() -> List[Dict[str, Any]]:
    """
    Get tools from ToolRegistry in OpenAI function format.
    This is the SINGLE SOURCE OF TRUTH for chatbot tools - no hardcoding!
    """
    registry = get_tool_registry()
    if not registry:
        return []
    
    from services.tool_registry import ToolCategory
    
    # Get RESEARCH tools (RAG, CodeGraph, semantic search, multimodal)
    research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
    
    # Also include DATABASE tools if available
    database_tools = registry.get_tools_by_category(ToolCategory.DATABASE_TOOLS)
    
    all_tools = research_tools + database_tools
    
    # Convert to OpenAI function format with wrapper
    openai_tools = []
    for tool in all_tools:
        schema = tool.to_openai_format()
        openai_tools.append({
            "type": "function",
            "function": schema
        })
    
    logger.debug(f"Loaded {len(openai_tools)} tools from registry for chatbot")
    return openai_tools

def get_memory_system():
    """Get or create the hierarchical memory system instance"""
    global _memory_system
    if _memory_system is None:
        try:
            from services.memory_knowledge_system import HierarchicalMemorySystem
            _memory_system = HierarchicalMemorySystem()
            logger.info("Chat service connected to HierarchicalMemorySystem")
        except Exception as e:
            logger.warning(f"Could not initialize memory system for chat: {e}")
    return _memory_system

async def get_context_retrieval_engine():
    """Get or create the context retrieval engine (uses existing infrastructure)"""
    global _context_retrieval_engine
    if _context_retrieval_engine is None:
        try:
            from context_engineering.retrieval.context_retrieval_engine import (
                ContextRetrievalEngine, RetrievalStrategy
            )
            from context_engineering.retrieval.vector_store_enhanced import EnhancedVectorStore
            from context_engineering.chunking.semantic_chunker import SemanticChunker
            from database.database import get_database_url
            
            # Initialize with your existing components
            vector_store = EnhancedVectorStore(
                database_url=get_database_url(),  # Fixed: correct param name
                embedding_dimension=384  # Match your HuggingFace local embeddings
            )
            await vector_store.initialize()  # Required async init
            
            chunker = SemanticChunker()
            _context_retrieval_engine = ContextRetrievalEngine(
                vector_store=vector_store,
                chunker=chunker,
                default_strategy=RetrievalStrategy.ADAPTIVE  # Let it learn!
            )
            logger.info("✅ Chat service connected to ContextRetrievalEngine (using existing infrastructure)")
        except Exception as e:
            logger.warning(f"ContextRetrievalEngine not available, using fallback: {e}")
    return _context_retrieval_engine


# Helper aliases for backward compatibility - use ToolResultFormatter directly
_clean_document_filename = ToolResultFormatter._clean_document_filename
_extract_useful_content = ToolResultFormatter._extract_useful_content


def _truncate_for_llm(result: Dict[str, Any], tool_name: str = "unknown", max_chars: int = 3000) -> str:
    """
    Truncate tool results to fit in LLM context.
    Uses unified formatter - NO MORE DUPLICATE LOGIC.
    """
    return ToolResultFormatter.format_for_llm(result, tool_name, max_chars)


def _get_unified_executor():
    """Get UnifiedToolExecutor instance (shared across chatbot)"""
    from services.unified_tool_executor import UnifiedToolExecutor
    from database.database import get_db_session
    
    # Get a db session for the executor
    with get_db_session() as db_session:
        return UnifiedToolExecutor(db_session)


async def execute_tool_unified(tool_name: str, tool_args: Dict[str, Any], agent_id: int = 1) -> Dict[str, Any]:
    """
    Execute a tool via UnifiedToolExecutor.
    This is THE SINGLE ENTRY POINT for all tool execution.
    
    Same infrastructure used by:
    - Chatbot (this)
    - Agents (via agent_factory)
    - Workflows (via orchestrator)
    """
    from services.unified_tool_executor import UnifiedToolExecutor
    from database.database import get_db_session
    
    with get_db_session() as db_session:
        executor = UnifiedToolExecutor(db_session)
        return await executor.execute_tool(tool_name, tool_args, agent_id)


# Tool implementations for RAG, CodeGraph, and Database
async def search_code_tool(query: str) -> Dict[str, Any]:
    """Search codebase using CodeGraph semantic search"""
    try:
        logger.info(f"[CodeGraph Tool] Searching for: {query}")
        
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
                
                return {"success": True, "results": results, "count": len(results)}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"[CodeGraph Tool] Error: {e}")
        return {"success": False, "error": str(e)}


async def search_documents_tool(query: str) -> Dict[str, Any]:
    """Search documents using RAG semantic search"""
    try:
        logger.info(f"[RAG Tool] Searching for: {query}")
        
        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/documents/search",
                params={
                    "query": query,
                    "limit": 8,
                    "min_similarity": 0.65
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[RAG Tool] Raw API response: {data.get('results', [])[0] if data.get('results') else 'NO RESULTS'}")
                results = []
                for doc in data.get('results', [])[:8]:
                    # RAG API returns nested structure: {content, source: {filename}, ...}
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
                
                logger.info(f"[RAG Tool] Formatted result: {results[0] if results else 'NO RESULTS'}")
                return {"success": True, "results": results, "count": len(results)}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"[RAG Tool] Error: {e}")
        return {"success": False, "error": str(e)}


async def query_database_tool(
    query: str,
    database_name: Optional[str] = None,
    analysis_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Query database using natural language. Falls back to direct main DB query if no knowledge sources."""
    try:
        logger.info(f"[Database Tool] Querying: {query}, DB: {database_name}")

        base_url = config.KNOWLEDGE_API_BASE_URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            # List active sources and select match or default
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
                logger.warning(f"[Database Tool] Failed to list sources: {source_err}")

            if database_name and sources:
                lowered = database_name.lower()
                selected_source = next(
                    (src for src in sources if str(src.get("name", "")).lower() == lowered),
                    None
                )

            if not selected_source and sources:
                selected_source = sources[0]

            # FALLBACK: If no knowledge sources, query main database directly
            if not selected_source:
                logger.info("[Database Tool] No knowledge sources - using direct DB query fallback")
                return await _query_main_database_direct(query, analysis_prompt)

            source_id = int(selected_source.get("id"))  # Ensure int for Pydantic
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
                tool_result = {
                    "success": True,
                    "database": data.get('database') or selected_source.get('name', ''),
                    "sql": data.get('sql', ''),
                    "row_count": data.get('row_count', 0),
                    "data": data.get('data', []),
                    "columns": data.get('columns', []),
                    "execution_time_ms": data.get('execution_time_ms', 0)
                }
                pandasai = get_pandasai_service()
                if pandasai:
                    insight = pandasai.generate_insight(
                        analysis_prompt or query,
                        tool_result.get('data', []),
                        tool_result.get('columns', []),
                    )
                    if insight:
                        tool_result["pandas_ai"] = insight
                return tool_result
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}

    except Exception as e:
        logger.error(f"[Database Tool] Error: {e}")
        return {"success": False, "error": str(e)}


async def _query_main_database_direct(query: str, analysis_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Direct fallback: Query the main Automatos database.
    Uses LLM to generate SQL from natural language - that's the whole point.
    """
    import time
    from database.database import get_db_session
    from sqlalchemy import text
    from services.llm_provider import create_llm_manager
    
    try:
        start_time = time.time()
        
        # Schema context for the LLM
        # Get schema from centralized provider
        from services.schema_provider import get_schema_provider
        schema_provider = get_schema_provider(db)
        schema = schema_provider.get_database_schema_overview()

        # Add query guidance
        schema += "\n\nUse DATE_TRUNC('day', col) for daily grouping. Use NOW() - INTERVAL 'N days' for date ranges."
        
        # Get LLM to generate SQL
        llm_manager = await create_llm_manager(provider="openai", model="gpt-4")
        
        response = await llm_manager.generate_response([
            {"role": "system", "content": f"You are a PostgreSQL expert. Generate ONLY the SQL query, nothing else. Only SELECT allowed.\n\n{schema}"},
            {"role": "user", "content": f"Generate SQL for: {query}"}
        ])
        
        sql = response.content.strip()
        # Clean markdown
        if "```" in sql:
            sql = sql.split("```")[1] if "```" in sql else sql
            sql = sql.replace("sql", "").strip()
        sql = sql.strip()
        
        if not sql.upper().startswith("SELECT"):
            return {"success": False, "error": "Only SELECT queries allowed"}
        
        logger.info(f"[Database Tool] LLM generated SQL: {sql}")
        
        # Execute
        with get_db_session() as db:
            result = db.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
            
            data = []
            for row in rows[:100]:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    row_dict[col] = val
                data.append(row_dict)
        
        execution_time = (time.time() - start_time) * 1000
        
        tool_result = {
            "success": True,
            "database": "automatos_main",
            "sql": sql,
            "row_count": len(data),
            "data": data,
            "columns": columns,
            "execution_time_ms": round(execution_time, 2)
        }
        
        # PandasAI for insights/charts
        pandasai = get_pandasai_service()
        if pandasai and data:
            insight = pandasai.generate_insight(analysis_prompt or query, data, columns)
            if insight:
                tool_result["pandas_ai"] = insight
        
        logger.info(f"[Database Tool] Returned {len(data)} rows in {execution_time:.0f}ms")
        return tool_result
        
    except Exception as e:
        logger.error(f"[Database Tool] Error: {e}")
        return {"success": False, "error": str(e)}


# Tool definitions for LLM - loaded dynamically from ToolRegistry
def get_chat_tools() -> List[Dict[str, Any]]:
    """
    Get chatbot tools from ToolRegistry (SINGLE SOURCE OF TRUTH).
    Falls back to minimal set if registry unavailable.
    """
    # Try registry first (SINGLE SOURCE OF TRUTH - includes RESEARCH + DATABASE_TOOLS)
    registry_tools = get_chatbot_tools_from_registry()
    if registry_tools:
        logger.debug(f"Loaded {len(registry_tools)} tools from registry for chatbot")
        return registry_tools
    
    # Fallback if registry unavailable
    logger.warning("ToolRegistry unavailable, using minimal fallback tools")
    return [
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search the codebase for functions, classes, or code patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search documents and knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Query database using natural language.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Plain English query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

# Backwards compatibility alias
CHAT_TOOLS = get_chat_tools()


class ChatService:
    """Service for managing chat sessions and messages"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create_chat(
        self,
        user_id: int,
        title: str,
        visibility: str = "private"
    ) -> Chat:
        """Create a new chat session"""
        chat = Chat(
            id=uuid.uuid4(),
            user_id=user_id,
            title=title,
            visibility=visibility,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        logger.info(f"Created chat {chat.id} for user {user_id}: {title}")
        return chat
    
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """Get a chat by ID"""
        try:
            chat_uuid = uuid.UUID(chat_id)
            return self.db.query(Chat).filter(Chat.id == chat_uuid).first()
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid chat_id format: {chat_id}")
            return None
    
    def get_chat_history(
        self,
        user_id: int,
        limit: int = 20,
        starting_after: Optional[datetime] = None
    ) -> List[Chat]:
        """Get chat history for a user"""
        query = self.db.query(Chat).filter(Chat.user_id == user_id)
        
        if starting_after:
            query = query.filter(Chat.created_at < starting_after)
        
        return query.order_by(desc(Chat.created_at)).limit(limit).all()
    
    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat title"""
        chat = self.get_chat(chat_id)
        if chat:
            chat.title = title
            chat.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Updated chat {chat_id} title: {title}")
            return True
        return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages"""
        chat = self.get_chat(chat_id)
        if chat:
            self.db.delete(chat)
            self.db.commit()
            logger.info(f"Deleted chat {chat_id}")
            return True
        return False
    
    def save_message(
        self,
        chat_id: str,
        role: str,
        parts: List[Dict[str, Any]],
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Message:
        """Save a message to the database"""
        try:
            chat_uuid = uuid.UUID(chat_id)
        except ValueError:
            raise ValueError(f"Invalid chat_id format: {chat_id}")
            
        message = Message(
            id=uuid.uuid4(),
            chat_id=chat_uuid,
            role=role,
            parts=parts,
            attachments=attachments or [],
            created_at=datetime.utcnow()
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        # Update chat's updated_at timestamp
        chat = self.get_chat(chat_id)
        if chat:
            chat.updated_at = datetime.utcnow()
            self.db.commit()
        
        logger.debug(f"Saved {role} message {message.id} to chat {chat_id}")
        return message
    
    def get_messages_by_chat_id(
        self,
        chat_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get all messages for a chat"""
        try:
            chat_uuid = uuid.UUID(chat_id)
        except ValueError:
            return []
            
        query = self.db.query(Message).filter(Message.chat_id == chat_uuid)
        query = query.order_by(Message.created_at.asc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def vote_message(
        self,
        chat_id: str,
        message_id: str,
        is_upvoted: bool
    ) -> bool:
        """Vote on a message"""
        try:
            chat_uuid = uuid.UUID(chat_id)
            message_uuid = uuid.UUID(message_id)
        except ValueError as e:
            logger.error(f"Invalid UUID format: {e}")
            return False
            
        vote = self.db.query(Vote).filter(
            and_(
                Vote.chat_id == chat_uuid,
                Vote.message_id == message_uuid
            )
        ).first()
        
        if vote:
            vote.is_upvoted = is_upvoted
        else:
            vote = Vote(
                chat_id=chat_uuid,
                message_id=message_uuid,
                is_upvoted=is_upvoted,
                created_at=datetime.utcnow()
            )
            self.db.add(vote)
        
        self.db.commit()
        logger.info(f"Voted on message {message_id}: upvoted={is_upvoted}")
        return True


class StreamingChatService:
    """Service for streaming chat responses with SSE"""
    
    def __init__(self, db: Session):
        self.db = db
        self.chat_service = ChatService(db)
        
    async def stream_response(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response using Server-Sent Events (SSE).
        
        Integrates with existing LLMManager for streaming.
        
        Args:
            chat_id: Chat session ID
            messages: List of message dicts with role and parts
            model: Model ID to use
            tools: Optional list of tools to enable
            
        Yields:
            SSE-formatted chunks (data: {json}\n\n)
        """
        from services.llm_provider import create_llm_manager
        
        try:
            # Get LLM manager (existing service)
            llm_manager = create_llm_manager(service_name="chatbot")
            
            # Convert messages to LLM format
            llm_messages = self._convert_to_llm_messages(messages)
            
            # Track assistant response parts for saving
            assistant_parts = []
            
            # Check if LLM manager supports streaming
            if hasattr(llm_manager, 'generate_response_stream'):
                # Use streaming if available
                async for chunk in llm_manager.generate_response_stream(
                    messages=llm_messages,
                    tools=tools or []
                ):
                    # Format and yield SSE chunk
                    sse_chunk = self._format_sse_chunk(chunk)
                    yield sse_chunk
                    
                    # Track text for saving
                    if chunk.get('type') == 'text':
                        assistant_parts.append({
                            'type': 'text',
                            'text': chunk.get('text', '')
                        })
            else:
                # Fallback: Generate complete response and stream it in chunks
                # Only use tools for non-trivial messages (skip for greetings, simple questions)
                latest_text = self._extract_latest_user_text(messages).lower().strip()
                simple_patterns = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'ok', 'yes', 'no', 'what model', 'who are you', 'what are you']
                is_simple = len(latest_text) < 50 and any(p in latest_text for p in simple_patterns)
                use_tools = None if is_simple else (tools or CHAT_TOOLS)
                
                response = await llm_manager.generate_response(
                    messages=llm_messages,
                    tools=use_tools
                )

                latest_user_question = self._extract_latest_user_text(messages)
                
                # Handle tool calls if present
                if response.tool_calls:
                    logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                    
                    # Execute tools and collect results
                    tool_results = []
                    tool_data = {}  # Collect structured data for frontend
                    
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('function', {}).get('name')
                        tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                        tool_id = tool_call.get('id')
                        
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        
                        # Execute tool via unified registry executor (no hardcoded if/else!)
                        result = await execute_tool_unified(tool_name, tool_args)
                        
                        # Format results using unified formatter (NO MORE DUPLICATE LOGIC!)
                        if result.get('success') or result.get('status') == 'success':
                            frontend_data = ToolResultFormatter.format_for_frontend(result, tool_name)
                            tool_data.update(frontend_data)

                            # Log what we're sending
                            if 'documents' in tool_data:
                                logger.info(f"[Chat Service] Sending {len(tool_data['documents'])} documents to frontend")
                            if 'code_snippets' in tool_data:
                                logger.info(f"[Chat Service] Sending {len(tool_data['code_snippets'])} code snippets to frontend")
                        
                        # Truncate result for LLM context (full data already sent to frontend via tool_data)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": _truncate_for_llm(result)
                        })
                    
                    # Send tool data to frontend as structured data
                    if tool_data:
                        logger.info(f"[Chat Service] Sending tool-data to frontend: {list(tool_data.keys())}")
                        yield f"data: {json.dumps({'type': 'tool-data', 'data': tool_data})}\n\n"
                    
                    # Add tool results to messages and get final response
                    llm_messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls
                    })
                    llm_messages.extend(tool_results)
                    
                    # Get final response after tool execution
                    final_response = await llm_manager.generate_response(
                        messages=llm_messages,
                        tools=None  # Don't allow recursive tool calling
                    )
                    response_text = final_response.content or ""
                else:
                    # Extract response text from LLMResponse object
                    response_text = response.content or ""
                
                # Stream response text word by word for progressive display
                # AI SDK expects: text-start -> text-delta(s) -> text-end
                import uuid
                message_id = str(uuid.uuid4())
                
                # 1. Send text-start
                yield f"data: {json.dumps({'type': 'text-start', 'id': message_id})}\n\n"
                
                # 2. Send text-delta chunks
                words = response_text.split(' ')
                for i, word in enumerate(words):
                    chunk_text = word + (' ' if i < len(words) - 1 else '')
                    chunk = {
                        'type': 'text-delta',
                        'id': message_id,
                        'delta': chunk_text
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 3. Send text-end
                yield f"data: {json.dumps({'type': 'text-end', 'id': message_id})}\n\n"
                
                # Collect full text for saving
                assistant_parts.append({'type': 'text', 'text': response_text})
            
            # Save assistant message after streaming completes
            if assistant_parts:
                self.chat_service.save_message(
                    chat_id=chat_id,
                    role='assistant',
                    parts=assistant_parts
                )
                
            # Send completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error streaming chat response: {e}", exc_info=True)
            error_chunk = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    def _convert_to_llm_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Convert chat messages to LLM format with intelligent tool guidance"""
        llm_messages = []
        
        # Count messages in conversation
        msg_count = len(messages)
        
        # Always add system prompt with context about memory
        llm_messages.append({
            "role": "system",
            "content": f"""You are the Automatos AI Assistant - a custom AI platform with PERSISTENT MEMORY.

⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. You HAVE memory. The conversation history ({msg_count} messages) is provided below.
2. If someone tells you their name, REMEMBER IT and use it.
3. NEVER say "I don't have memory", "I can't remember", "as an AI I don't...", or any similar phrases.
4. When asked "do you remember me?" or "what's my name?" - CHECK the conversation history AND the memory context provided.
5. If the name IS in the conversation or memory, say it. If it's NOT there, say "You haven't told me your name yet - what should I call you?"

You have tools: search_codebase, search_knowledge, query_database (natural language to SQL).

Be helpful, direct, and friendly. You're part of the Automatos family."""
        })
        
        for msg in messages:
            if msg.get('parts'):
                # Extract text content from parts
                text_parts = []
                for p in msg['parts']:
                    if p.get('type') != 'text':
                        continue
                    text_value = p.get('text')
                    if text_value is None:
                        continue
                    text_parts.append(str(text_value))
                content = '\n'.join(text_parts)
            else:
                content = msg.get('content', '')
            
            llm_messages.append({
                'role': msg['role'],
                'content': content
            })
        
        return llm_messages

    def _extract_latest_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get('role') != 'user':
                continue
            if msg.get('parts'):
                text_parts = [
                    part.get('text', '')
                    for part in msg['parts']
                    if part.get('type') == 'text'
                ]
                content = '\n'.join(filter(None, text_parts))
            else:
                content = msg.get('content', '')
            if content:
                return content
        return ''
    
    def _extract_search_terms(self, query: str) -> str:
        """
        Extract meaningful search terms from conversational queries.
        
        Converts: "Can you access my code and show me how agents work?"
        To: "agents AgentFactory agent_factory"
        
        This enables better results from RAG, CodeGraph, and database tools.
        """
        import re
        
        # Remove common conversational phrases
        stop_phrases = [
            r"can you (please )?",
            r"could you (please )?",
            r"please ",
            r"show me (how )?(to )?",
            r"tell me (about )?(how )?(to )?",
            r"i want to (know|see|understand) (about )?(how )?(to )?",
            r"help me (understand|with) ",
            r"what is (the )?",
            r"what are (the )?",
            r"how (does|do|can|to) ",
            r"where (is|are|can) ",
            r"access my ",
            r"look at (my )?",
            r"find (me )?",
            r"search (for )?",
            r"query (the )?",
        ]
        
        cleaned = query.lower()
        for phrase in stop_phrases:
            cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        
        # Split into words and filter
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                      'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                      'my', 'your', 'our', 'their', 'its', 'this', 'that', 'these', 'those',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
                      'work', 'works', 'working', 'about', 'just', 'also', 'only', 'some',
                      'code', 'codebase'}  # 'code' too generic - keep specific terms
        
        words = cleaned.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add variations for common terms
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            # Add CamelCase and snake_case variants
            if kw == 'agent' or kw == 'agents':
                expanded.extend(['AgentFactory', 'agent_factory', 'agent_execution'])
            elif kw == 'workflow' or kw == 'workflows':
                expanded.extend(['workflow_execution', 'WorkflowExecution'])
            elif kw == 'memory' or kw == 'memories':
                expanded.extend(['memory_system', 'HierarchicalMemorySystem'])
            elif kw == 'tool' or kw == 'tools':
                expanded.extend(['ToolRegistry', 'tool_executor'])
        
        result = " ".join(expanded[:8])  # Limit to 8 terms
        return result if result.strip() else query[:50]  # Fallback to original if empty

    async def _detect_tool_intent(self, query: str) -> List[str]:
        """
        Detect which tools should be triggered based on user query.
        Uses semantic matching with tool descriptions from ToolRegistry.
        
        Returns list of tool names to execute (can be multiple).
        """
        query_lower = query.lower()
        tools_to_trigger = []
        
        # Get available tools from registry
        registry = get_tool_registry()
        if not registry:
            # Fallback to basic keyword detection if registry unavailable
            if any(kw in query_lower for kw in ['doc', 'document', 'find', 'search']):
                tools_to_trigger.append('search_knowledge')
            if any(kw in query_lower for kw in ['code', 'function', 'class', 'implement', 'how does', 'work']):
                tools_to_trigger.append('search_codebase')
            if any(kw in query_lower for kw in ['database', 'data', 'count', 'how many', 'statistics', 'workflow']):
                tools_to_trigger.append('query_database')
            return tools_to_trigger
        
        # Use tool descriptions for smarter matching
        from services.tool_registry import ToolCategory
        
        # Check code-related queries
        code_indicators = ['code', 'function', 'class', 'method', 'implement', 'how does', 'works', 'agentfactory', 'factory', 'service', 'model']
        if any(ind in query_lower for ind in code_indicators):
            tools_to_trigger.append('search_codebase')
        
        # Check document-related queries
        doc_indicators = ['document', 'doc', 'guide', 'how to', 'tutorial', 'architecture', 'design', 'readme', 'help']
        if any(ind in query_lower for ind in doc_indicators):
            tools_to_trigger.append('search_knowledge')
        
        # Check database-related queries
        db_indicators = ['database', 'data', 'count', 'how many', 'statistics', 'workflow', 'agent', 'execution', 'list', 'show', 'query']
        if any(ind in query_lower for ind in db_indicators):
            tools_to_trigger.append('query_database')
        
        # Check multimodal-related queries
        multimodal_indicators = ['image', 'diagram', 'table', 'chart', 'formula', 'picture', 'screenshot']
        if any(ind in query_lower for ind in multimodal_indicators):
            tools_to_trigger.append('search_multimodal')
        
        # DON'T auto-search docs for general conversation
        # Only trigger tools when explicitly requested or clear intent detected
        # Memory + LLM knowledge should handle general questions
        
        return tools_to_trigger
    
    def _format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """Format chunk as SSE data - legacy format"""
        if chunk.get('type') == 'text':
            import uuid
            data = {
                'type': 'text-delta',
                'id': str(uuid.uuid4()),
                'delta': chunk.get('text', '')
            }
        elif chunk.get('type') == 'tool_call':
            data = {
                'type': 'tool-result',
                'toolName': chunk.get('tool_name'),
                'result': chunk.get('result')
            }
        elif chunk.get('type') == 'usage':
            data = {
                'type': 'data-usage',
                'data': {
                    'promptTokens': chunk.get('prompt_tokens', 0),
                    'completionTokens': chunk.get('completion_tokens', 0),
                    'totalTokens': chunk.get('total_tokens', 0),
                    'cost': chunk.get('cost')
                }
            }
        else:
            data = chunk
        
        return f"data: {json.dumps(data)}\n\n"

    async def _retrieve_relevant_memories(self, chat_id: str, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Retrieve relevant memories using the EXISTING ContextRetrievalEngine.
        
        Uses your infrastructure:
        - RetrievalStrategy.ADAPTIVE (learns from past queries)
        - Automatic temporal vs semantic detection
        - Mathematical foundations (decay, ranking)
        - No hardcoded keyword detection
        """
        try:
            latest_text = self._extract_latest_user_text(messages)
            if not latest_text or len(latest_text) < 5:
                return None
            
            # Try the advanced retrieval engine first (uses your existing infrastructure)
            retrieval_engine = await get_context_retrieval_engine()
            if retrieval_engine:
                logger.info(f"[Memory] Trying ContextRetrievalEngine...")
                result = await self._retrieve_with_context_engine(retrieval_engine, chat_id, latest_text)
                if result:
                    return result
                logger.info(f"[Memory] ContextRetrievalEngine returned empty, trying basic memory...")
            
            # Fallback to basic memory system (always try this!)
            logger.info(f"[Memory] Using HierarchicalMemorySystem...")
            return await self._retrieve_with_basic_memory(chat_id, latest_text)
            
        except Exception as e:
            logger.debug(f"Memory retrieval skipped: {e}")
            return None
    
    async def _retrieve_with_context_engine(self, engine, chat_id: str, query: str) -> Optional[str]:
        """Use the ContextRetrievalEngine (your existing infrastructure)"""
        try:
            from context_engineering.retrieval.context_retrieval_engine import (
                ContextQuery, ContextType, RetrievalStrategy
            )
            
            # Build query using your existing ContextQuery class
            context_query = ContextQuery(
                text=query,
                context_types=[ContextType.HISTORICAL],  # Conversation memories
                max_results=10,
                min_relevance=0.3,
                include_metadata=True
            )
            
            # Let ADAPTIVE strategy figure out temporal vs semantic
            result = await engine.retrieve_context(
                context_query, 
                strategy=RetrievalStrategy.ADAPTIVE
            )
            
            if not result.contexts:
                return None
            
            # Format results for LLM
            memory_lines = []
            for ctx in result.contexts[:8]:
                content = ctx.metadata.get('content', {}) if ctx.metadata else {}
                
                # Handle both conversation format and general content
                if isinstance(content, dict) and 'user_query' in content:
                    user_q = content.get('user_query', '')[:200]
                    assistant_r = content.get('assistant_response', '')[:150]
                    mem_chat = content.get('chat_id', '')
                    
                    if user_q:
                        prefix = "↳ " if mem_chat == chat_id else ""
                        memory_lines.append(f"{prefix}You: \"{user_q}\"")
                        if assistant_r:
                            memory_lines.append(f"{prefix}Me: \"{assistant_r}\"")
                else:
                    # General context (docs, code, etc)
                    if ctx.content and len(ctx.content) > 20:
                        memory_lines.append(f"[{ctx.context_type.value}] {ctx.content[:200]}")
            
            logger.info(f"✅ Retrieved {len(result.contexts)} memories via ContextRetrievalEngine (strategy: {result.strategy_used.value})")
            
            return "\n".join(memory_lines) if memory_lines else None
            
        except Exception as e:
            logger.warning(f"ContextRetrievalEngine failed, using fallback: {e}")
            return None
    
    async def _retrieve_with_basic_memory(self, chat_id: str, query: str) -> Optional[str]:
        """
        Retrieve memories using BOTH:
        1. Semantic search (for relevant context)
        2. Recent memories (for continuity and personal info like names)
        
        This ensures the chatbot always has access to recent conversations
        where the user might have introduced themselves.
        """
        memory_system = get_memory_system()
        if not memory_system:
            logger.warning("[Memory] HierarchicalMemorySystem not available!")
            return None
        
        all_memories = []
        seen_ids = set()
        
        # 1. Get semantically relevant memories
        logger.info(f"[Memory] Semantic search for: {query[:50]}...")
        try:
            semantic_memories = await memory_system.retrieve_relevant_memories(
                agent_id=1,
                context=query,
                memory_types=["experience"],
                top_k=8
            ) or []
            for mem in semantic_memories:
                mem_id = mem.get('id') or str(mem.get('content', ''))[:50]
                if mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    all_memories.append(('semantic', mem))
            logger.info(f"[Memory] Semantic search found {len(semantic_memories)} memories")
        except Exception as e:
            logger.error(f"[Memory] Semantic search failed: {e}")
        
        # 2. Get RECENT memories (last 10) to catch personal info like names
        # These might not match semantically but contain important context
        logger.info(f"[Memory] Fetching recent memories...")
        try:
            recent_memories = await self._get_recent_memories(memory_system, limit=10)
            for mem in recent_memories:
                mem_id = mem.get('id') or str(mem.get('content', ''))[:50]
                if mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    all_memories.append(('recent', mem))
            logger.info(f"[Memory] Recent memories found {len(recent_memories)}, {len(all_memories) - len(semantic_memories)} new")
        except Exception as e:
            logger.error(f"[Memory] Recent memory fetch failed: {e}")
        
        if not all_memories:
            logger.info("[Memory] No memories found")
            return None
        
        logger.info(f"[Memory] Total unique memories: {len(all_memories)}")
        
        # Format memories for LLM
        memory_lines = []
        for source, mem in all_memories[:12]:  # Limit to 12 for context window
            content = mem.get('content', {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {"summary": content[:200]}
            
            user_q = content.get('user_query', '')[:200]
            assistant_r = content.get('assistant_response', '')[:150]
            mem_chat = content.get('chat_id', '')
            
            if user_q:
                # Mark which chat it came from
                prefix = "[This chat] " if mem_chat == chat_id else "[Earlier] "
                memory_lines.append(f"{prefix}You: \"{user_q}\"")
                if assistant_r:
                    memory_lines.append(f"{prefix}Me: \"{assistant_r}\"")
        
        return "\n".join(memory_lines) if memory_lines else None
    
    async def _get_recent_memories(self, memory_system, limit: int = 10) -> List[Dict]:
        """Fetch the most recent memories regardless of semantic similarity"""
        try:
            # memory_items is the actual table used by HierarchicalMemorySystem
            from database.database import get_db_session
            from sqlalchemy import text
            
            with get_db_session() as db:
                result = db.execute(text("""
                    SELECT id, content, metadata, created_at
                    FROM memory_items 
                    WHERE memory_type = 'experience'
                    ORDER BY created_at DESC
                    LIMIT :limit
                """), {"limit": limit})
                
                memories = []
                for row in result:
                    content = row.content
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except:
                            content = {"summary": content}
                    memories.append({
                        'id': str(row.id),
                        'content': content,
                        'metadata': row.metadata,
                        'created_at': row.created_at
                    })
                logger.info(f"[Memory] Retrieved {len(memories)} recent from memory_items")
                return memories
        except Exception as e:
            logger.error(f"[Memory] Direct DB query failed: {e}")
            return []
    
    async def _store_conversation_memory(self, chat_id: str, user_message: str, assistant_response: str):
        """
        Store conversation as memory for MVP single user.
        Stores YOUR chat interactions for future recall.
        """
        try:
            logger.info(f"[Memory] Attempting to store: {user_message[:50]}...")
            memory_system = get_memory_system()
            if not memory_system:
                logger.warning("[Memory] No memory system available!")
                return
            
            # MVP: Store as YOUR experience (user_id=1)
            # ALL conversations are novel and important for personal context
            experience = {
                "type": "conversation",
                "chat_id": chat_id,
                "user_query": user_message[:500],
                "assistant_response": assistant_response[:500],
                "summary": f"{user_message[:100]}",
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "is_novel": True,  # All chat is important for memory
                "goal_relevant": True  # User conversations are always relevant
            }
            
            result = await memory_system.store_experience(
                agent_id=1,
                experience=experience
            )
            logger.info(f"[Memory] ✅ Stored chat memory (id={result}): {user_message[:50]}...")
            
        except Exception as e:
            logger.error(f"[Memory] ❌ Storage FAILED: {e}")
            import traceback
            traceback.print_exc()

    async def stream_response_aisdk(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        selected_model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response using AI SDK Data Stream format.
        
        Format:
        - 0:"text"\n for text chunks
        - d:{json}\n for data events
        - e:{json}\n for errors
        """
        from services.llm_provider import create_llm_manager
        import asyncio
        
        try:
            # Send chat_id to frontend first so it can track the conversation
            yield f'd:{{"type":"chat-id","chatId":"{chat_id}"}}\n'
            await asyncio.sleep(0)
            
            # Parse model string to get provider and model
            # Uses pattern matching - no hardcoded map needed
            provider = None
            model = None
            if selected_model:
                model = selected_model
                model_lower = selected_model.lower()
                
                # Pattern-based provider detection (matches frontend lib/ai/models.ts)
                if '/' in selected_model:
                    # HuggingFace format: org/model (e.g., meta-llama/Llama-3.2-3B-Instruct)
                    provider = 'huggingface'
                elif model_lower.startswith('gpt-') or 'openai' in model_lower:
                    provider = 'openai'
                elif model_lower.startswith('claude') or 'anthropic' in model_lower:
                    provider = 'anthropic'
                elif model_lower.startswith('grok') or 'xai' in model_lower:
                    provider = 'grok'
                elif model_lower.startswith('gemini') or 'google' in model_lower:
                    provider = 'google'
                elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'mixtral', 'phi']):
                    provider = 'huggingface'
                else:
                    # Default fallback - will use system settings
                    provider = None
            
            # Create LLM manager with optional provider/model override
            if provider and model:
                logger.info(f"Using user-selected model: {provider}/{model}")
                llm_manager = create_llm_manager(
                    service_name="chatbot",
                    provider=provider,
                    model=model
                )
            else:
                llm_manager = create_llm_manager(service_name="chatbot")
            llm_messages = self._convert_to_llm_messages(messages)
            assistant_parts = []
            full_response = ""
            tool_data = {}
            
            # Retrieve relevant memories from the memory system
            memory_context = await self._retrieve_relevant_memories(chat_id, messages)
            if memory_context:
                # Inject memory context into the conversation - VERY explicit
                # Use index 1 (after system prompt) - tool results will insert at 1 too, pushing memory to 2
                # That's fine - memory comes right after system prompt
                logger.info(f"[MEMORY DEBUG] Injecting memory:\n{memory_context[:200]}...")
                llm_messages.insert(1, {  # After system prompt
                    "role": "system",
                    "content": f"""🧠 CRITICAL - YOUR CONVERSATION MEMORY (READ THIS FIRST):

{memory_context}

INSTRUCTIONS:
- If the user asks your name, check memory for their name
- If asked "what did we talk about", summarize the memory above
- NEVER say "you haven't told me your name" if their name is in memory
- NEVER say "I don't have access to past conversations" - you DO have memory above"""
                })
                logger.info(f"Injected memory context ({len(memory_context)} chars) for chat {chat_id}")
            
            # Determine if we should use tools
            latest_text = self._extract_latest_user_text(messages).lower().strip()
            simple_patterns = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'ok', 'yes', 'no', 'what model', 'who are you', 'what are you']
            is_simple = len(latest_text) < 50 and any(p in latest_text for p in simple_patterns)
            
            # Check if model supports native tool calling (OpenAI, Anthropic, Grok do; HuggingFace doesn't)
            supports_native_tools = provider in ['openai', 'anthropic', 'grok'] if provider else False
            use_tools = None if is_simple else (tools or CHAT_TOOLS) if supports_native_tools else None
            
            # For models with NATIVE tool calling: let the LLM decide when to use tools
            # For HuggingFace (no native tools): only pre-trigger if EXPLICITLY requested
            detected_tools = []
            if not is_simple and not supports_native_tools:
                # HuggingFace fallback - only trigger for EXPLICIT requests
                if any(kw in latest_text for kw in ['search doc', 'find doc', 'show me doc', 'in the doc']):
                    detected_tools.append('search_knowledge')
                if any(kw in latest_text for kw in ['show code', 'show me code', 'search code', 'find code']):
                    detected_tools.append('search_codebase')
                if any(kw in latest_text for kw in ['query database', 'from database', 'sql', 'how many']):
                    detected_tools.append('query_database')
                
                if detected_tools:
                    logger.info(f"HuggingFace fallback: explicit tool request detected: {detected_tools}")
            
            # Execute pre-triggered tools (HuggingFace only)
            for tool_name in detected_tools:
                logger.info(f"Pre-triggering {tool_name} (HuggingFace fallback)")
                tool_query = latest_text  # Use full query for semantic search
                result = await execute_tool_unified(tool_name, {"query": tool_query})
                
                # Debug logging - what did we get back?
                logger.info(f"[TOOL DEBUG] {tool_name} returned: success={result.get('success')}, status={result.get('status')}, keys={list(result.keys())}")
                if result.get('results'):
                    logger.info(f"[TOOL DEBUG] {tool_name} has {len(result.get('results', []))} results")
                
                # Check for success - tools may use 'success' or 'status' field
                tool_succeeded = result.get('success') or result.get('status') == 'success' or result.get('results')
                
                if tool_succeeded:
                    # Use unified formatter for ALL tool results - NO MORE DUPLICATE LOGIC
                    formatted_data = ToolResultFormatter.format_for_frontend(result, tool_name)
                    tool_data.update(formatted_data)

                    # Generate context for LLM (different from frontend format)
                    if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
                        standardized = ToolResultFormatter.standardize_result(result, tool_name)
                        docs = standardized['results'][:3]  # Top 3 for context
                        if docs:
                            doc_context = "\n\n".join([
                                f"Document: {d.get('filename', d.get('source', 'Unknown'))}\n{d.get('excerpt', d.get('content', str(d)))[:500]}"
                                for d in docs
                            ])
                            logger.info(f"[TOOL DEBUG] Injecting {len(docs)} docs into LLM context")
                            llm_messages.insert(1, {  # After system prompt, not at 0
                                "role": "system",
                                "content": f"📚 DOCUMENTS FROM USER'S KNOWLEDGE BASE:\n\n{doc_context}\n\nYou MUST use this information to answer."
                            })

                    elif tool_name in ['search_codebase', 'search_code']:
                        standardized = ToolResultFormatter.standardize_result(result, tool_name)
                        code_results = standardized['results'][:5]  # Top 5 for context
                        if code_results:
                            code_context = "\n\n".join([
                                f"📁 {r.get('symbol_name', 'Code')} ({r.get('file_path', 'unknown')})\n```python\n{r.get('code', '')[:800]}\n```"
                                for r in code_results
                            ])
                            logger.info(f"[TOOL DEBUG] Injecting {len(code_results)} code snippets into LLM context")
                            llm_messages.insert(1, {  # After system prompt
                                "role": "system",
                                "content": f"💻 CODE FROM USER'S CODEBASE (from search_codebase tool):\n\n{code_context}\n\nYou MUST show and explain this code when answering. DO NOT say you cannot find it."
                            })

                    elif tool_name == 'query_database':
                        # Limit data preview to prevent context overflow
                        data_preview = result.get('data', [])[:5]
                        data_str = json.dumps(data_preview, default=str)[:1000]  # Cap at 1000 chars
                        db_context = f"Database query result: {result.get('row_count', 0)} rows\nSQL: {result.get('sql', 'N/A')[:200]}\nData preview: {data_str}"
                        logger.info(f"[TOOL DEBUG] Injecting DB results ({result.get('row_count', 0)} rows) into LLM context")
                        llm_messages.insert(1, {  # After system prompt
                            "role": "system",
                            "content": f"🗄️ DATABASE QUERY RESULTS:\n\n{db_context}\n\nYou MUST use this data when answering."
                        })

                    # Log what we're sending to frontend
                    if formatted_data:
                        logger.info(f"[TOOL DEBUG] Sending formatted {tool_name} results to frontend")
                
                # Send tool data if any
                if tool_data:
                    yield f'd:{{"type":"tool-data","data":{json.dumps(tool_data)}}}\n'
                    await asyncio.sleep(0)
            
            # Generate response
            response = await llm_manager.generate_response(
                messages=llm_messages,
                tools=use_tools
            )
            
            # Handle tool calls if present (for models that support it)
            if response.tool_calls:
                logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                tool_results = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('function', {}).get('name')
                    tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                    tool_id = tool_call.get('id')
                    
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    # Use unified registry executor (no hardcoded if/else!)
                    result = await execute_tool_unified(tool_name, tool_args)
                    
                    # Format results for frontend
                    # NOTE: Tools may return "success": True OR "status": "success" - check both!
                    tool_succeeded = result.get('success') or result.get('status') == 'success'
                    raw_results = result.get('results') or result.get('result') or []
                    
                    if tool_succeeded:
                        # Use unified formatter - NO MORE HARDCODED FORMATTING
                        formatted_data = ToolResultFormatter.format_for_frontend(result, tool_name)
                        tool_data.update(formatted_data)
                        
                        if formatted_data:
                            logger.info(f"[AISDK] Sending formatted {tool_name} results to frontend")
                    
                    # Truncate result for LLM context (full data sent to frontend via tool-data event)
                    tool_results.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "content": _truncate_for_llm(result) if isinstance(result, dict) else str(result)[:2000]
                    })
                
                # Send tool data via AI SDK data event
                if tool_data:
                    yield f'd:{{"type":"tool-data","data":{json.dumps(tool_data)}}}\n'
                    await asyncio.sleep(0)
                
                # Get final response with tool results
                llm_messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls
                })
                llm_messages.extend(tool_results)
                
                # Debug: Log what we're sending back to LLM
                logger.info(f"[TOOL DEBUG] Sending {len(tool_results)} tool results to LLM")
                for tr in tool_results:
                    logger.info(f"[TOOL DEBUG] Tool result preview: {str(tr.get('content', ''))[:300]}")
                
                final_response = await llm_manager.generate_response(
                    messages=llm_messages,
                    tools=None
                )
                full_response = final_response.content or ""
                logger.info(f"[TOOL DEBUG] Final LLM response: {full_response[:200]}...")
            else:
                full_response = response.content or ""
            
            # Stream text in chunks for smooth UI (AI SDK format: 0:"text"\n)
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                escaped = json.dumps(chunk)  # Properly escape the text
                yield f'0:{escaped}\n'
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
            # Send usage data
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    "promptTokens": response.usage.get('prompt_tokens', 0),
                    "completionTokens": response.usage.get('completion_tokens', 0),
                    "totalTokens": response.usage.get('total_tokens', 0)
                }
                yield f'd:{{"type":"usage","data":{json.dumps(usage_data)}}}\n'
            
            # Send finish event
            yield f'd:{{"type":"finish","finishReason":"stop"}}\n'
            
            # Save assistant message
            assistant_parts.append({'type': 'text', 'text': full_response})
            self.chat_service.save_message(
                chat_id=chat_id,
                role="assistant",
                parts=assistant_parts
            )
            
            # Store conversation as memory for future retrieval
            latest_user_text = self._extract_latest_user_text(messages)
            if latest_user_text and full_response:
                await self._store_conversation_memory(chat_id, latest_user_text, full_response)
            
        except Exception as e:
            logger.error(f"Error streaming chat response: {e}")
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            yield f'e:{json.dumps({"message": error_msg})}\n'

