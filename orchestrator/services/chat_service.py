"""
Chat Service for PRD-27
========================

Service layer for managing chat sessions, messages, and streaming responses.
Integrates with existing LLM service and tool ecosystem.
"""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import httpx

from services.pandas_ai_service import get_pandasai_service

# Import models - handle both models.py and models/ package
import sys
import importlib.util
from pathlib import Path

# Load models.py directly to get Chat models
models_path = Path(__file__).parent.parent / "models.py"
spec = importlib.util.spec_from_file_location("chat_models", models_path)
chat_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_models)

Chat = chat_models.Chat
Message = chat_models.Message  
Vote = chat_models.Vote
Artifact = chat_models.Artifact

logger = logging.getLogger(__name__)


# Tool implementations for RAG, CodeGraph, and Database
async def search_code_tool(query: str) -> Dict[str, Any]:
    """Search codebase using CodeGraph semantic search"""
    try:
        logger.info(f"[CodeGraph Tool] Searching for: {query}")
        
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
    """Query database using natural language via knowledge source endpoints."""
    try:
        logger.info(f"[Database Tool] Querying: {query}, DB: {database_name}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # List active sources and select match or default
            selected_source: Optional[Dict[str, Any]] = None
            sources: List[Dict[str, Any]] = []

            try:
                sources_response = await client.get(
                    "http://127.0.0.1:8000/api/knowledge/sources/database/",
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

            if not selected_source:
                return {"success": False, "error": "No active database sources available"}

            source_id = selected_source.get("id")
            response = await client.post(
                f"http://127.0.0.1:8000/api/knowledge/sources/database/{source_id}/query",
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


# Tool definitions for LLM
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the codebase for functions, classes, or code patterns. Use this when the user asks about code implementation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what code to find"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search documents and knowledge base. Use this when the user asks about documentation, guides, or architecture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what information to find"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query databases using natural language. Convert user questions into SQL queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language database query"
                    },
                    "database_name": {
                        "type": "string",
                        "description": "Optional: specific database name"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


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
                response = await llm_manager.generate_response(
                    messages=llm_messages,
                    tools=tools or CHAT_TOOLS
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
                        
                        # Execute the tool
                        if tool_name == 'search_code':
                            result = await search_code_tool(query=tool_args.get('query', ''))
                            if result.get('success') and result.get('results'):
                                tool_data['code_snippets'] = [{
                                    'symbol_name': r.get('symbol_name'),
                                    'symbol_type': r.get('symbol_type'),
                                    'file_path': r.get('file_path'),
                                    'line_number': r.get('line_number'),
                                    'code': r.get('code'),
                                    'language': 'python',  # TODO: detect from file
                                    'docstring': r.get('docstring')
                                } for r in result['results']]
                        elif tool_name == 'search_documents':
                            result = await search_documents_tool(query=tool_args.get('query', ''))
                            if result.get('success') and result.get('results'):
                                # Results are already formatted by search_documents_tool
                                tool_data['documents'] = result['results']
                                logger.info(f"[Chat Service] Sending {len(tool_data['documents'])} documents to frontend")
                        elif tool_name == 'query_database':
                            result = await query_database_tool(
                                query=tool_args.get('query', ''),
                                database_name=tool_args.get('database_name'),
                                analysis_prompt=latest_user_question,
                            )
                            if result.get('success'):
                                tool_data['database_results'] = [{
                                    'database': result.get('database'),
                                    'sql': result.get('sql'),
                                    'row_count': result.get('row_count'),
                                    'data': result.get('data'),
                                    'columns': result.get('columns'),
                                    'execution_time_ms': result.get('execution_time_ms'),
                                    'pandas_ai': result.get('pandas_ai')
                                }]
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}
                        
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result)
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
        
        # Add system prompt to guide intelligent tool usage (only on first conversion)
        if len(messages) <= 2:  # First user message
            llm_messages.append({
                "role": "system",
                "content": """You are the Automatos AI Assistant with access to powerful tools. Use them intelligently based on question intent:

🔍 **search_code** - Automatically use when users ask about:
   - Code implementation, functions, classes, methods
   - "How does X work?", "Show me the Y code", "Find Z implementation"
   - Any technical code-related questions

📚 **search_documents** - Automatically use when users ask about:
   - Documentation, guides, architecture, design docs
   - "Explain how to...", "What is the architecture?", "Show me documentation"
   - Conceptual or knowledge questions

💾 **query_database** - Automatically use when users ask about:
   - Data, statistics, counts, lists
   - "How many...", "Show me...", "List all...", "What's the total..."
   - Any data retrieval questions

**Key Rules:**
- NO keywords needed - analyze intent and use the right tool(s)
- Use MULTIPLE tools if the question would benefit from different sources
- Always explain what you found and provide actionable answers
- Cite sources (file paths, document names, table names) in your response"""
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
    
    def _format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """Format chunk as SSE data - compatible with @ai-sdk/react"""
        # Handle different chunk types from LLM manager
        if chunk.get('type') == 'text':
            # AI SDK expects text-delta with id and delta fields
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
            # Custom data type for usage stats
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

