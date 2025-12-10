"""
Chat Service - Consumer for Chat Functionality
===============================================

Consumes:
- shared.llm: LLM providers
- modules.memory: Memory injection/storage
- modules.tools: Tool execution via ToolRouter

Components:
- ChatService: Database operations for chats/messages
- StreamingChatService: SSE streaming orchestrator
"""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from core.models import Chat, Message, Vote
from config import config

# Import from consumer's own modules
from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.tool_router import get_tool_router, get_chat_tools

# Import from modules
from modules.memory.operations import get_memory_injector

logger = logging.getLogger(__name__)


# =============================================================================
# CHAT SERVICE - DATABASE OPERATIONS
# =============================================================================

class ChatService:
    """Service for managing chat sessions and messages in database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_chat(
        self,
        user_id: int,
        title: str,
        visibility: str = "private"
    ) -> Chat:
        """Create a new chat session."""
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
        """Get a chat by ID."""
        try:
            chat_uuid = uuid.UUID(chat_id)
            return self.db.query(Chat).filter(Chat.id == chat_uuid).first()
        except (ValueError, AttributeError):
            logger.error(f"Invalid chat_id format: {chat_id}")
            return None
    
    def get_chat_history(
        self,
        user_id: int,
        limit: int = 20,
        starting_after: Optional[datetime] = None
    ) -> List[Chat]:
        """Get chat history for a user."""
        query = self.db.query(Chat).filter(Chat.user_id == user_id)
        if starting_after:
            query = query.filter(Chat.created_at < starting_after)
        return query.order_by(desc(Chat.created_at)).limit(limit).all()
    
    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat title."""
        chat = self.get_chat(chat_id)
        if chat:
            chat.title = title
            chat.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Updated chat {chat_id} title: {title}")
            return True
        return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages."""
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
        """Save a message to the database."""
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
        """Get all messages for a chat."""
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
        """Vote on a message."""
        try:
            chat_uuid = uuid.UUID(chat_id)
            message_uuid = uuid.UUID(message_id)
        except ValueError as e:
            logger.error(f"Invalid UUID format: {e}")
            return False
        
        vote = self.db.query(Vote).filter(
            and_(Vote.chat_id == chat_uuid, Vote.message_id == message_uuid)
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


# =============================================================================
# STREAMING CHAT SERVICE - RESPONSE ORCHESTRATION
# =============================================================================

class StreamingChatService:
    """
    Service for streaming chat responses.
    Thin orchestrator consuming modules.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.chat_service = ChatService(db)
        self.prompt_analyzer = get_prompt_analyzer()
        self.memory_injector = get_memory_injector()
        self.tool_router = get_tool_router()
        self.streaming_handler = get_streaming_handler()
    
    async def stream_response_aisdk(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        selected_model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response using AI SDK Data Stream format.
        
        Format: 0:"text"\n for text, d:{json}\n for data, e:{json}\n for errors
        """
        # Import from core.llm
        from core.llm import create_llm_manager
        import asyncio
        
        # Get tools from modules.tools if not provided
        if tools is None:
            tools = get_chat_tools()
        
        try:
            # Send chat_id to frontend
            yield self.streaming_handler.format_aisdk_chat_id(chat_id)
            await asyncio.sleep(0)
            
            # Determine provider/model from selection
            provider, model = self._parse_model_selection(selected_model)
            
            # Create LLM manager via shared.llm
            if provider and model:
                logger.info(f"Using user-selected model: {provider}/{model}")
                llm_manager = create_llm_manager(service_name="chatbot", provider=provider, model=model)
            else:
                llm_manager = create_llm_manager(service_name="chatbot")
            
            # Convert messages to LLM format
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(messages)
            assistant_parts = []
            full_response = ""
            tool_data = {}
            
            # Inject memory context via modules.memory
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            memory_context = await self.memory_injector.retrieve_relevant_memories(chat_id, latest_text)
            if memory_context:
                logger.info(f"[Memory] Injecting {len(memory_context)} chars")
                llm_messages.insert(1, self.memory_injector.build_memory_injection_message(memory_context))
            
            # Determine tool usage
            is_simple = self.prompt_analyzer.is_simple_message(latest_text)
            supports_native_tools = provider in ['openai', 'anthropic', 'grok'] if provider else False
            use_tools = None if is_simple else tools if supports_native_tools else None
            
            # Pre-trigger tools for models without native tool calling
            if not is_simple and not supports_native_tools:
                detected_tools = self.prompt_analyzer.detect_explicit_tool_requests(latest_text)
                if detected_tools:
                    logger.info(f"Pre-triggering tools: {detected_tools}")
                    async for chunk in self._execute_pretriggered_tools(detected_tools, latest_text, llm_messages, tool_data):
                        yield chunk
                        await asyncio.sleep(0)
            
            # Generate response via shared.llm
            response = await llm_manager.generate_response(messages=llm_messages, tools=use_tools)
            
            # Handle tool calls from LLM (supports multi-turn)
            if response.tool_calls:
                full_response, tool_data = await self._handle_tool_calls(
                    response, llm_manager, llm_messages, tool_data,
                    tools=use_tools,  # Allow multi-turn tool calls
                    max_iterations=5  # Max 5 rounds of tool use
                )
                if tool_data:
                    yield self.streaming_handler.format_aisdk_tool_data(tool_data)
                    await asyncio.sleep(0)
            else:
                full_response = response.content or ""
            
            # Stream text response
            async for chunk in self.streaming_handler.stream_text_aisdk(full_response):
                yield chunk
            
            # Send usage data
            if hasattr(response, 'usage') and response.usage:
                yield self.streaming_handler.format_aisdk_usage(
                    response.usage.get('prompt_tokens', 0),
                    response.usage.get('completion_tokens', 0),
                    response.usage.get('total_tokens', 0)
                )
            
            # Send finish event
            yield self.streaming_handler.format_aisdk_finish()
            
            # Save assistant message
            assistant_parts.append({'type': 'text', 'text': full_response})
            self.chat_service.save_message(chat_id=chat_id, role="assistant", parts=assistant_parts)
            
            # Store memory via modules.memory
            if latest_text and full_response:
                await self.memory_injector.store_conversation_memory(chat_id, latest_text, full_response)
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            import traceback
            traceback.print_exc()
            yield self.streaming_handler.format_aisdk_error(str(e))
    
    async def stream_response(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response using legacy SSE format.
        """
        from core.llm import create_llm_manager
        
        # Get tools from modules.tools if not provided
        if tools is None:
            tools = get_chat_tools()
        
        try:
            llm_manager = create_llm_manager(service_name="chatbot")
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(messages)
            assistant_parts = []
            
            # Check for streaming support
            if hasattr(llm_manager, 'generate_response_stream'):
                async for chunk in llm_manager.generate_response_stream(messages=llm_messages, tools=tools):
                    yield self.streaming_handler.format_sse_chunk(chunk)
                    if chunk.get('type') == 'text':
                        assistant_parts.append({'type': 'text', 'text': chunk.get('text', '')})
            else:
                # Fallback: generate complete response
                latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
                is_simple = self.prompt_analyzer.is_simple_message(latest_text)
                use_tools = None if is_simple else tools
                
                response = await llm_manager.generate_response(messages=llm_messages, tools=use_tools)
                
                # Handle tool calls
                if response.tool_calls:
                    tool_data = {}
                    tool_results = []
                    
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('function', {}).get('name')
                        tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                        tool_id = tool_call.get('id')
                        
                        result = await self.tool_router.execute_and_format(tool_name, tool_args)
                        if result['success']:
                            tool_data.update(result['frontend_data'])
                        
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result['llm_context']
                        })
                    
                    if tool_data:
                        yield self.streaming_handler.format_sse_tool_data(tool_data)
                    
                    llm_messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls
                    })
                    llm_messages.extend(tool_results)
                    
                    final_response = await llm_manager.generate_response(messages=llm_messages, tools=None)
                    response_text = final_response.content or ""
                else:
                    response_text = response.content or ""
                
                # Stream text
                message_id = str(uuid.uuid4())
                async for chunk in self.streaming_handler.stream_text_legacy(response_text, message_id):
                    yield chunk
                
                assistant_parts.append({'type': 'text', 'text': response_text})
            
            # Save message
            if assistant_parts:
                self.chat_service.save_message(chat_id=chat_id, role='assistant', parts=assistant_parts)
            
            yield self.streaming_handler.format_sse_done()
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield self.streaming_handler.format_sse_error(str(e))
    
    def _parse_model_selection(self, selected_model: Optional[str]) -> tuple:
        """Parse model string to get provider and model."""
        if not selected_model:
            return None, None
        
        model = selected_model
        model_lower = selected_model.lower()
        
        if '/' in selected_model:
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
            provider = None
        
        return provider, model
    
    async def _execute_pretriggered_tools(
        self,
        detected_tools: List[str],
        query: str,
        llm_messages: List[Dict],
        tool_data: Dict
    ) -> AsyncGenerator[str, None]:
        """Execute pre-triggered tools and inject results."""
        for tool_name in detected_tools:
            logger.info(f"Pre-triggering {tool_name}")
            result = await self.tool_router.execute_and_format(tool_name, {"query": query})
            
            if result['success']:
                tool_data.update(result['frontend_data'])
                
                # Build context injection
                context_msg = self.tool_router.build_tool_context_message(tool_name, result)
                if context_msg:
                    llm_messages.insert(1, context_msg)
                
                # Send tool data to frontend
                if result['frontend_data']:
                    yield self.streaming_handler.format_aisdk_tool_data(result['frontend_data'])
    
    async def _handle_tool_calls(
        self,
        response,
        llm_manager,
        llm_messages: List[Dict],
        tool_data: Dict,
        tools: Optional[List[Any]] = None,
        max_iterations: int = 5
    ) -> tuple:
        """
        Handle tool calls from LLM response with multi-turn support.
        
        Allows the LLM to call tools multiple times (ReAct pattern).
        Stops when:
        - LLM generates text without tool calls
        - Max iterations reached
        """
        import asyncio
        
        iteration = 0
        current_response = response
        
        while current_response.tool_calls and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}: LLM requested {len(current_response.tool_calls)} tool calls")
            
            tool_results = []
            
            # Execute tools in parallel when possible
            async def execute_single_tool(tool_call):
                tool_name = tool_call.get('function', {}).get('name')
                tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                tool_id = tool_call.get('id')
                
                logger.info(f"Executing tool: {tool_name}")
                result = await self.tool_router.execute_and_format(tool_name, tool_args)
                
                return {
                    "tool_call_id": tool_id,
                    "role": "tool", 
                    "content": result['llm_context'],
                    "frontend_data": result.get('frontend_data', {}),
                    "success": result['success']
                }
            
            # Execute all tools in parallel
            results = await asyncio.gather(*[
                execute_single_tool(tc) for tc in current_response.tool_calls
            ])
            
            # Collect results
            for r in results:
                if r['success']:
                    tool_data.update(r['frontend_data'])
                tool_results.append({
                    "tool_call_id": r['tool_call_id'],
                    "role": "tool",
                    "content": r['content']
                })
            
            # Add assistant message with tool calls and results
            llm_messages.append({
                "role": "assistant",
                "content": current_response.content or "",
                "tool_calls": current_response.tool_calls
            })
            llm_messages.extend(tool_results)
            
            # Get next response - allow more tool calls if needed
            # Only pass tools if we haven't hit max iterations
            allow_more_tools = iteration < max_iterations - 1
            current_response = await llm_manager.generate_response(
                messages=llm_messages,
                tools=tools if allow_more_tools else None
            )
            
            logger.info(f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}")
        
        if iteration >= max_iterations and current_response.tool_calls:
            logger.warning(f"Hit max tool iterations ({max_iterations}). Forcing final response.")
        
        return current_response.content or "", tool_data

