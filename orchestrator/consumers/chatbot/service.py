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
import re
import hashlib
from typing import List, Optional, Dict, Any, AsyncGenerator, Set, Tuple
from datetime import datetime, timedelta, timezone
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, or_
from difflib import SequenceMatcher

from core.models import Chat, Message, Vote, Workspace
from core.services.image_store import get_image_store
from config import config

# Import from consumer's own modules
from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.tool_router import get_tool_router, get_chat_tools

# Import from modules
from modules.memory.operations import get_memory_injector

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL LOOP PREVENTION UTILITIES
# =============================================================================

def _normalize_query(query: str) -> str:
    """Normalize a search query for deduplication comparison."""
    if not query:
        return ""
    # Lowercase, remove extra whitespace, strip punctuation
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    normalized = ' '.join(normalized.split())
    return normalized


def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    """
    Check if two queries are semantically similar using string similarity.
    Returns True if similarity ratio >= threshold.
    """
    norm1 = _normalize_query(query1)
    norm2 = _normalize_query(query2)

    if not norm1 or not norm2:
        return False

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def _extract_query_from_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """Extract the search/query parameter from tool arguments."""
    # Common query parameter names
    query_keys = ['query', 'search_query', 'q', 'text', 'question', 'prompt']

    for key in query_keys:
        if key in tool_args and isinstance(tool_args[key], str):
            return tool_args[key]

    return None


class ToolExecutionTracker:
    """
    Tracks tool executions within a conversation turn to prevent looping.
    Implements:
    - Exact deduplication (same tool + same args)
    - Semantic deduplication for search tools (similar queries)
    - Per-tool retry limits
    """

    # Tools that use search queries and should have semantic deduplication
    SEARCH_TOOLS = {
        'search_knowledge', 'semantic_search', 'search_codebase',
        'search_tables', 'search_images', 'search_formulas',
        'search_multimodal', 'smart_query_database', 'query_database'
    }

    # Max retries per tool type
    TOOL_RETRY_LIMITS = {
        'composio_execute': 2,      # Composio gets 2 total attempts
        'search_knowledge': 2,
        'semantic_search': 2,
        'search_codebase': 2,
        'smart_query_database': 2,
        'query_database': 2,
        'list_directory': 2,
        'read_file': 3,
        'write_file': 2,
        'default': 3                # Default limit for unlisted tools
    }

    def __init__(self):
        # Track exact executions: set of (tool_name, args_hash)
        self.exact_executions: Set[Tuple[str, str]] = set()
        # Track queries per search tool for semantic dedup: {tool_name: [queries]}
        self.search_queries: Dict[str, List[str]] = {}
        # Track execution counts per tool: {tool_name: count}
        self.tool_counts: Dict[str, int] = {}

    def _hash_args(self, tool_args: Dict[str, Any]) -> str:
        """Create a hash of tool arguments for exact matching."""
        return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()

    def should_skip_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if a tool execution should be skipped.

        Returns:
            (should_skip, reason) - True if should skip, with explanation
        """
        # Check retry limit
        current_count = self.tool_counts.get(tool_name, 0)
        limit = self.TOOL_RETRY_LIMITS.get(tool_name, self.TOOL_RETRY_LIMITS['default'])

        if current_count >= limit:
            return True, f"Tool '{tool_name}' has reached its execution limit ({limit}) for this turn"

        # Check exact duplicate
        args_hash = self._hash_args(tool_args)
        exec_key = (tool_name, args_hash)

        if exec_key in self.exact_executions:
            return True, f"Tool '{tool_name}' was already executed with identical parameters"

        # Check semantic similarity for search tools
        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                previous_queries = self.search_queries.get(tool_name, [])
                for prev_query in previous_queries:
                    if _queries_are_similar(query, prev_query):
                        return True, f"Tool '{tool_name}' was already executed with a similar query"

        return False, ""

    def record_execution(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Record that a tool was executed."""
        # Record exact execution
        args_hash = self._hash_args(tool_args)
        self.exact_executions.add((tool_name, args_hash))

        # Increment count
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1

        # Record query for search tools
        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                if tool_name not in self.search_queries:
                    self.search_queries[tool_name] = []
                self.search_queries[tool_name].append(query)

    def get_execution_count(self, tool_name: str) -> int:
        """Get the number of times a tool has been executed."""
        return self.tool_counts.get(tool_name, 0)

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
        """Update chat title, handling unique constraint violations."""
        from sqlalchemy.exc import IntegrityError
        
        chat = self.get_chat(chat_id)
        if not chat:
            return False
        
        # If title is the same, no update needed
        if chat.title == title:
            return True
        
        # Try to update, handle unique constraint violation
        try:
            chat.title = title
            chat.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Updated chat {chat_id} title: {title}")
            return True
        except IntegrityError as e:
            # Handle unique constraint: make title unique by appending counter
            self.db.rollback()
            base_title = title
            counter = 1
            while True:
                unique_title = f"{base_title} ({counter})"
                # Check if this title already exists for this user
                existing = self.db.query(Chat).filter(
                    Chat.user_id == chat.user_id,
                    Chat.title == unique_title,
                    Chat.id != chat.id
                ).first()
                
                if not existing:
                    try:
                        chat.title = unique_title
                        chat.updated_at = datetime.utcnow()
                        self.db.commit()
                        logger.info(f"Updated chat {chat_id} title to unique: {unique_title}")
                        return True
                    except IntegrityError:
                        self.db.rollback()
                        counter += 1
                        continue
                counter += 1
                
                # Safety: prevent infinite loop
                if counter > 100:
                    logger.error(f"Failed to generate unique title for chat {chat_id} after 100 attempts")
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
        attachments: Optional[List[Dict[str, Any]]] = None,
        workspace_id: Optional[str] = None
    ) -> Message:
        """Save a message to the database."""
        try:
            chat_uuid = uuid.UUID(chat_id)
        except ValueError:
            raise ValueError(f"Invalid chat_id format: {chat_id}")
        
        if not workspace_id:
            raise ValueError("workspace_id is required to save messages")
        
        if isinstance(workspace_id, str):
            try:
                workspace_id = uuid.UUID(workspace_id)
            except ValueError:
                raise ValueError("Invalid workspace_id format")

        # Safety net: ensure the referenced workspace exists.
        # This prevents FK crashes when the request falls back to the dev workspace UUID.
        dev_fallback = UUID("00000000-0000-0000-0000-000000000001")
        existing_ws = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not existing_ws:
            if workspace_id == dev_fallback:
                ws = Workspace(
                    id=workspace_id,
                    name="Dev Workspace",
                    slug="dev",
                    plan="starter",
                    plan_limits={},
                    settings={},
                    is_personal=True,
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                self.db.add(ws)
                self.db.commit()
            else:
                raise ValueError(f"workspace_id does not exist: {workspace_id}")
        
        message = Message(
            id=uuid.uuid4(),
            chat_id=chat_uuid,
            workspace_id=workspace_id,
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
# =============================================================================
# IMAGE UPLOAD HELPER — Replace inline base64 images with S3 URLs
# =============================================================================

_BASE64_IMG_RE = re.compile(
    r'!\[([^\]]*)\]\((data:image/(jpeg|jpg|png|gif|webp);base64,([A-Za-z0-9+/=\s]+))\)'
)


async def _upload_inline_images(text: str, workspace_id: str = None) -> str:
    """Find base64 image markdown in text, upload to S3, replace with URLs."""
    matches = list(_BASE64_IMG_RE.finditer(text))
    if not matches:
        return text

    store = get_image_store()
    result = text
    for match in reversed(matches):  # reverse to preserve offsets
        alt = match.group(1)
        mime_type = f"image/{match.group(3)}"
        b64_data = match.group(4).replace("\n", "").replace(" ", "")
        try:
            image_id = await store.save_image(b64_data, mime_type, workspace_id)
            url = f"/api/generated-images/{image_id}"
            replacement = f"![{alt}]({url})"
            result = result[:match.start()] + replacement + result[match.end():]
            logger.info(f"Replaced inline base64 image with {url}")
        except Exception as e:
            logger.warning(f"Failed to upload inline image to S3: {e}")
    return result


# =============================================================================
# STREAMING CHAT SERVICE - RESPONSE ORCHESTRATION
# =============================================================================

class StreamingChatService:
    """
    Service for streaming chat responses.
    Thin orchestrator consuming modules.
    """
    
    def __init__(self, db: Session, workspace_id: Optional[str] = None):
        self.db = db
        self.chat_service = ChatService(db)
        self.prompt_analyzer = get_prompt_analyzer()
        self.memory_injector = get_memory_injector()
        self.tool_router = get_tool_router()
        self.streaming_handler = get_streaming_handler()
        self.workspace_id = workspace_id
        
        # PRD: Unified Agent-Chat System - Initialize AgentFactory
        from modules.agents.factory.agent_factory import AgentFactory
        self.agent_factory = AgentFactory(db_session=db)
        logger.info("StreamingChatService initialized with AgentFactory integration")

    def _resolve_workspace_id(self, agent_id: int) -> Optional[str]:
        logger.info(f"[chat] resolve_workspace_id current={self.workspace_id} agent={agent_id}")
        if self.workspace_id:
            return self.workspace_id
        try:
            from core.models import Agent as AgentModel
            agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent_row:
                logger.info(f"[chat] agent workspace_id={agent_row.workspace_id} agent={agent_id}")
            if agent_row and agent_row.workspace_id:
                self.workspace_id = agent_row.workspace_id
                logger.info(f"Resolved workspace_id from agent {agent_id}")
        except Exception as exc:
            logger.warning(f"Failed to resolve workspace_id for agent {agent_id}: {exc}")
        return self.workspace_id
    
    async def stream_response_aisdk(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        selected_model: Optional[str] = None,
        agent_id: int = 1
    ) -> AsyncGenerator[str, None]:
        """
        DEPRECATED: This method is no longer called from the chat API.
        All chat now flows through stream_response_with_agent() which uses
        SmartChatOrchestrator for personality, memory, and tool orchestration.

        Kept temporarily for backward compatibility with any direct callers.
        Will be removed in a future cleanup pass.
        """
        import warnings
        warnings.warn(
            "stream_response_aisdk is deprecated. Use stream_response_with_agent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from core.llm
        from core.llm import create_llm_manager
        import asyncio
        
        try:
            # Send chat_id to frontend
            yield self.streaming_handler.format_aisdk_chat_id(chat_id)
            await asyncio.sleep(0)

            # Ensure workspace_id for Composio tools permissions
            self._resolve_workspace_id(agent_id)
            
            # Get tools from modules.tools with permissions for this workspace
            if tools is None:
                tools = get_chat_tools(agent_id=agent_id, workspace_id=self.workspace_id)
            
            # Determine provider/model from selection
            provider, model = self._parse_model_selection(selected_model)
            
            # Create LLM manager via shared.llm
            if provider and model:
                logger.info(f"Using user-selected model: {provider}/{model}")
                llm_manager = create_llm_manager(service_name="chatbot", provider=provider, model=model)
            else:
                llm_manager = create_llm_manager(service_name="chatbot")
            
            # Optionally ignore prior context for a fresh run
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            fresh_start = self.prompt_analyzer.is_fresh_start_request(latest_text)
            if fresh_start:
                messages = [m for m in messages if m.get("role") == "user"][-1:]

            # Ensure workspace_id for Composio tools
            self._resolve_workspace_id(agent_id)

            # Determine tool usage
            is_simple = self.prompt_analyzer.is_simple_message(latest_text)
            supports_native_tools = provider in ['openai', 'anthropic', 'grok', 'openrouter', 'google'] if provider else False
            
            # --- TOOL FILTERING (PRD Refinement) ---
            # To prevent context overload (600+ tools), we filter to the top N relevant tools.
            use_tools = None
            if not is_simple and tools and supports_native_tools:
                # 1. Identify Core Tools (always included)
                core_tool_names = {
                    "search_knowledge", "semantic_search", "search_codebase",
                    "query_database", "smart_query_database",
                    "read_file", "write_file", "switch_context",
                    "search_tables", "search_images", "search_formulas", "search_multimodal",
                    "generate_document", "composio_execute",
                }
                
                # 2. Get relevant tools via ranking
                ranked_candidates = self.prompt_analyzer.rank_tools_for_query(
                    latest_text, 
                    tools, 
                    max_tools=25
                )
                
                # 3. Build final allowed list
                filtered_tools = []
                included_names = set()
                
                # Add core tools first
                for tool in tools:
                    t_name = tool.get("function", {}).get("name")
                    if t_name in core_tool_names:
                        filtered_tools.append(tool)
                        included_names.add(t_name)
                
                # Add top ranked tools
                for candidate in ranked_candidates:
                    c_name = candidate.get("name")
                    if c_name not in included_names:
                        # Find the full tool definition
                        full_tool = next((t for t in tools if t.get("function", {}).get("name") == c_name), None)
                        if full_tool:
                            filtered_tools.append(full_tool)
                            included_names.add(c_name)
                
                use_tools = filtered_tools
                logger.info(f"Filtering tools: {len(tools)} -> {len(use_tools)} relevant tools")
            
            # Use filtered tools (if applicable) for LLM context generation
            context_tools = use_tools if use_tools is not None else tools

            # Convert messages to LLM format (include tool names for stronger tool routing)
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages,
                available_tools=context_tools
            )
            assistant_parts = []
            full_response = ""
            tool_data = {}
            documents_tool_used = False
            
            # Inject memory context via modules.memory
            memory_context = None
            # Optimized: Check if we should even try to retrieve memories (save tokens/time)
            should_retrieve = await self.memory_injector.should_retrieve_memories(latest_text, chat_id)
            
            if not fresh_start and should_retrieve:
                memory_context = await self.memory_injector.retrieve_relevant_memories(
                    chat_id,
                    latest_text,
                    workspace_id=str(self.workspace_id) if self.workspace_id else None,
                    agent_id=agent_id
                )
                if memory_context:
                    logger.info(f"[Memory] Injecting {len(memory_context)} chars")
                    llm_messages.insert(1, self.memory_injector.build_memory_injection_message(memory_context))

            # Add dynamic tool candidates (hint) - only suggest, don't force
            # NOTE: Removed "Tool candidates for this request" phrasing which was triggering
            # force_tool_choice=required in OpenAI client. Tools should be optional.
            if context_tools and latest_text:
                candidates = self.prompt_analyzer.rank_tools_for_query(latest_text, context_tools)
                # Only hint at tools if there's a high-confidence match (score > 2)
                high_confidence_candidates = [c for c in candidates if c.get("score", 0) > 2]
                if high_confidence_candidates:
                    candidate_names = ", ".join([c["name"] for c in high_confidence_candidates[:3] if c.get("name")])
                    if candidate_names:
                        insert_at = 2 if memory_context else 1
                        llm_messages.insert(
                            insert_at,
                            {
                                "role": "system",
                                "content": (
                                    f"Available tools: {candidate_names}. "
                                    "When the user asks you to perform an action (fetch data, send messages, "
                                    "list items, etc.), call the appropriate tool immediately. "
                                    "For pure conversation or memory questions, respond naturally."
                                )
                            }
                        )

            # Composio per-action tools (primary) or hint fallback
            _composio_result = None
            try:
                if latest_text and agent_id and self.workspace_id:
                    from modules.tools.services.composio_tool_service import ComposioToolService

                    _composio_svc = ComposioToolService(self.db)
                    _composio_result = _composio_svc.get_tools_for_step(
                        agent_id=agent_id,
                        workspace_id=self.workspace_id,
                        task_prompt=latest_text,
                    )
                    insert_at = 2 if memory_context else 1
                    if _composio_result and _composio_result.tools:
                        # Strip composio_execute, add per-action tools
                        if use_tools:
                            use_tools = [
                                t for t in use_tools
                                if t.get("function", {}).get("name") != "composio_execute"
                            ] + _composio_result.tools
                        elif context_tools:
                            use_tools = [
                                t for t in context_tools
                                if t.get("function", {}).get("name") != "composio_execute"
                            ] + _composio_result.tools
                        else:
                            use_tools = _composio_result.tools
                        from api.recipe_executor import _composio_scope_message
                        llm_messages.insert(insert_at, {
                            "role": "system",
                            "content": _composio_scope_message(_composio_result.app_names),
                        })
                        logger.info(
                            f"[ComposioToolService] strategy={_composio_result.strategy} "
                            f"actions={len(_composio_result.action_set)} search_ms={_composio_result.search_ms}"
                        )
                    else:
                        # Fallback: ComposioHintService (composio_execute mega-tool)
                        from modules.tools.services.composio_hint_service import ComposioHintService

                        hint_service = ComposioHintService(self.db)
                        hint_result = hint_service.build_hints(
                            agent_id=agent_id,
                            prompt=latest_text,
                            workspace_id=self.workspace_id,
                        )
                        if hint_result.hint_lines:
                            llm_messages.insert(insert_at, {"role": "system", "content": "\n".join(hint_result.hint_lines)})
                            logger.info(f"[Composio Hints fallback] strategy={hint_result.strategy_used} apps={hint_result.allowed_apps} matches={len(hint_result.matched_actions)}")
            except Exception as exc:
                logger.warning(f"Composio tool injection failed: {exc}", exc_info=True)

            # Explicit tool call bypass (e.g., "Use tool X with params {...}")
            explicit_call = self.prompt_analyzer.parse_explicit_tool_call(latest_text)
            if explicit_call and tools:
                available_tool_names = {
                    t.get("function", {}).get("name")
                    for t in tools
                    if isinstance(t, dict)
                }
                tool_name = explicit_call["tool_name"]
                tool_args = explicit_call["tool_args"]
                parse_error = explicit_call["parse_error"]

                if tool_name in available_tool_names:
                    if parse_error:
                        full_response = f"Invalid tool params for {tool_name}: {parse_error}"
                    else:
                        import time
                        tool_call_id = str(uuid.uuid4())
                        start_time = time.time()

                        yield self.streaming_handler.format_aisdk_tool_start(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_input=tool_args,
                        )
                        await asyncio.sleep(0)

                        result = await self.tool_router.execute_and_format(
                            tool_name,
                            tool_args,
                            agent_id=agent_id,
                            workspace_id=self.workspace_id,
                            original_intent=latest_text,
                        )

                        yield self.streaming_handler.format_aisdk_tool_end(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            success=result.get("success", False),
                            error=(result.get("raw_result", {}) or {}).get("error"),
                            duration_ms=int((time.time() - start_time) * 1000),
                        )
                        await asyncio.sleep(0)

                        if result.get("frontend_data"):
                            tool_data.update(result["frontend_data"])
                            yield self.streaming_handler.format_aisdk_tool_data(result["frontend_data"])
                            await asyncio.sleep(0)

                        if result.get("success"):
                            full_response = f"✅ Ran {tool_name} successfully."
                        else:
                            error_msg = (result.get("raw_result", {}) or {}).get("error", "Unknown error")
                            full_response = f"❌ {tool_name} failed: {error_msg}"

                    # Stream text response
                    async for chunk in self.streaming_handler.stream_text_aisdk(full_response):
                        yield chunk

                    # Send finish event
                    yield self.streaming_handler.format_aisdk_finish()

                    # Save assistant message
                    assistant_parts.append({'type': 'text', 'text': full_response})
                    self.chat_service.save_message(
                        chat_id=chat_id,
                        role="assistant",
                        parts=assistant_parts,
                        workspace_id=self.workspace_id
                    )

                    # Store memory
                    if latest_text and full_response:
                        await self.memory_injector.store_conversation_memory(
                            chat_id,
                            latest_text,
                            full_response,
                            workspace_id=str(self.workspace_id) if self.workspace_id else None,
                            agent_id=agent_id
                        )

                    # FutureAGI live traffic eval (fire-and-forget)
                    if latest_text and full_response:
                        try:
                            from core.services.futureagi_service import futureagi_service
                            if futureagi_service.is_available:
                                asyncio.create_task(
                                    futureagi_service.eval_live_traffic(
                                        input_text=latest_text,
                                        output_text=full_response,
                                    )
                                )
                        except Exception:
                            pass  # Never block chat for eval

                    return
            
            # Pre-trigger tools for models without native tool calling
            if not is_simple and not supports_native_tools:
                detected_tools = self.prompt_analyzer.detect_explicit_tool_requests(latest_text)
                if detected_tools:
                    logger.info(f"Pre-triggering tools: {detected_tools}")
                    async for chunk in self._execute_pretriggered_tools(
                        detected_tools,
                        latest_text,
                        llm_messages,
                        tool_data,
                        agent_id
                    ):
                        yield chunk
                        await asyncio.sleep(0)
            
            # Generate response via shared.llm
            response = await llm_manager.generate_response(messages=llm_messages, tools=use_tools)
            
            # DEBUG: Log LLM response to understand why tools aren't being called
            logger.info(f"🔍 LLM Response - has_tool_calls: {bool(response.tool_calls)}, content_length: {len(response.content or '')}, finish_reason: {getattr(response, 'finish_reason', 'unknown')}")
            if response.tool_calls:
                logger.info(f"✅ LLM requested {len(response.tool_calls)} tool calls")
            elif use_tools:
                logger.warning(f"⚠️ LLM did NOT call tools despite {len(use_tools)} tools being available. Response: {response.content[:200] if response.content else 'No content'}")
            
            # Handle tool calls from LLM (supports multi-turn)
            if response.tool_calls:
                # Emit tool lifecycle events + stream tool-data incrementally
                import time

                max_iterations = 5
                iteration = 0
                current_response = response
                sent_tool_data = False
                # Allow one recovery if the model calls composio_execute with missing args.
                composio_invalid_parameters_retry_budget = 1

                # Use enhanced tool execution tracker for loop prevention
                tool_tracker = ToolExecutionTracker()

                while current_response.tool_calls and iteration < max_iterations:
                    iteration += 1
                    logger.info(
                        f"Tool iteration {iteration}: LLM requested {len(current_response.tool_calls)} tool calls"
                    )

                    # Emit tool-start for all requested tools
                    start_times: Dict[str, float] = {}
                    tool_calls_prepared = []

                    for tool_call in current_response.tool_calls:
                        tool_name = tool_call.get("function", {}).get("name") or "unknown_tool"
                        tool_args_raw = tool_call.get("function", {}).get("arguments", "{}")
                        tool_id = tool_call.get("id") or str(uuid.uuid4())

                        try:
                            tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else (tool_args_raw or {})
                        except Exception:
                            tool_args = {"raw": tool_args_raw}
                        
                        start_times[tool_id] = time.time()

                        # Enhanced deduplication check with semantic similarity
                        should_skip, skip_reason = tool_tracker.should_skip_execution(tool_name, tool_args)
                        
                        # Determine if this skip should be silent (hidden from UI)
                        # We silently hide "identical parameters" duplicates to prevent "printing twice"
                        silent_suppression = False
                        if should_skip and "identical parameters" in skip_reason:
                            silent_suppression = True
                            logger.info(f"🚫 Silently suppressing duplicate tool: {tool_name}")
                        elif should_skip:
                            logger.warning(f"⚠️ Tool loop prevention: {skip_reason}")
                        else:
                            # Record the execution before it happens
                            tool_tracker.record_execution(tool_name, tool_args)

                        tool_calls_prepared.append((tool_id, tool_name, tool_args, should_skip, skip_reason, silent_suppression))

                        # Only emit start event if NOT silently suppressed
                        if not silent_suppression:
                            yield self.streaming_handler.format_aisdk_tool_start(
                                tool_call_id=tool_id,
                                tool_name=tool_name,
                                tool_input=tool_args,
                            )
                        await asyncio.sleep(0)

                    # Execute all tools in parallel
                    async def execute_single_tool(tool_id: str, tool_name: str, tool_args: Dict[str, Any], should_skip: bool, skip_reason: str, silent: bool):
                        if should_skip:
                            if not silent:
                                logger.warning(f"⚠️ Skipping tool execution: {tool_name} - {skip_reason}")
                            
                            return {
                                "tool_call_id": tool_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "role": "tool",
                                "content": f"Error: {skip_reason}. Do not call this tool again with similar parameters.",
                                "frontend_data": {},
                                "success": False,
                                "error": skip_reason,
                                "silent": silent
                            }

                        # ------------------------------------------------------------------
                        # Generic safety guard for Composio:
                        # Do NOT allow destructive actions unless the user explicitly asked.
                        # This prevents "archive/delete/clear" style side-effects for requests
                        # like "send a message".
                        # ------------------------------------------------------------------
                        if tool_name == "composio_execute" and isinstance(tool_args, dict):
                            action = str(tool_args.get("action") or "").upper().strip()
                            if action:
                                user_text = (latest_text or "").lower()
                                wants_destructive = bool(re.search(r"\b(archive|delete|remove|revoke|clear|close|disable)\b", user_text))
                                is_destructive = bool(re.search(r"(ARCHIVE|DELETE|REMOVE|REVOKE|CLEAR|CLOSE|DISABLE)", action))
                                is_messaging = bool(re.search(r"\b(send|message|post|dm|chat)\b", user_text))
                                if is_messaging and is_destructive and not wants_destructive:
                                    return {
                                        "tool_call_id": tool_id,
                                        "tool_name": tool_name,
                                        "tool_args": tool_args,
                                        "role": "tool",
                                        "content": (
                                            f"Refused to execute destructive action '{action}' for a messaging request. "
                                            "Pick a non-destructive mapped action that sends a message."
                                        ),
                                        "frontend_data": {},
                                        "success": False,
                                        "error": "Refused destructive action for messaging intent",
                                        "silent": False
                                    }

                        # Direct Composio action execution (per-action tools)
                        _is_composio = (
                            _composio_result and _composio_result.entity_id and (
                                tool_name in _composio_result.action_set
                                or any(tool_name.startswith(f"{app}_") for app in (_composio_result.app_names or []))
                            )
                        )
                        if _is_composio:
                            try:
                                from modules.tools.services.composio_tool_service import ComposioToolService
                                _exec_svc = ComposioToolService(self.db)
                                exec_result = _exec_svc.execute_action(
                                    action_name=tool_name,
                                    params=tool_args,
                                    entity_id=_composio_result.entity_id,
                                )
                                success = exec_result.get("success", False)
                                data = exec_result.get("data")
                                error = exec_result.get("error")
                                if success:
                                    content = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or "")
                                else:
                                    content = f"Error executing {tool_name}: {error or 'unknown error'}"
                                logger.info(f"[Composio direct] {tool_name}: success={success}")
                            except Exception as exc:
                                content = f"Error executing {tool_name}: {exc}"
                                success = False
                                error = str(exc)
                                logger.error(f"[Composio direct] {tool_name} exception: {exc}", exc_info=True)

                            if len(content) > 4000:
                                content = content[:4000] + "\n... (truncated)"

                            return {
                                "tool_call_id": tool_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "role": "tool",
                                "content": content,
                                "frontend_data": {},
                                "success": success,
                                "error": error if not success else None,
                                "silent": False,
                            }

                        logger.info(f"Executing tool: {tool_name}")
                        result = await self.tool_router.execute_and_format(
                            tool_name,
                            tool_args,
                            agent_id=agent_id,
                            workspace_id=self.workspace_id,
                            original_intent=latest_text,
                        )
                        return {
                            "tool_call_id": tool_id,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "role": "tool",
                            "content": result.get("llm_context", ""),
                            "frontend_data": result.get("frontend_data", {}),
                            "success": bool(result.get("success")),
                            "error": (result.get("raw_result", {}) or {}).get("error"),
                            "silent": False
                        }

                    results = await asyncio.gather(*[
                        execute_single_tool(tool_id, tool_name, tool_args, should_skip, skip_reason, silent)
                        for (tool_id, tool_name, tool_args, should_skip, skip_reason, silent) in tool_calls_prepared
                    ])

                    tool_results = []
                    followup_system_messages: List[Dict[str, Any]] = []

                    # Emit tool-end + stream tool-data for each tool
                    for r in results:
                        # Skip UI events for silently suppressed duplicates
                        if r.get("silent"):
                            # We still append to tool_results below so the LLM knows what happened
                            # (or thinks it errored, so it stops trying), but frontend sees nothing.
                            pass
                        else:
                            tool_id = r["tool_call_id"]
                            tool_name = r["tool_name"]
                            duration_ms = int((time.time() - start_times.get(tool_id, time.time())) * 1000)

                            yield self.streaming_handler.format_aisdk_tool_end(
                                tool_call_id=tool_id,
                                tool_name=tool_name,
                                success=r["success"],
                                error=r.get("error"),
                                duration_ms=duration_ms,
                            )
                            await asyncio.sleep(0)

                            if r["success"] and r.get("frontend_data"):
                                tool_data.update(r["frontend_data"])
                                if isinstance(r["frontend_data"], dict) and r["frontend_data"].get("documents"):
                                    documents_tool_used = True
                                logger.info(f"[TOOL-DATA] Yielding tool-data for {r['tool_name']}: keys={list(r['frontend_data'].keys())}")
                                yield self.streaming_handler.format_aisdk_tool_data(r["frontend_data"])
                                sent_tool_data = True
                                await asyncio.sleep(0)
                            else:
                                logger.warning(f"[TOOL-DATA] NOT yielding tool-data - success={r.get('success')}, has_frontend_data={bool(r.get('frontend_data'))}")

                        tool_results.append({
                            "tool_call_id": r["tool_call_id"],
                            "role": "tool",
                            "content": r.get("content", ""),
                        })

                        # If Composio execution failed due to invalid parameters (often the model sent {}),
                        # inject ONE follow-up instruction with candidate mapped actions from DB.
                        if (
                            (not r.get("success"))
                            and r.get("tool_name") == "composio_execute"
                            and (r.get("error") or "").lower().find("missing") != -1
                            and composio_invalid_parameters_retry_budget > 0
                        ):
                            composio_invalid_parameters_retry_budget -= 1
                            try:
                                from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
                                from core.composio.entity_manager import EntityManager
                                from sqlalchemy import or_

                                q = (latest_text or "").lower()
                                q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
                                stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
                                q_tokens = [t for t in q_tokens if t not in stop][:10]

                                # Allowed apps = assigned EXTERNAL apps, optionally intersect with connected apps
                                assigned = (
                                    self.db.query(AgentAppAssignment)
                                    .filter(
                                        AgentAppAssignment.agent_id == agent_id,
                                        AgentAppAssignment.is_active == True,  # noqa: E712
                                        AgentAppAssignment.app_type == "EXTERNAL",
                                    )
                                    .all()
                                )
                                assigned_apps = [(a.app_name or "").upper() for a in assigned if a.app_name]
                                allowed_apps = assigned_apps
                                if self.workspace_id:
                                    manager = EntityManager(self.db)
                                    entity = manager.get_entity_by_workspace(self.workspace_id)
                                    if entity:
                                        connected_apps = [
                                            (c.get("app_name") or "").upper()
                                            for c in manager.get_entity_connections(entity["id"])
                                            if c.get("status") == "active"
                                        ]
                                        if connected_apps:
                                            connected_set = set(connected_apps)
                                            allowed_apps = [a for a in assigned_apps if a in connected_set]

                                suggestions: List[str] = []
                                if q_tokens and allowed_apps:
                                    for app in allowed_apps[:12]:
                                        token_filters = []
                                        for tok in q_tokens:
                                            like = f"%{tok}%"
                                            token_filters.append(ComposioActionCache.action_name.ilike(like))
                                            token_filters.append(ComposioActionCache.description.ilike(like))
                                        rows = (
                                            self.db.query(ComposioActionCache.action_name)
                                            .filter(ComposioActionCache.app_name == app)
                                            .filter(or_(*token_filters))
                                            .limit(12)
                                            .all()
                                        )
                                        for (action_name,) in rows:
                                            if action_name:
                                                suggestions.append(str(action_name))
                                        if len(suggestions) >= 12:
                                            break
                                suggestions = list(dict.fromkeys(suggestions))[:10]

                                followup_system_messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            "Your previous `composio_execute` call was missing required fields. "
                                            "Retry `composio_execute` with an explicit mapped `action` from "
                                            "`composio_actions_cache` plus the required `params`.\n"
                                            + (
                                                f"Candidate actions for this request: {', '.join(suggestions)}"
                                                if suggestions
                                                else "Pick the correct mapped action for the assigned app (e.g., Slack send message)."
                                            )
                                        ),
                                    }
                                )
                            except Exception:
                                followup_system_messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            "Your previous `composio_execute` call was missing required fields. "
                                            "Retry with an explicit mapped `action` and any required `params`."
                                        ),
                                    }
                                )

                    # Add assistant message with tool calls and results
                    llm_messages.append({
                        "role": "assistant",
                        "content": current_response.content or "",
                        "tool_calls": current_response.tool_calls,
                    })
                    llm_messages.extend(tool_results)
                    if followup_system_messages:
                        llm_messages.extend(followup_system_messages)

                    # Get next response - allow more tool calls if needed
                    allow_more_tools = iteration < max_iterations - 1
                    current_response = await llm_manager.generate_response(
                        messages=llm_messages,
                        tools=use_tools if allow_more_tools else None,
                    )

                    logger.info(
                        f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}, "
                        f"Has content: {bool(current_response.content)}"
                    )
                    # Log tool execution summary for debugging
                    if tool_tracker.tool_counts:
                        logger.info(f"📊 Tool execution counts this turn: {dict(tool_tracker.tool_counts)}")

                if iteration >= max_iterations and current_response.tool_calls:
                    logger.warning(f"Hit max tool iterations ({max_iterations}). Forcing final response.")

                full_response = current_response.content or ""

                # Safety: if we aggregated but never streamed tool-data incrementally
                if tool_data and not sent_tool_data:
                    yield self.streaming_handler.format_aisdk_tool_data(tool_data)
                    await asyncio.sleep(0)
            else:
                full_response = response.content or ""

            # ------------------------------------------------------------------
            # Document-answer shaping (prevent filename/link dumps in chat text)
            # ------------------------------------------------------------------
            def _infer_doc_topic(user_text: str) -> str:
                tl = (user_text or "").lower()
                if "agentfactory" in tl or "agent factory" in tl:
                    return "AgentFactory"
                cleaned = re.sub(
                    r"^(show|give|list|find|search)\s+(me\s+)?(the\s+)?(docs|documents)\s+(for|about)\s+",
                    "",
                    (user_text or "").strip(),
                    flags=re.I,
                )
                cleaned = cleaned.strip().strip("?.!")
                return cleaned[:60] if cleaned else "this topic"

            def _enforce_documents_shape(text: str, topic: str) -> str:
                """
                Enforce:
                - short plain-text summary (no 1./2./3. lists)
                - then exactly: "Here are some documents that discuss <topic>:"
                - stop (no filename list; UI cards below are the list)
                """
                text = (text or "").strip()
                lines = text.splitlines()
                out_lines: List[str] = []

                filename_re = re.compile(r"\b[\w\-. ]+\.(md|pdf|txt|docx?)\b", re.I)
                ordered_re = re.compile(r"^\s*\d+\.\s+")
                bullet_re = re.compile(r"^\s*[-*]\s+")
                md_link_re = re.compile(r"\[[^\]]+\]\([^)]+\)")

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        out_lines.append("")
                        continue

                    # Drop ordered/bulleted list items (removes the 1/2/3 sections)
                    if ordered_re.match(stripped) or bullet_re.match(stripped):
                        # If it's clearly a filename/link list item, drop it
                        if filename_re.search(stripped) or md_link_re.search(stripped):
                            continue
                        # Otherwise drop anyway for doc answers (keep summary plain text)
                        continue

                    # Drop standalone filename-ish lines (prevents echoed file lists)
                    if filename_re.fullmatch(stripped) or (filename_re.search(stripped) and len(stripped) <= 90):
                        continue

                    out_lines.append(line)

                cleaned = "\n".join(out_lines).strip()
                header = f"Here are some documents that discuss {topic}:"

                # Truncate at the first occurrence of the header intent if present
                m = re.search(r"here are some documents that discuss[^:]*:", cleaned, flags=re.I)
                if m:
                    before = cleaned[: m.start()].strip()
                    before = before.split("\n\n")[0].strip()  # first paragraph only
                    return (before + "\n\n" + header).strip()

                summary = cleaned.split("\n\n")[0].strip()
                if len(summary) > 700:
                    summary = summary[:700].rsplit(" ", 1)[0].strip() + "…"
                return (summary + "\n\n" + header).strip()

            # Only apply document shaping for DIRECT document requests
            # Don't apply for research/synthesis queries like "write a report"
            is_direct_doc_request = bool(re.search(
                r'\b(show|give|list|find|search)\s+(me\s+)?(the\s+)?(docs?|documents?)\b',
                (latest_text or '').lower()
            ))
            
            if tool_data.get("documents") and is_direct_doc_request:
                topic = _infer_doc_topic(latest_text)
                full_response = _enforce_documents_shape(full_response, topic)

            # Upload inline base64 images to S3 and replace with URLs
            full_response = await _upload_inline_images(
                full_response,
                workspace_id=str(self.workspace_id) if self.workspace_id else None,
            )

            # Stream text response
            async for chunk in self.streaming_handler.stream_text_aisdk(full_response):
                yield chunk

            # Send usage data (tracking now handled by LLMManager)
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
            self.chat_service.save_message(
                chat_id=chat_id,
                role="assistant",
                parts=assistant_parts,
                workspace_id=self.workspace_id
            )

            # Store memory via modules.memory
            if latest_text and full_response:
                await self.memory_injector.store_conversation_memory(
                    chat_id,
                    latest_text,
                    full_response,
                    workspace_id=str(self.workspace_id) if self.workspace_id else None,
                    agent_id=agent_id
                )

            # FutureAGI live traffic eval (fire-and-forget)
            if latest_text and full_response:
                try:
                    from core.services.futureagi_service import futureagi_service
                    if futureagi_service.is_available:
                        asyncio.create_task(
                            futureagi_service.eval_live_traffic(
                                input_text=latest_text,
                                output_text=full_response,
                            )
                        )
                except Exception:
                    pass  # Never block chat for eval

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            import traceback
            traceback.print_exc()
            yield self.streaming_handler.format_aisdk_error(str(e))

    async def stream_response_with_agent(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        agent_id: int,
        user_id: int,
        use_system_llm: bool = False,
        skip_composio: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response using a specialized agent from AgentFactory.

        PRD: Unified Agent-Chat System
        - Activates agent from factory
        - Uses agent's LLM manager, skills, and tools
        - Builds system prompt from agent's skills
        - Uses shared user-level memory

        Args:
            chat_id: Chat session ID
            messages: Chat messages
            agent_id: ID of agent to use
            user_id: User ID for memory
            use_system_llm: Use orchestrator LLM settings instead of agent's model

        Yields:
            AI SDK formatted response chunks
        """
        import asyncio
        
        try:
            # Ensure workspace_id is available for Memory and Composio tools
            if not self.workspace_id:
                try:
                    from core.models import Agent as AgentModel
                    # Use a new session or the existing one if safe
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and agent_row.workspace_id:
                        self.workspace_id = agent_row.workspace_id
                        logger.info(f"Resolved workspace_id from agent {agent_id}: {self.workspace_id}")
                except Exception as exc:
                    logger.warning(f"Failed to resolve workspace_id for agent {agent_id}: {exc}")

            # Start agent activation
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            fresh_start = self.prompt_analyzer.is_fresh_start_request(latest_text)
            if fresh_start:
                messages = [m for m in messages if m.get("role") == "user"][-1:]

            agent_task = asyncio.create_task(self.agent_factory.activate_agent(agent_id, use_system_llm=use_system_llm))

            # Send chat_id to frontend
            yield self.streaming_handler.format_aisdk_chat_id(chat_id)
            await asyncio.sleep(0)

            # Await agent activation
            agent_runtime = await agent_task
            if not agent_runtime:
                raise Exception(f"Failed to activate agent {agent_id}")

            logger.info(f"Activating agent {agent_id} for chat {chat_id}")

            # Send agent info to frontend
            yield self.streaming_handler.format_aisdk_data({
                "type": "agent-info",
                "agent": {
                    "id": agent_runtime.agent_id,
                    "name": agent_runtime.metadata.name,
                    "type": agent_runtime.metadata.agent_type,
                    "skills": agent_runtime.metadata.skills
                }
            })
            await asyncio.sleep(0)

            # ── SmartChatIntegration (personality + memory + tool filtering) ──
            from consumers.chatbot.integration import SmartChatIntegration, apply_orchestration_to_messages
            smart_chat = SmartChatIntegration(
                workspace_id=str(self.workspace_id) if self.workspace_id else self.workspace_id,
                agent_id=agent_id,
                agent_name=agent_runtime.metadata.name
            )

            # Load agent context: persona, description, skill tools
            agent_ctx = await self._load_agent_context(agent_runtime)
            skill_tools = agent_ctx["skill_tools"]
            from consumers.chatbot.tool_router import get_chat_tools
            is_simple = self.prompt_analyzer.is_simple_message(latest_text)
            all_tools = None if is_simple else get_chat_tools(agent_id=agent_id, workspace_id=self.workspace_id)
            if not is_simple and skill_tools:
                all_tools = (all_tools or []) + skill_tools

            # Convert messages to LLM format (no system prompt — orchestrator builds it)
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages,
                system_prompt="",
                available_tools=all_tools
            )

            # Orchestrate: memory + personality + tool filtering in one call
            orchestrated = await smart_chat.prepare(llm_messages, all_tools or [], chat_id)
            llm_messages = apply_orchestration_to_messages(orchestrated)
            use_tools = orchestrated.tools if orchestrated.requires_tools else None

            logger.info(
                f"[SmartChat] intent={orchestrated.intent.value} "
                f"tools={len(use_tools) if use_tools else 0} "
                f"memory={'yes' if orchestrated.memory_context else 'no'} "
                f"prep={orchestrated.preparation_time_ms:.0f}ms"
            )

            # Inject agent persona + description (after orchestrator system prompt)
            agent_identity_parts = []
            if agent_ctx.get("description"):
                agent_identity_parts.append(agent_ctx["description"])
            if agent_ctx.get("persona"):
                agent_identity_parts.append(f"## Persona & Communication Style\n{agent_ctx['persona']}")
            if agent_ctx.get("extra_context"):
                agent_identity_parts.append(agent_ctx["extra_context"])
            if agent_identity_parts:
                llm_messages.insert(1, {
                    "role": "system",
                    "content": "\n\n".join(agent_identity_parts),
                })

            # Multi-step execution policy
            insert_pos = 2 if agent_identity_parts else 1
            llm_messages.insert(
                insert_pos,
                {
                    "role": "system",
                    "content": (
                        "Execution policy: If the user requests multiple distinct tasks, you may call tools "
                        "multiple times to complete ALL tasks before producing your final answer. "
                        "Prefer data-gathering (read/list/fetch) steps before side-effect (send/post/create/update) steps. "
                        "Only send/post after you have the final content to send."
                    ),
                },
            )

            # Composio per-action tools (primary) or hint fallback
            _composio_result = None
            try:
                if latest_text and agent_id and self.workspace_id and not skip_composio:
                    from modules.tools.services.composio_tool_service import ComposioToolService

                    _composio_svc = ComposioToolService(self.db)
                    _composio_result = _composio_svc.get_tools_for_step(
                        agent_id=agent_id,
                        workspace_id=self.workspace_id,
                        task_prompt=latest_text,
                    )
                    if _composio_result and _composio_result.tools:
                        # Strip composio_execute mega-tool, add per-action tools
                        if use_tools:
                            use_tools = [
                                t for t in use_tools
                                if t.get("function", {}).get("name") != "composio_execute"
                            ] + _composio_result.tools
                        else:
                            use_tools = _composio_result.tools
                        # Inject scope message so LLM knows to call actions directly
                        from api.recipe_executor import _composio_scope_message
                        llm_messages.insert(2, {
                            "role": "system",
                            "content": _composio_scope_message(_composio_result.app_names),
                        })
                        logger.info(
                            f"[ComposioToolService] Agent {agent_id}: strategy={_composio_result.strategy} "
                            f"actions={len(_composio_result.action_set)} search_ms={_composio_result.search_ms}"
                        )
                    else:
                        # Fallback: ComposioHintService (composio_execute mega-tool)
                        from modules.tools.services.composio_hint_service import ComposioHintService

                        hint_service = ComposioHintService(self.db)
                        hint_result = hint_service.build_hints(
                            agent_id=agent_id,
                            prompt=latest_text,
                            workspace_id=self.workspace_id,
                        )
                        if hint_result.hint_lines:
                            llm_messages.insert(2, {"role": "system", "content": "\n".join(hint_result.hint_lines)})
                            logger.info(f"[Composio Hints fallback] Agent {agent_id}: strategy={hint_result.strategy_used} apps={hint_result.allowed_apps} matches={len(hint_result.matched_actions)}")
            except Exception as exc:
                logger.warning(f"Composio tool injection failed for agent {agent_id}: {exc}")

            # Context Window Guard — auto-compact if approaching context limit
            from core.context_guard import ContextGuard
            _guard = ContextGuard()
            _model_name = getattr(agent_runtime.llm_manager, 'config', None)
            _model_name = getattr(_model_name, 'model', 'gpt-4') if _model_name else 'gpt-4'
            llm_messages, _was_compacted = await _guard.check_and_compact(
                messages=llm_messages,
                model_name=_model_name,
                llm_manager=agent_runtime.llm_manager,
                workspace_id=str(self.workspace_id) if self.workspace_id else None,
                agent_id=agent_id,
                db_session=self.db,
            )
            if _was_compacted:
                logger.info("[ContextGuard] Messages compacted before LLM call")

            # Generate response using agent's LLM manager
            logger.info(f"Generating response with agent {agent_runtime.metadata.name}")
            logger.info(f"Agent tools - count: {len(use_tools) if use_tools else 0}, is_simple: {is_simple}")
            if use_tools:
                tool_names = [t.get("function", {}).get("name") for t in use_tools if isinstance(t, dict)]
                logger.info(f"Available tools: {tool_names}")

            response = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages,
                tools=use_tools
            )

            # Emit warning if LLM fell back to a different model (dead primary)
            if getattr(response, '_used_fallback', False):
                failed_model = getattr(response, '_failed_model', 'unknown')
                fallback_model = getattr(response, '_fallback_model', 'unknown')
                logger.warning(
                    "Chat response used fallback model '%s' (primary '%s' is dead). "
                    "User should update Settings > Orchestrator.",
                    fallback_model, failed_model,
                )
                yield self.streaming_handler.format_aisdk_data(
                    "model-warning",
                    {
                        "message": (
                            f"Your configured model ({failed_model}) is unavailable. "
                            f"Using {fallback_model} as fallback. "
                            f"Please update your model in Settings > Orchestrator."
                        ),
                        "failedModel": failed_model,
                        "fallbackModel": fallback_model,
                    },
                )
                await asyncio.sleep(0)

            # DEBUG: Log LLM response to understand why tools aren't being called
            logger.info(f"🔍 Agent LLM Response - has_tool_calls: {bool(response.tool_calls)}, content_length: {len(response.content or '')}, finish_reason: {getattr(response, 'finish_reason', 'unknown')}")
            if response.tool_calls:
                logger.info(f"✅ Agent LLM requested {len(response.tool_calls)} tool calls")
            elif use_tools:
                logger.warning(f"⚠️ Agent LLM did NOT call tools despite {len(use_tools)} tools being available. Response preview: {response.content[:200] if response.content else 'No content'}")
            
            # Track response parts
            assistant_parts = []
            full_response = ""
            tool_data = {}
            
            # Handle tool calls if any
            if response.tool_calls:
                logger.info(f"Agent requested {len(response.tool_calls)} tool calls")
                final_response = None
                async for chunk in self._handle_tool_calls_aisdk(response, llm_messages, agent_runtime, tool_data, use_tools, _composio_result=_composio_result):
                    # Check if this is the final response
                    if isinstance(chunk, dict) and chunk.get('_final_response'):
                        final_response = chunk['_final_response']
                        logger.info(f"Received final response from tool loop: {bool(final_response.content)}")
                    else:
                        yield chunk
                    await asyncio.sleep(0)
                
                # CRITICAL: Ensure we ALWAYS have a final response
                if final_response and final_response.content:
                    logger.info(f"Using final response from tool loop ({len(final_response.content)} chars)")
                    full_response = final_response.content
                else:
                    # Tool loop completed but no final response - force one
                    logger.warning("Tool loop completed without final response - forcing synthesis")
                    # Add instruction to synthesize
                    llm_messages.append({
                        'role': 'system',
                        'content': 'Based on the tool results above, provide a comprehensive response to the user. Synthesize the information and create the requested output.'
                    })
                    forced_response = await agent_runtime.llm_manager.generate_response(
                        messages=llm_messages,
                        tools=None  # No tools - force text response
                    )
                    full_response = forced_response.content if forced_response.content else "I apologize, but I encountered an issue generating a response. Please try again."
                    logger.info(f"Forced response generated ({len(full_response)} chars)")
            else:
                # No tool calls, use response content directly
                if response.content:
                    full_response = response.content

            # Upload inline base64 images to S3 and replace with URLs
            _ws_id_img = getattr(agent_runtime, 'workspace_id', None) or self.workspace_id
            full_response = await _upload_inline_images(
                full_response,
                workspace_id=str(_ws_id_img) if _ws_id_img else None,
            )

            # Stream text response
            async for chunk in self.streaming_handler.stream_text_aisdk(full_response):
                yield chunk

            # Send usage data (tracking now handled by LLMManager)
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
            self.chat_service.save_message(
                chat_id=chat_id,
                role="assistant",
                parts=assistant_parts,
                workspace_id=self.workspace_id
            )

            # Store memory via SmartChatIntegration (two-tier Mem0)
            if latest_text and full_response:
                try:
                    await smart_chat.store(latest_text, full_response, chat_id)
                except Exception as mem_err:
                    logger.warning(f"Failed to store memory exchange: {mem_err}")

            # FutureAGI live traffic eval (fire-and-forget, scores ALL enabled prompts)
            if latest_text and full_response:
                try:
                    from core.services.futureagi_service import futureagi_service
                    if futureagi_service.is_available:
                        asyncio.create_task(
                            futureagi_service.eval_live_traffic(
                                input_text=latest_text,
                                output_text=full_response,
                                context_text=orchestrated.system_prompt,
                            )
                        )
                except Exception:
                    pass  # Never block chat for eval

            # Update agent metrics
            if hasattr(agent_runtime, 'update_metrics'):
                tokens_used = response.usage.get('total_tokens', 0) if response.usage else 0
                agent_runtime.update_metrics(
                    execution_time=1.0,  # TODO: Track actual time
                    tokens_used=tokens_used,
                    success=True
                )
            
        except Exception as e:
            logger.error(f"Error streaming response with agent: {e}", exc_info=True)
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
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            if self.prompt_analyzer.is_fresh_start_request(latest_text):
                messages = [m for m in messages if m.get("role") == "user"][-1:]
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages,
                available_tools=tools
            )
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
                        
                        result = await self.tool_router.execute_and_format(
                            tool_name,
                            tool_args,
                            agent_id=1,
                            workspace_id=self.workspace_id,
                            original_intent=latest_text,
                        )
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
                self.chat_service.save_message(
                    chat_id=chat_id,
                    role='assistant',
                    parts=assistant_parts,
                    workspace_id=self.workspace_id
                )
            
            yield self.streaming_handler.format_sse_done()
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield self.streaming_handler.format_sse_error(str(e))
    
    def _parse_model_selection(self, selected_model: Optional[str]) -> tuple:
        """Parse model string to get provider and model.

        Models with vendor/model format (e.g. qwen/qwen3-coder-next) are
        OpenRouter marketplace models and route through the OpenRouter API.
        """
        if not selected_model:
            return None, None

        model = selected_model
        model_lower = selected_model.lower()

        # Check direct provider models first (no slash in model ID)
        if model_lower.startswith('gpt-') or model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
            provider = 'openai'
        elif model_lower.startswith('claude') or 'anthropic' in model_lower:
            provider = 'anthropic'
        elif model_lower.startswith('grok') or 'xai' in model_lower:
            provider = 'grok'
        elif model_lower.startswith('gemini') or 'google' in model_lower:
            provider = 'google'
        elif '/' in selected_model:
            # Slash format = OpenRouter marketplace model (e.g. qwen/qwen3-coder-next,
            # meta-llama/llama-3.1-70b, mistralai/mistral-large)
            provider = 'openrouter'
        else:
            provider = None

        return provider, model
    async def _load_agent_context(self, agent_runtime) -> dict:
        """
        Load agent-specific context: persona, description, plugins/skills, tool schemas.

        Returns a dict with keys:
            - persona: str (agent's persona/communication style prompt)
            - description: str (agent name + type + description)
            - skill_tools: list (tool schemas from skills)
        """
        from core.models import Skill

        # Load persona from agent's DB record
        persona = ""
        try:
            from core.models import Agent as AgentModel
            db_agent = self.db.query(AgentModel).filter(AgentModel.id == agent_runtime.agent_id).first()
            if db_agent:
                if getattr(db_agent, "use_custom_persona", False) and getattr(db_agent, "custom_persona_prompt", None):
                    persona = db_agent.custom_persona_prompt
                    logger.info(f"Loaded custom persona for agent {agent_runtime.agent_id}")
                elif getattr(db_agent, "persona_id", None) and getattr(db_agent, "persona", None):
                    persona = db_agent.persona.system_prompt or ""
                    logger.info(f"Loaded persona '{db_agent.persona.name}' for agent {agent_runtime.agent_id}")
        except Exception as e:
            logger.warning(f"Failed to load persona for agent {agent_runtime.agent_id}: {e}")

        # Agent description
        description = (
            f"You are {agent_runtime.metadata.name}, "
            f"a specialized {agent_runtime.metadata.agent_type} agent.\n"
            f"{agent_runtime.metadata.description or ''}"
        )

        # Load plugins OR skills for additional context
        extra_context = ""
        has_plugins = False
        try:
            from core.services.plugin_context_service import PluginContextService
            plugin_svc = PluginContextService(self.db)
            plugin_rows = plugin_svc.get_assigned_plugins(agent_runtime.agent_id)
            if plugin_rows:
                has_plugins = True
                tier1 = plugin_svc.build_tier1_summary(plugin_rows)
                tier2 = await plugin_svc.build_tier2_content(
                    plugin_rows,
                    task_context=agent_runtime.metadata.description,
                )
                extra_context = f"\n{tier1}\n{tier2}" if tier2 else f"\n{tier1}"
                logger.info(
                    "Loaded plugin context for agent %s (%d plugins)",
                    agent_runtime.agent_id, len(plugin_rows),
                )
        except Exception as e:
            logger.warning(f"Failed to load plugins for agent {agent_runtime.agent_id}: {e}")

        if not has_plugins and agent_runtime.metadata.skills:
            skills = self.db.query(Skill).filter(
                Skill.name.in_(agent_runtime.metadata.skills),
                Skill.is_active.is_(True),
            ).all()
            parts = []
            for skill in skills:
                if skill.prompt_template:
                    parts.append(skill.prompt_template)
                elif skill.description:
                    parts.append(f"- {skill.name}: {skill.description}")
            if parts:
                extra_context = "\n".join(parts)

        # Extract tool schemas from skills
        tool_schemas = []
        if not has_plugins and agent_runtime.metadata.skills:
            skills = self.db.query(Skill).filter(
                Skill.name.in_(agent_runtime.metadata.skills),
                Skill.is_active == True,  # noqa: E712
            ).all()
            for skill in skills:
                if skill.content and isinstance(skill.content, dict):
                    schemas = skill.content.get("tools_schema", [])
                    if schemas:
                        tool_schemas.extend(schemas)

        return {
            "persona": persona,
            "description": description,
            "extra_context": extra_context,
            "skill_tools": tool_schemas,
        }
    
    async def _handle_tool_calls_aisdk(
        self,
        response,
        llm_messages: List[Dict],
        agent_runtime,
        tool_data: Dict,
        use_tools: List = None,
        _composio_result=None,
    ) -> AsyncGenerator[str, None]:
        """
        Handle tool calls from agent's LLM response.

        Args:
            response: LLM response with tool_calls
            llm_messages: Current message history
            agent_runtime: Agent runtime with tools
            tool_data: Dict to store tool results
            use_tools: Tools to pass to LLM for next iteration
            _composio_result: ComposioToolResult for per-action tool execution
            
        Yields:
            AI SDK formatted tool execution chunks
        """
        import asyncio
        import time
        
        max_iterations = 5
        iteration = 0
        current_response = response
        # Allow ONE recovery attempt when the model picks an unmapped action name.
        action_not_mapped_retry_budget = 1
        # Allow ONE recovery attempt when the model calls composio_execute without required args.
        invalid_parameters_retry_budget = 1
        # Detect repeated tool calls (prevents infinite loops / API spam).
        seen_call_keys: set[str] = set()
        # Stop "search tool spirals": if the model keeps calling the same search tool
        # with different queries but keeps getting 0 results, force a final answer.
        last_tool_name: Optional[str] = None
        empty_same_tool_streak = 0
        # Stop "rephrase and retry" loops: for most internal tools, we only allow one attempt
        # per user request (multi-step exceptions: composio_execute and file tools).
        tool_attempts: dict[str, int] = {}
        
        while current_response.tool_calls and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}: {len(current_response.tool_calls)} tool calls")
            
            start_times = {}
            tool_calls_prepared = []
            fatal_errors: List[Dict[str, Any]] = []
            
            for tool_call in current_response.tool_calls:
                tool_name = tool_call.get('function', {}).get('name', 'unknown')
                tool_id = tool_call.get('id', f'call_{int(time.time() * 1000)}')
                
                # Emit tool-start
                try:
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    tool_input = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    tool_input = {}
                yield self.streaming_handler.format_aisdk_tool_start(tool_id, tool_name, tool_input=tool_input)
                await asyncio.sleep(0)
                
                start_times[tool_id] = time.time()
                tool_calls_prepared.append((tool_id, tool_name, tool_call))
            
            # Execute tools
            tool_results = []
            followup_system_messages: List[Dict[str, Any]] = []
            executed_call_key_repeat = False
            for tool_id, tool_name, tool_call in tool_calls_prepared:
                try:
                    # Parse arguments
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    tool_args = json.loads(args_str) if isinstance(args_str, str) else args_str

                    # Loop detection key (same tool + same args)
                    try:
                        args_key = json.dumps(tool_args or {}, sort_keys=True, default=str)
                    except Exception:
                        args_key = str(tool_args)
                    call_key = f"{tool_name}:{args_key}"
                    if call_key in seen_call_keys:
                        executed_call_key_repeat = True
                        # IMPORTANT: skip executing duplicates to prevent repeated side-effects
                        # (e.g., posting the same Slack message twice).
                        llm_context = (
                            "Skipped duplicate tool call (same tool + same arguments were already executed in this request)."
                        )
                        tool_results.append(
                            {
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": tool_name,
                                "content": llm_context,
                            }
                        )
                        yield self.streaming_handler.format_aisdk_data(
                            "tool-result",
                            {
                                "toolCallId": tool_id,
                                "toolName": tool_name,
                                "result": llm_context,
                            },
                        )
                        await asyncio.sleep(0)
                        continue
                    else:
                        seen_call_keys.add(call_key)
                    
                    # Execute tool via tool router
                    # (pass original user intent for deterministic guards like relative-date rewriting)
                    user_text = ""
                    for m in reversed(llm_messages):
                        if m.get("role") == "user":
                            user_text = m.get("content") or ""
                            break

                    # Direct Composio action execution (per-action tools from ComposioToolService).
                    # If the tool name is in the resolved action set (or matches a connected app prefix),
                    # execute via ComposioToolService.execute_action() instead of tool_router.
                    _is_composio_action = (
                        _composio_result and _composio_result.entity_id and (
                            tool_name in _composio_result.action_set
                            or any(tool_name.startswith(f"{app}_") for app in _composio_result.app_names)
                        )
                    )
                    if _is_composio_action:
                        try:
                            from modules.tools.services.composio_tool_service import ComposioToolService
                            _exec_svc = ComposioToolService(self.db)
                            exec_result = _exec_svc.execute_action(
                                action_name=tool_name,
                                params=tool_args,
                                entity_id=_composio_result.entity_id,
                            )
                            success = exec_result.get("success", False)
                            data = exec_result.get("data")
                            error = exec_result.get("error")
                            if success:
                                llm_context = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or "")
                            else:
                                llm_context = f"Error executing {tool_name}: {error or 'unknown error'}"
                            logger.info(f"[Composio direct] {tool_name}: success={success}")
                        except Exception as exc:
                            llm_context = f"Error executing {tool_name}: {exc}"
                            logger.error(f"[Composio direct] {tool_name} exception: {exc}", exc_info=True)

                        if len(llm_context) > 4000:
                            llm_context = llm_context[:4000] + f"\n... (truncated)"

                        tool_results.append({
                            'tool_call_id': tool_id,
                            'role': 'tool',
                            'name': tool_name,
                            'content': llm_context,
                        })

                        yield self.streaming_handler.format_aisdk_tool_end(
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            success=True,
                            duration_ms=int((time.time() - start_times.get(tool_id, time.time())) * 1000),
                        )
                        yield self.streaming_handler.format_aisdk_data(
                            "tool-result",
                            {"toolCallId": tool_id, "toolName": tool_name, "result": llm_context[:500]},
                        )
                        await asyncio.sleep(0)
                        continue

                    result = await self.tool_router.execute_and_format(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_id=agent_runtime.agent_id if hasattr(agent_runtime, 'agent_id') else 1,
                        workspace_id=self.workspace_id,
                        original_intent=user_text,
                    )

                    # Detect repeated empty search attempts (prevents "try again" loops)
                    try:
                        raw = result.get("raw_result") or {}
                        count = raw.get("count")
                        if count is None:
                            rr = raw.get("results")
                            if isinstance(rr, list):
                                count = len(rr)
                        is_search_tool = tool_name.startswith("search_") or tool_name in {
                            "semantic_search",
                        }
                        is_empty = (isinstance(count, int) and count == 0)
                        if is_search_tool and is_empty:
                            if last_tool_name == tool_name:
                                empty_same_tool_streak += 1
                            else:
                                empty_same_tool_streak = 1
                            last_tool_name = tool_name
                        else:
                            last_tool_name = tool_name
                            empty_same_tool_streak = 0
                    except Exception:
                        pass

                    # Track attempts per tool name (used to stop wasteful re-tries).
                    try:
                        tool_attempts[tool_name] = int(tool_attempts.get(tool_name, 0)) + 1
                    except Exception:
                        pass

                    # Extract LLM context from result and truncate to prevent context overflow
                    llm_context = result.get('llm_context', str(result.get('raw_result', '')))
                    # Truncate to max 1000 chars to keep context manageable
                    if len(llm_context) > 1000:
                        llm_context = llm_context[:1000] + f"\n... (truncated {len(llm_context) - 1000} chars)"
                    
                    # Store result
                    tool_results.append({
                        'tool_call_id': tool_id,
                        'role': 'tool',
                        'name': tool_name,
                        'content': llm_context
                    })
                    # CRITICAL: Yield tool-data for frontend widgets (documents, code, etc.)
                    # This was missing for selected agents - widgets only worked with default agent
                    frontend_data = result.get("frontend_data", {})
                    if result.get("success") and frontend_data:
                        tool_data.update(frontend_data)
                        logger.info(f"[TOOL-DATA] Yielding tool-data for {tool_name}: keys={list(frontend_data.keys())}")
                        yield self.streaming_handler.format_aisdk_tool_data(frontend_data)
                        await asyncio.sleep(0)

                    # Stop immediately on deterministic "unavailable" errors (no point retrying),
                    # EXCEPT: allow a single recovery attempt for "action not mapped" so the model
                    # can pick from the provided mapped action examples.
                    error_type = result.get("error_type")
                    raw_error = (result.get("raw_result") or {}).get("error") if isinstance(result.get("raw_result"), dict) else None
                    if (not result.get("success")) and error_type == "composio_action_not_mapped" and action_not_mapped_retry_budget > 0:
                        action_not_mapped_retry_budget -= 1
                        # Add a follow-up system instruction to pick an exact mapped action name.
                        if raw_error and "Examples of mapped actions:" in raw_error:
                            examples = raw_error.split("Examples of mapped actions:", 1)[1].strip()

                            # IMPORTANT: don't let the model "pick any Slack action" if the requested
                            # capability isn't actually present in cache (prevents wrong side-effects).
                            user_text = ""
                            for m in reversed(llm_messages):
                                if m.get("role") == "user":
                                    user_text = m.get("content") or ""
                                    break
                            q = (user_text or "").lower()
                            q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
                            stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
                            q_tokens = [t for t in q_tokens if t not in stop][:12]

                            candidates = [c.strip() for c in examples.split(",") if c.strip()]
                            scored: List[tuple[int, str]] = []
                            for c in candidates:
                                ct = c.lower()
                                score = sum(1 for tok in q_tokens if tok in ct)
                                scored.append((score, c))
                            scored.sort(key=lambda x: (-x[0], x[1]))
                            top = [c for score, c in scored if score > 0][:8]

                            # If none of the suggested actions match the user's request at all,
                            # stop and ask for a cache sync rather than executing unrelated actions.
                            if not top:
                                from types import SimpleNamespace
                                message = (
                                    "That action is not available in the local integrations cache for this workspace/agent. "
                                    "The system won't guess a different action (to avoid doing the wrong thing). "
                                    "Please run a Composio sync to refresh `composio_actions_cache` for this app, then retry."
                                )
                                yield {"_final_response": SimpleNamespace(content=message, tool_calls=None, usage=None)}
                                return

                            followup_system_messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "The previous Composio action name was not mapped. "
                                        "Retry using ONE of these exact mapped action names that best matches the user's request:\n"
                                        f"{', '.join(top)}\n"
                                        "Use `composio_execute` again with the corrected `action`."
                                    ),
                                }
                            )
                        else:
                            followup_system_messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "The previous Composio action name was not mapped. "
                                        "Retry using a valid mapped action from `composio_actions_cache`."
                                    ),
                                }
                            )
                    else:
                        # Recovery: composio_execute called with missing/invalid parameters.
                        # Give ONE chance to retry with an explicit mapped action.
                        if (
                            (not result.get("success"))
                            and tool_name == "composio_execute"
                            and error_type == "invalid_parameters"
                            and invalid_parameters_retry_budget > 0
                        ):
                            invalid_parameters_retry_budget -= 1
                            try:
                                from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
                                from core.composio.entity_manager import EntityManager

                                # Find last user message
                                user_text = ""
                                for m in reversed(llm_messages):
                                    if m.get("role") == "user":
                                        user_text = m.get("content") or ""
                                        break

                                q = (user_text or "").lower()
                                q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
                                stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
                                q_tokens = [t for t in q_tokens if t not in stop][:10]

                                # Allowed apps = assigned EXTERNAL apps, optionally intersect with connected apps
                                aid = agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else 0
                                assigned = (
                                    self.db.query(AgentAppAssignment)
                                    .filter(
                                        AgentAppAssignment.agent_id == aid,
                                        AgentAppAssignment.is_active == True,  # noqa: E712
                                        AgentAppAssignment.app_type == "EXTERNAL",
                                    )
                                    .all()
                                )
                                assigned_apps = [(a.app_name or "").upper() for a in assigned if a.app_name]
                                allowed_apps = assigned_apps

                                if self.workspace_id:
                                    manager = EntityManager(self.db)
                                    entity = manager.get_entity_by_workspace(self.workspace_id)
                                    if entity:
                                        connected_apps = [
                                            (c.get("app_name") or "").upper()
                                            for c in manager.get_entity_connections(entity["id"])
                                            if c.get("status") == "active"
                                        ]
                                        if connected_apps:
                                            connected_set = set(connected_apps)
                                            allowed_apps = [a for a in assigned_apps if a in connected_set]

                                suggestions: List[str] = []
                                if q_tokens and allowed_apps:
                                    for app in allowed_apps[:12]:
                                        token_filters = []
                                        for tok in q_tokens:
                                            like = f"%{tok}%"
                                            token_filters.append(ComposioActionCache.action_name.ilike(like))
                                            token_filters.append(ComposioActionCache.description.ilike(like))
                                        rows = (
                                            self.db.query(ComposioActionCache.action_name)
                                            .filter(ComposioActionCache.app_name == app)
                                            .filter(or_(*token_filters))
                                            .limit(12)
                                            .all()
                                        )
                                        for (action_name,) in rows:
                                            if action_name:
                                                suggestions.append(str(action_name))
                                        if len(suggestions) >= 12:
                                            break

                                suggestions = list(dict.fromkeys(suggestions))[:10]
                                followup_system_messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            "Your previous `composio_execute` call was missing required parameters. "
                                            "Retry by calling `composio_execute` again with an explicit mapped `action` "
                                            "from `composio_actions_cache` and any required `params`. "
                                            + (
                                                f"Candidate actions for this request: {', '.join(suggestions)}"
                                                if suggestions
                                                else "Pick a valid mapped action for one of the agent's assigned + connected apps."
                                            )
                                        ),
                                    }
                                )
                            except Exception:
                                followup_system_messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            "Your previous `composio_execute` call was missing required parameters. "
                                            "Retry with an explicit mapped `action` and any required `params`."
                                        ),
                                    }
                                )

                        deterministic_error_types = {
                            # Composio gating
                            "composio_not_assigned",
                            "composio_not_connected",
                            "composio_action_not_allowed",
                            "composio_missing_workspace",
                            # Generic parameter issues
                            "invalid_parameters",
                        }
                        # If we injected a follow-up retry instruction, don't stop here.
                        if followup_system_messages:
                            pass
                        elif (not result.get("success")) and error_type in deterministic_error_types:
                            from types import SimpleNamespace
                            message = raw_error or llm_context or "That tool is not available for this agent/workspace."
                            yield {"_final_response": SimpleNamespace(content=message, tool_calls=None, usage=None)}
                            return

                    if result.get('fatal_error'):
                        fatal_errors.append(result)
                    
                    # Emit tool-result as data event
                    yield self.streaming_handler.format_aisdk_data('tool-result', {
                        'toolCallId': tool_id,
                        'toolName': tool_name,
                        'result': llm_context[:500]  # Truncate for streaming
                    })
                    await asyncio.sleep(0)

                    # --- Hard stop conditions to prevent looping for single requests ---
                    #
                    # 1) Search tools: stop after FIRST empty result.
                    if empty_same_tool_streak >= 1 and (
                        tool_name.startswith("search_") or tool_name in {"semantic_search"}
                    ):
                        from types import SimpleNamespace
                        message = (
                            "I couldn't find any matching results and will stop retrying to avoid looping. "
                            "If you expected results, it likely means the underlying knowledge/code index isn't ingested "
                            "or the query is targeting the wrong index/type."
                        )
                        yield {"_final_response": SimpleNamespace(content=message, tool_calls=None, usage=None)}
                        return

                    # 2) Database tools: stop after FIRST successful result (avoid paraphrase loops).
                    if tool_name in {"query_database", "smart_query_database"} and result.get("success"):
                        from types import SimpleNamespace
                        llm_messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "You now have the database result. Do NOT call the database tool again. "
                                    "Write the final answer using the tool output above."
                                ),
                            }
                        )
                        final_response = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                        yield {"_final_response": SimpleNamespace(content=final_response.content or "", tool_calls=None, usage=getattr(final_response, "usage", None))}
                        return

                    # 3) Generic: for most non-Composio tools, don't allow multiple attempts in one request.
                    # This prevents the model from re-issuing the same tool with slightly different phrasing.
                    if tool_name not in {"composio_execute"} and not tool_name.startswith("composio_"):
                        # Allow file operations to be multi-step for report saving.
                        if tool_name not in {"read_file", "write_file", "list_directory", "create_directory", "delete_file"}:
                            if tool_attempts.get(tool_name, 0) >= 2:
                                from types import SimpleNamespace
                                message = (
                                    f"I already tried `{tool_name}` and won’t retry again in the same request "
                                    "to avoid looping. If you want me to try a different approach, tell me what to change."
                                )
                                yield {"_final_response": SimpleNamespace(content=message, tool_calls=None, usage=None)}
                                return
                    
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    tool_results.append({
                        'tool_call_id': tool_id,
                        'role': 'tool',
                        'name': tool_name,
                        'content': error_msg
                    })
                    
                    yield self.streaming_handler.format_aisdk_data('tool-result', {
                        'toolCallId': tool_id,
                        'toolName': tool_name,
                        'result': error_msg
                    })
                    await asyncio.sleep(0)
            
            # Add assistant message with tool_calls first (required by OpenAI API)
            assistant_tool_message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': tc[0],
                        'type': 'function',
                        'function': {
                            'name': tc[1],
                            'arguments': tc[2].get('function', {}).get('arguments', '{}')
                        }
                    }
                    for tc in tool_calls_prepared
                ]
            }
            llm_messages.append(assistant_tool_message)
            
            # Then add tool results to message history
            llm_messages.extend(tool_results)

            # Add any follow-up instructions (e.g., one retry for action-not-mapped)
            if followup_system_messages:
                llm_messages.extend(followup_system_messages)

            # If the model is repeating the same tool call, force synthesis WITHOUT tools
            # to prevent repeated external API calls.
            #
            # IMPORTANT:
            # Do NOT stop tool use merely because one tool call succeeded: many user requests are
            # multi-step (e.g., "fetch emails THEN post summary to Slack"). Stopping after the first
            # success prevents completion of later side-effects.
            if executed_call_key_repeat:
                from types import SimpleNamespace
                llm_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You now have the tool results needed. "
                            "Do NOT call any more tools. "
                            "Write the final answer for the user using the tool output above."
                        ),
                    }
                )
                final_response = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                yield {"_final_response": SimpleNamespace(content=final_response.content or "", tool_calls=None, usage=getattr(final_response, "usage", None))}
                return

            if fatal_errors:
                from types import SimpleNamespace
                message = (
                    "I ran into a server configuration issue while executing that tool. "
                    "Please restart the backend and try again."
                )
                yield {'_final_response': SimpleNamespace(content=message, tool_calls=None, usage=None)}
                return
            
            # Get next LLM response
            current_response = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages,
                tools=use_tools
            )
            
            # Log iteration completion
            logger.info(f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}, Has content: {bool(current_response.content)}")
            
            # If no more tool calls, this is the final response
            if not current_response.tool_calls:
                logger.info(f"✅ Tool loop complete after {iteration} iterations. Returning final response.")
                # Yield final response marker
                yield {'_final_response': current_response}
                return  # Exit the generator
        
        # If max iterations reached, force final response
        if iteration >= max_iterations:
            logger.warning(f"Max tool iterations ({max_iterations}) reached. Forcing final response.")
            # Make one final call without tools to get synthesized response
            final_response = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages,
                tools=None  # No tools - force text response
            )
            yield {'_final_response': final_response}

    
    async def _execute_pretriggered_tools(
        self,
        detected_tools: List[str],
        query: str,
        llm_messages: List[Dict],
        tool_data: Dict,
        agent_id: int
    ) -> AsyncGenerator[str, None]:
        """Execute pre-triggered tools and inject results."""
        for tool_name in detected_tools:
            import time
            tool_call_id = str(uuid.uuid4())
            start_time = time.time()

            logger.info(f"Pre-triggering {tool_name}")
            yield self.streaming_handler.format_aisdk_tool_start(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_input={"query": query},
            )

            result = await self.tool_router.execute_and_format(
                tool_name,
                {"query": query},
                agent_id=agent_id,
                workspace_id=self.workspace_id,
                original_intent=query,
            )
            
            if result['success']:
                tool_data.update(result['frontend_data'])
                
                # Build context injection
                context_msg = self.tool_router.build_tool_context_message(tool_name, result)
                if context_msg:
                    llm_messages.insert(1, context_msg)
                
                # Send tool data to frontend
                if result['frontend_data']:
                    yield self.streaming_handler.format_aisdk_tool_data(result['frontend_data'])

                yield self.streaming_handler.format_aisdk_tool_end(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    success=True,
                    duration_ms=int((time.time() - start_time) * 1000),
                )
            else:
                yield self.streaming_handler.format_aisdk_tool_end(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    success=False,
                    error=(result.get("raw_result", {}) or {}).get("error") if isinstance(result, dict) else "Tool failed",
                    duration_ms=int((time.time() - start_time) * 1000),
                )
    
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
                user_text = ""
                for m in reversed(llm_messages):
                    if m.get("role") == "user":
                        user_text = m.get("content") or ""
                        break
                result = await self.tool_router.execute_and_format(
                    tool_name,
                    tool_args,
                    agent_id=1,
                    workspace_id=self.workspace_id,
                    original_intent=user_text,
                )
                
                return {
                    "tool_call_id": tool_id,
                    "role": "tool", 
                    "content": result['llm_context'],
                    "frontend_data": result.get('frontend_data', {}),
                    "success": result['success'],
                    "fatal_error": result.get('fatal_error', False),
                    "error_type": result.get('error_type')
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

            fatal_errors = [r for r in results if r.get("fatal_error")]
            if fatal_errors:
                return (
                    "I ran into a server configuration issue while executing that tool. "
                    "Please restart the backend and try again."
                ), tool_data
            
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

