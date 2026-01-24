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
        Stream chat response using AI SDK Data Stream format.
        
        Format: 0:"text"\n for text, d:{json}\n for data, e:{json}\n for errors
        """
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
            supports_native_tools = provider in ['openai', 'anthropic', 'grok'] if provider else False
            
            # --- TOOL FILTERING (PRD Refinement) ---
            # To prevent context overload (600+ tools), we filter to the top N relevant tools.
            use_tools = None
            if not is_simple and tools and supports_native_tools:
                # 1. Identify Core Tools (always included)
                core_tool_names = {
                    "search_knowledge", "semantic_search", "search_codebase", 
                    "query_database", "smart_query_database", 
                    "read_file", "write_file", "switch_context",
                    "search_tables", "search_images", "search_formulas", "search_multimodal"
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
            if not fresh_start:
                memory_context = await self.memory_injector.retrieve_relevant_memories(
                    chat_id,
                    latest_text,
                    workspace_id=str(self.workspace_id) if self.workspace_id else None,
                    agent_id=agent_id
                )
                if memory_context:
                    logger.info(f"[Memory] Injecting {len(memory_context)} chars")
                    llm_messages.insert(1, self.memory_injector.build_memory_injection_message(memory_context))

            # Add dynamic tool candidates (hint)
            if context_tools and latest_text:
                candidates = self.prompt_analyzer.rank_tools_for_query(latest_text, context_tools)
                if candidates:
                    candidate_names = ", ".join([c["name"] for c in candidates if c.get("name")])
                    if candidate_names:
                        insert_at = 2 if memory_context else 1
                        llm_messages.insert(
                            insert_at,
                            {
                                "role": "system",
                                "content": (
                                    "Tool candidates for this request: "
                                    f"{candidate_names}. Use the most relevant tool instead of answering from memory."
                                )
                            }
                        )

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
                            workspace_id=self.workspace_id
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
                            workspace_id=str(self.workspace_id) if self.workspace_id else None
                        )

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

                executed_tool_signatures = set()

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
                        
                        # Deduplication check
                        tool_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                        is_duplicate = tool_sig in executed_tool_signatures
                        if not is_duplicate:
                            executed_tool_signatures.add(tool_sig)
                        
                        tool_calls_prepared.append((tool_id, tool_name, tool_args, is_duplicate))

                        yield self.streaming_handler.format_aisdk_tool_start(
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            tool_input=tool_args,
                        )
                        await asyncio.sleep(0)

                    # Execute all tools in parallel
                    async def execute_single_tool(tool_id: str, tool_name: str, tool_args: Dict[str, Any], is_duplicate: bool):
                        if is_duplicate:
                            logger.warning(f"⚠️ Skipping duplicate tool execution: {tool_name}")
                            return {
                                "tool_call_id": tool_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "role": "tool",
                                "content": f"Error: Tool '{tool_name}' was already executed with these parameters in this turn. Do not call it again.",
                                "frontend_data": {},
                                "success": False,
                                "error": "Duplicate execution skipped",
                            }

                        logger.info(f"Executing tool: {tool_name}")
                        result = await self.tool_router.execute_and_format(
                            tool_name,
                            tool_args,
                            agent_id=agent_id,
                            workspace_id=self.workspace_id
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
                        }

                    results = await asyncio.gather(*[
                        execute_single_tool(tool_id, tool_name, tool_args, is_dup)
                        for (tool_id, tool_name, tool_args, is_dup) in tool_calls_prepared
                    ])

                    tool_results = []

                    # Emit tool-end + stream tool-data for each tool
                    for r in results:
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
                            yield self.streaming_handler.format_aisdk_tool_data(r["frontend_data"])
                            sent_tool_data = True
                            await asyncio.sleep(0)

                        tool_results.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "content": r.get("content", ""),
                        })

                    # Add assistant message with tool calls and results
                    llm_messages.append({
                        "role": "assistant",
                        "content": current_response.content or "",
                        "tool_calls": current_response.tool_calls,
                    })
                    llm_messages.extend(tool_results)

                    # Get next response - allow more tool calls if needed
                    allow_more_tools = iteration < max_iterations - 1
                    current_response = await llm_manager.generate_response(
                        messages=llm_messages,
                        tools=use_tools if allow_more_tools else None,
                    )

                    logger.info(
                        f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}"
                    )

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
                    workspace_id=str(self.workspace_id) if self.workspace_id else None
                )
            
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
        user_id: int
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
            
        Yields:
            AI SDK formatted response chunks
        """
        import asyncio
        
        try:
            # Start parallel tasks
            agent_task = asyncio.create_task(self.agent_factory.activate_agent(agent_id))
            
            # Start memory retrieval concurrently
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            fresh_start = self.prompt_analyzer.is_fresh_start_request(latest_text)
            if fresh_start:
                messages = [m for m in messages if m.get("role") == "user"][-1:]
                memory_task = None
            else:
                memory_task = asyncio.create_task(self.memory_injector.retrieve_relevant_memories(chat_id, latest_text))
            
            # Send chat_id to frontend
            yield self.streaming_handler.format_aisdk_chat_id(chat_id)
            await asyncio.sleep(0)
            
            # Await agent activation
            try:
                agent_runtime = await agent_task
                logger.info(f"Activating agent {agent_id} for chat {chat_id}")
            except Exception as e:
                # Cancel memory task if agent fails
                if memory_task:
                    memory_task.cancel()
                raise e
            
            if not agent_runtime:
                if memory_task:
                    memory_task.cancel()
                raise Exception(f"Failed to activate agent {agent_id}")

            # Ensure workspace_id is available for Composio tools
            if not self.workspace_id:
                try:
                    from core.models import Agent as AgentModel
                    agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
                    if agent_row and agent_row.workspace_id:
                        self.workspace_id = agent_row.workspace_id
                        logger.info(f"Resolved workspace_id from agent {agent_id}")
                except Exception as exc:
                    logger.warning(f"Failed to resolve workspace_id for agent {agent_id}: {exc}")
            
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
            
            # Build system prompt from agent's skills and extract tool schemas
            system_prompt, skill_tools = await self._build_agent_system_prompt(agent_runtime)
            
            # Determine tool usage
            is_simple = self.prompt_analyzer.is_simple_message(latest_text)
            
            # CRITICAL: Use tool_router to get properly formatted function schemas
            # agent_runtime.tools contains METADATA, not OpenAI function schemas!
            # get_chat_tools returns proper OpenAI function schemas based on agent permissions
            from consumers.chatbot.tool_router import get_chat_tools
            use_tools = None if is_simple else get_chat_tools(agent_id=agent_id)
            
            # Add skill tools if available
            if not is_simple and skill_tools:
                if use_tools is None:
                    use_tools = skill_tools
                else:
                    use_tools = use_tools + skill_tools
            
            # Convert messages to LLM format and inject system prompt
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages,
                available_tools=use_tools
            )
            
            # Replace or prepend system message with agent's system prompt
            if llm_messages and llm_messages[0].get('role') == 'system':
                llm_messages[0]['content'] = system_prompt
            else:
                llm_messages.insert(0, {'role': 'system', 'content': system_prompt})
            
            # Await memory result
            memory_context = None
            if memory_task:
                memory_context = await memory_task
                if memory_context:
                    logger.info(f"[Memory] Injecting {len(memory_context)} chars of shared memory")
                    llm_messages.insert(1, self.memory_injector.build_memory_injection_message(memory_context))

            # Add dynamic tool candidates (no hardcoded app mapping)
            if use_tools and latest_text:
                candidates = self.prompt_analyzer.rank_tools_for_query(latest_text, use_tools)
                if candidates:
                    candidate_names = [c["name"] for c in candidates if c.get("name")]
                    if candidate_names:
                        # Narrow tool list to top candidates to reduce overload
                        candidate_set = set(candidate_names)
                        filtered_tools = [
                            tool for tool in use_tools
                            if tool.get("function", {}).get("name") in candidate_set
                        ]
                        if filtered_tools:
                            logger.info(
                                f"🔍 Narrowing tools from {len(use_tools)} to {len(filtered_tools)} "
                                f"based on ranked candidates"
                            )
                            use_tools = filtered_tools

                        insert_at = 2 if memory_context else 1
                        llm_messages.insert(
                            insert_at,
                            {
                                "role": "system",
                                "content": (
                                    "Tool candidates for this request: "
                                    f"{', '.join(candidate_names)}. "
                                    "You MUST call one of these tools to answer."
                                )
                            }
                        )
            
            # Generate response using agent's LLM manager
            logger.info(f"Generating response with agent {agent_runtime.metadata.name}")
            logger.info(f"🔍 Agent tools - count: {len(use_tools) if use_tools else 0}, is_simple: {is_simple}")
            if use_tools:
                tool_names = [t.get("function", {}).get("name") for t in use_tools if isinstance(t, dict)]
                logger.info(f"🔍 Available tools: {tool_names}")
            
            response = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages,
                tools=use_tools
            )
            
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
                async for chunk in self._handle_tool_calls_aisdk(response, llm_messages, agent_runtime, tool_data, use_tools):
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
            self.chat_service.save_message(
                chat_id=chat_id,
                role="assistant",
                parts=assistant_parts,
                workspace_id=self.workspace_id
            )
            
            # Store memory (shared user-level)
            if latest_text and full_response:
                await self.memory_injector.store_conversation_memory(
                    chat_id,
                    latest_text,
                    full_response,
                    workspace_id=str(self.workspace_id) if self.workspace_id else None
                )
            
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
    
    async def _build_agent_system_prompt(self, agent_runtime) -> tuple[str, list]:
        """
        Build system prompt from agent's skills.
        
        Combines:
        1. Base chatbot prompt
        2. Agent-specific prompt (from description/type)
        3. Skills (prompt templates from database)
        
        Args:
            agent_runtime: AgentRuntime from factory
            
        Returns:
            Tuple of (system_prompt, tool_schemas)
        """
        from core.models import Skill
        
        # Base chatbot system prompt
        base_prompt = """You are a helpful AI assistant with access to various tools and knowledge.

CRITICAL INSTRUCTIONS:
- When asked to CREATE, WRITE, GENERATE, POST, SEND, or MESSAGE something, you MUST use your tools to actually do it
- DO NOT explain how to do something - ACTUALLY DO IT using function calls
- If you have a write_file or create_document tool, USE IT to create files
- If you have database query tools, USE THEM to get real data
- If you have communication tools (adapter_slack, adapter_github, etc.), USE THEM to send messages, post updates, create issues, etc.
- EXECUTE first, explain later (if needed)

Examples:
- "Create a report" → Call write_file tool with the report content
- "Query the database" → Call smart_query_database tool
- "Search for code" → Call search_codebase tool
- "Post a message to Slack" → Call adapter_slack tool with operation="chat.postMessage"
- "Send a message" → Call adapter_slack tool to actually send it
- "Create a GitHub issue" → Call adapter_github tool with appropriate operation

COMMUNICATION TOOLS:
- adapter_slack: Use this to POST messages to Slack channels, read conversations, upload files, add reactions
  - When user asks to "post", "send", "message" in Slack → Use adapter_slack with operation="chat.postMessage"
  - Required params: {"channel": "C123456", "text": "your message"}
- adapter_github: Use this to interact with GitHub repositories, create issues, manage PRs

Always be clear, concise, and ACTION-ORIENTED in your responses. When tools are available, USE THEM instead of explaining what you would do."""
        
        # Add agent-specific context
        agent_context = f"""

You are {agent_runtime.metadata.name}, a specialized {agent_runtime.metadata.agent_type} agent.

AGENT DESCRIPTION:
{agent_runtime.metadata.description or f"Specialized {agent_runtime.metadata.agent_type} with expertise in specific domains."}

YOUR SPECIALIZED SKILLS:
"""
        
        # Load and inject skills from database
        skills_text = ""
        if agent_runtime.metadata.skills:
            # Query database for skill details
            skills = self.db.query(Skill).filter(
                Skill.name.in_(agent_runtime.metadata.skills),
                Skill.is_active == True
            ).all()
            
            for skill in skills:
                if skill.prompt_template:
                    skills_text += f"\n{skill.prompt_template}\n"
                elif skill.description:
                    skills_text += f"\n- {skill.name}: {skill.description}\n"
        
        # Build complete prompt
        complete_prompt = f"{base_prompt}\n\n{agent_context}\n\n{skills_text}"
        
        # Extract tool schemas from skills for function calling
        tool_schemas = []
        if agent_runtime.metadata.skills:
            skills = self.db.query(Skill).filter(
                Skill.name.in_(agent_runtime.metadata.skills),
                Skill.is_active == True
            ).all()
            
            for skill in skills:
                if skill.content and isinstance(skill.content, dict):
                    tools_schema = skill.content.get('tools_schema', [])
                    if tools_schema:
                        logger.info(f"Extracted {len(tools_schema)} tools from skill '{skill.name}'")
                        tool_schemas.extend(tools_schema)
        
        return (complete_prompt, tool_schemas)
    
    async def _handle_tool_calls_aisdk(
        self,
        response,
        llm_messages: List[Dict],
        agent_runtime,
        tool_data: Dict,
        use_tools: List = None
    ) -> AsyncGenerator[str, None]:
        """
        Handle tool calls from agent's LLM response.
        
        Args:
            response: LLM response with tool_calls
            llm_messages: Current message history
            agent_runtime: Agent runtime with tools
            tool_data: Dict to store tool results
            use_tools: Tools to pass to LLM for next iteration
            
        Yields:
            AI SDK formatted tool execution chunks
        """
        import asyncio
        import time
        
        max_iterations = 5
        iteration = 0
        current_response = response
        
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
                yield self.streaming_handler.format_aisdk_tool_start(tool_id, tool_name)
                await asyncio.sleep(0)
                
                start_times[tool_id] = time.time()
                tool_calls_prepared.append((tool_id, tool_name, tool_call))
            
            # Execute tools
            tool_results = []
            for tool_id, tool_name, tool_call in tool_calls_prepared:
                try:
                    # Parse arguments
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    tool_args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    
                    # Execute tool via tool router
                    result = await self.tool_router.execute_and_format(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_id=agent_runtime.agent_id if hasattr(agent_runtime, 'agent_id') else 1
                    )
                    
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

                    if result.get('fatal_error'):
                        fatal_errors.append(result)
                    
                    # Emit tool-result as data event
                    yield self.streaming_handler.format_aisdk_data('tool-result', {
                        'toolCallId': tool_id,
                        'toolName': tool_name,
                        'result': llm_context[:500]  # Truncate for streaming
                    })
                    await asyncio.sleep(0)
                    
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
                agent_id=agent_id
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
                result = await self.tool_router.execute_and_format(tool_name, tool_args)
                
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

