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
import time
from types import SimpleNamespace
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
from consumers.chatbot.tool_router import get_tool_router

# Import from modules — SINGLE SOURCE for tool schemas
from modules.tools.tool_router import get_tools_for_agent

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL LOOP PREVENTION UTILITIES
# =============================================================================

def _normalize_query(query: str) -> str:
    """Normalize a search query for deduplication comparison."""
    if not query:
        return ""
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    normalized = ' '.join(normalized.split())
    return normalized


def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    """Check if two queries are semantically similar using string similarity."""
    norm1 = _normalize_query(query1)
    norm2 = _normalize_query(query2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def _extract_query_from_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """Extract the search/query parameter from tool arguments."""
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

    SEARCH_TOOLS = {
        'search_knowledge', 'semantic_search', 'search_codebase',
        'search_tables', 'search_images', 'search_formulas',
        'search_multimodal', 'smart_query_database', 'query_database'
    }

    TOOL_RETRY_LIMITS = {
        'composio_execute': 2,
        'search_knowledge': 2,
        'semantic_search': 2,
        'search_codebase': 2,
        'smart_query_database': 2,
        'query_database': 2,
        'list_directory': 2,
        'read_file': 3,
        'write_file': 2,
        'default': 3
    }

    def __init__(self):
        self.exact_executions: Set[Tuple[str, str]] = set()
        self.search_queries: Dict[str, List[str]] = {}
        self.tool_counts: Dict[str, int] = {}

    def _hash_args(self, tool_args: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()

    def should_skip_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if a tool execution should be skipped. Returns (should_skip, reason)."""
        current_count = self.tool_counts.get(tool_name, 0)
        limit = self.TOOL_RETRY_LIMITS.get(tool_name, self.TOOL_RETRY_LIMITS['default'])

        if current_count >= limit:
            return True, f"Tool '{tool_name}' has reached its execution limit ({limit}) for this turn"

        args_hash = self._hash_args(tool_args)
        exec_key = (tool_name, args_hash)
        if exec_key in self.exact_executions:
            return True, f"Tool '{tool_name}' was already executed with identical parameters"

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
        args_hash = self._hash_args(tool_args)
        self.exact_executions.add((tool_name, args_hash))
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                if tool_name not in self.search_queries:
                    self.search_queries[tool_name] = []
                self.search_queries[tool_name].append(query)

    def get_execution_count(self, tool_name: str) -> int:
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

        if chat.title == title:
            return True

        try:
            chat.title = title
            chat.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Updated chat {chat_id} title: {title}")
            return True
        except IntegrityError:
            self.db.rollback()
            base_title = title
            counter = 1
            while True:
                unique_title = f"{base_title} ({counter})"
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
    for match in reversed(matches):
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

    def __init__(self, db: Session, workspace_id: Optional[str] = None, widget_mode: bool = False):
        self.db = db
        self.chat_service = ChatService(db)
        self.prompt_analyzer = get_prompt_analyzer()
        self.tool_router = get_tool_router()
        self.streaming_handler = get_streaming_handler()
        self.workspace_id = workspace_id
        self.widget_mode = widget_mode

        from modules.agents.factory.agent_factory import AgentFactory
        self.agent_factory = AgentFactory(db_session=db)
        logger.info("StreamingChatService initialized with AgentFactory integration")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_file_parts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve document:// file parts to inline text content."""
        from sqlalchemy import text as sa_text

        resolved = []
        for msg in messages:
            parts = msg.get("parts")
            if not parts:
                resolved.append(msg)
                continue

            new_parts = []
            for part in parts:
                if part.get("type") == "file" and part.get("url", "").startswith("document://"):
                    doc_id_str = part["url"].replace("document://", "")
                    filename = part.get("filename", "uploaded file")
                    try:
                        doc_id = int(doc_id_str)
                        result = self.db.execute(
                            sa_text(
                                "SELECT content FROM document_chunks "
                                "WHERE document_id = :doc_id ORDER BY chunk_index"
                            ),
                            {"doc_id": doc_id}
                        )
                        chunks = result.fetchall()
                        if chunks:
                            content = "\n\n".join(row.content for row in chunks)
                            new_parts.append({
                                "type": "text",
                                "text": f"[Attached file: {filename}]\n{content}"
                            })
                            logger.info(f"[file-resolve] Injected document {doc_id} ({filename}, {len(chunks)} chunks) into chat context")
                        else:
                            new_parts.append({
                                "type": "text",
                                "text": f"[Attached file: {filename} — document is still being processed, content not yet available]"
                            })
                            logger.warning(f"[file-resolve] Document {doc_id} has no chunks yet")
                    except Exception as e:
                        logger.error(f"[file-resolve] Failed to resolve document {doc_id_str}: {e}")
                        new_parts.append({
                            "type": "text",
                            "text": f"[Attached file: {filename} — could not load content]"
                        })
                else:
                    new_parts.append(part)

            resolved.append({**msg, "parts": new_parts})

        return resolved

    def _resolve_workspace_id(self, agent_id: int) -> Optional[str]:
        logger.info(f"[chat] resolve_workspace_id current={self.workspace_id} agent={agent_id}")
        if self.workspace_id:
            return self.workspace_id
        try:
            from core.models import Agent as AgentModel
            agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent_row and agent_row.workspace_id:
                self.workspace_id = agent_row.workspace_id
                logger.info(f"Resolved workspace_id from agent {agent_id}")
        except Exception as exc:
            logger.warning(f"Failed to resolve workspace_id for agent {agent_id}: {exc}")
        return self.workspace_id

    def _extract_user_text(self, llm_messages: List[Dict[str, Any]]) -> str:
        """Extract the latest user message text from LLM messages."""
        for m in reversed(llm_messages):
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""

    def _parse_model_selection(self, selected_model: Optional[str]) -> tuple:
        """Parse model string to get provider and model."""
        if not selected_model:
            return None, None

        model = selected_model
        model_lower = selected_model.lower()

        if model_lower.startswith('gpt-') or model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
            provider = 'openai'
        elif model_lower.startswith('claude') or 'anthropic' in model_lower:
            provider = 'anthropic'
        elif model_lower.startswith('grok') or 'xai' in model_lower:
            provider = 'grok'
        elif model_lower.startswith('gemini') or 'google' in model_lower:
            provider = 'google'
        elif '/' in selected_model:
            provider = 'openrouter'
        else:
            provider = None

        return provider, model

    async def _load_agent_context(self, agent_runtime) -> dict:
        """
        Load agent-specific context: persona, description for chatbot identity injection.

        PRD-81: System prompt cache removed from AgentRuntime — ContextService is
        now the single prompt builder. This method only provides lightweight identity
        context for the chatbot's _inject_agent_identity().
        """
        persona = ""
        try:
            from core.models import Agent as AgentModel
            db_agent = self.db.query(AgentModel).filter(AgentModel.id == agent_runtime.agent_id).first()
            if db_agent:
                if getattr(db_agent, "use_custom_persona", False) and getattr(db_agent, "custom_persona_prompt", None):
                    persona = db_agent.custom_persona_prompt
                elif getattr(db_agent, "persona_id", None) and getattr(db_agent, "persona", None):
                    persona = db_agent.persona.system_prompt or ""
        except Exception as e:
            logger.warning(f"Failed to load persona for agent {agent_runtime.agent_id}: {e}")

        description = (
            f"You are {agent_runtime.metadata.name}, "
            f"a specialized {agent_runtime.metadata.agent_type} agent.\n"
            f"{agent_runtime.metadata.description or ''}"
        )

        return {
            "persona": persona,
            "description": description,
            "extra_context": "",
            "skill_tools": [],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Tool source — SINGLE SOURCE OF TRUTH
    # ─────────────────────────────────────────────────────────────────────

    def _get_tools(
        self,
        agent_id: int,
        skill_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all tools for an agent from the SINGLE source: modules.tools.tool_router.

        Returns full OpenAI-format tool schemas (ToolRegistry + ActionRegistry + Composio).
        Appends any skill-specific tool schemas from the agent runtime.
        """
        all_tools = get_tools_for_agent(
            agent_id=agent_id,
            db_session=self.db,
            workspace_id=self.workspace_id,
        )
        if skill_tools:
            all_tools = (all_tools or []) + skill_tools
        return all_tools

    # ─────────────────────────────────────────────────────────────────────
    # Message preparation
    # ─────────────────────────────────────────────────────────────────────

    async def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        agent_runtime,
        agent_ctx: dict,
        all_tools: List[Dict[str, Any]],
        chat_id: str,
        complexity_assessment: Optional[Any],
        is_cto_agent: bool,
        cto_check_result: Any,
        mission_mode: bool = False,
        plan_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Any]:
        """
        Prepare LLM messages with orchestration, persona, CTO override, and context guard.

        Returns:
            (llm_messages, use_tools, orchestrated)
        """
        import asyncio
        from consumers.chatbot.auto import Complexity
        from consumers.chatbot.integration import SmartChatIntegration, apply_orchestration_to_messages

        latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
        smart_chat = SmartChatIntegration(
            workspace_id=str(self.workspace_id) if self.workspace_id else self.workspace_id,
            agent_id=agent_runtime.agent_id,
            agent_name=agent_runtime.metadata.name,
            widget_mode=self.widget_mode,
            db_session=self.db,
        )

        _complexity = (
            complexity_assessment.complexity
            if complexity_assessment
            else Complexity.MOLECULE
        )

        if _complexity == Complexity.ATOM:
            llm_messages, use_tools, orchestrated = await self._prepare_atom_path(
                messages, agent_runtime, smart_chat
            )
        else:
            llm_messages, use_tools, orchestrated = await self._prepare_full_path(
                messages, agent_runtime, agent_ctx, all_tools,
                smart_chat, chat_id, complexity_assessment,
            )

        # PRD-67: CTO Agent system prompt override
        if is_cto_agent:
            self._apply_cto_override(
                llm_messages, smart_chat, cto_check_result,
                messages, use_tools, agent_runtime.agent_id,
            )

        # Inject agent persona + description (skip for CTO — soul document already includes it)
        if not is_cto_agent and orchestrated:
            self._inject_agent_identity(llm_messages, agent_ctx)

        # Multi-step execution policy
        insert_pos = 2 if (not is_cto_agent and agent_ctx.get("extra_context")) else 1
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

        # Plan mode: inject research-focused system prompt, disable tools
        if plan_mode:
            llm_messages.insert(
                1,
                {
                    "role": "system",
                    "content": (
                        "You are in PLAN MODE. Your role is to research, analyze, and produce a structured plan. "
                        "Do NOT execute actions, call tools, or make changes. Focus entirely on strategy and planning.\n\n"
                        "Output format:\n"
                        "1. **Goal Summary** — 2-3 sentence overview of what will be accomplished\n"
                        "2. **Steps** — Numbered list of concrete steps with agent assignments and tool recommendations\n"
                        "3. **Dependencies** — What needs to happen before each step\n"
                        "4. **Risks & Mitigations** — Potential failure points and how to handle them\n"
                        "5. **Success Criteria** — How to verify the plan worked\n\n"
                        "The user will iterate on this plan conversationally. When they are satisfied, "
                        "they can launch it as a Mission. Keep plans actionable and specific — "
                        "name the agents, tools, and data sources that should be used."
                    ),
                },
            )
            use_tools = None  # Disable tool execution in plan mode

        # Context Window Guard
        from core.context_guard import ContextGuard
        _guard = ContextGuard()
        _model_name = getattr(agent_runtime.llm_manager, 'config', None)
        _model_name = getattr(_model_name, 'model', config.LLM_MODEL) if _model_name else config.LLM_MODEL
        llm_messages, _was_compacted, use_tools = await _guard.check_and_compact(
            messages=llm_messages,
            model_name=_model_name,
            llm_manager=agent_runtime.llm_manager,
            workspace_id=str(self.workspace_id) if self.workspace_id else None,
            agent_id=agent_runtime.agent_id,
            db_session=self.db,
            tools=use_tools,
        )
        if _was_compacted:
            logger.info("[ContextGuard] Messages compacted before LLM call")

        # Stash smart_chat on self for memory storage in post_response
        self._smart_chat = smart_chat

        return llm_messages, use_tools, orchestrated

    async def _prepare_atom_path(
        self,
        messages: List[Dict[str, Any]],
        agent_runtime,
        smart_chat,
    ) -> Tuple[List[Dict[str, Any]], None, None]:
        """ATOM path: no tools, no orchestration, lightweight memory only."""
        logger.info("[PRD-68] ATOM path — skipping tools/orchestration, retrieving memory")
        _now = datetime.utcnow()
        _time_ctx = (
            "Good morning" if _now.hour < 12
            else "Good afternoon" if _now.hour < 18
            else "Good evening"
        )

        _memory_block = ""
        try:
            _user_msg = next(
                (m.get("content", "") for m in reversed(messages)
                 if isinstance(m, dict) and m.get("role") == "user"),
                ""
            )
            if _user_msg and smart_chat.orchestrator and smart_chat.orchestrator.memory_manager:
                _mem_result = await smart_chat.orchestrator.memory_manager.retrieve_memories(
                    workspace_id=str(self.workspace_id),
                    agent_id=agent_runtime.agent_id,
                    query=_user_msg if len(_user_msg) > 5 else "user context",
                    widget_mode=self.widget_mode,
                )
                if _mem_result and _mem_result.formatted_context:
                    _memory_block = f"\n\n## What you remember about this user:\n{_mem_result.formatted_context}\n"
                    logger.info(f"[PRD-68] ATOM memory: {len(_mem_result.memories)} memories injected")
        except Exception as _mem_err:
            logger.debug(f"[PRD-68] ATOM memory retrieval skipped: {_mem_err}")

        _atom_prompt = (
            f"You are {agent_runtime.metadata.name}, an AI assistant on the Automatos platform.\n\n"
            f"{_time_ctx}. Read the conversation and match the user's energy. "
            "If they're frustrated, be direct — skip the niceties and lead with the answer. "
            "If they're curious, explain the why. If they're casual, be casual back. "
            "If they're formal, match it. Never be artificially cheerful when someone is having a bad time. "
            "Never be robotic when someone is being warm.\n\n"
            "You adapt. That's what makes you good at this.\n"
            f"{_memory_block}"
        )
        llm_messages = self.prompt_analyzer.convert_to_llm_messages(
            messages, system_prompt=_atom_prompt, available_tools=None
        )
        return llm_messages, None, None

    async def _prepare_full_path(
        self,
        messages: List[Dict[str, Any]],
        agent_runtime,
        agent_ctx: dict,
        all_tools: List[Dict[str, Any]],
        smart_chat,
        chat_id: str,
        complexity_assessment: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Any]:
        """Full pipeline: MOLECULE / CELL / ORGAN / ORGANISM."""
        from consumers.chatbot.integration import apply_orchestration_to_messages

        llm_messages = self.prompt_analyzer.convert_to_llm_messages(
            messages, system_prompt="", available_tools=all_tools
        )

        orchestrated = await smart_chat.prepare(
            messages=llm_messages,
            available_tools=all_tools or [],
            chat_id=chat_id,
            complexity_assessment=complexity_assessment,
        )
        llm_messages = apply_orchestration_to_messages(orchestrated)
        use_tools = orchestrated.tools if orchestrated.requires_tools else None

        return llm_messages, use_tools, orchestrated

    def _apply_cto_override(
        self,
        llm_messages: List[Dict[str, Any]],
        smart_chat,
        cto_check_result,
        messages: List[Dict[str, Any]],
        use_tools: Optional[List[Dict[str, Any]]],
        agent_id: int,
    ) -> None:
        """PRD-67: Replace system prompt with CTO soul document."""
        from consumers.chatbot.cto_prompt_builder import CtoPromptBuilder

        _cto_memories = []
        _mem_result = getattr(smart_chat.orchestrator, '_last_memory_result', None)
        if _mem_result:
            _cto_memories = [m.get("memory", "") for m in _mem_result.memories if m.get("memory")]

        _platform_state = CtoPromptBuilder.get_platform_state_snapshot(self.db)
        _soul = cto_check_result.custom_persona_prompt or ""
        _config = cto_check_result.configuration
        _arch_ctx = ""
        if _config and isinstance(_config, dict):
            _arch_ctx = _config.get("extra_context", "")

        _cto_prompt = CtoPromptBuilder.build(
            soul_document=_soul,
            architecture_context=_arch_ctx,
            user_name=None,
            msg_count=len(messages),
            memories=_cto_memories,
            tool_names=[t.get("function", {}).get("name", "") for t in (use_tools or []) if isinstance(t, dict)],
            platform_state=_platform_state,
        )

        if llm_messages and llm_messages[0].get("role") == "system":
            llm_messages[0]["content"] = _cto_prompt
        else:
            llm_messages.insert(0, {"role": "system", "content": _cto_prompt})

        logger.info(f"[CTO] System prompt replaced for CTO Agent (agent_id={agent_id})")

    def _inject_agent_identity(
        self,
        llm_messages: List[Dict[str, Any]],
        agent_ctx: dict,
    ) -> None:
        """Inject agent persona + description after orchestrator system prompt."""
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

    # ─────────────────────────────────────────────────────────────────────
    # Composio per-action tool injection
    # ─────────────────────────────────────────────────────────────────────

    def _inject_composio_tools(
        self,
        llm_messages: List[Dict[str, Any]],
        use_tools: Optional[List[Dict[str, Any]]],
        latest_text: str,
        agent_id: int,
        agent_runtime,
        skip_composio: bool,
        complexity_assessment: Optional[Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Any]:
        """
        Inject Composio per-action tools (primary) or hint fallback.
        Returns (updated use_tools, composio_result).
        """
        _composio_result = None
        _tool_hints = (
            complexity_assessment.tool_hints
            if complexity_assessment and hasattr(complexity_assessment, 'tool_hints')
            else []
        )
        try:
            if latest_text and agent_id and self.workspace_id and not skip_composio:
                from modules.tools.services.composio_tool_service import ComposioToolService

                _composio_svc = ComposioToolService(self.db)
                _search_prompt = (
                    " ".join(_tool_hints) if _tool_hints else latest_text
                )
                _composio_result = _composio_svc.get_tools_for_step(
                    agent_id=agent_id,
                    workspace_id=self.workspace_id,
                    task_prompt=_search_prompt,
                    tool_hints=_tool_hints,
                )
                if _composio_result and _composio_result.tools:
                    if use_tools:
                        use_tools = [
                            t for t in use_tools
                            if t.get("function", {}).get("name") != "composio_execute"
                        ] + _composio_result.tools
                    else:
                        use_tools = _composio_result.tools
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
                    from modules.tools.services.composio_hint_service import ComposioHintService

                    hint_service = ComposioHintService(self.db)
                    hint_result = hint_service.build_hints(
                        agent_id=agent_id,
                        prompt=latest_text,
                        workspace_id=self.workspace_id,
                    )
                    if hint_result.hint_lines:
                        llm_messages.insert(2, {"role": "system", "content": "\n".join(hint_result.hint_lines)})
                        logger.info(
                            f"[Composio Hints fallback] Agent {agent_id}: strategy={hint_result.strategy_used} "
                            f"apps={hint_result.allowed_apps} matches={len(hint_result.matched_actions)}"
                        )
        except Exception as exc:
            logger.warning(f"Composio tool injection failed for agent {agent_id}: {exc}")

        return use_tools, _composio_result

    # ─────────────────────────────────────────────────────────────────────
    # Post-response: memory, metrics, eval
    # ─────────────────────────────────────────────────────────────────────

    async def _post_response(
        self,
        latest_text: str,
        full_response: str,
        chat_id: str,
        agent_runtime,
        agent_id: int,
        response,
        orchestrated: Any,
    ) -> AsyncGenerator[str, None]:
        """Store memory, emit memory-stored event, update metrics, fire eval."""
        import asyncio

        smart_chat = getattr(self, '_smart_chat', None)

        # Store memory via SmartChatIntegration
        if latest_text and full_response and smart_chat:
            try:
                _stored = await smart_chat.store(latest_text, full_response, chat_id)
                if _stored:
                    _tier = getattr(smart_chat.orchestrator.memory_manager, '_last_tier', 'conversation')
                    yield self.streaming_handler.format_aisdk_memory_stored(
                        memory={
                            "userMessage": latest_text[:200],
                            "assistantResponse": full_response[:200],
                            "chatId": chat_id,
                        },
                        reason=_tier if isinstance(_tier, str) else "conversation",
                    )
                    await asyncio.sleep(0)
            except Exception as mem_err:
                logger.warning(f"Failed to store memory exchange: {mem_err}")

        # FutureAGI live traffic eval (fire-and-forget)
        if latest_text and full_response:
            try:
                from core.services.futureagi_service import futureagi_service
                if futureagi_service.is_available:
                    asyncio.create_task(
                        futureagi_service.eval_live_traffic(
                            input_text=latest_text,
                            output_text=full_response,
                            context_text=orchestrated.system_prompt if orchestrated else "",
                        )
                    )
            except Exception:
                pass

        # Update agent metrics
        if hasattr(agent_runtime, 'update_metrics'):
            tokens_used = response.usage.get('total_tokens', 0) if response.usage else 0
            agent_runtime.update_metrics(
                execution_time=1.0,
                tokens_used=tokens_used,
                success=True
            )

        # Persist task counter to DB
        try:
            from core.models import Agent as AgentModel
            from sqlalchemy.orm.attributes import flag_modified
            agent_row = self.db.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent_row:
                metrics = dict(agent_row.performance_metrics or {})
                total = metrics.get("total_tasks_executed", 0) + 1
                successes = metrics.get("success_count", 0) + 1
                metrics["total_tasks_executed"] = total
                metrics["tasks_completed"] = total
                metrics["success_count"] = successes
                metrics["success_rate"] = round(successes / total, 4) if total > 0 else 0
                metrics["last_task_at"] = datetime.now(timezone.utc).isoformat()
                metrics["last_task_success"] = True
                agent_row.performance_metrics = metrics
                flag_modified(agent_row, "performance_metrics")
                self.db.commit()
        except Exception as metric_err:
            logger.warning(f"Failed to persist agent task counter: {metric_err}")
            try:
                self.db.rollback()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    # Unified tool loop
    # ─────────────────────────────────────────────────────────────────────

    async def _run_tool_loop(
        self,
        response,
        llm_messages: List[Dict[str, Any]],
        agent_runtime,
        tool_data: Dict[str, Any],
        use_tools: Optional[List[Dict[str, Any]]],
        composio_result: Any = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Unified tool execution loop with dedup, retry limits, and Composio recovery.

        Yields SSE chunks and a final {'_final_response': response} dict.
        Consolidates _handle_tool_calls_aisdk and _handle_tool_calls.
        """
        import asyncio

        max_iterations = 10
        iteration = 0
        current_response = response
        tracker = ToolExecutionTracker()

        # Recovery budgets
        action_not_mapped_retry_budget = 1
        invalid_parameters_retry_budget = 1

        # Search spiral detection
        last_tool_name: Optional[str] = None
        empty_same_tool_streak = 0

        # Per-tool attempt tracking for loop prevention
        tool_attempts: Dict[str, int] = {}

        # Multi-step tools that get a higher retry cap
        _MULTI_STEP_TOOLS = {
            "composio_execute",
            "read_file", "write_file", "list_directory", "create_directory", "delete_file",
            "generate_document",
            "workspace_read_file", "workspace_grep", "workspace_list_dir",
            "workspace_write_file", "workspace_create_directory",
        }

        while current_response.tool_calls and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}: {len(current_response.tool_calls)} tool calls")

            start_times: Dict[str, float] = {}
            tool_calls_prepared: List[Tuple[str, str, Dict]] = []
            fatal_errors: List[Dict[str, Any]] = []
            followup_system_messages: List[Dict[str, Any]] = []
            executed_call_key_repeat = False

            # Phase 1: Emit tool-start events
            for tool_call in current_response.tool_calls:
                tool_name = tool_call.get('function', {}).get('name', 'unknown')
                tool_id = tool_call.get('id', f'call_{int(time.time() * 1000)}')

                try:
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    tool_input = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    tool_input = {}

                yield self.streaming_handler.format_aisdk_tool_start(tool_id, tool_name, tool_input=tool_input)
                await asyncio.sleep(0)

                start_times[tool_id] = time.time()
                tool_calls_prepared.append((tool_id, tool_name, tool_call))

            # Phase 2: Execute each tool
            # Concurrency classification: log batch composition for future
            # parallel execution optimization (free-code pattern).
            try:
                from modules.tools.execution.concurrency import partition_tool_batch
                _read_safe, _mutating = partition_tool_batch(tool_calls_prepared)
                if len(_read_safe) > 1 and len(_mutating) == 0:
                    logger.info(
                        f"[tool-batch] All {len(_read_safe)} tools are read-safe — "
                        f"eligible for parallel execution"
                    )
            except Exception:
                pass  # Classification is non-critical

            tool_results: List[Dict[str, Any]] = []
            for tool_id, tool_name, tool_call in tool_calls_prepared:
                try:
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    tool_args = json.loads(args_str or '{}') if isinstance(args_str, str) else (args_str or {})

                    # ── Dedup check via ToolExecutionTracker ──
                    should_skip, skip_reason = tracker.should_skip_execution(tool_name, tool_args)
                    if should_skip:
                        executed_call_key_repeat = True
                        llm_context = f"Skipped: {skip_reason}"
                        tool_results.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": llm_context,
                        })
                        yield self.streaming_handler.format_aisdk_data(
                            "tool-result",
                            {"toolCallId": tool_id, "toolName": tool_name, "result": llm_context},
                        )
                        await asyncio.sleep(0)
                        continue

                    # Record execution
                    tracker.record_execution(tool_name, tool_args)

                    # ── Execute via tool_router.execute_and_format ──
                    user_text = self._extract_user_text(llm_messages)

                    # Direct Composio action execution for per-action tools
                    _is_composio_action = (
                        composio_result and composio_result.entity_id and (
                            tool_name in composio_result.action_set
                            or any(tool_name.startswith(f"{app}_") for app in composio_result.app_names)
                        )
                    )
                    if _is_composio_action:
                        llm_context = await self._execute_composio_action(
                            tool_name, tool_args, composio_result
                        )
                    else:
                        result = await self.tool_router.execute_and_format(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            agent_id=agent_runtime.agent_id if hasattr(agent_runtime, 'agent_id') else 1,
                            workspace_id=self.workspace_id,
                            original_intent=user_text,
                        )

                        # Track empty search results for spiral detection
                        empty_same_tool_streak = self._track_search_spiral(
                            result, tool_name, last_tool_name, empty_same_tool_streak
                        )
                        last_tool_name = tool_name

                        # Track per-tool attempts
                        tool_attempts[tool_name] = tool_attempts.get(tool_name, 0) + 1

                        llm_context = result.get('llm_context', str(result.get('raw_result', '')))
                        if len(llm_context) > 6000:
                            llm_context = llm_context[:6000] + f"\n... (truncated {len(llm_context) - 6000} chars)"

                        # Emit frontend data (tool-data for widgets)
                        frontend_data = result.get("frontend_data", {})
                        if result.get("success") and frontend_data:
                            tool_data.update(frontend_data)
                            yield self.streaming_handler.format_aisdk_tool_data(frontend_data)
                            await asyncio.sleep(0)

                        # Emit workflow-update for recipe/workflow tools
                        _WORKFLOW_TOOL_PREFIXES = ("platform_list_recipes", "platform_create_recipe", "platform_execute_recipe")
                        if tool_name.startswith(_WORKFLOW_TOOL_PREFIXES) or "workflow" in tool_name.lower():
                            _raw = result.get("raw_result") or {}
                            _wf_id = str(_raw.get("id") or _raw.get("workflow_id") or _raw.get("recipe_id") or tool_id)
                            _wf_status = "completed" if result.get("success") else "failed"
                            yield self.streaming_handler.format_aisdk_workflow_update(
                                workflow_id=_wf_id, status=_wf_status, current_step=tool_name,
                            )
                            await asyncio.sleep(0)

                        # ── Composio error recovery ──
                        recovery_result = self._handle_composio_error_recovery(
                            result, tool_name, llm_messages, agent_runtime,
                            action_not_mapped_retry_budget, invalid_parameters_retry_budget,
                            followup_system_messages,
                        )
                        if recovery_result is not None:
                            if recovery_result.get("_early_return"):
                                yield {"_final_response": SimpleNamespace(
                                    content=recovery_result["message"],
                                    tool_calls=None, usage=None,
                                )}
                                return
                            action_not_mapped_retry_budget = recovery_result.get(
                                "action_not_mapped_retry_budget", action_not_mapped_retry_budget
                            )
                            invalid_parameters_retry_budget = recovery_result.get(
                                "invalid_parameters_retry_budget", invalid_parameters_retry_budget
                            )

                        if result.get('fatal_error'):
                            fatal_errors.append(result)

                    # Store tool result
                    tool_results.append({
                        'tool_call_id': tool_id,
                        'role': 'tool',
                        'name': tool_name,
                        'content': llm_context,
                    })

                    # Emit tool-end + tool-result events
                    duration_ms = int((time.time() - start_times.get(tool_id, time.time())) * 1000)
                    yield self.streaming_handler.format_aisdk_tool_end(
                        tool_call_id=tool_id, tool_name=tool_name, success=True, duration_ms=duration_ms,
                    )
                    yield self.streaming_handler.format_aisdk_data('tool-result', {
                        'toolCallId': tool_id,
                        'toolName': tool_name,
                        'result': llm_context[:500],
                    })
                    await asyncio.sleep(0)

                    # ── Loop prevention: inject proceed instructions ──
                    self._inject_loop_prevention(
                        llm_messages, tool_name, tool_attempts,
                        empty_same_tool_streak, result if not _is_composio_action else {"success": True},
                        agent_runtime, _MULTI_STEP_TOOLS,
                    )
                    # Database tool: force synthesis after first success
                    if (not _is_composio_action
                            and tool_name in {"query_database", "smart_query_database"}
                            and result.get("success")):
                        llm_messages.append({
                            "role": "system",
                            "content": (
                                "You now have the database result. Do NOT call the database tool again. "
                                "Write the final answer using the tool output above."
                            ),
                        })
                        # Append tool exchange before forcing synthesis
                        llm_messages.append(self._build_assistant_tool_message(tool_calls_prepared))
                        llm_messages.extend(tool_results)
                        final = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                        yield {"_final_response": SimpleNamespace(
                            content=final.content or "", tool_calls=None,
                            usage=getattr(final, "usage", None),
                        )}
                        return

                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    tool_results.append({
                        'tool_call_id': tool_id,
                        'role': 'tool',
                        'name': tool_name,
                        'content': error_msg,
                    })
                    yield self.streaming_handler.format_aisdk_data('tool-result', {
                        'toolCallId': tool_id,
                        'toolName': tool_name,
                        'result': error_msg,
                    })
                    await asyncio.sleep(0)

            # Phase 3: Append tool exchange to message history
            llm_messages.append(self._build_assistant_tool_message(tool_calls_prepared))
            llm_messages.extend(tool_results)

            if followup_system_messages:
                llm_messages.extend(followup_system_messages)

            # Phase 4: Force synthesis if duplicate or exhausted
            _any_tool_exhausted = any(v >= 8 for v in tool_attempts.values())
            if executed_call_key_repeat or _any_tool_exhausted:
                if _any_tool_exhausted:
                    logger.warning(f"[tool-loop] Tool hard cap reached — forcing synthesis (attempts: {dict(tool_attempts)})")
                llm_messages.append({
                    "role": "system",
                    "content": (
                        "You now have the tool results needed. "
                        "Do NOT call any more tools. "
                        "Write the final answer for the user using the tool output above."
                    ),
                })
                final = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                yield {"_final_response": SimpleNamespace(
                    content=final.content or "", tool_calls=None,
                    usage=getattr(final, "usage", None),
                )}
                return

            if fatal_errors:
                yield {"_final_response": SimpleNamespace(
                    content=(
                        "I ran into a server configuration issue while executing that tool. "
                        "Please restart the backend and try again."
                    ),
                    tool_calls=None, usage=None,
                )}
                return

            # Phase 5: Next LLM call — withhold errors during recovery
            try:
                current_response = await agent_runtime.llm_manager.generate_response(
                    messages=llm_messages, tools=use_tools,
                )
            except Exception as llm_err:
                # Withhold error: attempt compaction + retry before surfacing
                logger.warning(
                    f"LLM call failed (iteration {iteration}), attempting recovery: {llm_err}"
                )
                try:
                    from core.context_guard import ContextGuard
                    guard = ContextGuard()
                    llm_messages, compacted, use_tools = await guard.check_and_compact(
                        llm_messages, use_tools, agent_runtime.llm_manager,
                        workspace_id=self.workspace_id,
                    )
                    if compacted:
                        logger.info("Recovery compaction succeeded, retrying LLM call")
                        current_response = await agent_runtime.llm_manager.generate_response(
                            messages=llm_messages, tools=use_tools,
                        )
                    else:
                        raise  # No compaction possible, surface original error
                except Exception:
                    logger.error(f"Recovery failed, surfacing error: {llm_err}", exc_info=True)
                    raise llm_err
            logger.info(f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}, Has content: {bool(current_response.content)}")

            if not current_response.tool_calls:
                yield {'_final_response': current_response}
                return

        # Max iterations reached
        if iteration >= max_iterations:
            logger.warning(f"Max tool iterations ({max_iterations}) reached. Forcing final response.")
            final = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages, tools=None,
            )
            yield {'_final_response': final}

    def _build_assistant_tool_message(
        self,
        tool_calls_prepared: List[Tuple[str, str, Dict]],
    ) -> Dict[str, Any]:
        """Build the assistant message with tool_calls (required by OpenAI API)."""
        return {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': tc[0],
                    'type': 'function',
                    'function': {
                        'name': tc[1],
                        'arguments': tc[2].get('function', {}).get('arguments', '{}'),
                    },
                }
                for tc in tool_calls_prepared
            ],
        }

    async def _execute_composio_action(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        composio_result: Any,
    ) -> str:
        """Execute a Composio per-action tool directly."""
        try:
            from modules.tools.services.composio_tool_service import ComposioToolService
            _exec_svc = ComposioToolService(self.db)
            exec_result = _exec_svc.execute_action(
                action_name=tool_name,
                params=tool_args,
                entity_id=composio_result.entity_id,
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
            llm_context = llm_context[:4000] + "\n... (truncated)"
        return llm_context

    def _track_search_spiral(
        self,
        result: Dict[str, Any],
        tool_name: str,
        last_tool_name: Optional[str],
        empty_same_tool_streak: int,
    ) -> int:
        """Track consecutive empty search results for spiral detection."""
        try:
            raw = result.get("raw_result") or {}
            count = raw.get("count")
            if count is None:
                rr = raw.get("results")
                if isinstance(rr, list):
                    count = len(rr)
            is_search_tool = tool_name.startswith("search_") or tool_name in {"semantic_search"}
            is_empty = isinstance(count, int) and count == 0
            if is_search_tool and is_empty:
                if last_tool_name == tool_name:
                    return empty_same_tool_streak + 1
                return 1
        except Exception:
            pass
        return 0

    def _handle_composio_error_recovery(
        self,
        result: Dict[str, Any],
        tool_name: str,
        llm_messages: List[Dict[str, Any]],
        agent_runtime,
        action_not_mapped_retry_budget: int,
        invalid_parameters_retry_budget: int,
        followup_system_messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Handle Composio-specific error recovery (action-not-mapped, invalid-parameters).
        Returns None if no recovery needed, dict with recovery state otherwise.
        """
        if result.get("success"):
            return None

        error_type = result.get("error_type")
        raw_error = (result.get("raw_result") or {}).get("error") if isinstance(result.get("raw_result"), dict) else None

        # Recovery for action_not_mapped
        if error_type == "composio_action_not_mapped" and action_not_mapped_retry_budget > 0:
            user_text = self._extract_user_text(llm_messages)
            if raw_error and "Examples of mapped actions:" in raw_error:
                examples = raw_error.split("Examples of mapped actions:", 1)[1].strip()
                top = self._score_composio_candidates(user_text, examples)
                if not top:
                    return {
                        "_early_return": True,
                        "message": (
                            "That action is not available in the local integrations cache for this workspace/agent. "
                            "The system won't guess a different action (to avoid doing the wrong thing). "
                            "Please run a Composio sync to refresh `composio_actions_cache` for this app, then retry."
                        ),
                    }
                followup_system_messages.append({
                    "role": "system",
                    "content": (
                        "The previous Composio action name was not mapped. "
                        "Retry using ONE of these exact mapped action names that best matches the user's request:\n"
                        f"{', '.join(top)}\n"
                        "Use `composio_execute` again with the corrected `action`."
                    ),
                })
            else:
                followup_system_messages.append({
                    "role": "system",
                    "content": (
                        "The previous Composio action name was not mapped. "
                        "Retry using a valid mapped action from `composio_actions_cache`."
                    ),
                })
            return {"action_not_mapped_retry_budget": action_not_mapped_retry_budget - 1}

        # Recovery for invalid_parameters on composio_execute
        if (
            tool_name == "composio_execute"
            and error_type == "invalid_parameters"
            and invalid_parameters_retry_budget > 0
        ):
            self._build_composio_param_recovery(
                llm_messages, agent_runtime, followup_system_messages
            )
            return {"invalid_parameters_retry_budget": invalid_parameters_retry_budget - 1}

        # Deterministic errors — stop immediately (unless we have followup instructions)
        deterministic_error_types = {
            "composio_not_assigned",
            "composio_not_connected",
            "composio_action_not_allowed",
            "composio_missing_workspace",
            "invalid_parameters",
        }
        if followup_system_messages:
            return None
        if error_type in deterministic_error_types:
            raw_error_msg = raw_error or result.get('llm_context', '') or "That tool is not available for this agent/workspace."
            return {"_early_return": True, "message": raw_error_msg}

        return None

    def _score_composio_candidates(self, user_text: str, examples_str: str) -> List[str]:
        """Score Composio action candidates against the user's query."""
        q = (user_text or "").lower()
        q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
        stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
        q_tokens = [t for t in q_tokens if t not in stop][:12]

        candidates = [c.strip() for c in examples_str.split(",") if c.strip()]
        scored = []
        for c in candidates:
            ct = c.lower()
            score = sum(1 for tok in q_tokens if tok in ct)
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for score, c in scored if score > 0][:8]

    def _build_composio_param_recovery(
        self,
        llm_messages: List[Dict[str, Any]],
        agent_runtime,
        followup_system_messages: List[Dict[str, Any]],
    ) -> None:
        """Build recovery instructions for composio_execute missing parameters."""
        try:
            from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
            from core.composio.entity_manager import EntityManager

            user_text = self._extract_user_text(llm_messages)
            q = (user_text or "").lower()
            q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
            stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
            q_tokens = [t for t in q_tokens if t not in stop][:10]

            aid = agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else 0
            assigned = (
                self.db.query(AgentAppAssignment)
                .filter(
                    AgentAppAssignment.agent_id == aid,
                    AgentAppAssignment.is_active == True,
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
            followup_system_messages.append({
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
            })
        except Exception:
            followup_system_messages.append({
                "role": "system",
                "content": (
                    "Your previous `composio_execute` call was missing required parameters. "
                    "Retry with an explicit mapped `action` and any required `params`."
                ),
            })

    def _inject_loop_prevention(
        self,
        llm_messages: List[Dict[str, Any]],
        tool_name: str,
        tool_attempts: Dict[str, int],
        empty_same_tool_streak: int,
        result: Dict[str, Any],
        agent_runtime,
        multi_step_tools: set,
    ) -> None:
        """Inject system messages to prevent tool loops."""
        # Search spiral: 2+ consecutive empty results from same tool
        if empty_same_tool_streak >= 2 and (
            tool_name.startswith("search_") or tool_name in {"semantic_search"}
        ):
            llm_messages.append({
                "role": "system",
                "content": (
                    f"The tool `{tool_name}` returned no results after multiple attempts. "
                    "STOP calling search tools. Use the information you already have "
                    "and proceed to fulfill the user's request with your other available tools."
                ),
            })
            logger.info(f"[tool-loop] Search spiral detected for {tool_name} — injecting proceed instruction")

        # Per-tool retry limits
        _is_multi_step = (
            tool_name in multi_step_tools
            or tool_name.startswith("composio_")
            or tool_name.startswith("workspace_")
        )
        _attempts = tool_attempts.get(tool_name, 0)

        if _is_multi_step and _attempts >= 8:
            llm_messages.append({
                "role": "system",
                "content": (
                    f"STOP: `{tool_name}` has been called {_attempts} times. "
                    "You MUST now synthesize a response from the results you have. "
                    "Do NOT call any more tools."
                ),
            })
            logger.warning(f"[tool-loop] Multi-step tool {tool_name} hit hard cap ({_attempts} calls) — forcing synthesis")
        elif not _is_multi_step and _attempts >= 2:
            llm_messages.append({
                "role": "system",
                "content": (
                    f"You have already called `{tool_name}` multiple times. "
                    f"Do NOT call `{tool_name}` again. "
                    "Use the results you already have and proceed to fulfill the user's request "
                    "with your other available tools."
                ),
            })
            logger.info(f"[tool-loop] Tool {tool_name} hit retry limit — injecting proceed instruction")

    # ─────────────────────────────────────────────────────────────────────
    # Main streaming methods
    # ─────────────────────────────────────────────────────────────────────

    async def stream_response_with_agent(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        agent_id: int,
        user_id: int,
        use_system_llm: bool = False,
        skip_composio: bool = False,
        complexity_assessment: Optional[Any] = None,
        mission_mode: bool = False,
        plan_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response produced by the specified agent.
        Yields AISDK-formatted chunks for frontend consumption.
        """
        import asyncio

        try:
            # Ensure workspace_id is available
            if not self.workspace_id:
                self._resolve_workspace_id(agent_id)

            # Parse user text and handle fresh start
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            fresh_start = self.prompt_analyzer.is_fresh_start_request(latest_text)
            if fresh_start:
                messages = [m for m in messages if m.get("role") == "user"][-1:]

            # Start agent activation (concurrent with chat-id emission)
            agent_task = asyncio.create_task(self.agent_factory.activate_agent(agent_id, use_system_llm=use_system_llm))

            # Send chat_id to frontend
            yield self.streaming_handler.format_aisdk_chat_id(chat_id)
            await asyncio.sleep(0)

            # Await agent activation
            agent_runtime = await agent_task
            if not agent_runtime:
                raise Exception(f"Failed to activate agent {agent_id}")

            logger.info(f"Activating agent {agent_id} for chat {chat_id}")

            # PRD-67: Single query for CTO detection
            _cto_check_result = None
            try:
                from core.models import Agent as _AgentModel
                _cto_check_result = self.db.query(
                    _AgentModel.slug, _AgentModel.is_system_agent,
                    _AgentModel.custom_persona_prompt, _AgentModel.configuration,
                ).filter(_AgentModel.id == agent_id).first()
            except Exception:
                pass

            # Send agent info to frontend
            _agent_info = {
                "type": "agent-info",
                "agent": {
                    "id": agent_runtime.agent_id,
                    "name": agent_runtime.metadata.name,
                    "type": agent_runtime.metadata.agent_type,
                    "skills": agent_runtime.metadata.skills
                }
            }
            _is_cto_agent = bool(
                _cto_check_result
                and _cto_check_result.is_system_agent
                and _cto_check_result.slug == "auto-cto"
            )
            if _is_cto_agent:
                _agent_info["agent"]["is_cto"] = True
                _agent_info["agent"]["name"] = "Auto CTO"
            yield self.streaming_handler.format_aisdk_data(_agent_info)
            await asyncio.sleep(0)

            # Resolve file attachments
            messages = self._resolve_file_parts(messages)

            # Load agent context (persona, skills, etc.)
            from consumers.chatbot.auto import Complexity
            _complexity = (
                complexity_assessment.complexity
                if complexity_assessment
                else Complexity.MOLECULE
            )
            agent_ctx = {}
            all_tools = []
            if _complexity != Complexity.ATOM:
                agent_ctx = await self._load_agent_context(agent_runtime)
                all_tools = self._get_tools(agent_id, agent_ctx.get("skill_tools"))
            else:
                # ATOM path skips full tool loading, but always include
                # platform_execute so the agent can respond to platform
                # queries even when the classifier under-estimates complexity.
                try:
                    from modules.tools.discovery.action_registry import get_action_registry
                    _dispatcher = get_action_registry().to_dispatcher_schema(exclude_admin=True)
                    all_tools = [_dispatcher]
                except Exception:
                    logger.debug("[chat] Could not load platform_execute for ATOM path")

            # Prepare messages (orchestration, persona, CTO override, context guard)
            llm_messages, use_tools, orchestrated = await self._prepare_messages(
                messages, agent_runtime, agent_ctx, all_tools,
                chat_id, complexity_assessment, _is_cto_agent, _cto_check_result,
                mission_mode=mission_mode,
                plan_mode=plan_mode,
            )

            if orchestrated:
                logger.info(
                    f"[SmartChat] intent={orchestrated.intent.value} "
                    f"tools={len(use_tools) if use_tools else 0} "
                    f"memory={'yes' if orchestrated.memory_context else 'no'} "
                    f"prep={orchestrated.preparation_time_ms:.0f}ms"
                )
            else:
                logger.info("[PRD-68] ATOM — no orchestration, direct LLM")

            # US-015: Emit memory-injected SSE event
            if orchestrated and orchestrated.memory_context:
                smart_chat = getattr(self, '_smart_chat', None)
                if smart_chat:
                    _mem_result = getattr(smart_chat.orchestrator, '_last_memory_result', None)
                    _memories_list = _mem_result.memories if _mem_result else []
                    _total_matched = len(_memories_list)
                    _mem_summaries = [
                        {"id": m.get("id", ""), "memory": m.get("memory", ""), "tier": m.get("_tier", "global")}
                        for m in _memories_list[:10]
                    ]
                    yield self.streaming_handler.format_aisdk_memory_injected(
                        memories=_mem_summaries, total_matched=_total_matched,
                    )
                    await asyncio.sleep(0)

            # Inject Composio per-action tools
            if _complexity != Complexity.ATOM:
                use_tools, _composio_result = self._inject_composio_tools(
                    llm_messages, use_tools, latest_text,
                    agent_id, agent_runtime, skip_composio, complexity_assessment,
                )
            else:
                _composio_result = None

            # Generate LLM response
            logger.info(f"Generating response with agent {agent_runtime.metadata.name}")
            logger.info(f"Agent tools - count: {len(use_tools) if use_tools else 0}")
            if use_tools:
                tool_names = [t.get("function", {}).get("name") for t in use_tools if isinstance(t, dict)]
                logger.info(f"Available tools: {tool_names}")

            response = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages, tools=use_tools,
            )

            logger.info(
                f"Agent LLM Response - has_tool_calls: {bool(response.tool_calls)}, "
                f"content_length: {len(response.content or '')}, "
                f"finish_reason: {getattr(response, 'finish_reason', 'unknown')}"
            )

            # Track response
            assistant_parts = []
            full_response = ""
            tool_data = {}

            # Handle tool calls via unified tool loop
            if response.tool_calls:
                logger.info(f"Agent requested {len(response.tool_calls)} tool calls")
                final_response = None
                async for chunk in self._run_tool_loop(
                    response, llm_messages, agent_runtime, tool_data, use_tools,
                    composio_result=_composio_result,
                ):
                    if isinstance(chunk, dict) and chunk.get('_final_response'):
                        final_response = chunk['_final_response']
                    else:
                        yield chunk
                    await asyncio.sleep(0)

                if final_response and final_response.content:
                    full_response = final_response.content
                else:
                    logger.warning("Tool loop completed without final response - forcing synthesis")
                    llm_messages.append({
                        'role': 'system',
                        'content': 'Based on the tool results above, provide a comprehensive response to the user.',
                    })
                    forced = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                    full_response = forced.content or "I apologize, but I encountered an issue generating a response. Please try again."
            else:
                if response.content:
                    full_response = response.content

            # Upload inline base64 images to S3
            _ws_id_img = getattr(agent_runtime, 'workspace_id', None) or self.workspace_id
            full_response = await _upload_inline_images(
                full_response, workspace_id=str(_ws_id_img) if _ws_id_img else None,
            )

            # Stream text response
            async for chunk in self.streaming_handler.stream_text_aisdk(full_response):
                yield chunk

            # Send usage data
            if hasattr(response, 'usage') and response.usage:
                yield self.streaming_handler.format_aisdk_usage(
                    response.usage.get('prompt_tokens', 0),
                    response.usage.get('completion_tokens', 0),
                    response.usage.get('total_tokens', 0),
                )

            # Send finish event
            yield self.streaming_handler.format_aisdk_finish()

            # Save assistant message
            assistant_parts.append({'type': 'text', 'text': full_response})
            self.chat_service.save_message(
                chat_id=chat_id, role="assistant",
                parts=assistant_parts, workspace_id=self.workspace_id,
            )

            # Post-response: memory, metrics, eval
            async for chunk in self._post_response(
                latest_text, full_response, chat_id,
                agent_runtime, agent_id, response, orchestrated,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming response with agent: {e}", exc_info=True)
            yield self.streaming_handler.format_aisdk_error(str(e))

    async def stream_response(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response using legacy SSE format."""
        from core.llm import create_llm_manager

        # Get tools from SINGLE SOURCE if not provided
        if tools is None:
            tools = get_tools_for_agent()

        try:
            llm_manager = create_llm_manager(service_name="chatbot")
            messages = self._resolve_file_parts(messages)
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            if self.prompt_analyzer.is_fresh_start_request(latest_text):
                messages = [m for m in messages if m.get("role") == "user"][-1:]
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages, available_tools=tools,
            )
            assistant_parts = []

            if hasattr(llm_manager, 'generate_response_stream'):
                async for chunk in llm_manager.generate_response_stream(messages=llm_messages, tools=tools):
                    yield self.streaming_handler.format_sse_chunk(chunk)
                    if chunk.get('type') == 'text':
                        assistant_parts.append({'type': 'text', 'text': chunk.get('text', '')})
            else:
                latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
                is_simple = self.prompt_analyzer.is_simple_message(latest_text)
                use_tools = None if is_simple else tools

                response = await llm_manager.generate_response(messages=llm_messages, tools=use_tools)

                if response.tool_calls:
                    tool_data = {}
                    tool_results = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('function', {}).get('name')
                        tool_args = json.loads(tool_call.get('function', {}).get('arguments', '') or '{}')
                        tool_id = tool_call.get('id')

                        result = await self.tool_router.execute_and_format(
                            tool_name, tool_args,
                            agent_id=1, workspace_id=self.workspace_id,
                            original_intent=latest_text,
                        )
                        if result['success']:
                            tool_data.update(result['frontend_data'])

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result['llm_context'],
                        })

                    if tool_data:
                        yield self.streaming_handler.format_sse_tool_data(tool_data)

                    llm_messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls,
                    })
                    llm_messages.extend(tool_results)

                    final_response = await llm_manager.generate_response(messages=llm_messages, tools=None)
                    response_text = final_response.content or ""
                else:
                    response_text = response.content or ""

                message_id = str(uuid.uuid4())
                async for chunk in self.streaming_handler.stream_text_legacy(response_text, message_id):
                    yield chunk

                assistant_parts.append({'type': 'text', 'text': response_text})

            if assistant_parts:
                self.chat_service.save_message(
                    chat_id=chat_id, role='assistant',
                    parts=assistant_parts, workspace_id=self.workspace_id,
                )

            yield self.streaming_handler.format_sse_done()

        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield self.streaming_handler.format_sse_error(str(e))

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
            tool_call_id = str(uuid.uuid4())
            start_time = time.time()

            logger.info(f"Pre-triggering {tool_name}")
            yield self.streaming_handler.format_aisdk_tool_start(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_input={"query": query},
            )

            result = await self.tool_router.execute_and_format(
                tool_name, {"query": query},
                agent_id=agent_id, workspace_id=self.workspace_id,
                original_intent=query,
            )

            if result['success']:
                tool_data.update(result['frontend_data'])

                context_msg = self.tool_router.build_tool_context_message(tool_name, result)
                if context_msg:
                    llm_messages.insert(1, context_msg)

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
