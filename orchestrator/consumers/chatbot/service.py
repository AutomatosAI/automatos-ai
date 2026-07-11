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

import asyncio
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

# Converged tool-loop spine — chat + agent share this (PRD-142 W3-S4 / G6)
from modules.tools.execution.tool_loop import (
    RoundState,
    ToolLoopExecutor,
    ToolPostResult,
)
from modules.tools.execution.telemetry import resolve_action_name

from core.models import Chat, Message, Vote, Workspace
from core.services.image_store import get_image_store
from config import config

# Import from consumer's own modules
from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
from consumers.chatbot.primitive_heartbeat import _emit_chat_primitive
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.tool_router import get_tool_router

# Import from modules — SINGLE SOURCE for tool schemas
# Async-native entries: the chat hot path must never bridge the narrowing
# embed through a helper thread (freezes the event loop for its duration).
from modules.tools.tool_router import (
    _rank_actions_for_dispatcher_async,
    _semantic_routing_enabled,
    _semantic_routing_top_k,
    get_tools_for_agent_async,
)

logger = logging.getLogger(__name__)

# PRD-157 S3: token budget for a single tool result fed back into the LLM loop
# (replaces the former 6000/4000-char cuts). Truncation is token-aware.
_TOOL_RESULT_TOKEN_BUDGET = 2000


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


def build_tool_caller_context(
    *,
    user_query: Optional[str],
    conversation_id: Optional[str],
    turn_id: Optional[str],
    driving_clerk: Optional[str],
    prior_action: Optional[str],
    model_id: Optional[str] = None,
    est_input_tokens: int = 0,
    est_output_tokens: int = 0,
) -> Optional[Dict[str, Any]]:
    """Build the caller_context threaded into every chat tool execution (PRD-177 S2 / F017).

    Before this, the chat tool-callback threaded only ``{user_id}`` — so
    ``user_query`` and the conversation/turn grouping never reached the edge
    builder, and ``succeeds_for_intent`` intent affinities never materialized
    from real traffic. This populates the fields telemetry.py already consumes:

    * ``user_query``       — clusters intent (drives succeeds/fails_for_intent).
    * ``conversation_id``  — groups a conversation's tool calls (router_decision).
    * ``turn_id``          — groups ONE user turn's sequential calls; the edge
      builder prefers turn_id for used_after pairing, so per-turn ordering is
      exact rather than time-bucketed.
    * ``user_id``          — the driving clerk (unchanged from before F017).
    * ``prior_action``     — the previous tool this turn, for the signal recorder.
    * ``model_id`` / ``est_input_tokens`` / ``est_output_tokens`` — PRD-192 S3:
      the turn-level budget estimate (driving model, prompt tokens of the
      assembled context, configured output cap) the policy chokepoint lifts
      into ``ToolCall`` so budget admission prices the pending call.

    Empty fields are omitted (not written as None) to keep the telemetry row
    clean. Returns ``None`` when there is genuinely nothing to record, matching
    the previous ``{...} if _driving_clerk else None`` contract.
    """
    ctx: Dict[str, Any] = {}
    if user_query:
        ctx["user_query"] = user_query
    if conversation_id:
        ctx["conversation_id"] = conversation_id
    if turn_id:
        ctx["turn_id"] = turn_id
    if driving_clerk:
        ctx["user_id"] = driving_clerk
    if prior_action:
        ctx["prior_action"] = prior_action
    if model_id:
        ctx["model_id"] = model_id
        ctx["est_input_tokens"] = int(est_input_tokens or 0)
        ctx["est_output_tokens"] = int(est_output_tokens or 0)
    return ctx or None


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
        'search_multimodal',
        # PRD-160 S1: NL2SQL re-enabled workspace-scoped & in-process. Treated as
        # a search tool so semantically-similar repeat questions are deduped.
        'smart_query_database', 'query_database',
    }

    TOOL_RETRY_LIMITS = {
        'composio_execute': 5,
        'search_knowledge': 5,
        'semantic_search': 5,
        'search_codebase': 5,
        'list_directory': 5,
        'read_file': 8,
        'write_file': 5,
        # PRD-160 S1: NL2SQL is expensive and self-corrects internally
        # (max_retries=2); cap turn-level reuse low to match the 2-attempt
        # contract advertised in the tool description.
        'smart_query_database': 2,
        'query_database': 2,
        'platform_default': 25,
        'workspace_default': 8,
        'default': 5,
    }

    def __init__(self):
        self.exact_executions: Set[Tuple[str, str]] = set()
        self.search_queries: Dict[str, List[str]] = {}
        self.tool_counts: Dict[str, int] = {}

    def _hash_args(self, tool_args: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _counting_key(tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Return the key used for per-tool call counting.

        For the ``platform_execute`` dispatcher, count by inner action so
        that ``list_agents → get_settings → update_agent`` is three
        distinct actions, not three calls to the same tool.
        """
        if tool_name == "platform_execute":
            action = tool_args.get("action") or tool_args.get("name")
            if action:
                return f"platform_execute:{action}"
        return tool_name

    def _resolve_limit(self, counting_key: str) -> int:
        """Resolve the retry limit for a counting key, honouring prefix-based defaults.

        Handles dispatched actions like ``platform_execute:workspace_read_file``
        — the inner action name determines the prefix, not the dispatcher.
        """
        if counting_key in self.TOOL_RETRY_LIMITS:
            return self.TOOL_RETRY_LIMITS[counting_key]
        effective_key = counting_key.split(":", 1)[-1] if ":" in counting_key else counting_key
        if effective_key.startswith('workspace_'):
            return self.TOOL_RETRY_LIMITS.get('workspace_default', self.TOOL_RETRY_LIMITS['default'])
        if effective_key.startswith('platform_') or counting_key.startswith('platform_'):
            return self.TOOL_RETRY_LIMITS.get('platform_default', self.TOOL_RETRY_LIMITS['default'])
        return self.TOOL_RETRY_LIMITS['default']

    def should_skip_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if a tool execution should be skipped. Returns (should_skip, reason)."""
        key = self._counting_key(tool_name, tool_args)
        current_count = self.tool_counts.get(key, 0)
        limit = self._resolve_limit(key)

        if current_count >= limit:
            return True, f"Tool '{key}' has reached its execution limit ({limit}) for this turn"

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
        key = self._counting_key(tool_name, tool_args)
        self.tool_counts[key] = self.tool_counts.get(key, 0) + 1
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
        visibility: str = "private",
        workspace_id: Optional[uuid.UUID] = None,
    ) -> Chat:
        """Create a new chat session scoped to a workspace."""
        chat = Chat(
            id=uuid.uuid4(),
            user_id=user_id,
            workspace_id=workspace_id,
            title=title,
            visibility=visibility,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        logger.info(f"Created chat {chat.id} for user {user_id} workspace {workspace_id}: {title}")
        return chat

    def get_chat(self, chat_id: str, workspace_id: Optional[uuid.UUID] = None) -> Optional[Chat]:
        """Get a chat by ID, optionally scoped to a workspace."""
        try:
            chat_uuid = uuid.UUID(chat_id)
            query = self.db.query(Chat).filter(Chat.id == chat_uuid)
            if workspace_id is not None:
                query = query.filter(Chat.workspace_id == workspace_id)
            return query.first()
        except (ValueError, AttributeError):
            logger.error(f"Invalid chat_id format: {chat_id}")
            return None

    def get_chat_history(
        self,
        user_id: int,
        limit: int = 20,
        starting_after: Optional[datetime] = None,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> List[Chat]:
        """Get chat history for a user within a workspace."""
        query = self.db.query(Chat).filter(Chat.user_id == user_id)
        if workspace_id is not None:
            query = query.filter(Chat.workspace_id == workspace_id)
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
        workspace_id: Optional[str] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Save a message to the database.

        ``retrieval_context`` (PRD-185 S7) carries the turn's retrieved
        ``{document_ids, chunk_ids, query}`` on assistant messages; NULL for
        turns that retrieved nothing. Read back at vote time to feed rag_feedback.
        """
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
            retrieval_context=retrieval_context,
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

    def get_message(self, chat_id: str, message_id: str) -> Optional[Message]:
        """Fetch a single message by (chat_id, message_id). None on bad id/miss.

        PRD-185 S7: the vote path reads the assistant message's
        ``retrieval_context`` to write a complete rag_feedback row.
        """
        try:
            chat_uuid = uuid.UUID(chat_id)
            message_uuid = uuid.UUID(message_id)
        except ValueError:
            return None
        return self.db.query(Message).filter(
            and_(Message.chat_id == chat_uuid, Message.id == message_uuid)
        ).first()

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

        # PRD-185 S7: per-turn retrieval provenance. The instance is constructed
        # per request (one request == one turn), so these accumulate the turn's
        # retrieved ids (pinned docs + retrieval-tool results) and are read at the
        # assistant-message save. Reset defensively at each stream entrypoint.
        self._turn_document_ids: Set[int] = set()
        self._turn_chunk_ids: Set[int] = set()

        from modules.agents.factory.agent_factory import AgentFactory
        self.agent_factory = AgentFactory(db_session=db)
        logger.info("StreamingChatService initialized with AgentFactory integration")

    def _reset_turn_retrieval(self) -> None:
        """Clear per-turn retrieval provenance at the start of a turn."""
        self._turn_document_ids = set()
        self._turn_chunk_ids = set()

    def _collect_tool_retrieval(self, tool_name: Optional[str], result: Any) -> None:
        """Accumulate retrieved doc/chunk ids from a retrieval-tool result.

        Best-effort and gated to retrieval tools — never raises into the turn.
        """
        try:
            from modules.rag.retrieval_provenance import (
                is_retrieval_tool, collect_doc_ids_from_tool_result,
            )
            if not is_retrieval_tool(tool_name):
                return
            docs, chunks = collect_doc_ids_from_tool_result(result)
            self._turn_document_ids |= docs
            self._turn_chunk_ids |= chunks
        except Exception:
            logger.debug("[PRD-185 S7] tool retrieval provenance collect failed", exc_info=True)

    def _turn_retrieval_context(self, query: Optional[str]) -> Optional[Dict[str, Any]]:
        """Build the retrieval_context blob for the assistant message, or None."""
        try:
            from modules.rag.retrieval_provenance import build_retrieval_context
            return build_retrieval_context(
                self._turn_document_ids, self._turn_chunk_ids, query,
            )
        except Exception:
            logger.debug("[PRD-185 S7] retrieval_context build failed", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_file_parts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        DEPRECATED (PRD-127): Resolve document:// file parts to inline text content.

        This is a 30-day compat layer for old messages with document:// URLs.
        New messages use attachment_ids which are resolved via ContextService.build_context().
        Sunset date: 2026-05-10 (30 days after PRD-127 ship).
        """
        from sqlalchemy import text as sa_text

        resolved = []
        for msg in messages:
            parts = msg.get("parts")
            if not parts:
                resolved.append(msg)
                continue

            new_parts = []
            for part in parts:
                url = part.get("url") or ""
                if part.get("type") == "file" and url.startswith("document://"):
                    doc_id_str = url.replace("document://", "")
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

    def _extract_attachment_ids(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        PRD-127: Extract attachment_ids from the latest user message.

        Messages may have attachment_ids in their payload (new format).
        Returns list of attachment UUIDs to pass to build_context().
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                # Check for attachment_ids in message metadata
                attachment_ids = msg.get("attachment_ids", [])
                if attachment_ids:
                    return attachment_ids
                # Also check attachments field (may have attachment_id key)
                attachments = msg.get("attachments", [])
                if attachments:
                    ids = [
                        att.get("attachment_id")
                        for att in attachments
                        if att.get("attachment_id")
                    ]
                    if ids:
                        return ids
        return []

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

    async def _load_agent_context(self, agent_runtime) -> dict:
        """
        Load agent-specific context: persona, description for chatbot identity injection.

        PRD-81: System prompt cache removed from AgentRuntime — ContextService is
        now the single prompt builder.

        PRD-137 Fix #3: identity is now injected by IdentitySection (single owner).
        This method still produces an agent_ctx dict for non-identity callers
        (e.g. CTO override path that reads `extra_context`).
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

    async def _get_tools(
        self,
        agent_id: int,
        skill_tools: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        is_super_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get all tools for an agent from the SINGLE source: modules.tools.tool_router.

        Returns full OpenAI-format tool schemas (ToolRegistry + ActionRegistry + Composio).
        Appends any skill-specific tool schemas from the agent runtime.

        Args:
            query: Latest user turn — when set and SEMANTIC_TOOL_ROUTING is on,
                the platform_execute dispatcher's action.enum is narrowed to
                top-K relevant actions (PRD-138 US-009).
            is_super_admin: PRD-143 — True ONLY when the driving chat principal
                is system_role == 'super_admin'. Fail-closed default excludes
                the su tool tier from the surface.
        """
        all_tools = await get_tools_for_agent_async(
            agent_id=agent_id,
            db_session=self.db,
            workspace_id=self.workspace_id,
            query=query,
            is_super_admin=is_super_admin,
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
        attachment_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        force_text_only: bool = False,
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
        # Proactive openers (force_text_only) are self-contained: page-context
        # directive in, one line of text out. They carry no complexity_assessment,
        # so without this they'd default to MOLECULE and take the full
        # ContextService path (internal tool load + Mem0 read) — exactly the
        # work stream_response_with_agent already decided to skip. Pin to ATOM.
        if force_text_only:
            _complexity = Complexity.ATOM

        if _complexity == Complexity.ATOM:
            llm_messages, use_tools, orchestrated = await self._prepare_atom_path(
                messages, agent_runtime, smart_chat,
                atom_tools=all_tools,
                attachment_ids=attachment_ids,
                model_id=model_id,
                force_text_only=force_text_only,
            )
        else:
            llm_messages, use_tools, orchestrated = await self._prepare_full_path(
                messages, agent_runtime, agent_ctx, all_tools,
                smart_chat, chat_id, complexity_assessment,
                attachment_ids=attachment_ids,
                model_id=model_id,
            )

        # PRD-67: CTO Agent system prompt override
        if is_cto_agent:
            self._apply_cto_override(
                llm_messages, smart_chat, cto_check_result,
                messages, use_tools, agent_runtime.agent_id,
            )

        # PRD-137 Fix #3: agent description + persona are injected by
        # IdentitySection (modules/context/sections/identity.py) into the
        # orchestrated system prompt, for both chatbot and non-chatbot modes.
        # Do NOT inject again here — double injection causes the model to echo
        # its intro twice (observed on Shopify widget).

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
        atom_tools: Optional[List[Dict[str, Any]]] = None,
        attachment_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        force_text_only: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], None]:
        """ATOM path: lightweight memory only, but keeps platform_execute tool.

        force_text_only (proactive openers) skips memory retrieval entirely —
        the opener is self-contained and the Mem0 read only adds latency.
        """
        logger.info(
            "[PRD-68] ATOM path — lightweight (tools=%d, memory=%s)",
            len(atom_tools or []),
            "skipped" if force_text_only else "on",
        )
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
            if (
                not force_text_only
                and _user_msg
                and smart_chat.orchestrator
                and smart_chat.orchestrator.memory_manager
            ):
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

        _persona_block = ""
        if agent_runtime.metadata.persona:
            _persona_block = f"\n\n{agent_runtime.metadata.persona}\n"

        _description_block = ""
        if agent_runtime.metadata.description and str(agent_runtime.metadata.description).strip():
            _description_block = f"\n\n## Agent Description\n{str(agent_runtime.metadata.description).strip()}\n"

        _atom_prompt = (
            f"You are {agent_runtime.metadata.name}, an AI assistant on the Automatos platform.\n\n"
            f"{_time_ctx}. Read the conversation and match the user's energy. "
            "If they're frustrated, be direct — skip the niceties and lead with the answer. "
            "If they're curious, explain the why. If they're casual, be casual back. "
            "If they're formal, match it. Never be artificially cheerful when someone is having a bad time. "
            "Never be robotic when someone is being warm.\n\n"
            "You adapt. That's what makes you good at this.\n"
            f"{_description_block}"
            f"{_persona_block}"
            f"{_memory_block}"
        )
        llm_messages = self.prompt_analyzer.convert_to_llm_messages(
            messages, system_prompt=_atom_prompt, available_tools=atom_tools
        )

        # PRD-127: Resolve ephemeral attachments for ATOM path.
        # Full path goes through ContextService.build_context which handles this;
        # ATOM bypasses ContextService, so we resolve directly here.
        if attachment_ids:
            try:
                from uuid import UUID
                from modules.attachments.resolver import (
                    AttachmentResolver,
                    VisionNotSupportedError,
                    inject_parts_into_last_user_message,
                )
                resolver = AttachmentResolver(db_session=self.db)
                parts = await resolver.resolve(
                    attachment_ids=[UUID(a) for a in attachment_ids],
                    workspace_id=UUID(str(self.workspace_id)),
                    model_id=model_id or "",
                )
                if parts:
                    inject_parts_into_last_user_message(llm_messages, parts)
                    logger.info(
                        f"[PRD-127] ATOM path: resolved {len(parts)} attachment parts "
                        f"from {len(attachment_ids)} ids"
                    )
            except VisionNotSupportedError as _vne:
                logger.warning(f"[PRD-127] ATOM vision not supported: {_vne}")
            except Exception as _att_err:
                logger.error(
                    f"[PRD-127] ATOM attachment resolution failed: {_att_err}",
                    exc_info=True,
                )

        return llm_messages, atom_tools, None

    async def _prepare_full_path(
        self,
        messages: List[Dict[str, Any]],
        agent_runtime,
        agent_ctx: dict,
        all_tools: List[Dict[str, Any]],
        smart_chat,
        chat_id: str,
        complexity_assessment: Optional[Any],
        attachment_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
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
            attachment_ids=attachment_ids,
            model_id=model_id,
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

    # PRD-137 Fix #3: removed _inject_agent_identity. IdentitySection now
    # owns description+persona injection for both chatbot and non-chatbot
    # modes — see modules/context/sections/identity.py.

    def _inject_pinned_documents(
        self, llm_messages: List[Dict[str, Any]], chat_id: Optional[str]
    ) -> None:
        """PRD-157 S5: prepend the chat's pinned-document content as a system
        message so it is always present in context (within the token budget).

        Inserted after any leading system prompt and before user/history. Best
        effort — a failure never breaks the turn.
        """
        if not chat_id or self.workspace_id is None:
            return
        try:
            from modules.rag.pinned_context import build_pinned_system_message

            content = build_pinned_system_message(
                self.db, chat_id=chat_id, workspace_id=self.workspace_id
            )
            if not content:
                return
            insert_at = 1 if (llm_messages and llm_messages[0].get("role") == "system") else 0
            llm_messages.insert(insert_at, {"role": "system", "content": content})
            logger.info("[PRD-157 S5] injected pinned-document context for chat %s", chat_id)

            # PRD-185 S7: pinned documents are retrieved context for this turn —
            # record their ids so a later vote writes complete rag_feedback.
            try:
                from modules.rag.pinned_context import list_pinned
                for row in list_pinned(self.db, chat_id=chat_id, workspace_id=self.workspace_id):
                    did = row.get("document_id")
                    if isinstance(did, int):
                        self._turn_document_ids.add(did)
            except Exception:
                logger.debug("[PRD-185 S7] pinned provenance collect failed", exc_info=True)
        except Exception:
            logger.warning("[PRD-157 S5] pinned-document injection failed", exc_info=True)

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
                _mm = smart_chat.orchestrator.memory_manager
                _facts_stored = getattr(_mm, '_last_l3_facts_stored', 0)
                # PRD-159 S5: honest event — emit ONLY after durable facts were
                # actually persisted to L3, with the real tier. Zero-fact turns
                # (e.g. "user said hello") produce NO memory_stored event.
                if _stored and _facts_stored > 0:
                    _tier = getattr(_mm, '_last_tier', 'conversation')
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
    # Streaming tool loop — delegates to the converged ToolLoopExecutor
    # (PRD-142 W3-S4 / G6: one tool loop, shared with agent_factory).
    # ─────────────────────────────────────────────────────────────────────

    async def _stream_tool_loop(
        self,
        response,
        llm_messages: List[Dict[str, Any]],
        agent_runtime,
        tool_data: Dict[str, Any],
        use_tools: Optional[List[Dict[str, Any]]],
        composio_result: Any = None,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """Drive :class:`ToolLoopExecutor` from the chat surface.

        Yields SSE chunks as the executor runs and ends with
        ``{'_final_response': response}`` — the exact contract the chat
        caller consumes (see ``stream_response_with_agent`` below).

        Chat-specific behaviour (Composio per-action shortcut, Composio
        error recovery, fatal_error short-circuit, force-synth on dedup,
        ContextGuard compaction recovery, frontend-data + workflow-update
        SSE emissions) is layered on top of the converged spine via the
        executor's callback hooks. The spine itself — dedup, per-tool
        attempt caps, finish_reason=length recovery, iteration cap — is
        shared with the agent ``execute_with_prompt`` inner loop.
        """
        max_iterations = config.CHATBOT_MAX_TOOL_ITERATIONS
        action_budget = config.CHATBOT_ACTION_RETRY_BUDGET
        param_budget = config.CHATBOT_PARAM_RETRY_BUDGET

        # State shared by callbacks within this turn.
        last_tool_name: Optional[str] = None
        empty_streak = 0

        # PRD-163 S1/Q56: resolve the chatting user's clerk id once, so a mission
        # created mid-chat is attributed to THEM (created_by) — not the agent — and
        # plan-ready / awaiting-approval notifications land for the right person.
        _driving_clerk: Optional[str] = None
        if user_id:
            try:
                from core.models import User
                _row = self.db.query(User.clerk_user_id).filter(User.id == user_id).first()
                _driving_clerk = _row[0] if _row else None
            except Exception:
                _driving_clerk = None

        # PRD-177 S2 (F017): one turn_id per user turn (this stream invocation).
        # All sequential tool calls in this turn share it, so the edge builder
        # pairs used_after edges by exact turn order (it prefers turn_id over the
        # conversation grouping). conversation_id groups the whole conversation.
        _turn_id: str = uuid.uuid4().hex
        _prior_action: Optional[str] = None
        cumulative_attempts: Dict[str, int] = {}
        followup_messages: List[Dict[str, Any]] = []

        _MULTI_STEP_TOOLS = {
            "composio_execute", "generate_document",
            "workspace_read_file", "workspace_grep", "workspace_list_dir",
            "workspace_write_file", "workspace_exec", "workspace_git",
        }
        _WORKFLOW_PREFIXES = (
            "platform_list_recipes",
            "platform_create_recipe",
            "platform_execute_recipe",
        )

        # SSE bridge: executor on_event puts AI SDK chunks here, this generator drains.
        sse_queue: "asyncio.Queue[Any]" = asyncio.Queue()
        DONE = object()

        async def _on_event(event: Dict[str, Any]) -> None:
            et = event.get("type")
            if et == "tool-start":
                await sse_queue.put(self.streaming_handler.format_aisdk_tool_start(
                    event["tool_call_id"], event["tool_name"],
                    tool_input=event.get("tool_input", {}),
                ))
            elif et == "tool-end":
                # tool-result frame mirrors what the legacy loop emitted.
                if not event.get("skipped"):
                    await sse_queue.put(self.streaming_handler.format_aisdk_tool_end(
                        tool_call_id=event["tool_call_id"],
                        tool_name=event["tool_name"],
                        success=bool(event.get("success")),
                        duration_ms=int(event.get("duration_ms", 0)),
                    ))

        def _is_chat_composio_action(name: str) -> bool:
            """A per-action Composio tool from this turn's SDK schema set."""
            return bool(
                composio_result and composio_result.entity_id and (
                    name in composio_result.action_set
                    or any(name.startswith(f"{app}_") for app in composio_result.app_names)
                )
            )

        async def _tool_callback(name: str, args: Dict[str, Any], call_id: str, ws_id) -> Dict[str, Any]:
            nonlocal last_tool_name, empty_streak, _prior_action

            # PRD-192 S4: per-action Composio calls ride the SPINE. The legacy
            # shortcut here called ComposioToolService.execute_action raw and
            # returned success:True unconditionally — no policy gate, no
            # telemetry, no outcome capture, no scope enforcement, and a
            # dishonest envelope (tool-runtime C.3a). They now dispatch through
            # execute_and_format like every other tool in this callback, as the
            # composio_execute meta-tool (the executor's registry route — the
            # per-action name travels in `action`, so the tracker, the policy
            # gate's effective-name resolution, and the routing graph all see
            # GMAIL_SEND_EMAIL, not the wrapper). The gate classifies it
            # external_side_effect via the S1 is_composio hint.
            dispatch_name, dispatch_args = name, args
            if _is_chat_composio_action(name):
                dispatch_name = "composio_execute"
                dispatch_args = {
                    "action": name,
                    "params": args if isinstance(args, dict) else {},
                }

            user_text = self._extract_user_text(llm_messages)
            # PRD-192 S3: turn-level budget estimate at the loop boundary —
            # the driving model + prompt tokens + output cap, so the policy
            # gate prices this call instead of admitting at a structural $0.
            from core.context_guard import estimate_turn_budget
            _turn_budget = estimate_turn_budget(agent_runtime.llm_manager, llm_messages)
            # PRD-177 S2 (F017): thread user_query + conversation/turn ids so the
            # edge builder can cluster intent and pair per-turn used_after edges.
            result = await self.tool_router.execute_and_format(
                tool_name=dispatch_name,
                tool_args=dispatch_args,
                agent_id=agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else 1,
                workspace_id=ws_id,
                original_intent=user_text,
                caller_context=build_tool_caller_context(
                    user_query=user_text,
                    conversation_id=conversation_id,
                    turn_id=_turn_id,
                    driving_clerk=_driving_clerk,
                    prior_action=_prior_action,
                    model_id=_turn_budget.get("model_id"),
                    est_input_tokens=_turn_budget.get("est_input_tokens", 0),
                    est_output_tokens=_turn_budget.get("est_output_tokens", 0),
                ),
            )
            # PRD-185 S7: capture retrieved doc ids (retrieval tools only).
            self._collect_tool_retrieval(name, result)
            # Record this call as the prior_action for the NEXT tool in the turn
            # (resolved to the per-action name so composio chains pair correctly).
            _prior_action = resolve_action_name(name, args if isinstance(args, dict) else {})

            # Search-spiral detection (chat-only signal).
            empty_streak = self._track_search_spiral(
                result, name, last_tool_name, empty_streak,
            )
            last_tool_name = name

            cumulative_attempts[name] = cumulative_attempts.get(name, 0) + 1

            llm_context = result.get("llm_context", str(result.get("raw_result", "")))
            # PRD-157 S3: token-budgeted truncation (model-aware), not a char cut.
            from modules.rag.budget import truncate_to_token_budget
            llm_context = truncate_to_token_budget(llm_context, _TOOL_RESULT_TOKEN_BUDGET)

            # Frontend data emission (widget tool-data).
            # PRD-193 S3 (P2-12): an approval ask (success=False +
            # requires_confirmation) still carries the tool_approval card —
            # the human must see it live even though the tool did not run.
            frontend_data = result.get("frontend_data", {})
            _is_approval_ask = bool(
                (result.get("raw_result") or {}).get("requires_confirmation")
            )
            if frontend_data and (result.get("success") or _is_approval_ask):
                tool_data.update(frontend_data)
                await sse_queue.put(self.streaming_handler.format_aisdk_tool_data(frontend_data))

            # Recipe / workflow tool-update emission.
            if name.startswith(_WORKFLOW_PREFIXES) or "workflow" in name.lower():
                _raw = result.get("raw_result") or {}
                _wf_id = str(_raw.get("id") or _raw.get("workflow_id") or _raw.get("recipe_id") or call_id)
                _wf_status = "completed" if result.get("success") else "failed"
                await sse_queue.put(self.streaming_handler.format_aisdk_workflow_update(
                    workflow_id=_wf_id, status=_wf_status, current_step=name,
                ))

            # tool-result excerpt frame (legacy parity).
            await sse_queue.put(self.streaming_handler.format_aisdk_data(
                "tool-result",
                {"toolCallId": call_id, "toolName": name, "result": llm_context[:500]},
            ))

            # Loop-prevention proceed instructions (mutates llm_messages).
            self._inject_loop_prevention(
                llm_messages, name, cumulative_attempts,
                empty_streak, result, agent_runtime, _MULTI_STEP_TOOLS,
            )

            # Hand back to executor with the truncated llm_context already prepared
            # so it does not re-truncate.
            return {
                "success": result.get("success", True),
                "llm_context": llm_context,
                "raw_result": result.get("raw_result", {}),
                "frontend_data": frontend_data,
                "error_type": result.get("error_type"),
                "fatal_error": result.get("fatal_error", False),
            }

        async def _on_tool_result(
            name: str, args: Dict[str, Any], result: Any,
        ) -> Optional[ToolPostResult]:
            nonlocal action_budget, param_budget
            if not isinstance(result, dict):
                return None
            # PRD-192 S4: per-action Composio calls dispatch as the
            # composio_execute meta-tool — the recovery hook reads the same
            # honest envelope for both shapes.
            recovery_name = (
                "composio_execute" if _is_chat_composio_action(name) else name
            )
            recovery = self._handle_composio_error_recovery(
                result, recovery_name, llm_messages, agent_runtime,
                action_budget, param_budget, followup_messages,
            )
            if recovery is None:
                return None
            if recovery.get("_early_return"):
                return ToolPostResult(
                    force_final=True,
                    final_content=recovery.get("message"),
                )
            action_budget = recovery.get(
                "action_not_mapped_retry_budget", action_budget,
            )
            param_budget = recovery.get(
                "invalid_parameters_retry_budget", param_budget,
            )
            return None

        async def _on_round_end(state: RoundState) -> Optional[ToolPostResult]:
            # Flush any Composio follow-up system messages into llm_messages
            # BEFORE the next LLM call (legacy ordering).
            nonlocal followup_messages
            if followup_messages:
                llm_messages.extend(followup_messages)
                followup_messages = []

            # Fatal error short-circuit (chat-only behaviour).
            if state.had_fatal_errors:
                return ToolPostResult(
                    force_final=True,
                    final_content=(
                        "I ran into a server configuration issue while executing that tool. "
                        "Please restart the backend and try again."
                    ),
                )

            # Force synthesis on dedup skip OR per-tool exhaustion (>=8 attempts).
            any_exhausted = any(v >= 8 for v in cumulative_attempts.values())
            if state.had_skips or any_exhausted:
                if any_exhausted:
                    logger.warning(
                        f"[tool-loop] Tool hard cap reached — forcing synthesis "
                        f"(attempts: {dict(cumulative_attempts)})"
                    )
                return ToolPostResult(force_final=True)
            return None

        async def _llm_callback(messages, tools):
            try:
                return await agent_runtime.llm_manager.generate_response(
                    messages=messages, tools=tools,
                )
            except Exception as llm_err:
                logger.warning(
                    f"LLM call failed in tool loop, attempting recovery: {llm_err}"
                )
                try:
                    from core.context_guard import ContextGuard
                    guard = ContextGuard()
                    messages_new, compacted, tools_new = await guard.check_and_compact(
                        messages, tools, agent_runtime.llm_manager,
                        workspace_id=self.workspace_id,
                    )
                    if compacted:
                        logger.info("Recovery compaction succeeded, retrying LLM call")
                        return await agent_runtime.llm_manager.generate_response(
                            messages=messages_new, tools=tools_new,
                        )
                    raise
                except Exception:
                    logger.error(
                        f"Recovery failed, surfacing error: {llm_err}", exc_info=True
                    )
                    raise llm_err

        executor = ToolLoopExecutor(
            llm_callback=_llm_callback,
            tool_callback=_tool_callback,
            max_iterations=max_iterations,
            content_truncate_tokens=2000,
        )

        async def _runner():
            try:
                return await executor.run(
                    initial_response=response,
                    messages=llm_messages,
                    tools=use_tools,
                    workspace_id=self.workspace_id,
                    on_event=_on_event,
                    on_tool_result=_on_tool_result,
                    on_round_end=_on_round_end,
                )
            finally:
                await sse_queue.put(DONE)

        runner_task = asyncio.create_task(_runner())

        while True:
            item = await sse_queue.get()
            if item is DONE:
                break
            yield item
            await asyncio.sleep(0)

        try:
            result = await runner_task
        except Exception as loop_err:
            logger.error(f"Tool loop failed: {loop_err}", exc_info=True)
            yield {"_final_response": SimpleNamespace(
                content=f"Error: {loop_err}", tool_calls=None, usage=None,
            )}
            return

        # Max-iterations reached → emit limit_reached SSE + synthesize.
        if result.max_iterations_reached:
            yield self.streaming_handler.format_aisdk_limit_reached(
                limit="max_tool_iterations",
                value=max_iterations,
                message=(
                    f"I reached the maximum of {max_iterations} tool steps for a "
                    "single response, so I'm answering with what I have so far. "
                    "An admin can raise this via the CHATBOT_MAX_TOOL_ITERATIONS "
                    "setting (or the workspace power-mode caps)."
                ),
            )
            final = await agent_runtime.llm_manager.generate_response(
                messages=llm_messages, tools=None,
            )
            yield {"_final_response": final}
            return

        yield {"_final_response": result.response}

    # PRD-192 S4: `_execute_composio_action` (the raw ComposioToolService
    # shortcut) is DELETED — per-action Composio calls dispatch through
    # execute_and_format → UnifiedToolExecutor like every other chat tool, so
    # the policy gate, telemetry, outcome capture, and scope validation all
    # fire on this lane and failures surface honestly (no unconditional
    # success:True). No shim (CLAUDE.md §5).

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
        # Search spiral: 4+ consecutive empty results from same tool
        if empty_same_tool_streak >= 4 and (
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
            or tool_name.startswith("platform_")
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
        use_orchestrator_llm: bool = False,
        skip_composio: bool = False,
        complexity_assessment: Optional[Any] = None,
        mission_mode: bool = False,
        plan_mode: bool = False,
        team: Optional[str] = None,
        suggest_mission: bool = False,
        force_text_only: bool = False,
        is_super_admin: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response produced by the specified agent.
        Yields AISDK-formatted chunks for frontend consumption.

        PRD-137 Fix #2: parameter renamed from ``use_system_llm`` —
        when True, the orchestrator-tier defaults are used; when
        False, the agent's own model_config drives the LLM.

        PRD-143: ``is_super_admin`` is derived by the caller from the driving
        principal's system_role == 'super_admin' ONLY — it widens the tool
        surface to the su tier and is never inferred from workspace roles.
        """
        import asyncio

        # PRD-185 S7: start the turn with clean retrieval provenance.
        self._reset_turn_retrieval()

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
            agent_task = asyncio.create_task(
                self.agent_factory.activate_agent(agent_id, use_orchestrator_llm=use_orchestrator_llm)
            )

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

            # Resolve file attachments (legacy document:// URLs - sunset 2026-05-10)
            messages = self._resolve_file_parts(messages)

            # PRD-127: Extract attachment_ids and model_id for multimodal resolution
            _attachment_ids = self._extract_attachment_ids(messages)
            if _attachment_ids:
                logger.info(
                    f"[PRD-127] Extracted {len(_attachment_ids)} attachment_ids from messages: {_attachment_ids}"
                )
            _model_id = None
            _llm_config = getattr(agent_runtime.llm_manager, 'config', None)
            if _llm_config:
                _model_id = getattr(_llm_config, 'model', None)

            # Load agent context — persona is always loaded (it's who the agent IS)
            from consumers.chatbot.auto import Complexity
            _complexity = (
                complexity_assessment.complexity
                if complexity_assessment
                else Complexity.MOLECULE
            )
            # PRD-007 v0.5 — proactive openers force ATOM path. The 45-tool
            # discovery in _get_tools() takes ~12s and the agent doesn't need
            # tools to produce a one-sentence opener from the directive.
            if force_text_only:
                _complexity = Complexity.ATOM
            agent_ctx = await self._load_agent_context(agent_runtime)
            all_tools = []
            if _complexity != Complexity.ATOM:
                all_tools = await self._get_tools(
                    agent_id,
                    agent_ctx.get("skill_tools"),
                    query=latest_text,
                    is_super_admin=is_super_admin,
                )
            else:
                # ATOM path skips full tool loading, but always include
                # platform_execute so the agent can respond to platform
                # queries even when the classifier under-estimates complexity.
                # PRD-138 US-009: ATOM path also narrows the dispatcher's
                # action enum when SEMANTIC_TOOL_ROUTING is on, so a model
                # in the lightweight ATOM lane sees the same focused
                # surface as the full path.
                try:
                    from modules.tools.discovery.action_registry import get_action_registry
                    _allowed = None
                    if _semantic_routing_enabled() and latest_text:
                        _allowed = await _rank_actions_for_dispatcher_async(
                            query=latest_text,
                            top_k=_semantic_routing_top_k(),
                            exclude_admin=True,
                            exclude_promoted=True,
                            include_super_admin=is_super_admin,
                        )
                    _dispatcher = get_action_registry().to_dispatcher_schema(
                        exclude_admin=True,
                        allowed_names=_allowed,
                        include_super_admin=is_super_admin,
                    )
                    # PRD-007 v0.5: proactive openers get zero tools — directive
                    # is self-contained (page context + graph related products).
                    all_tools = [] if force_text_only else [_dispatcher]
                except Exception:
                    logger.debug("[chat] Could not load platform_execute for ATOM path")

            # Prepare messages (orchestration, persona, CTO override, context guard)
            llm_messages, use_tools, orchestrated = await self._prepare_messages(
                messages, agent_runtime, agent_ctx, all_tools,
                chat_id, complexity_assessment, _is_cto_agent, _cto_check_result,
                mission_mode=mission_mode,
                plan_mode=plan_mode,
                attachment_ids=_attachment_ids,
                model_id=_model_id,
                force_text_only=force_text_only,
            )

            # PRD-157 S5: keep the chat's pinned documents always in context.
            self._inject_pinned_documents(llm_messages, chat_id)

            if orchestrated:
                logger.info(
                    f"[SmartChat] intent={orchestrated.intent.value} "
                    f"tools={len(use_tools) if use_tools else 0} "
                    f"memory={'yes' if orchestrated.memory_context else 'no'} "
                    f"prep={orchestrated.preparation_time_ms:.0f}ms"
                )
            else:
                logger.info("[PRD-68] ATOM — no orchestration, direct LLM")

            # PRD-125: Inject mission suggestion directive into system prompt
            # Must happen AFTER _prepare_messages() since it rebuilds the system prompt.
            if suggest_mission and llm_messages:
                _mission_directive = (
                    "\n\n## IMPORTANT — Mission Suggestion Required\n"
                    "This task has been classified as complex (multi-step, multi-tool). "
                    "You MUST include the following in your response:\n"
                    "1. Briefly acknowledge the task\n"
                    "2. Tell the user: \"This is a complex task that could benefit from a "
                    "**Multi-Agent Mission**. A mission coordinates multiple specialized agents, "
                    "plans the steps, executes them in sequence, and verifies the results. "
                    "Would you like me to launch a mission for this?\"\n"
                    "3. Then proceed to help with what you can as a single agent.\n"
                    "Do NOT skip the mission suggestion — it is mandatory."
                )
                if llm_messages[0].get("role") == "system":
                    llm_messages[0]["content"] = llm_messages[0].get("content", "") + _mission_directive
                else:
                    llm_messages.insert(0, {"role": "system", "content": _mission_directive})

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

            # PRD-007 v0.5 — proactive openers must produce plain text, never tool calls.
            # The skill forbids tools, but with 40+ tools wired in the agent still calls
            # one and emits 0 text chars (STREAM_NO_TEXT). Force-clear here so the LLM
            # literally has nothing to call.
            if force_text_only and use_tools:
                logger.info(
                    f"[force_text_only] Clearing {len(use_tools)} tools for text-only generation"
                )
                use_tools = None
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
                async for chunk in self._stream_tool_loop(
                    response, llm_messages, agent_runtime, tool_data, use_tools,
                    composio_result=_composio_result,
                    user_id=user_id,
                    conversation_id=chat_id,
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
                # PRD-185 S7: stamp the turn's retrieved doc ids for vote feedback.
                retrieval_context=self._turn_retrieval_context(latest_text),
            )

            # Post-response: memory, metrics, eval
            async for chunk in self._post_response(
                latest_text, full_response, chat_id,
                agent_runtime, agent_id, response, orchestrated,
            ):
                yield chunk

            # PRD-142 W3-S6: chat primitive heartbeat — green on clean turn.
            _emit_chat_primitive(
                self.workspace_id, success=True, detail="chat turn completed",
            )

        except Exception as e:
            logger.error(f"Error streaming response with agent: {e}", exc_info=True)
            yield self.streaming_handler.format_aisdk_error(str(e))
            # PRD-142 W3-S6: chat primitive heartbeat — down on caught error.
            _emit_chat_primitive(
                self.workspace_id, success=False, detail=str(e),
            )

    async def stream_response(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response using legacy SSE format."""
        from core.llm import create_llm_manager

        # PRD-185 S7: start the turn with clean retrieval provenance.
        self._reset_turn_retrieval()

        # Get tools from SINGLE SOURCE if not provided.
        # PRD-138 US-009: extract latest user turn first so the dispatcher
        # enum can be narrowed to relevant actions for this query.
        if tools is None:
            _query = self.prompt_analyzer.extract_latest_user_text(messages)
            tools = await get_tools_for_agent_async(query=_query)

        try:
            llm_manager = create_llm_manager(service_name="chatbot", workspace_id=self.workspace_id, request_type="chat")
            messages = self._resolve_file_parts(messages)
            latest_text = self.prompt_analyzer.extract_latest_user_text(messages)
            if self.prompt_analyzer.is_fresh_start_request(latest_text):
                messages = [m for m in messages if m.get("role") == "user"][-1:]
            llm_messages = self.prompt_analyzer.convert_to_llm_messages(
                messages, available_tools=tools,
            )
            # PRD-157 S5: keep the chat's pinned documents always in context.
            self._inject_pinned_documents(llm_messages, chat_id)
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
                        # PRD-185 S7: capture retrieved doc ids (retrieval tools only).
                        self._collect_tool_retrieval(tool_name, result)
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
                    # PRD-185 S7: stamp the turn's retrieved doc ids for vote feedback.
                    retrieval_context=self._turn_retrieval_context(latest_text),
                )

            yield self.streaming_handler.format_sse_done()

            # PRD-142 W3-S6: chat primitive heartbeat — green on clean turn.
            _emit_chat_primitive(
                self.workspace_id, success=True, detail="chat turn completed",
            )

        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield self.streaming_handler.format_sse_error(str(e))
            # PRD-142 W3-S6: chat primitive heartbeat — down on caught error.
            _emit_chat_primitive(
                self.workspace_id, success=False, detail=str(e),
            )

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
            # PRD-185 S7: capture retrieved doc ids (retrieval tools only).
            self._collect_tool_retrieval(tool_name, result)

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
