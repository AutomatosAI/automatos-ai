"""
Stream Orchestrator - Public Streaming Methods
===============================================

StreamingChatService: the main public class that orchestrates chat streaming.
Delegates to message_prep, tool_integration, and tool_loop_handler modules.
"""

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from config import config
from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.tool_router import get_tool_router
from modules.tools.tool_router import get_tools_for_agent

from .chat_crud import ChatService
from .message_prep import (
    extract_user_text,
    load_agent_context,
    parse_model_selection,
    prepare_messages,
    resolve_file_parts,
    resolve_workspace_id,
)
from .tool_integration import get_tools, inject_composio_tools
from .tool_loop_handler import run_tool_loop

logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE UPLOAD HELPER — Replace inline base64 images with S3 URLs
# =============================================================================

_BASE64_IMG_RE = re.compile(
    r'!\[([^\]]*)\]\((data:image/(jpeg|jpg|png|gif|webp);base64,([A-Za-z0-9+/=\s]+))\)'
)


async def _upload_inline_images(text: str, workspace_id: str = None) -> str:
    """Find base64 image markdown in text, upload to S3, replace with URLs."""
    from core.services.image_store import get_image_store

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
    # Delegation wrappers (maintain self.* interface for backward compat)
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_file_parts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return resolve_file_parts(self.db, messages)

    def _resolve_workspace_id(self, agent_id: int) -> Optional[str]:
        self.workspace_id = resolve_workspace_id(self.db, self.workspace_id, agent_id)
        return self.workspace_id

    def _extract_user_text(self, llm_messages: List[Dict[str, Any]]) -> str:
        return extract_user_text(llm_messages)

    def _parse_model_selection(self, selected_model: Optional[str]) -> tuple:
        return parse_model_selection(selected_model)

    async def _load_agent_context(self, agent_runtime) -> dict:
        return await load_agent_context(self.db, agent_runtime)

    def _get_tools(
        self,
        agent_id: int,
        skill_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return get_tools(agent_id, self.db, self.workspace_id, skill_tools)

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
    ):
        return await prepare_messages(
            self, messages, agent_runtime, agent_ctx, all_tools,
            chat_id, complexity_assessment, is_cto_agent, cto_check_result,
        )

    def _inject_composio_tools(
        self,
        llm_messages: List[Dict[str, Any]],
        use_tools: Optional[List[Dict[str, Any]]],
        latest_text: str,
        agent_id: int,
        agent_runtime,
        skip_composio: bool,
        complexity_assessment: Optional[Any],
    ):
        return inject_composio_tools(
            self.db, llm_messages, use_tools, latest_text,
            agent_id, agent_runtime, skip_composio, complexity_assessment,
            self.workspace_id,
        )

    async def _run_tool_loop(
        self,
        response,
        llm_messages: List[Dict[str, Any]],
        agent_runtime,
        tool_data: Dict[str, Any],
        use_tools: Optional[List[Dict[str, Any]]],
        composio_result: Any = None,
    ) -> AsyncGenerator[Any, None]:
        async for chunk in run_tool_loop(
            self, response, llm_messages, agent_runtime, tool_data,
            use_tools, composio_result,
        ):
            yield chunk

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
        from datetime import timezone

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
            from datetime import datetime, timezone
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

            # Prepare messages (orchestration, persona, CTO override, context guard)
            llm_messages, use_tools, orchestrated = await self._prepare_messages(
                messages, agent_runtime, agent_ctx, all_tools,
                chat_id, complexity_assessment, _is_cto_agent, _cto_check_result,
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
