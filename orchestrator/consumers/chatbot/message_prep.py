"""
Message Preparation Helpers
============================

Functions extracted from StreamingChatService for preparing messages
before LLM calls: file resolution, workspace resolution, text extraction,
model parsing, and the full message preparation pipeline (ATOM vs FULL path).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)


def resolve_file_parts(db_session, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    result = db_session.execute(
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


def resolve_workspace_id(db_session, current_workspace_id: Optional[str], agent_id: int) -> Optional[str]:
    """Resolve workspace_id from agent if not already set."""
    logger.info(f"[chat] resolve_workspace_id current={current_workspace_id} agent={agent_id}")
    if current_workspace_id:
        return current_workspace_id
    try:
        from core.models import Agent as AgentModel
        agent_row = db_session.query(AgentModel).filter(AgentModel.id == agent_id).first()
        if agent_row and agent_row.workspace_id:
            logger.info(f"Resolved workspace_id from agent {agent_id}")
            return agent_row.workspace_id
    except Exception as exc:
        logger.warning(f"Failed to resolve workspace_id for agent {agent_id}: {exc}")
    return current_workspace_id


def extract_user_text(llm_messages: List[Dict[str, Any]]) -> str:
    """Extract the latest user message text from LLM messages."""
    for m in reversed(llm_messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def parse_model_selection(selected_model: Optional[str]) -> tuple:
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


async def load_agent_context(db_session, agent_runtime) -> dict:
    """
    Load agent-specific context: persona, description for chatbot identity injection.

    PRD-81: System prompt cache removed from AgentRuntime -- ContextService is
    now the single prompt builder. This method only provides lightweight identity
    context for the chatbot's _inject_agent_identity().
    """
    persona = ""
    try:
        from core.models import Agent as AgentModel
        db_agent = db_session.query(AgentModel).filter(AgentModel.id == agent_runtime.agent_id).first()
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


async def prepare_atom_path(
    messages: List[Dict[str, Any]],
    agent_runtime,
    smart_chat,
    workspace_id: Optional[str],
    widget_mode: bool,
    prompt_analyzer,
) -> Tuple[List[Dict[str, Any]], None, None]:
    """ATOM path: no tools, no orchestration, lightweight memory only."""
    logger.info("[PRD-68] ATOM path -- skipping tools/orchestration, retrieving memory")
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
                workspace_id=str(workspace_id),
                agent_id=agent_runtime.agent_id,
                query=_user_msg if len(_user_msg) > 5 else "user context",
                widget_mode=widget_mode,
            )
            if _mem_result and _mem_result.formatted_context:
                _memory_block = f"\n\n## What you remember about this user:\n{_mem_result.formatted_context}\n"
                logger.info(f"[PRD-68] ATOM memory: {len(_mem_result.memories)} memories injected")
    except Exception as _mem_err:
        logger.debug(f"[PRD-68] ATOM memory retrieval skipped: {_mem_err}")

    _atom_prompt = (
        f"You are {agent_runtime.metadata.name}, a warm and helpful AI assistant "
        f"on the Automatos platform. {_time_ctx}! "
        "Respond naturally and conversationally -- be friendly, be brief. "
        "You're chatting, not executing tasks."
        f"{_memory_block}"
    )
    llm_messages = prompt_analyzer.convert_to_llm_messages(
        messages, system_prompt=_atom_prompt, available_tools=None
    )
    return llm_messages, None, None


async def prepare_full_path(
    messages: List[Dict[str, Any]],
    agent_runtime,
    agent_ctx: dict,
    all_tools: List[Dict[str, Any]],
    smart_chat,
    chat_id: str,
    complexity_assessment: Optional[Any],
    prompt_analyzer,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Any]:
    """Full pipeline: MOLECULE / CELL / ORGAN / ORGANISM."""
    from consumers.chatbot.integration import apply_orchestration_to_messages

    llm_messages = prompt_analyzer.convert_to_llm_messages(
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


async def prepare_messages(
    service,
    messages: List[Dict[str, Any]],
    agent_runtime,
    agent_ctx: dict,
    all_tools: List[Dict[str, Any]],
    chat_id: str,
    complexity_assessment: Optional[Any],
    is_cto_agent: bool,
    cto_check_result: Any,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Any]:
    """
    Prepare LLM messages with orchestration, persona, CTO override, and context guard.

    Args:
        service: The StreamingChatService instance (for accessing self.* state).

    Returns:
        (llm_messages, use_tools, orchestrated)
    """
    from consumers.chatbot.auto import Complexity
    from consumers.chatbot.integration import SmartChatIntegration

    from .tool_integration import apply_cto_override, inject_agent_identity

    latest_text = service.prompt_analyzer.extract_latest_user_text(messages)
    smart_chat = SmartChatIntegration(
        workspace_id=str(service.workspace_id) if service.workspace_id else service.workspace_id,
        agent_id=agent_runtime.agent_id,
        agent_name=agent_runtime.metadata.name,
        widget_mode=service.widget_mode,
        db_session=service.db,
    )

    _complexity = (
        complexity_assessment.complexity
        if complexity_assessment
        else Complexity.MOLECULE
    )

    if _complexity == Complexity.ATOM:
        llm_messages, use_tools, orchestrated = await prepare_atom_path(
            messages, agent_runtime, smart_chat,
            service.workspace_id, service.widget_mode, service.prompt_analyzer,
        )
    else:
        llm_messages, use_tools, orchestrated = await prepare_full_path(
            messages, agent_runtime, agent_ctx, all_tools,
            smart_chat, chat_id, complexity_assessment, service.prompt_analyzer,
        )

    # PRD-67: CTO Agent system prompt override
    if is_cto_agent:
        apply_cto_override(
            service.db, llm_messages, smart_chat, cto_check_result,
            messages, use_tools, agent_runtime.agent_id,
        )

    # Inject agent persona + description (skip for CTO -- soul document already includes it)
    if not is_cto_agent and orchestrated:
        inject_agent_identity(llm_messages, agent_ctx)

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

    # Context Window Guard
    from core.context_guard import ContextGuard
    _guard = ContextGuard()
    _model_name = getattr(agent_runtime.llm_manager, 'config', None)
    _model_name = getattr(_model_name, 'model', config.LLM_MODEL) if _model_name else config.LLM_MODEL
    llm_messages, _was_compacted, use_tools = await _guard.check_and_compact(
        messages=llm_messages,
        model_name=_model_name,
        llm_manager=agent_runtime.llm_manager,
        workspace_id=str(service.workspace_id) if service.workspace_id else None,
        agent_id=agent_runtime.agent_id,
        db_session=service.db,
        tools=use_tools,
    )
    if _was_compacted:
        logger.info("[ContextGuard] Messages compacted before LLM call")

    # Stash smart_chat on service for memory storage in post_response
    service._smart_chat = smart_chat

    return llm_messages, use_tools, orchestrated
