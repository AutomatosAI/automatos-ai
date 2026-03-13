"""
Tool Integration Helpers
========================

Tool-related methods extracted from StreamingChatService:
- Tool schema retrieval (get_tools)
- Composio per-action tool injection
- CTO system prompt override
- Agent identity injection
- Tool message builders
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)


def get_tools(
    agent_id: int,
    db_session: Session,
    workspace_id: Optional[str],
    skill_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get all tools for an agent from the SINGLE source: modules.tools.tool_router.

    Returns full OpenAI-format tool schemas (ToolRegistry + ActionRegistry + Composio).
    Appends any skill-specific tool schemas from the agent runtime.
    """
    from modules.tools.tool_router import get_tools_for_agent

    all_tools = get_tools_for_agent(
        agent_id=agent_id,
        db_session=db_session,
        workspace_id=workspace_id,
    )
    if skill_tools:
        all_tools = (all_tools or []) + skill_tools
    return all_tools


def apply_cto_override(
    db_session: Session,
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

    _platform_state = CtoPromptBuilder.get_platform_state_snapshot(db_session)
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


def inject_agent_identity(
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


def inject_composio_tools(
    db_session: Session,
    llm_messages: List[Dict[str, Any]],
    use_tools: Optional[List[Dict[str, Any]]],
    latest_text: str,
    agent_id: int,
    agent_runtime,
    skip_composio: bool,
    complexity_assessment: Optional[Any],
    workspace_id: Optional[str],
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
        if latest_text and agent_id and workspace_id and not skip_composio:
            from modules.tools.services.composio_tool_service import ComposioToolService

            _composio_svc = ComposioToolService(db_session)
            _search_prompt = (
                " ".join(_tool_hints) if _tool_hints else latest_text
            )
            _composio_result = _composio_svc.get_tools_for_step(
                agent_id=agent_id,
                workspace_id=workspace_id,
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

                hint_service = ComposioHintService(db_session)
                hint_result = hint_service.build_hints(
                    agent_id=agent_id,
                    prompt=latest_text,
                    workspace_id=workspace_id,
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


def build_assistant_tool_message(
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


def extract_user_text_from_messages(llm_messages: List[Dict[str, Any]]) -> str:
    """Extract the latest user message text from LLM messages."""
    for m in reversed(llm_messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def score_composio_candidates(user_text: str, examples_str: str) -> List[str]:
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


def build_composio_param_recovery(
    db_session: Session,
    llm_messages: List[Dict[str, Any]],
    agent_runtime,
    followup_system_messages: List[Dict[str, Any]],
    workspace_id: Optional[str],
) -> None:
    """Build recovery instructions for composio_execute missing parameters."""
    try:
        from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
        from core.composio.entity_manager import EntityManager

        user_text = extract_user_text_from_messages(llm_messages)
        q = (user_text or "").lower()
        q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
        stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
        q_tokens = [t for t in q_tokens if t not in stop][:10]

        aid = agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else 0
        assigned = (
            db_session.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == aid,
                AgentAppAssignment.is_active == True,
                AgentAppAssignment.app_type == "EXTERNAL",
            )
            .all()
        )
        assigned_apps = [(a.app_name or "").upper() for a in assigned if a.app_name]
        allowed_apps = assigned_apps

        if workspace_id:
            manager = EntityManager(db_session)
            entity = manager.get_entity_by_workspace(workspace_id)
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
                    db_session.query(ComposioActionCache.action_name)
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
