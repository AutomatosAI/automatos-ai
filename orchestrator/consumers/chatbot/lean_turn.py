"""Lean turns: a runtime profile for agents that answer inside another product.

PRD-274, "Automatos for products", Wave 2 (§5.3). An agent's ``configuration`` names
its ``runtime_profile``: ``full`` (today's turn, and the default) or ``lean``. With
``LEAN_TURNS_ENABLED`` on, a lean agent's chat turn:

- assembles the ``LEAN`` context mode: its own identity, what it recalls, the date,
  the owner's standing rules and the conversation. Auto's personality, platform skill,
  skills, app and action catalogues, the knowledge prefetch, pinned documents and the
  brand kit stay out;
- is offered only the tools its session tool groups grant (the same switches as its
  sessions, ``services.session_tool_groups``), described in its prompt, and the executor
  refuses anything else (``core.security.lean_grants``);
- runs no per-turn app search, so no Composio action is offered.

A system agent (Auto, the CTO) is never lean: its turn is the platform's own.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

RUNTIME_PROFILE_KEY = "runtime_profile"
PROFILE_FULL = "full"
PROFILE_LEAN = "lean"
RUNTIME_PROFILES = (PROFILE_FULL, PROFILE_LEAN)

# The always-on session tools that are not about a board ticket: a lean agent can
# search and read what the workspace knows. The rest (board_summary, update_ticket,
# ask_human, record_memory, composio_execute, …) belong to a session's ticket.
LEAN_CORE_TOOLS = ("search_knowledge", "search_documents", "read_document")

GRANTED_ACTIONS_HEADING = "## Your tools"


def profile_errors(configuration: Any) -> List[str]:
    """Errors (empty when valid) for ``configuration.runtime_profile``; absent is ``full``."""
    raw = configuration.get(RUNTIME_PROFILE_KEY) if isinstance(configuration, Mapping) else None
    if raw is None or raw in RUNTIME_PROFILES:
        return []
    return [f"configuration.{RUNTIME_PROFILE_KEY} must be one of {list(RUNTIME_PROFILES)}, got {raw!r}"]


def is_lean(agent: Any) -> bool:
    """The agent's turns are lean: the dial is on, it is not a system agent, and its
    configuration says ``lean``."""
    from config import config

    if not getattr(config, "LEAN_TURNS_ENABLED", False) or getattr(agent, "is_system_agent", False):
        return False
    configuration = getattr(agent, "configuration", None)
    return isinstance(configuration, Mapping) and configuration.get(RUNTIME_PROFILE_KEY) == PROFILE_LEAN


def granted_names(agent: Any) -> List[str]:
    """The platform actions and tools the agent's session tool groups grant, plus the
    non-ticket core tools, in catalogue order without repeats."""
    from services import session_tool_groups as groups
    from services import session_tools as st

    chosen = set(groups.effective_groups(agent))
    names = [*LEAN_CORE_TOOLS, *(n for g in groups.SESSION_TOOL_GROUPS if g.id in chosen for n in g.tools)]
    granted: List[str] = []
    for name in names:
        tool = st.get_tool(name)
        if tool is not None and tool.action not in granted:
            granted.append(tool.action)
    return granted


def lean_grant_for(db: Any, agent_id: Any) -> Optional[List[str]]:
    """The turn's grants when ``agent_id`` is a lean agent; None for any other turn. The tools
    of the partner packs the agent lists (PRD-275) are grants too."""
    from config import config

    if not getattr(config, "LEAN_TURNS_ENABLED", False):
        return None
    from core.models import Agent
    from modules.tools.partner.offer import partner_tool_names

    agent = db.query(Agent).filter(Agent.id == agent_id).first() if db is not None else None
    if agent is None or not is_lean(agent):
        return None
    return [*granted_names(agent), *partner_tool_names(db, agent)]


def lean_tool_surface(granted: List[str], *, is_super_admin: bool, db: Any = None) -> List[Dict[str, Any]]:
    """The tool schemas a lean turn is offered: the platform dispatcher narrowed to the
    granted actions, and each granted registry tool's own schema."""
    from core.security.widget_scopes import narrowed_dispatcher
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    actions = [name for name in granted if registry.get(name) is not None]
    surface: List[Dict[str, Any]] = []
    if actions:
        full = registry.to_dispatcher_schema(
            exclude_admin=True, exclude_promoted=False, include_super_admin=is_super_admin,
        )
        dispatcher = narrowed_dispatcher(full, actions)
        if dispatcher is not None:
            surface.append(dispatcher)
    surface.extend(_registry_schemas([name for name in granted if name not in actions], db))
    return surface


def _registry_schemas(names: List[str], db: Any) -> List[Dict[str, Any]]:
    if not names:
        return []
    from modules.tools.registry.tool_registry import get_tool_registry

    registry = get_tool_registry(db)
    specs = [registry.get_tool(name) for name in names]
    return [spec.to_openai_format() for spec in specs if spec is not None]


def offered_names(surface: List[Dict[str, Any]]) -> List[str]:
    """The action and tool names a surface offers: the dispatcher's enum, then the rest."""
    names: List[str] = []
    for schema in surface:
        function = (schema or {}).get("function") or {}
        if function.get("name") == "platform_execute":
            names.extend(function.get("parameters", {}).get("properties", {}).get("action", {}).get("enum", []))
        elif function.get("name"):
            names.append(function["name"])
    return names


def with_granted_actions(llm_messages: List[Dict[str, Any]], surface: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``llm_messages`` with the offered platform actions described at the end of the
    system prompt (the dispatcher names them; this says what each takes). A new list.
    The surface was already filtered for the caller, so every action it offers is described."""
    from modules.tools.discovery.action_registry import get_action_registry

    actions = [name for name in offered_names(surface) if name.startswith("platform_")]
    if not actions:
        return llm_messages
    summary = get_action_registry().build_filtered_prompt_summary(
        actions, exclude_admin=True, include_super_admin=True,
    )
    block = f"{GRANTED_ACTIONS_HEADING}\nCall these through platform_execute.\n\n{summary}".strip()
    first = llm_messages[0] if llm_messages else {}
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": f"{first['content']}\n\n{block}"}, *llm_messages[1:]]
    return [{"role": "system", "content": block}, *llm_messages]


# --- the chat turn's hooks -------------------------------------------------------------------
# ``StreamingChatService`` marks a lean agent's turn with its grants (``_lean_actions``) and the
# executor's gate (core.security.lean_grants); each step below is a wrapper on the service
# method it changes, so the turn's own long function is not touched.

def lean_turn_of(chat: Any) -> bool:
    """``chat``'s current turn is a lean agent's."""
    return getattr(chat, "_lean_actions", None) is not None


def offers_a_lean_turn_its_grants(get_tools: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``StreamingChatService._get_tools``: a lean turn is offered its grants, nothing else.
    Its granted partner tools (PRD-275) are workspace data, read on a session of their own, off
    the event loop."""
    @functools.wraps(get_tools)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        if not lean_turn_of(chat):
            return await get_tools(chat, *args, **kwargs)
        from modules.tools.partner.offer import granted_partner_schemas, partner_tools_enabled
        from modules.tools.partner.store import on_own_session

        surface = lean_tool_surface(chat._lean_actions, is_super_admin=bool(kwargs.get("is_super_admin")),
                                    db=getattr(chat, "db", None))
        if not partner_tools_enabled():
            return surface
        agent_id = kwargs.get("agent_id", args[0] if args else None)
        partner = await asyncio.to_thread(on_own_session, granted_partner_schemas, agent_id, chat._lean_actions)
        return [*surface, *partner]
    return wrapped


def a_lean_turns_prompt(prepare: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``StreamingChatService._prepare_messages``: a lean turn's model call carries its
    granted tools (the LEAN mode loads none) and their descriptions, and the request's app
    context (consumers.chatbot.lean_request) for this turn only; the turn's trace keeps the
    app context's size, never its words."""
    @functools.wraps(prepare)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> Any:
        from consumers.chatbot.lean_request import NOTHING_ASKED, with_app_context, with_app_context_size

        llm_messages, use_tools, orchestrated = await prepare(chat, *args, **kwargs)
        if not lean_turn_of(chat):
            return llm_messages, use_tools, orchestrated
        all_tools = kwargs.get("all_tools", args[3] if len(args) > 3 else None) or []
        app_context = getattr(chat, "_lean_request", NOTHING_ASKED).app_context
        llm_messages = with_app_context(with_granted_actions(llm_messages, all_tools), app_context)
        if app_context and dataclasses.is_dataclass(orchestrated):
            trace = with_app_context_size(getattr(orchestrated, "context_trace", None), app_context)
            orchestrated = dataclasses.replace(orchestrated, context_trace=trace)
        return llm_messages, all_tools or None, orchestrated
    return wrapped


def no_pinned_documents_on_a_lean_turn(inject: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``StreamingChatService._inject_pinned_documents``: not on a lean turn."""
    @functools.wraps(inject)
    def wrapped(chat: Any, *args: Any, **kwargs: Any) -> None:
        if not lean_turn_of(chat):
            inject(chat, *args, **kwargs)
    return wrapped


def no_app_search_on_a_lean_turn(inject: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``StreamingChatService._inject_composio_tools``: a lean turn runs no per-turn app
    search, so it is offered no Composio action; its tools stay as they are."""
    @functools.wraps(inject)
    async def wrapped(chat: Any, llm_messages: Any, use_tools: Any, *args: Any, **kwargs: Any) -> Any:
        if lean_turn_of(chat):
            return use_tools, None
        return await inject(chat, llm_messages, use_tools, *args, **kwargs)
    return wrapped


def no_prefetch_on_a_lean_turn(prefetch: Callable[..., AsyncGenerator[Any, None]]) -> Callable[..., Any]:
    """Wrap ``StreamingChatService._retrieval_first``: a lean turn searches nothing before its
    first call; its agent searches with its granted tools when it needs to."""
    @functools.wraps(prefetch)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        if lean_turn_of(chat):
            return
        async for frame in prefetch(chat, *args, **kwargs):
            yield frame
    return wrapped
