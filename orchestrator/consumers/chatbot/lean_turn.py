"""Lean turns: a runtime profile for agents that answer inside another product.

PRD draft "Automatos for products", Wave 2 (§5.3). An agent's ``configuration`` names
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

import logging
from typing import Any, Dict, List, Mapping, Optional

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
    """The turn's grants when ``agent_id`` is a lean agent; None for any other turn."""
    from config import config

    if not getattr(config, "LEAN_TURNS_ENABLED", False):
        return None
    from core.models import Agent

    agent = db.query(Agent).filter(Agent.id == agent_id).first() if db is not None else None
    return granted_names(agent) if agent is not None and is_lean(agent) else None


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


def with_granted_actions(llm_messages: List[Dict[str, Any]], surface: List[Dict[str, Any]],
                        *, is_super_admin: bool) -> List[Dict[str, Any]]:
    """``llm_messages`` with the offered platform actions described at the end of the
    system prompt (the dispatcher names them; this says what each takes). A new list."""
    from modules.tools.discovery.action_registry import get_action_registry

    actions = [name for name in offered_names(surface) if name.startswith("platform_")]
    if not actions:
        return llm_messages
    summary = get_action_registry().build_filtered_prompt_summary(
        actions, exclude_admin=True, include_super_admin=is_super_admin,
    )
    block = f"{GRANTED_ACTIONS_HEADING}\nCall these through platform_execute.\n\n{summary}".strip()
    first = llm_messages[0] if llm_messages else {}
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": f"{first['content']}\n\n{block}"}, *llm_messages[1:]]
    return [{"role": "system", "content": block}, *llm_messages]
