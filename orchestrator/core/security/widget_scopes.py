"""F155 — what a public widget turn may call, by its key's scopes.

Gerard (25 Sep): the key's scopes decide, as the owner grants them per key on
Settings → API keys. Every widget turn gets the base lookups (knowledge search,
blog reads, the callback form); each scope below adds its family, and a scope
this map does not name adds nothing. Anything not listed is never called on a
widget turn:

- writes that are the owner's: agents, playbooks, skills, schedules, watches,
  memory, marketplace installs, the harness, widget config, onboarding, the
  mission lifecycle, board tasks, blog posts, reports, messaging the owner;
- reads of the owner's private data: members, API keys, channels, settings,
  analytics, memory, other conversations, reports, code;
- the owner's connected apps (composio_*) and the workspace's files, shell and
  network tools.

The tool executor refuses on the resolved action (platform_execute's inner
action included), and the chat trims the tool list the model is offered to
the same set.

missions:execute and playbooks:execute unlock nothing on a widget turn (F155
review, 25 Sep). A mission's tasks and a playbook's steps run later, outside
the turn; work a widget turn starts runs under its key's restrictions again
(core.security.surface.origin_surface), and whether a key may start it at all
is not decided yet.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional

WIDGET_REFUSAL = "That isn't available in this website chat."

BASE_TOOLS: FrozenSet[str] = frozenset({
    "search_knowledge",
    "semantic_search",
    "search_multimodal",
    "search_tables",
    "search_images",
    "search_formulas",
    "platform_list_blog_posts",
    "platform_get_blog_post",
    "platform_find_tools",
    "widget_open_callback_form",
})

SCOPE_TOOLS: Mapping[str, FrozenSet[str]] = {
    "documents:read": frozenset({
        "platform_search_documents",
        "platform_list_documents",
        "platform_read_document",
        "platform_grep_documents",
        "platform_list_templates",
        "platform_get_template_schema",
        "platform_query_graph",
        "platform_graph_communities",
        "platform_graph_impact",
        "platform_graph_neighbors",
        "platform_graph_path",
        "platform_graph_stats",
    }),
    "documents:write": frozenset({"platform_upload_document", "platform_reprocess_document"}),
    "data:query": frozenset({"smart_query_database", "platform_query_data"}),
    # Who the agents are (a visitor's view: name, description, status); never
    # their heartbeats, live work or models.
    "agents:read": frozenset({"platform_list_agents", "platform_get_agent"}),
    "missions:read": frozenset({"platform_list_missions", "platform_get_mission"}),
    "playbooks:read": frozenset({"platform_list_playbooks", "platform_get_playbook", "platform_get_playbook_execution"}),
    "tasks:read": frozenset({
        "platform_list_tasks",
        "platform_get_task",
        "platform_board_snapshot",
        "platform_board_summary",
        "platform_wait_for_task",
    }),
}


def allowed_tools(scopes: Iterable[str]) -> FrozenSet[str]:
    """The tools a widget turn may call with these key scopes."""
    granted = set(BASE_TOOLS)
    for scope in scopes or ():
        granted |= SCOPE_TOOLS.get(scope, frozenset())
    return frozenset(granted)


def widget_may_call(tool_name: str, scopes: Iterable[str]) -> bool:
    """A widget turn with these key scopes may call ``tool_name`` (the resolved
    action, for platform_execute)."""
    return tool_name in allowed_tools(scopes)


def widget_tool_surface(tools: Optional[List[Dict[str, Any]]], scopes: Iterable[str]) -> List[Dict[str, Any]]:
    """The tool schemas a widget turn is offered: the allowed tools, and the
    platform_execute dispatcher narrowed to the allowed platform actions
    (dropped when none is allowed). A new list; no schema is mutated."""
    allowed = allowed_tools(scopes)
    actions = [name for name in sorted(allowed) if name.startswith("platform_")]
    surface: List[Dict[str, Any]] = []
    for schema in tools or []:
        name = ((schema or {}).get("function") or {}).get("name", "")
        if name == "platform_execute":
            dispatcher = _narrowed_dispatcher(schema, actions)
            if dispatcher is not None:
                surface.append(dispatcher)
        elif name in allowed:
            surface.append(schema)
    return surface


def _narrowed_dispatcher(schema: Dict[str, Any], actions: List[str]) -> Optional[Dict[str, Any]]:
    """A copy of the dispatcher whose action enum holds only ``actions`` (and
    only those it already offered, when it carried an enum); None when none
    is left."""
    try:
        offered = schema["function"]["parameters"]["properties"]["action"].get("enum")
    except (KeyError, TypeError, AttributeError):
        return None
    kept = [name for name in actions if not offered or name in offered]
    if not kept:
        return None
    narrowed = copy.deepcopy(schema)
    narrowed["function"]["parameters"]["properties"]["action"]["enum"] = kept
    return narrowed
