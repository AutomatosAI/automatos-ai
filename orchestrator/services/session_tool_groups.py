"""#942: the owner chooses, per agent, which platform tool groups its sessions get.

``agent.configuration["session_tool_groups"]`` is a list of group ids. Absent means
``DEFAULT_SESSION_TOOL_GROUPS`` (all six: a fixed constant, never derived from the
workspace, so what an agent is offered moves only when the owner changes the
agent); ``[]`` means the core tools only. PRD-245 D2's prompt-cache rule still
holds, one level down: the advertised list is fixed PER AGENT across all its
sessions. Scope is still applied per call, and a call to a tool outside the
agent's groups is refused in words (``not_offered_text``).

Everything that tells a session its tools reads this module: the host's claim
payload and policy names (``agent_tool_names``), ``tools/list`` and the manifest
(``offered_definitions``), and the agent page (``groups_payload``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services import session_tools as st

GROUPS_KEY = "session_tool_groups"


@dataclass(frozen=True)
class ToolGroup:
    """One switch on the agent page: the session tools it turns on."""

    id: str
    label: str
    description: str
    tools: Tuple[str, ...]


# Display order, and the order the tools are advertised in after the core ones.
SESSION_TOOL_GROUPS: Tuple[ToolGroup, ...] = (
    ToolGroup("data", "Data",
              "Ask the workspace's connected database in plain words; read-only.",
              ("query_database",)),
    ToolGroup("graph", "Knowledge Graph",
              "Who supplies, buys or is part of what, from the owner's documents; read-only.",
              ("query_graph",)),
    ToolGroup("documents", "Documents",
              "Make PDF, Word and Excel files, and social images from templates, saved to Deliverables; "
              "list the templates, read what each needs, draw a page to look at it, and make or change "
              "document templates (never a starter).",
              ("generate_document", "list_templates", "get_template_schema", "render_preview",
               "create_template", "update_template")),
    ToolGroup("playbooks", "Playbooks",
              "List and read the workspace's playbooks, and start a run of one.",
              ("list_playbooks", "get_playbook", "run_playbook")),
    ToolGroup("reports", "Reports",
              "Read another agent's latest report.",
              ("get_latest_report",)),
    ToolGroup("missions", "Missions",
              "Read missions and their steps, and what the mission's other agents found.",
              ("list_missions", "get_mission", "search_mission_findings")),
)
GROUP_IDS: Tuple[str, ...] = tuple(g.id for g in SESSION_TOOL_GROUPS)
DEFAULT_SESSION_TOOL_GROUPS: Tuple[str, ...] = GROUP_IDS
_GROUPED = {name for g in SESSION_TOOL_GROUPS for name in g.tools}
# Always on: every session tool no group owns, in the list's own order.
CORE_TOOLS: Tuple[str, ...] = tuple(n for n in st.tool_names() if n not in _GROUPED)


class UnknownToolGroup(ValueError):
    """A group id the catalogue does not have; the message names the valid ones."""


def parse_groups(value: Any) -> List[str]:
    """``value`` (a list, or a comma-separated string) as group ids in catalogue
    order. Raises :class:`UnknownToolGroup` for anything else."""
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise UnknownToolGroup(f"{GROUPS_KEY} must be a list of group ids: {', '.join(GROUP_IDS)}")
    asked = [str(i).strip() for i in items if str(i).strip()]
    unknown = [i for i in asked if i not in GROUP_IDS]
    if unknown:
        raise UnknownToolGroup(
            f"unknown session tool group(s): {', '.join(unknown)}. The groups are: {', '.join(GROUP_IDS)}."
        )
    return [g for g in GROUP_IDS if g in asked]


def check_configuration(configuration: Any) -> None:
    """Refuse an agent configuration whose ``session_tool_groups`` names an
    unknown group; absent (or null) is the default and passes."""
    raw = configuration.get(GROUPS_KEY) if isinstance(configuration, dict) else None
    if raw is None:
        return
    if not isinstance(raw, (list, tuple)):
        raise UnknownToolGroup(f"{GROUPS_KEY} must be a list of group ids: {', '.join(GROUP_IDS)}")
    parse_groups(raw)


def is_default(agent: Any) -> bool:
    cfg = getattr(agent, "configuration", None)
    return not isinstance(cfg, dict) or cfg.get(GROUPS_KEY) is None


def effective_groups(agent: Any) -> List[str]:
    """The groups this agent's sessions get: its own choice, else the default.
    A stored value that does not parse (written before validation) counts as
    the known ids in it, in catalogue order."""
    if is_default(agent):
        return list(DEFAULT_SESSION_TOOL_GROUPS)
    raw = agent.configuration.get(GROUPS_KEY)
    known = {str(i).strip() for i in raw} if isinstance(raw, (list, tuple)) else set()
    return [g for g in GROUP_IDS if g in known]


def tools_for_groups(groups: Iterable[str]) -> List[str]:
    """The tool names those groups give: core first, then each group's tools in
    catalogue order, whatever order ``groups`` came in."""
    chosen = set(groups)
    grouped = [name for g in SESSION_TOOL_GROUPS if g.id in chosen for name in g.tools]
    return list(CORE_TOOLS) + grouped


def agent_tool_names(agent: Any) -> Tuple[str, ...]:
    """What this agent's sessions are offered (the claim payload, the host's gate)."""
    return tuple(tools_for_groups(effective_groups(agent)))


def session_tools_for_agent(agent: Any) -> Tuple[st.SessionTool, ...]:
    """The session tools themselves, in the advertised order."""
    return tuple(st.get_tool(name) for name in agent_tool_names(agent))


def _offered(ctx: st.SessionContext) -> Sequence[str]:
    return ctx.offered if ctx.offered is not None else st.tool_names()


def offered_definitions(ctx: st.SessionContext) -> List[Dict[str, Any]]:
    """``tools/list`` for this session: its agent's tools, stable text and order."""
    names = set(_offered(ctx))
    return [dict(d) for d in st.definitions() if d["name"] in names]


def default_definitions() -> List[Dict[str, Any]]:
    """The list an agent on the default groups is offered: Settings → Session mode
    shows it (that page has no one agent; the agent page shows each agent's own)."""
    return offered_definitions(st.SessionContext(task_id=0, agent_id=None, agent_name=None, workspace_id=None,
                                                 offered=tuple(tools_for_groups(DEFAULT_SESSION_TOOL_GROUPS))))


def offered_tool(ctx: st.SessionContext, name: Any) -> Optional[st.SessionTool]:
    """The tool, when this session is offered it; ``None`` otherwise."""
    tool = st.get_tool(name)
    return tool if tool is not None and tool.name in _offered(ctx) else None


def group_of(name: str) -> Optional[ToolGroup]:
    return next((g for g in SESSION_TOOL_GROUPS if name in g.tools), None)


def not_offered_text(ctx: st.SessionContext, name: Any) -> str:
    """Why a named tool cannot run here, in words the model can act on."""
    available = ", ".join(_offered(ctx))
    group = group_of(str(name or ""))
    if group is None:
        return f"{name!r} is not a tool this session has. Available: {available}."
    return (f"{name!r} is not turned on for this agent: it is in the {group.label} tool group, which the "
            f"owner turns on in the agent's settings. Do the work without it and say so in your result. "
            f"Available: {available}.")


def groups_payload(agent: Any, enabled: Optional[List[str]] = None) -> Dict[str, Any]:
    """``session_tool_groups`` on GET /api/agents/{id}: the enabled ids (the
    agent's own, or ``enabled`` when the page previews other groups), whether
    that is the default, and the catalogue."""
    return {
        "enabled": list(effective_groups(agent) if enabled is None else enabled),
        "is_default": is_default(agent),
        "available": [
            {"id": g.id, "label": g.label, "description": g.description, "tools": list(g.tools)}
            for g in SESSION_TOOL_GROUPS
        ],
    }
