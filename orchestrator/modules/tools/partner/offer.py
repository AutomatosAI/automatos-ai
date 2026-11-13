"""Partner tools: which agents are offered which tools (PRD-275).

A pack installed in a workspace is offered to an agent only when the agent's
``configuration.partner_packs`` lists it *and* the install allows that agent
(``store.may_use``), so Auto and every other agent keep the tools they had, and an agent
someone edits to list a pack gets nothing the manager who installed it did not allow. Both
are read again when a call runs (``runner.resolve``): offering a tool is not what lets it run.

Three places read this module, each on its own seam so no long function is touched:

* ``tool_router._get_tools_for_agent_core`` (``offers_partner_tools``): the one source of an
  agent's tool schemas, for chat turns and agent runs;
* a lean turn's grants and surface (``consumers.chatbot.lean_turn``): a lean agent is offered
  only what it is granted, so its packs' tools are named there;
* ``api.agents`` (``partner_pack_errors``): the shape of ``configuration.partner_packs``.

Everything is a no-op while ``PARTNER_TOOLS_ENABLED`` is off.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from modules.tools.partner.pack import PACK_NAME, PartnerPack, full_tool_name, pack_schemas
from modules.tools.partner.store import installed_packs, may_use, same_workspace

logger = logging.getLogger(__name__)

CONFIG_KEY = "partner_packs"
MAX_PACKS_PER_AGENT = 10
_PACK_NAME = re.compile(PACK_NAME)

Schemas = List[Dict[str, Any]]


def partner_tools_enabled() -> bool:
    """The server offers and runs partner tools (``PARTNER_TOOLS_ENABLED``)."""
    from config import config

    return bool(getattr(config, "PARTNER_TOOLS_ENABLED", False))


def packs_named(configuration: Any) -> Tuple[str, ...]:
    """The pack names an agent's configuration lists, without repeats; anything else is no pack."""
    raw = configuration.get(CONFIG_KEY) if isinstance(configuration, Mapping) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(name for name in raw if isinstance(name, str) and _PACK_NAME.fullmatch(name)))


def partner_pack_errors(configuration: Any) -> List[str]:
    """Errors (empty when valid) for ``configuration.partner_packs``; absent is no packs. While
    partner tools are off the key is not read, so it is not judged either."""
    raw = configuration.get(CONFIG_KEY) if isinstance(configuration, Mapping) else None
    if raw is None or not partner_tools_enabled():
        return []
    names_ok = isinstance(raw, (list, tuple)) and len(raw) <= MAX_PACKS_PER_AGENT and all(
        isinstance(name, str) and _PACK_NAME.fullmatch(name) for name in raw)
    if names_ok:
        return []
    return [f"configuration.{CONFIG_KEY} must be a list of at most {MAX_PACKS_PER_AGENT} pack names"]


def granted_packs(db: Any, agent_id: Any, workspace_id: Any) -> Tuple[str, ...]:
    """The packs the agent lists, when the agent is this workspace's own; none otherwise."""
    from core.models import Agent

    if agent_id is None or workspace_id is None:
        return ()
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None or not same_workspace(agent.workspace_id, workspace_id):
        return ()
    return packs_named(agent.configuration)


def schema_name(schema: Any) -> Optional[str]:
    """The name a tool schema is called by."""
    function = schema.get("function") if isinstance(schema, Mapping) else None
    return function.get("name") if isinstance(function, Mapping) else None


def _schemas_for(db: Any, workspace_id: Any, names: Iterable[str], agent_id: Any) -> Schemas:
    """The tools of the named packs the workspace installed for ``agent_id``."""
    return [schema for installed in installed_packs(db, workspace_id, names) if may_use(installed, agent_id)
            for schema in pack_schemas(installed.pack)]


def _read(db: Any, what: str, read: Callable[[], Any], fallback: Any) -> Any:
    """``read()``, or ``fallback`` when it fails: logged, and the caller's session is left usable.
    A turn is never lost because its partner tools could not be read."""
    from core.database.session_health import rollback_if_aborted

    try:
        return read()
    except Exception:  # noqa: BLE001 — logged; the turn goes on without partner tools
        logger.exception("[partner-tools] could not read %s; no partner tool is offered", what)
        rollback_if_aborted(db, "partner tools")
        return fallback


def with_partner_tools(tools: Schemas, db: Any, agent_id: Any, workspace_id: Any) -> Schemas:
    """``tools`` and the agent's partner tools, as a new list. A name already offered is not
    offered again: two tools of one name would fail the model call."""
    if not partner_tools_enabled() or db is None:
        return tools
    names = _read(db, f"agent {agent_id}'s packs", lambda: granted_packs(db, agent_id, workspace_id), ())
    if not names:
        return tools
    taken = {schema_name(schema) for schema in tools}
    fresh: Schemas = []
    for schema in _read(db, f"agent {agent_id}'s partner tools",
                        lambda: _schemas_for(db, workspace_id, names, agent_id), []):
        if schema_name(schema) not in taken:
            taken.add(schema_name(schema))
            fresh.append(schema)
    return [*tools, *fresh]


def offers_partner_tools(core: Callable[..., Schemas]) -> Callable[..., Schemas]:
    """Wrap ``tool_router._get_tools_for_agent_core``: an agent's own partner tools join its tools."""

    @functools.wraps(core)
    def wrapped(**kwargs: Any) -> Schemas:
        return with_partner_tools(core(**kwargs), kwargs.get("session_used"), kwargs.get("agent_id"),
                                  kwargs.get("workspace_id"))

    return wrapped


def _agent_schemas(db: Any, agent: Any) -> Schemas:
    """The tools of the packs the agent lists and its own workspace installed for it."""
    names = packs_named(getattr(agent, "configuration", None))
    workspace_id, agent_id = getattr(agent, "workspace_id", None), getattr(agent, "id", None)
    if not names or workspace_id is None:
        return []
    return _read(db, f"agent {agent_id}'s partner tools", lambda: _schemas_for(db, workspace_id, names, agent_id), [])


def partner_tool_names(db: Any, agent: Any) -> List[str]:
    """A lean turn's grants: the names of the agent's partner tools."""
    if not partner_tools_enabled() or db is None or agent is None:
        return []
    return [name for name in (schema_name(schema) for schema in _agent_schemas(db, agent)) if name]


def granted_partner_schemas(db: Any, agent_id: Any, granted: Optional[Iterable[str]]) -> Schemas:
    """A lean turn's surface: the agent's partner tools that its grants name. The workspace is
    the agent's own, as it is for the grants (``partner_tool_names``)."""
    from core.models import Agent

    allowed = set(granted or ())
    if not partner_tools_enabled() or db is None or not allowed:
        return []
    agent = _read(db, f"agent {agent_id}", lambda: db.query(Agent).filter(Agent.id == agent_id).first(), None)
    if agent is None:
        return []
    return [schema for schema in _agent_schemas(db, agent) if schema_name(schema) in allowed]


def taken_names(db: Any, pack: PartnerPack) -> List[str]:
    """The pack's tool names that one of the platform's own tools already has."""
    from modules.tools.execution.unified_executor import TOOL_ALIASES
    from modules.tools.registry.tool_registry import get_tool_registry

    registry = get_tool_registry(db)
    names = [full_tool_name(pack.name, tool) for tool in pack.tools]
    return [name for name in names if name in TOOL_ALIASES or registry.get_tool(name) is not None]


__all__ = ["CONFIG_KEY", "granted_packs", "granted_partner_schemas", "offers_partner_tools", "packs_named",
           "partner_pack_errors", "partner_tool_names", "partner_tools_enabled", "schema_name", "taken_names",
           "with_partner_tools"]
