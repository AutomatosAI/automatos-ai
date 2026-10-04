"""F305 (night 9): the ``scope`` argument that asks a search for past work, declared in
each schema an agent or Auto reads (services/past_work.py does the search).

Three schemas declare the searches, each inside one long registration function:
search_knowledge's ToolSpec (``ToolRegistry._register_core_tools``), its inline chat
schema (``AgentPlatformTools.get_available_tools``) and platform_search_documents'
ActionDefinition (``register_documents_actions``). Each is wrapped here, so the
argument is added once the function has built its schema, as a new definition.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, List

SEARCH_KNOWLEDGE = "search_knowledge"
SEARCH_DOCUMENTS = "platform_search_documents"
SCOPE = "scope"
SCOPE_DESCRIPTION = (
    "Leave out to search the owner's documents. 'past_work' searches earlier answers agents wrote on approved "
    "cards instead: labelled with who wrote them and when, never the owner's facts."
)
SCOPE_PROPERTY: Dict[str, Any] = {"type": "string", "enum": ["past_work"], "description": SCOPE_DESCRIPTION}


def with_scope(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-schema ``parameters`` object with the ``scope`` property added."""
    return {**parameters, "properties": {**(parameters.get("properties") or {}), SCOPE: dict(SCOPE_PROPERTY)}}


def chat_search_takes_a_scope(build: Callable[..., List[Dict[str, Any]]]) -> Callable[..., List[Dict[str, Any]]]:
    """Wrap ``AgentPlatformTools.get_available_tools``: search_knowledge takes ``scope``."""
    @functools.wraps(build)
    def wrapped(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [{**tool, "parameters": with_scope(tool.get("parameters") or {})}
                if isinstance(tool, dict) and tool.get("name") == SEARCH_KNOWLEDGE else tool
                for tool in build(*args, **kwargs)]
    return wrapped


def core_search_takes_a_scope(register: Callable[[Any], None]) -> Callable[[Any], None]:
    """Wrap ``ToolRegistry._register_core_tools``: search_knowledge's ToolSpec takes ``scope``."""
    @functools.wraps(register)
    def wrapped(self: Any) -> None:
        register(self)
        spec = self.tools.get(SEARCH_KNOWLEDGE)
        if spec is None or any(p.name == SCOPE for p in spec.parameters):
            return
        from modules.tools.registry.tool_registry import ToolParameter

        scope = ToolParameter(name=SCOPE, type="string", description=SCOPE_DESCRIPTION, required=False,
                              enum=list(SCOPE_PROPERTY["enum"]))
        self.tools[SEARCH_KNOWLEDGE] = dataclasses.replace(spec, parameters=[*spec.parameters, scope])
    return wrapped


def documents_search_takes_a_scope(register: Callable[[Any], None]) -> Callable[[Any], None]:
    """Wrap ``register_documents_actions``: platform_search_documents takes ``scope``."""
    @functools.wraps(register)
    def wrapped(registry: Any) -> None:
        register(registry)
        actions = getattr(registry, "_actions", None)
        found = actions.get(SEARCH_DOCUMENTS) if isinstance(actions, dict) else None
        if found is not None:
            actions[SEARCH_DOCUMENTS] = dataclasses.replace(found, parameters=with_scope(found.parameters or {}))
    return wrapped


__all__ = ["SCOPE_DESCRIPTION", "chat_search_takes_a_scope", "core_search_takes_a_scope",
           "documents_search_takes_a_scope", "with_scope"]
