"""platform_browse_memories and platform_delete_memory, registered with the other memory actions
(``platform_actions._register_memory_actions``). P256-FIX-N14-4 (F418) moved them out of
``register_workspace_actions_defs``: browsing lists the asking person's standing rules first, with
the ids delete takes, and delete finds a memory wherever in the workspace it was stored."""

from .action_registry import ActionDefinition, ActionRegistry

_BROWSE_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query for semantic search (optional)."},
        "limit": {"type": "integer", "description": "Max results to return (default 20)."},
    },
    "required": [],
}
_DELETE_PARAMETERS = {
    "type": "object",
    "properties": {"memory_id": {"type": "string", "description": "ID of the memory to delete."}},
    "required": ["memory_id"],
}


def register_memory_browse_actions(registry: ActionRegistry) -> None:
    """Register platform_browse_memories (read) and platform_delete_memory (destructive)."""
    registry.register(ActionDefinition(
        name="platform_browse_memories",
        description=(
            "Browse all stored memories with optional keyword filter. Returns "
            "paginated list with content, agent, date. With no query, the asking "
            "person's standing rules come first (standing_rule: true), each with the "
            "id platform_delete_memory takes. For storing new memories, "
            "use platform_store_memory instead."
        ),
        category="memory",
        parameters=_BROWSE_PARAMETERS,
        permission_level="read",
        tags=["memory", "read", "browse"],
        examples=[
            "show all memories",
            "browse memories",
            "what has been remembered",
            "search memories for user preferences",
            "what standing rules do you have for me",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_memory",
        description=("Permanently delete a specific memory by the id platform_browse_memories "
                     "shows (a standing rule included). Cannot be undone."),
        category="memory",
        parameters=_DELETE_PARAMETERS,
        permission_level="destructive",
        requires_confirmation=True,
        tags=["memory", "write", "destructive"],
        examples=["delete memory abc-123", "remove that incorrect memory", "forget that standing rule"],
    ))
