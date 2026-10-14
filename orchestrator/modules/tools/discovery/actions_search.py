"""Search-related ActionDefinitions (chat history search, memory search)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_search_actions(registry: ActionRegistry) -> None:
    """Register search-related platform actions."""
    _register_search_chat_history(registry)
    _register_search_memory(registry)


def _register_search_chat_history(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_search_chat_history",
        description=(
            "Search this person's own past chats in this workspace by keyword: a message "
            "matches when it holds every word of the query, in any order. Returns matching "
            "messages with the chat title, role (user/assistant), content snippet, and "
            "timestamp (UTC). Use when the user asks about previous conversations, wants to "
            "find something they discussed before, or references past chats."
        ),
        category="chat",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for in chat messages (each must appear; a few distinctive ones work best).",
                },
                "days": {
                    "type": "integer",
                    "description": "How far back to search in days (default 30, max 365).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20, max 100).",
                },
            },
            "required": ["query"],
        },
        permission_level="read",
        tags=["chat", "history", "search", "conversation"],
        examples=[
            "what did we talk about yesterday?",
            "search my chats for 'Jira'",
            "find conversations about the deployment",
            "did I mention anything about Redis?",
        ],
    ))


def _register_search_memory(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_search_memory",
        description=(
            "Search the agent's long-term memory (Mem0) for stored facts, preferences, "
            "and past context. Searches both global workspace memories and per-agent "
            "memories. Use when the user asks what the system remembers, or to look up "
            "specific stored facts."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50).",
                },
            },
            "required": ["query"],
        },
        permission_level="read",
        promoted=True,
        tags=["memory", "search", "recall", "context"],
        examples=[
            "what do you remember about me?",
            "search memory for Slack channel",
            "what's stored about the deployment?",
            "do you remember my name?",
        ],
    ))
