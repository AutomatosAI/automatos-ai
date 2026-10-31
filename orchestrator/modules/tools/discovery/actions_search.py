"""Search-related ActionDefinitions (chat history search, memory search)."""

from .action_registry import ActionDefinition, ActionRegistry
from .my_actions_window import WINDOWS


def register_search_actions(registry: ActionRegistry) -> None:
    """Register search-related platform actions."""
    _register_search_chat_history(registry)
    _register_search_memory(registry)


# P256-FIX-N17-11 (F443): a window's messages, oldest or newest first, with no query (chat_history_read).
ORDERS = ("newest", "oldest")      # the default first
ROLES = ("user", "assistant")
CHAT_HISTORY_DESCRIPTION = (
    "Search this person's own past chats in this workspace: with query, the messages holding every word of "
    "it; with no query, every message in a window or since ('tonight' runs from 18:00 UTC). Each result: "
    "date (UTC), chat title, role, up to 300 characters, and the gap before it when 30+ minutes passed with "
    "nothing. For 'what did I ask first tonight?' read last_24_hours, role user, order oldest, and answer "
    "with times, naming any gap ('first after 18:00 UTC was X at 18:29; first after the 2 h gap was Y at "
    "17:03'). Use for past conversations and what the user said before."
)
CHAT_HISTORY_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Words to search for in chat messages (each must appear; a few distinctive ones "
                           "work best). Leave it out to list a window's messages.",
        },
        "window": {
            "type": "string",
            "enum": list(WINDOWS),
            "description": "The time to read: 'tonight' (from 18:00 UTC), 'today', 'yesterday', "
                           "'last_24_hours' or 'this_week'.",
        },
        "since": {
            "type": "string",
            "description": "Read from this time on (ISO 8601, UTC, e.g. 2026-10-14T17:00:00Z); overrides "
                           "window; at most 90 days back.",
        },
        "order": {
            "type": "string",
            "enum": list(ORDERS),
            "description": "newest first (the default) or oldest first.",
        },
        "role": {
            "type": "string",
            "enum": list(ROLES),
            "description": "user: only what this person said; assistant: only the replies. Leave it out "
                           "for both.",
        },
        "days": {
            "type": "integer",
            "description": "With a query and no window or since: how far back to search in days "
                           "(default 30, max 365).",
        },
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 20, max 100).",
        },
    },
    "required": [],
}


def _register_search_chat_history(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_search_chat_history",
        description=CHAT_HISTORY_DESCRIPTION,
        category="chat",
        parameters=CHAT_HISTORY_PARAMETERS,
        permission_level="read",
        tags=["chat", "history", "search", "conversation"],
        examples=[
            "what did we talk about yesterday?",
            "search my chats for 'Jira'",
            "find conversations about the deployment",
            "did I mention anything about Redis?",
            "what did I ask you first tonight?",
            "what was the first thing I asked you today?",
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
