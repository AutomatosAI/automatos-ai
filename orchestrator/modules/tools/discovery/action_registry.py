"""
Action Registry (PRD-64)
========================

Central registry for platform actions that Auto can discover and execute.
Each ActionDefinition describes a platform operation with its parameters,
permission level, and OpenAI function schema.

Usage:
    registry = get_action_registry()
    all_actions = registry.get_all()
    openai_tools = registry.to_openai_tools()
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-safe singleton
_registry_lock = threading.Lock()
_registry_instance: Optional["ActionRegistry"] = None


@dataclass
class ActionDefinition:
    """Definition of a platform action that Auto can execute."""

    name: str                           # e.g. "platform_list_agents"
    description: str                    # For embedding + LLM tool description
    category: str                       # "agents", "analytics", "recipes", etc.
    parameters: Dict[str, Any]          # OpenAI function parameters schema
    permission_level: str = "read"      # "read" | "write" | "destructive"
    requires_confirmation: bool = False
    workspace_scoped: bool = True
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ActionRegistry:
    """
    Registry of all platform actions available to Auto.

    Thread-safe. Actions are registered at startup and queried at runtime.
    Supports filtering by category and permission level.
    """

    def __init__(self):
        self._actions: Dict[str, ActionDefinition] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-load platform actions on first access."""
        if not self._initialized:
            from .platform_actions import register_all_actions
            from .workspace_actions import register_workspace_actions
            register_all_actions(self)
            register_workspace_actions(self)
            self._initialized = True
            logger.info(f"[ActionRegistry] Initialized with {len(self._actions)} actions (platform + workspace)")

    def register(self, action: ActionDefinition) -> None:
        """Register a platform action."""
        if action.name in self._actions:
            logger.warning(f"[ActionRegistry] Overwriting existing action: {action.name}")
        self._actions[action.name] = action

    def get(self, name: str) -> Optional[ActionDefinition]:
        """Get an action by name."""
        self._ensure_initialized()
        return self._actions.get(name)

    def get_all(self) -> List[ActionDefinition]:
        """Get all registered actions."""
        self._ensure_initialized()
        return list(self._actions.values())

    def get_by_category(self, category: str) -> List[ActionDefinition]:
        """Get actions filtered by category."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.category == category]

    def get_by_permission(self, level: str) -> List[ActionDefinition]:
        """Get actions filtered by permission level."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.permission_level == level]

    def to_openai_tools(self, permission_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert all actions to OpenAI function calling format.

        Args:
            permission_filter: Optional - only include actions with this permission level
        """
        self._ensure_initialized()
        actions = self._actions.values()
        if permission_filter:
            actions = [a for a in actions if a.permission_level == permission_filter]
        return [a.to_openai_schema() for a in actions]


def get_action_registry() -> ActionRegistry:
    """Get or create the global ActionRegistry singleton (thread-safe)."""
    global _registry_instance

    if _registry_instance is not None:
        return _registry_instance

    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = ActionRegistry()

    return _registry_instance
