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
    admin_only: bool = False
    promoted: bool = False
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
        """Lazy-load platform + workspace actions on first access."""
        if not self._initialized:
            from .platform_actions import register_all_actions
            register_all_actions(self)  # includes workspace actions
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

    def get_promoted(self) -> List[ActionDefinition]:
        """Get all actions marked as promoted."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.promoted]

    def to_first_class_schemas(self, exclude_admin: bool = False) -> List[Dict[str, Any]]:
        """
        Return OpenAI function schemas for promoted actions.

        Promoted actions get their own first-class tool schemas instead of
        going through the platform_execute dispatcher.

        Args:
            exclude_admin: If True, admin_only promoted actions are excluded
                (non-admin callers won't get schemas for admin tools).
        """
        self._ensure_initialized()
        promoted = [a for a in self._actions.values() if a.promoted]
        if exclude_admin:
            promoted = [a for a in promoted if not a.admin_only]
        return [a.to_openai_schema() for a in promoted]

    def to_dispatcher_schema(self, exclude_admin: bool = False, exclude_promoted: bool = True) -> Dict[str, Any]:
        """
        Return a SINGLE OpenAI tool schema (platform_execute) that wraps
        all platform actions behind one dispatcher.

        The LLM learns available actions from the system prompt (markdown),
        not from the schema.  This keeps the tool payload small.

        Args:
            exclude_admin: If True, admin_only actions are excluded from the
                dispatcher (non-admin callers won't see them).
            exclude_promoted: If True (default), promoted actions are excluded
                from the dispatcher since they have first-class schemas.
        """
        self._ensure_initialized()

        # Build enum of valid action names for the dispatcher
        valid_actions = sorted(
            a.name for a in self._actions.values()
            if (not exclude_promoted or not a.promoted)
            and (not exclude_admin or not a.admin_only)
        )

        action_property: Dict[str, Any] = {
            "type": "string",
            "description": "The exact platform action name (e.g. 'platform_configure_agent_heartbeat')",
        }
        if valid_actions:
            action_property["enum"] = valid_actions

        return {
            "type": "function",
            "function": {
                "name": "platform_execute",
                "description": (
                    "Execute an internal Automatos platform action. "
                    "You MUST pass both 'action' and 'params'. "
                    "Example: platform_execute(action='platform_configure_agent_heartbeat', "
                    "params={'agent_id': 147, 'enabled': true, 'interval_minutes': 15}). "
                    "See the 'Available Platform Actions' section in your system prompt "
                    "for the full list of actions and their required parameters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": action_property,
                        "params": {
                            "type": "object",
                            "description": (
                                "Parameters for the action as a JSON object. "
                                "Always include required params from the action's definition. "
                                "Example: {'agent_id': 147, 'enabled': true, 'interval_minutes': 60}"
                            ),
                        },
                    },
                    "required": ["action", "params"],
                },
            },
        }

    def build_prompt_summary(self, exclude_admin: bool = False, exclude_promoted: bool = True) -> str:
        """
        Build a markdown summary of all platform actions for injection
        into the agent's system prompt.  Grouped by category.

        Args:
            exclude_admin: If True, admin_only actions are skipped from the
                summary (non-admin callers won't see them).
            exclude_promoted: If True (default), promoted actions are skipped
                since they have their own first-class schemas.
        """
        self._ensure_initialized()
        by_category: Dict[str, List[ActionDefinition]] = {}
        for action in self._actions.values():
            if exclude_admin and action.admin_only:
                continue
            if exclude_promoted and action.promoted:
                continue
            by_category.setdefault(action.category, []).append(action)

        lines = ["\n## Available Platform Actions\n"]
        lines.append(
            "Use `platform_execute(action, params)` to call these. "
            "The `action` field must be the exact action name.\n"
        )
        for category in sorted(by_category.keys()):
            lines.append(f"### {category.replace('_', ' ').title()}")
            for action in sorted(by_category[category], key=lambda a: a.name):
                # Extract required params from schema
                props = action.parameters.get("properties", {})
                required = action.parameters.get("required", [])
                param_hints = []
                for pname, pdef in props.items():
                    req_marker = " (required)" if pname in required else ""
                    param_hints.append(f"`{pname}`{req_marker}")
                param_str = f" — params: {', '.join(param_hints)}" if param_hints else ""
                lines.append(f"- `{action.name}`: {action.description}{param_str}")
            lines.append("")

        return "\n".join(lines)


def get_action_registry() -> ActionRegistry:
    """Get or create the global ActionRegistry singleton (thread-safe)."""
    global _registry_instance

    if _registry_instance is not None:
        return _registry_instance

    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = ActionRegistry()

    return _registry_instance
