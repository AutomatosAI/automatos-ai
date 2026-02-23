"""
Platform Action Discovery (PRD-64)
===================================

Provides the ActionRegistry and ActionDefinition system for platform self-awareness.
Auto can discover and execute platform operations (list agents, query analytics, etc.)
through the same tool pipeline as Composio and internal tools.
"""

from .action_registry import ActionDefinition, ActionRegistry, get_action_registry

__all__ = [
    "ActionDefinition",
    "ActionRegistry",
    "get_action_registry",
]
