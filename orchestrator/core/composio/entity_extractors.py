"""
Entity Extractors for Composio Tool Results (PRD-41: Context-Aware Suggestions)
================================================================================

Extracts meaningful entities from Composio tool execution results to enable
context-aware suggestions. Each tool type has a specialized extractor that knows
how to parse its specific API response format.

Entities are stored in Mem0 and used to generate contextual suggestions like:
- "Reply to Sarah's urgent email" (from Gmail results)
- "Send message to #general" (from Slack results)
- "Review PR #123" (from GitHub results)

Architecture:
    Base EntityExtractor abstract class defines the interface.
    Tool-specific extractors (GmailExtractor, SlackExtractor, etc.) implement
    the extraction logic for their respective API formats.

Usage:
    from core.composio.entity_extractors import get_extractor

    # After tool execution
    extractor = get_extractor("GMAIL")
    if extractor:
        entities = extractor.extract(tool_result)
        # entities = {"senders": ["Sarah"], "subjects": ["Deadline"], ...}
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EntityExtractor(ABC):
    """
    Abstract base class for entity extraction from tool results.

    Each tool type (Gmail, Slack, GitHub, etc.) implements this interface
    to extract relevant entities from its specific API response format.
    """

    @abstractmethod
    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract entities from a tool execution result.

        Args:
            tool_result: Raw response from tool execution

        Returns:
            Dictionary mapping entity types to lists of extracted values.
            Example: {"senders": ["Sarah"], "subjects": ["Meeting"], "email_ids": ["msg123"]}

        Note:
            Implementations should handle malformed/empty results gracefully
            and return empty lists for missing data rather than raising exceptions.
        """
        pass

    def _safe_get(self, data: Dict, *keys: str, default: Any = None) -> Any:
        """
        Safely navigate nested dictionary keys.

        Args:
            data: Dictionary to navigate
            *keys: Sequence of keys to traverse
            default: Value to return if any key is missing

        Returns:
            Value at the nested key path, or default if not found

        Example:
            _safe_get(data, "payload", "headers", 0, "value")
        """
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            elif isinstance(current, list) and isinstance(key, int):
                try:
                    current = current[key]
                except (IndexError, TypeError):
                    return default
            else:
                return default

            if current is None:
                return default

        return current if current is not None else default


def get_extractor(app_name: str) -> Optional[EntityExtractor]:
    """
    Factory function to get the appropriate extractor for an app.

    Args:
        app_name: Name of the Composio app (e.g., "GMAIL", "SLACK", "GITHUB")

    Returns:
        EntityExtractor instance for the app, or None if no extractor exists

    Usage:
        extractor = get_extractor("GMAIL")
        if extractor:
            entities = extractor.extract(result)
    """
    # Import here to avoid circular dependencies and allow lazy loading
    # Extractors will be added in subsequent user stories
    extractors = {
        # "GMAIL": GmailExtractor(),
        # "SLACK": SlackExtractor(),
        # "GITHUB": GitHubExtractor(),
    }

    normalized_app = app_name.upper() if app_name else None
    return extractors.get(normalized_app)
