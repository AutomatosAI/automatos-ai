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


class GmailExtractor(EntityExtractor):
    """
    Extracts entities from Gmail API responses.

    Entities extracted:
    - senders: Email sender names (e.g., "Sarah Johnson")
    - subjects: Email subject lines
    - labels: Gmail labels (IMPORTANT, INBOX, etc.)
    - email_ids: Message IDs for reference

    Gmail API Response Format:
        {
            "messages": [
                {
                    "id": "msg_123",
                    "labelIds": ["INBOX", "IMPORTANT"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Sarah <sarah@example.com>"},
                            {"name": "Subject", "value": "Urgent: Deadline"}
                        ]
                    }
                }
            ]
        }
    """

    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from Gmail API response."""
        entities = {
            "senders": [],
            "subjects": [],
            "labels": [],
            "email_ids": []
        }

        # Handle both "messages" array and single message format
        messages = tool_result.get("messages", [])
        if not messages and "id" in tool_result:
            # Single message format
            messages = [tool_result]

        for msg in messages:
            try:
                # Extract sender from headers
                sender = self._extract_sender(msg)
                if sender:
                    entities["senders"].append(sender)

                # Extract subject
                subject = self._extract_subject(msg)
                if subject:
                    entities["subjects"].append(subject)

                # Extract labels
                labels = msg.get("labelIds", [])
                if labels:
                    entities["labels"].extend(labels)

                # Store email ID
                email_id = msg.get("id")
                if email_id:
                    entities["email_ids"].append(email_id)

            except Exception as e:
                logger.warning(f"Failed to extract entities from Gmail message: {e}")
                continue

        return entities

    def _extract_sender(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Extract sender name from email headers.

        Args:
            message: Gmail message object

        Returns:
            Sender name (e.g., "Sarah Johnson") or None
        """
        headers = self._safe_get(message, "payload", "headers", default=[])
        if not isinstance(headers, list):
            return None

        for header in headers:
            if header.get("name") == "From":
                from_value = header.get("value", "")
                # Extract name from "Sarah Johnson <sarah@example.com>" format
                if "<" in from_value:
                    name = from_value.split("<")[0].strip()
                    # Remove quotes if present
                    name = name.strip('"').strip("'")
                    return name if name else from_value
                return from_value

        return None

    def _extract_subject(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Extract subject from email headers.

        Args:
            message: Gmail message object

        Returns:
            Email subject line or None
        """
        headers = self._safe_get(message, "payload", "headers", default=[])
        if not isinstance(headers, list):
            return None

        for header in headers:
            if header.get("name") == "Subject":
                return header.get("value")

        return None


class SlackExtractor(EntityExtractor):
    """
    Extracts entities from Slack API responses.

    Entities extracted:
    - channels: Channel names (e.g., "#general", "#engineering")
    - mentions: User mentions (e.g., "@john", "@sarah")
    - message_ids: Message timestamps/IDs for reference

    Slack API Response Format:
        {
            "messages": [
                {
                    "ts": "1234567890.123456",
                    "text": "Hey <@U123> can you check #general?",
                    "channel": "C456",
                    "channel_name": "engineering",
                    "user": "U789"
                }
            ],
            "channels": [
                {"id": "C456", "name": "engineering"}
            ]
        }
    """

    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from Slack API response."""
        entities = {
            "channels": [],
            "mentions": [],
            "message_ids": []
        }

        # Handle messages array
        messages = tool_result.get("messages", [])
        if not messages and "ts" in tool_result:
            # Single message format
            messages = [tool_result]

        for msg in messages:
            try:
                # Extract channel name
                channel_name = msg.get("channel_name") or self._extract_channel_from_text(msg.get("text", ""))
                if channel_name:
                    # Add # prefix if not present
                    if not channel_name.startswith("#"):
                        channel_name = f"#{channel_name}"
                    if channel_name not in entities["channels"]:
                        entities["channels"].append(channel_name)

                # Extract mentions from text
                mentions = self._extract_mentions(msg.get("text", ""))
                for mention in mentions:
                    if mention not in entities["mentions"]:
                        entities["mentions"].append(mention)

                # Store message timestamp as ID
                ts = msg.get("ts")
                if ts:
                    entities["message_ids"].append(ts)

            except Exception as e:
                logger.warning(f"Failed to extract entities from Slack message: {e}")
                continue

        # Also check top-level channels array
        channels_list = tool_result.get("channels", [])
        for channel in channels_list:
            try:
                name = channel.get("name")
                if name:
                    channel_name = f"#{name}" if not name.startswith("#") else name
                    if channel_name not in entities["channels"]:
                        entities["channels"].append(channel_name)
            except Exception as e:
                logger.warning(f"Failed to extract channel from list: {e}")
                continue

        return entities

    def _extract_channel_from_text(self, text: str) -> Optional[str]:
        """Extract channel reference from message text (#channel)."""
        import re
        # Match #channel pattern
        match = re.search(r'#([a-z0-9_-]+)', text, re.IGNORECASE)
        if match:
            return f"#{match.group(1)}"
        return None

    def _extract_mentions(self, text: str) -> List[str]:
        """
        Extract user mentions from Slack message text.

        Slack mentions format: <@U123> or <@U123|username>

        Args:
            text: Slack message text

        Returns:
            List of mentions as @username or @U123
        """
        import re
        mentions = []

        # Match <@U123> or <@U123|username> pattern
        pattern = r'<@([A-Z0-9]+)(?:\|([a-z0-9._-]+))?>'
        matches = re.findall(pattern, text, re.IGNORECASE)

        for user_id, username in matches:
            # Prefer username if available, otherwise use user ID
            mention = f"@{username}" if username else f"@{user_id}"
            mentions.append(mention)

        # Also match plain @mentions (less common but possible)
        plain_mentions = re.findall(r'(?:^|\s)@([a-z0-9._-]+)', text, re.IGNORECASE)
        for username in plain_mentions:
            mention = f"@{username}"
            if mention not in mentions:
                mentions.append(mention)

        return mentions


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
    extractors = {
        "GMAIL": GmailExtractor(),
        "SLACK": SlackExtractor(),
        # "GITHUB": GitHubExtractor(),
    }

    normalized_app = app_name.upper() if app_name else None
    return extractors.get(normalized_app)
