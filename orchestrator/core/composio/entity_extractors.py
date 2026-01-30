"""
Entity Extractors for Composio Tool Results (PRD-41: Context-Aware Suggestions)
================================================================================

Extracts meaningful entities from Composio tool execution results to enable
context-aware suggestions. Works with ALL 850+ tools automatically.

Entities are stored in Redis (10-minute TTL) and used to generate contextual suggestions:
- "Reply to Sarah's urgent email" (from Gmail results)
- "Send message to #general" (from Slack results)
- "Review PR #123" (from GitHub results)
- "View details for '[item name]'" (from ANY tool via GenericExtractor)

Architecture:
    - Base EntityExtractor abstract class defines the interface
    - Specialized extractors (Gmail, Slack, GitHub) for optimized precision
    - GenericExtractor handles ALL other tools automatically (no hardcoding needed)

Usage:
    from core.composio.entity_extractors import get_extractor

    # After tool execution (works with ANY tool)
    extractor = get_extractor("GMAIL")   # Returns GmailExtractor
    extractor = get_extractor("LINEAR")  # Returns GenericExtractor
    extractor = get_extractor("JIRA")    # Returns GenericExtractor
    entities = extractor.extract(tool_result)
    # entities = {"names": [...], "ids": [...], "emails": [...], ...}
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

        # Unwrap Composio's nested "data" structure if present
        data = tool_result
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Handle both "messages" array and single message format
        messages = data.get("messages", [])
        if not messages and "id" in data:
            # Single message format
            messages = [data]

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


class GenericExtractor(EntityExtractor):
    """
    Universal extractor that works with ANY tool using heuristics.

    This extractor scans tool responses for common patterns and extracts:
    - names: Any field containing name/title/subject
    - ids: Any field containing id/identifier
    - emails: Email addresses found anywhere
    - urls: URLs found in responses
    - key_values: Other important-looking fields

    Works with 850+ tools without hardcoding anything.
    """

    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities using generic heuristics."""
        entities = {
            "names": [],      # Names, titles, subjects
            "ids": [],        # IDs, identifiers, references
            "emails": [],     # Email addresses
            "urls": [],       # URLs
            "key_values": []  # Other important values
        }

        # Unwrap Composio's nested "data" structure if present
        data = tool_result
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Recursively scan the response
        self._scan_dict(data, entities)

        # Deduplicate all lists
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # Limit to 10 per type

        return entities

    def _scan_dict(self, data: Any, entities: Dict[str, List[str]], depth: int = 0) -> None:
        """
        Recursively scan dictionary/list for extractable entities.

        Args:
            data: Data structure to scan
            entities: Dictionary to populate with extracted entities
            depth: Current recursion depth (limit to 5 levels)
        """
        if depth > 5:  # Prevent infinite recursion
            return

        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()

                # Extract based on key names (field names tell us what the data is)
                if isinstance(value, str) and value:
                    # Names/Titles/Subjects
                    if any(pattern in key_lower for pattern in ['name', 'title', 'subject', 'label', 'summary']):
                        if len(value) < 200:  # Skip long text
                            entities["names"].append(value)

                    # IDs/References
                    elif any(pattern in key_lower for pattern in ['id', 'identifier', 'reference', 'number', 'key']):
                        if len(value) < 100:  # IDs are usually short
                            entities["ids"].append(str(value))

                    # Emails (extract from any field)
                    if '@' in value and '.' in value:
                        email = self._extract_email(value)
                        if email:
                            entities["emails"].append(email)

                    # URLs (extract from any field)
                    if value.startswith(('http://', 'https://')):
                        entities["urls"].append(value[:200])  # Truncate long URLs

                    # Generic important-looking short strings
                    elif len(value) < 100 and not key_lower.startswith('_'):
                        # Skip internal/meta fields (those starting with _)
                        if any(pattern in key_lower for pattern in ['status', 'type', 'category', 'priority', 'state']):
                            entities["key_values"].append(f"{key}: {value}")

                # Recurse into nested structures
                elif isinstance(value, (dict, list)):
                    self._scan_dict(value, entities, depth + 1)

                # Extract numeric values for important fields
                elif isinstance(value, (int, float)) and key_lower in ['number', 'count', 'total', 'id']:
                    entities["ids"].append(str(value))

        elif isinstance(data, list):
            # Process first few items in arrays (don't scan entire 1000-item lists)
            for item in data[:20]:  # Limit to first 20 items
                if isinstance(item, (dict, list)):
                    self._scan_dict(item, entities, depth + 1)

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text."""
        import re
        # Simple email regex
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return match.group(0) if match else None


class GitHubExtractor(EntityExtractor):
    """
    Extracts entities from GitHub API responses.

    Entities extracted:
    - pr_numbers: Pull request numbers
    - issue_numbers: Issue numbers
    - repos: Repository names (e.g., "owner/repo")
    - authors: Usernames of PR/issue authors

    GitHub API Response Format:
        {
            "pull_requests": [
                {
                    "number": 123,
                    "title": "Fix bug",
                    "user": {"login": "johndoe"},
                    "base": {"repo": {"full_name": "org/repo"}}
                }
            ],
            "issues": [
                {
                    "number": 456,
                    "title": "Feature request",
                    "user": {"login": "jane"}
                }
            ]
        }
    """

    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from GitHub API response."""
        entities = {
            "pr_numbers": [],
            "issue_numbers": [],
            "repos": [],
            "authors": []
        }

        # Extract from pull requests
        prs = tool_result.get("pull_requests", [])
        if not prs and "number" in tool_result and ("pull_request" in tool_result or "head" in tool_result):
            # Single PR format
            prs = [tool_result]

        for pr in prs:
            try:
                # Extract PR number
                pr_num = pr.get("number")
                if pr_num:
                    entities["pr_numbers"].append(pr_num)

                # Extract repo name
                repo = self._safe_get(pr, "base", "repo", "full_name") or self._safe_get(pr, "repository", "full_name")
                if repo and repo not in entities["repos"]:
                    entities["repos"].append(repo)

                # Extract author
                author = self._safe_get(pr, "user", "login")
                if author and author not in entities["authors"]:
                    entities["authors"].append(author)

            except Exception as e:
                logger.warning(f"Failed to extract entities from GitHub PR: {e}")
                continue

        # Extract from issues
        issues = tool_result.get("issues", [])
        if not issues and "number" in tool_result and "pull_request" not in tool_result:
            # Single issue format
            issues = [tool_result]

        for issue in issues:
            try:
                # Extract issue number
                issue_num = issue.get("number")
                if issue_num:
                    entities["issue_numbers"].append(issue_num)

                # Extract repo name
                repo = self._safe_get(issue, "repository", "full_name") or self._safe_get(issue, "repo", "full_name")
                if repo and repo not in entities["repos"]:
                    entities["repos"].append(repo)

                # Extract author
                author = self._safe_get(issue, "user", "login")
                if author and author not in entities["authors"]:
                    entities["authors"].append(author)

            except Exception as e:
                logger.warning(f"Failed to extract entities from GitHub issue: {e}")
                continue

        # Handle top-level repositories array
        repos_list = tool_result.get("repositories", [])
        for repo in repos_list:
            try:
                repo_name = repo.get("full_name")
                if repo_name and repo_name not in entities["repos"]:
                    entities["repos"].append(repo_name)
            except Exception as e:
                logger.warning(f"Failed to extract repo from list: {e}")
                continue

        return entities


def get_extractor(app_name: str) -> EntityExtractor:
    """
    Factory function to get the appropriate extractor for an app.

    This function is fully agnostic - it works with ALL 850+ tools.
    Specialized extractors (Gmail, Slack, GitHub) provide better precision,
    but GenericExtractor handles everything else automatically.

    Args:
        app_name: Name of the Composio app (e.g., "GMAIL", "SLACK", "LINEAR", "JIRA", etc.)

    Returns:
        EntityExtractor instance - ALWAYS returns an extractor (never None)
        - Specialized extractor if one exists for this tool
        - GenericExtractor as fallback for all other tools

    Usage:
        extractor = get_extractor("GMAIL")  # Returns GmailExtractor
        extractor = get_extractor("LINEAR")  # Returns GenericExtractor
        extractor = get_extractor("JIRA")    # Returns GenericExtractor
        entities = extractor.extract(result)
    """
    # Specialized extractors for tools we've optimized
    specialized_extractors = {
        "GMAIL": GmailExtractor(),
        "SLACK": SlackExtractor(),
        "GITHUB": GitHubExtractor(),
    }

    normalized_app = app_name.upper() if app_name else ""

    # Return specialized extractor if available, otherwise use GenericExtractor
    extractor = specialized_extractors.get(normalized_app)
    if extractor:
        logger.debug(f"Using specialized {extractor.__class__.__name__} for {app_name}")
        return extractor
    else:
        logger.debug(f"Using GenericExtractor for {app_name} (agnostic mode)")
        return GenericExtractor()
