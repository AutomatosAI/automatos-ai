"""
Context Manager for Tool Results (PRD-41: Context-Aware Suggestions)
====================================================================

Retrieves recent tool execution context from Mem0 to enable context-aware
suggestions. When a user clicks a tool icon, the system can fetch what that
tool recently showed them and generate relevant suggestions.

Example:
    User runs Gmail tool → sees "urgent email from Sarah"
    User clicks Gmail icon → system retrieves context → suggests "Reply to Sarah's urgent email"

Usage:
    from modules.memory.context_manager import get_recent_tool_context

    context = get_recent_tool_context(
        user_id=workspace_id,
        tool_name="GMAIL",
        max_age_minutes=10
    )
    # Returns: [{"tool": "GMAIL", "action": "...", "entities": {...}, "timestamp": "..."}]
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json

logger = logging.getLogger(__name__)


def get_recent_tool_context(
    user_id: str,
    tool_name: Optional[str] = None,
    max_age_minutes: int = 10,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve recent tool execution results from Mem0.

    Args:
        user_id: User/workspace identifier (workspace_id in our case)
        tool_name: Optional filter for specific tool (e.g., "GMAIL", "SLACK")
        max_age_minutes: Only return results from last N minutes (default: 10)
        limit: Maximum number of results to return

    Returns:
        List of tool result contexts with entities and timestamps.
        Each item contains: {"tool": str, "action": str, "entities": dict, "timestamp": str}

    Example:
        >>> context = get_recent_tool_context("workspace_123", tool_name="GMAIL")
        >>> context[0]
        {
            "tool": "GMAIL",
            "action": "GMAIL_GET_EMAILS",
            "entities": {
                "senders": ["Sarah Johnson"],
                "subjects": ["Urgent: Deadline"],
                "labels": ["IMPORTANT"],
                "email_ids": ["msg_123"]
            },
            "timestamp": "2026-01-29T10:30:00Z"
        }
    """
    try:
        from modules.memory.integrations.mem0_client import Mem0Client

        mem0_client = Mem0Client()

        # Search for tool_result memories
        # Note: Mem0 search doesn't have perfect filtering, so we filter client-side
        query = f"tool_result {tool_name}" if tool_name else "tool_result"

        memories = mem0_client.search(
            query=query,
            user_id=user_id,
            limit=50  # Fetch more to allow for filtering
        )

        if not memories:
            logger.debug(f"No tool context found in Mem0 for user {user_id}")
            return []

        # Filter and parse memories
        results = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

        for memory in memories:
            try:
                metadata = memory.get("metadata", {})

                # Filter by type
                if metadata.get("type") != "tool_result":
                    continue

                # Filter by tool name if specified
                memory_tool = metadata.get("tool", "").upper()
                if tool_name and memory_tool != tool_name.upper():
                    continue

                # Filter by age
                created_at = memory.get("created_at")
                if created_at:
                    # Parse timestamp (could be string or timestamp)
                    if isinstance(created_at, str):
                        try:
                            memory_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        except:
                            # Try parsing as timestamp
                            memory_time = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
                    else:
                        memory_time = datetime.fromtimestamp(created_at, tz=timezone.utc)

                    if memory_time < cutoff_time:
                        continue  # Too old

                # Parse entities from metadata
                entities_json = metadata.get("entities_json", "{}")
                try:
                    entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
                except:
                    entities = {}

                # Build context object
                context = {
                    "tool": memory_tool,
                    "action": metadata.get("action", ""),
                    "entities": entities,
                    "timestamp": created_at,
                    "agent_id": metadata.get("agent_id")
                }

                results.append(context)

                if len(results) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to parse tool context from Mem0 memory: {e}")
                continue

        logger.info(f"Retrieved {len(results)} tool contexts from Mem0 for {tool_name or 'all tools'}")
        return results

    except Exception as e:
        logger.error(f"Failed to retrieve tool context from Mem0: {e}")
        # Return empty list on failure (graceful degradation)
        return []


def clear_old_tool_context(
    user_id: str,
    max_age_hours: int = 24
) -> int:
    """
    Clear tool context older than specified hours.

    This is a maintenance function to prevent Mem0 from filling up
    with old tool results. Can be called periodically.

    Args:
        user_id: User/workspace identifier
        max_age_hours: Clear contexts older than this many hours

    Returns:
        Number of contexts cleared

    Note:
        Currently Mem0 doesn't support deletion by query, so this is a placeholder.
        In production, you'd need to implement batch deletion via Mem0 API.
    """
    # TODO: Implement when Mem0 supports deletion API
    logger.info(f"Context cleanup requested for user {user_id} (not yet implemented)")
    return 0
