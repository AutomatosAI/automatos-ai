"""
Context Manager for Tool Results (PRD-41: Context-Aware Suggestions)
====================================================================

Retrieves recent tool execution context from Redis to enable context-aware
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
    Retrieve recent tool execution results from Redis.

    Args:
        user_id: User/workspace identifier (format: ws_{workspace_id})
        tool_name: Optional filter for specific tool (e.g., "GMAIL", "SLACK")
        max_age_minutes: Only return results from last N minutes (default: 10)
        limit: Maximum number of results to return

    Returns:
        List of tool result contexts with entities and timestamps.
        Each item contains: {"tool": str, "action": str, "entities": dict, "timestamp": str}

    Example:
        >>> context = get_recent_tool_context("ws_workspace_123", tool_name="GMAIL")
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
    def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _within_max_age(context_data: Dict[str, Any]) -> bool:
        ts = _parse_timestamp(context_data.get("timestamp"))
        if ts is None:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        return ts >= cutoff

    try:
        from core.redis.client import get_redis_client

        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("[ContextManager] Redis client not available")
            return []

        redis_conn = redis_client.get_redis()
        try:
            # Redis key: tool_context:{workspace_id}:{tool_name}
            workspace_id = user_id.replace("ws_", "") if user_id.startswith("ws_") else user_id

            if tool_name:
                redis_key = f"tool_context:{workspace_id}:{tool_name}"
                logger.info(f"[ContextManager] Fetching from Redis: {redis_key}")

                data = redis_conn.get(redis_key)

                if data:
                    context_data = json.loads(data)
                    if _within_max_age(context_data):
                        logger.info(f"[ContextManager] ✅ Found tool context in Redis for {tool_name}")
                        return [context_data]
                    logger.info(f"[ContextManager] Tool context for {tool_name} exceeded max_age_minutes")
                else:
                    logger.info(f"[ContextManager] No tool context found in Redis for {tool_name}")
                return []
            else:
                pattern = f"tool_context:{workspace_id}:*"
                keys = list(redis_conn.scan_iter(match=pattern, count=100))

                results = []
                for key in keys[:limit]:
                    try:
                        data = redis_conn.get(key)
                        if data:
                            context_data = json.loads(data)
                            if _within_max_age(context_data):
                                results.append(context_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse Redis key {key}: {e}")
                        continue

                logger.info(f"[ContextManager] Found {len(results)} tool contexts in Redis")
                return results[:limit]
        finally:
            redis_conn.close()

    except Exception as e:
        logger.error(f"Failed to retrieve tool context from Redis: {e}")
        return []
