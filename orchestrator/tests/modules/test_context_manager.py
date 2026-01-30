"""
Unit Tests for Context Manager (PRD-41)
========================================

Tests retrieval of tool context from Redis.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
import json


def _make_context_payload(tool="GMAIL", action="GMAIL_GET_EMAILS", entities=None, minutes_ago=0):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return {
        "tool": tool,
        "action": action,
        "entities": entities or {},
        "timestamp": ts,
    }


class TestGetRecentToolContext:
    """Tests for get_recent_tool_context function"""

    @patch('core.redis.client.get_redis_client')
    def test_retrieve_gmail_context(self, mock_get_redis):
        """Test retrieving Gmail context from Redis"""
        from modules.memory.context_manager import get_recent_tool_context

        payload = _make_context_payload(
            entities={"senders": ["Sarah"], "subjects": ["Urgent"], "labels": ["IMPORTANT"], "email_ids": ["msg_123"]}
        )
        mock_conn = Mock()
        mock_conn.get.return_value = json.dumps(payload)
        mock_conn.scan_iter.return_value = iter([])
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL",
            max_age_minutes=10
        )

        assert len(context) == 1
        assert context[0]["tool"] == "GMAIL"
        assert context[0]["action"] == "GMAIL_GET_EMAILS"
        assert "Sarah" in context[0]["entities"]["senders"]
        assert context[0]["entities"]["labels"] == ["IMPORTANT"]
        mock_conn.close.assert_called_once()

    @patch('core.redis.client.get_redis_client')
    def test_filter_by_tool_name(self, mock_get_redis):
        """Test filtering contexts by tool name"""
        from modules.memory.context_manager import get_recent_tool_context

        payload = _make_context_payload(tool="GMAIL", action="GET_EMAILS", entities={})
        mock_conn = Mock()
        mock_conn.get.return_value = json.dumps(payload)
        mock_conn.scan_iter.return_value = iter([])
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        assert len(context) == 1
        assert context[0]["tool"] == "GMAIL"
        mock_conn.close.assert_called_once()

    @patch('core.redis.client.get_redis_client')
    def test_filter_by_age(self, mock_get_redis):
        """Test filtering out old contexts"""
        from modules.memory.context_manager import get_recent_tool_context

        recent_payload = _make_context_payload(entities={"senders": ["Alice"]}, minutes_ago=0)
        old_payload = _make_context_payload(entities={"senders": ["Bob"]}, minutes_ago=15)
        mock_conn = Mock()
        mock_conn.scan_iter.return_value = iter([
            b"tool_context:workspace_123:GMAIL",
            b"tool_context:workspace_123:GMAIL_OLD",
        ])
        call_count = [0]
        def get_side_effect(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(recent_payload)
            return json.dumps(old_payload)
        mock_conn.get.side_effect = get_side_effect
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            max_age_minutes=10
        )

        assert len(context) == 1
        assert context[0]["entities"]["senders"] == ["Alice"]
        mock_conn.close.assert_called_once()

    @patch('core.redis.client.get_redis_client')
    def test_empty_results(self, mock_get_redis):
        """Test handling empty Redis results"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_conn = Mock()
        mock_conn.get.return_value = None
        mock_conn.scan_iter.return_value = iter([])
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        assert context == []
        mock_conn.close.assert_called_once()

    @patch('core.redis.client.get_redis_client')
    def test_redis_failure_graceful(self, mock_get_redis):
        """Test graceful handling of Redis failure"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_client.get_redis.side_effect = Exception("Redis connection failed")
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        assert context == []

    @patch('core.redis.client.get_redis_client')
    def test_malformed_entry_skipped(self, mock_get_redis):
        """Test that malformed Redis entries are skipped in scan branch"""
        from modules.memory.context_manager import get_recent_tool_context

        good_payload = _make_context_payload(entities={"senders": ["Alice"]})
        mock_conn = Mock()
        mock_conn.scan_iter.return_value = iter([
            b"tool_context:workspace_123:GMAIL",
            b"tool_context:workspace_123:GMAIL_2",
        ])
        call_count = [0]
        def get_side_effect(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(good_payload)
            raise ValueError("Invalid JSON")
        mock_conn.get.side_effect = get_side_effect
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            max_age_minutes=10
        )

        assert len(context) == 1
        assert "Alice" in context[0]["entities"]["senders"]
        mock_conn.close.assert_called_once()

    @patch('core.redis.client.get_redis_client')
    def test_limit_results(self, mock_get_redis):
        """Test limiting number of results returned"""
        from modules.memory.context_manager import get_recent_tool_context

        keys = [f"tool_context:workspace_123:GMAIL_{i}".encode() for i in range(10)]
        mock_conn = Mock()
        mock_conn.scan_iter.return_value = iter(keys)
        mock_conn.get.side_effect = lambda k: json.dumps(
            _make_context_payload(entities={"senders": ["User"]})
        )
        mock_client = Mock()
        mock_client.get_redis.return_value = mock_conn
        mock_get_redis.return_value = mock_client

        context = get_recent_tool_context(
            user_id="workspace_123",
            limit=3
        )

        assert len(context) == 3
        mock_conn.close.assert_called_once()
