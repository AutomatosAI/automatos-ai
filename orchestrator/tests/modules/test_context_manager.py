"""
Unit Tests for Context Manager (PRD-41)
========================================

Tests retrieval of tool context from Mem0.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
import json


class TestGetRecentToolContext:
    """Tests for get_recent_tool_context function"""

    @patch('modules.memory.context_manager.Mem0Client')
    def test_retrieve_gmail_context(self, mock_mem0_class):
        """Test retrieving Gmail context from Mem0"""
        from modules.memory.context_manager import get_recent_tool_context

        # Mock Mem0 response
        mock_client = Mock()
        mock_mem0_class.return_value = mock_client

        mock_client.search.return_value = [
            {
                "id": "mem_1",
                "memory": "GMAIL GMAIL_GET_EMAILS: emails from Sarah",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GMAIL_GET_EMAILS",
                    "agent_id": 123,
                    "entities_json": json.dumps({
                        "senders": ["Sarah"],
                        "subjects": ["Urgent"],
                        "labels": ["IMPORTANT"],
                        "email_ids": ["msg_123"]
                    })
                }
            }
        ]

        # Call function
        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL",
            max_age_minutes=10
        )

        # Verify
        assert len(context) == 1
        assert context[0]["tool"] == "GMAIL"
        assert context[0]["action"] == "GMAIL_GET_EMAILS"
        assert "Sarah" in context[0]["entities"]["senders"]
        assert context[0]["entities"]["labels"] == ["IMPORTANT"]

    @patch('modules.memory.context_manager.Mem0Client')
    def test_filter_by_tool_name(self, mock_mem0_class):
        """Test filtering contexts by tool name"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client

        # Return multiple tool results
        mock_client.search.return_value = [
            {
                "id": "mem_1",
                "memory": "GMAIL context",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": "{}"
                }
            },
            {
                "id": "mem_2",
                "memory": "SLACK context",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "SLACK",
                    "action": "SEND_MESSAGE",
                    "entities_json": "{}"
                }
            }
        ]

        # Request only GMAIL
        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        # Should only return GMAIL context
        assert len(context) == 1
        assert context[0]["tool"] == "GMAIL"

    @patch('modules.memory.context_manager.Mem0Client')
    def test_filter_by_age(self, mock_mem0_class):
        """Test filtering out old contexts"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client

        # One recent, one old
        recent_time = datetime.now(timezone.utc)
        old_time = recent_time - timedelta(minutes=15)

        mock_client.search.return_value = [
            {
                "id": "mem_recent",
                "memory": "Recent context",
                "created_at": recent_time.isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": "{}"
                }
            },
            {
                "id": "mem_old",
                "memory": "Old context",
                "created_at": old_time.isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": "{}"
                }
            }
        ]

        # Request with 10 minute age limit
        context = get_recent_tool_context(
            user_id="workspace_123",
            max_age_minutes=10
        )

        # Should only return recent context
        assert len(context) == 1

    @patch('modules.memory.context_manager.Mem0Client')
    def test_empty_results(self, mock_mem0_class):
        """Test handling empty Mem0 results"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client
        mock_client.search.return_value = []

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        assert context == []

    @patch('modules.memory.context_manager.Mem0Client')
    def test_mem0_failure_graceful(self, mock_mem0_class):
        """Test graceful handling of Mem0 failure"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client
        mock_client.search.side_effect = Exception("Mem0 connection failed")

        # Should return empty list, not crash
        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        assert context == []

    @patch('modules.memory.context_manager.Mem0Client')
    def test_malformed_memory_skipped(self, mock_mem0_class):
        """Test that malformed memories are skipped"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client

        mock_client.search.return_value = [
            {
                "id": "mem_good",
                "memory": "Good memory",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": json.dumps({"senders": ["Alice"]})
                }
            },
            {
                "id": "mem_bad",
                "memory": "Bad memory",
                # Missing created_at and metadata
            },
            {
                "id": "mem_also_good",
                "memory": "Another good memory",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": json.dumps({"senders": ["Bob"]})
                }
            }
        ]

        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL"
        )

        # Should get 2 good contexts, skip bad one
        assert len(context) == 2
        assert "Alice" in context[0]["entities"]["senders"]
        assert "Bob" in context[1]["entities"]["senders"]

    @patch('modules.memory.context_manager.Mem0Client')
    def test_limit_results(self, mock_mem0_class):
        """Test limiting number of results returned"""
        from modules.memory.context_manager import get_recent_tool_context

        mock_client = Mock()
        mock_mem0_class.return_value = mock_client

        # Return many results
        mock_client.search.return_value = [
            {
                "id": f"mem_{i}",
                "memory": f"Memory {i}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "type": "tool_result",
                    "tool": "GMAIL",
                    "action": "GET_EMAILS",
                    "entities_json": "{}"
                }
            }
            for i in range(10)
        ]

        # Request limit of 3
        context = get_recent_tool_context(
            user_id="workspace_123",
            tool_name="GMAIL",
            limit=3
        )

        # Should only return 3
        assert len(context) == 3
