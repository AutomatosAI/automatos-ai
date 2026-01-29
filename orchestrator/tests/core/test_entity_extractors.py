"""
Unit Tests for Entity Extractors (PRD-41)
==========================================

Tests entity extraction from Composio tool results.
"""

import pytest
from core.composio.entity_extractors import (
    get_extractor,
    GmailExtractor,
    EntityExtractor
)


class TestGmailExtractor:
    """Tests for Gmail entity extraction"""

    def test_extract_with_multiple_messages(self):
        """Test extracting entities from Gmail API response with multiple messages"""
        sample_response = {
            "messages": [
                {
                    "id": "msg_123",
                    "labelIds": ["INBOX", "IMPORTANT"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Sarah Johnson <sarah@example.com>"},
                            {"name": "Subject", "value": "Urgent: Deadline Update"}
                        ]
                    }
                },
                {
                    "id": "msg_456",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "John Doe <john@example.com>"},
                            {"name": "Subject", "value": "Meeting Invite"}
                        ]
                    }
                }
            ]
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        assert "Sarah Johnson" in entities["senders"]
        assert "John Doe" in entities["senders"]
        assert "Urgent: Deadline Update" in entities["subjects"]
        assert "Meeting Invite" in entities["subjects"]
        assert "IMPORTANT" in entities["labels"]
        assert "INBOX" in entities["labels"]
        assert "msg_123" in entities["email_ids"]
        assert "msg_456" in entities["email_ids"]

    def test_extract_with_single_message(self):
        """Test extracting from single message format (not in array)"""
        sample_response = {
            "id": "msg_789",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "Subject", "value": "Quick Question"}
                ]
            }
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        assert "Alice" in entities["senders"]
        assert "Quick Question" in entities["subjects"]
        assert "msg_789" in entities["email_ids"]

    def test_extract_with_quoted_sender_name(self):
        """Test extracting sender with quoted name format"""
        sample_response = {
            "messages": [{
                "id": "msg_111",
                "labelIds": [],
                "payload": {
                    "headers": [
                        {"name": "From", "value": '"Bob Smith" <bob@example.com>'},
                        {"name": "Subject", "value": "Test"}
                    ]
                }
            }]
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        assert "Bob Smith" in entities["senders"]

    def test_extract_with_email_only_from(self):
        """Test extracting when From header has only email address"""
        sample_response = {
            "messages": [{
                "id": "msg_222",
                "labelIds": [],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "noreply@example.com"},
                        {"name": "Subject", "value": "Notification"}
                    ]
                }
            }]
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        assert "noreply@example.com" in entities["senders"]

    def test_extract_with_empty_messages(self):
        """Test extracting from empty message list"""
        sample_response = {"messages": []}

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        assert entities["senders"] == []
        assert entities["subjects"] == []
        assert entities["labels"] == []
        assert entities["email_ids"] == []

    def test_extract_with_malformed_message(self):
        """Test graceful handling of malformed message"""
        sample_response = {
            "messages": [
                {
                    "id": "msg_good",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Good <good@example.com>"},
                            {"name": "Subject", "value": "Good Email"}
                        ]
                    }
                },
                {
                    "id": "msg_bad",
                    # Missing payload
                },
                {
                    "id": "msg_also_good",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Also Good <also@example.com>"},
                            {"name": "Subject", "value": "Another Email"}
                        ]
                    }
                }
            ]
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        # Should successfully extract from good messages
        assert "Good" in entities["senders"]
        assert "Also Good" in entities["senders"]
        # Bad message should be skipped, not crash
        assert len(entities["email_ids"]) == 3  # Still records the ID

    def test_extract_with_missing_headers(self):
        """Test handling of message with no From/Subject headers"""
        sample_response = {
            "messages": [{
                "id": "msg_333",
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "Date", "value": "2025-01-29"}
                    ]
                }
            }]
        }

        extractor = GmailExtractor()
        entities = extractor.extract(sample_response)

        # Should not crash, just have empty sender/subject
        assert len(entities["senders"]) == 0
        assert len(entities["subjects"]) == 0
        assert "msg_333" in entities["email_ids"]


class TestGetExtractor:
    """Tests for get_extractor factory function"""

    def test_get_gmail_extractor(self):
        """Test getting Gmail extractor"""
        extractor = get_extractor("GMAIL")
        assert extractor is not None
        assert isinstance(extractor, GmailExtractor)

    def test_get_extractor_case_insensitive(self):
        """Test that app name is case-insensitive"""
        extractor = get_extractor("gmail")
        assert extractor is not None
        assert isinstance(extractor, GmailExtractor)

    def test_get_unknown_extractor(self):
        """Test getting extractor for unknown app"""
        extractor = get_extractor("UNKNOWN_APP")
        assert extractor is None

    def test_get_extractor_with_none(self):
        """Test getting extractor with None app name"""
        extractor = get_extractor(None)
        assert extractor is None
