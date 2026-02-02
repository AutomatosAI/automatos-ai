"""
Unit Tests for Entity Extractors (PRD-41)
==========================================

Tests entity extraction from Composio tool results.
"""

import pytest
from core.composio.entity_extractors import (
    get_extractor,
    GmailExtractor,
    SlackExtractor,
    GitHubExtractor,
    GenericExtractor,
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


class TestSlackExtractor:
    """Tests for Slack entity extraction"""

    def test_extract_with_multiple_messages(self):
        """Test extracting entities from Slack messages"""
        sample_response = {
            "messages": [
                {
                    "ts": "1234567890.123456",
                    "text": "Hey <@U123|john> can you check #general?",
                    "channel_name": "engineering"
                },
                {
                    "ts": "1234567891.123456",
                    "text": "Thanks <@U456|sarah>!",
                    "channel_name": "engineering"
                }
            ]
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert "#engineering" in entities["channels"]
        assert "#general" in entities["channels"]
        assert "@john" in entities["mentions"]
        assert "@sarah" in entities["mentions"]
        assert "1234567890.123456" in entities["message_ids"]
        assert "1234567891.123456" in entities["message_ids"]

    def test_extract_with_channels_array(self):
        """Test extracting from top-level channels array"""
        sample_response = {
            "channels": [
                {"id": "C123", "name": "general"},
                {"id": "C456", "name": "random"}
            ],
            "messages": []
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert "#general" in entities["channels"]
        assert "#random" in entities["channels"]

    def test_extract_mentions_with_user_ids(self):
        """Test extracting mentions with only user IDs (no username)"""
        sample_response = {
            "messages": [{
                "ts": "1234567892.123456",
                "text": "Hey <@U999> are you there?",
                "channel_name": "general"
            }]
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert "@U999" in entities["mentions"]

    def test_extract_plain_mentions(self):
        """Test extracting plain @mention format"""
        sample_response = {
            "messages": [{
                "ts": "1234567893.123456",
                "text": "Thanks @alice and @bob for the help!",
                "channel_name": "general"
            }]
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert "@alice" in entities["mentions"]
        assert "@bob" in entities["mentions"]

    def test_extract_with_empty_messages(self):
        """Test extracting from empty message list"""
        sample_response = {"messages": []}

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert entities["channels"] == []
        assert entities["mentions"] == []
        assert entities["message_ids"] == []

    def test_extract_single_message_format(self):
        """Test extracting from single message (not in array)"""
        sample_response = {
            "ts": "1234567894.123456",
            "text": "Check #announcements please",
            "channel_name": "general"
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        assert "#general" in entities["channels"]
        assert "#announcements" in entities["channels"]
        assert "1234567894.123456" in entities["message_ids"]

    def test_extract_duplicate_channels(self):
        """Test that duplicate channels are not added"""
        sample_response = {
            "messages": [
                {"ts": "1", "text": "Message in #general", "channel_name": "general"},
                {"ts": "2", "text": "Another in #general", "channel_name": "general"}
            ]
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        # Should only have one instance of #general
        assert entities["channels"].count("#general") == 1

    def test_extract_with_malformed_message(self):
        """Test graceful handling of malformed message"""
        sample_response = {
            "messages": [
                {"ts": "1", "text": "Good message", "channel_name": "general"},
                {"ts": "2"},  # Missing text
                {"ts": "3", "text": "Also good", "channel_name": "random"}
            ]
        }

        extractor = SlackExtractor()
        entities = extractor.extract(sample_response)

        # Should extract from good messages
        assert "#general" in entities["channels"]
        assert "#random" in entities["channels"]
        assert len(entities["message_ids"]) == 3


class TestGitHubExtractor:
    """Tests for GitHub entity extraction"""

    def test_extract_pull_requests(self):
        """Test extracting entities from GitHub pull requests"""
        sample_response = {
            "pull_requests": [
                {
                    "number": 123,
                    "title": "Fix authentication bug",
                    "user": {"login": "johndoe"},
                    "base": {"repo": {"full_name": "org/myrepo"}}
                },
                {
                    "number": 456,
                    "title": "Add new feature",
                    "user": {"login": "janedoe"},
                    "base": {"repo": {"full_name": "org/myrepo"}}
                }
            ]
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        assert 123 in entities["pr_numbers"]
        assert 456 in entities["pr_numbers"]
        assert "org/myrepo" in entities["repos"]
        assert "johndoe" in entities["authors"]
        assert "janedoe" in entities["authors"]

    def test_extract_issues(self):
        """Test extracting entities from GitHub issues"""
        sample_response = {
            "issues": [
                {
                    "number": 789,
                    "title": "Bug report",
                    "user": {"login": "alice"},
                    "repository": {"full_name": "org/another-repo"}
                }
            ]
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        assert 789 in entities["issue_numbers"]
        assert "org/another-repo" in entities["repos"]
        assert "alice" in entities["authors"]

    def test_extract_single_pr(self):
        """Test extracting from single PR format"""
        sample_response = {
            "number": 111,
            "title": "Fix typo",
            "user": {"login": "bob"},
            "head": {"repo": {"full_name": "org/repo"}},
            "base": {"repo": {"full_name": "org/repo"}}
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        assert 111 in entities["pr_numbers"]
        assert "org/repo" in entities["repos"]
        assert "bob" in entities["authors"]

    def test_extract_repositories_array(self):
        """Test extracting from top-level repositories array"""
        sample_response = {
            "repositories": [
                {"full_name": "org/repo1"},
                {"full_name": "org/repo2"}
            ]
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        assert "org/repo1" in entities["repos"]
        assert "org/repo2" in entities["repos"]

    def test_extract_duplicate_repos(self):
        """Test that duplicate repos are not added"""
        sample_response = {
            "pull_requests": [
                {
                    "number": 1,
                    "user": {"login": "user1"},
                    "base": {"repo": {"full_name": "org/repo"}}
                },
                {
                    "number": 2,
                    "user": {"login": "user2"},
                    "base": {"repo": {"full_name": "org/repo"}}
                }
            ]
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        # Should only have one instance of org/repo
        assert entities["repos"].count("org/repo") == 1

    def test_extract_with_empty_lists(self):
        """Test extracting from empty lists"""
        sample_response = {
            "pull_requests": [],
            "issues": []
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        assert entities["pr_numbers"] == []
        assert entities["issue_numbers"] == []
        assert entities["repos"] == []
        assert entities["authors"] == []

    def test_extract_with_malformed_pr(self):
        """Test graceful handling of malformed PR"""
        sample_response = {
            "pull_requests": [
                {
                    "number": 100,
                    "user": {"login": "good"},
                    "base": {"repo": {"full_name": "org/repo"}}
                },
                {
                    "number": 200
                    # Missing user and repo
                },
                {
                    "number": 300,
                    "user": {"login": "alsogood"},
                    "base": {"repo": {"full_name": "org/repo"}}
                }
            ]
        }

        extractor = GitHubExtractor()
        entities = extractor.extract(sample_response)

        # Should extract from all PRs, even malformed ones (partial data)
        assert 100 in entities["pr_numbers"]
        assert 200 in entities["pr_numbers"]
        assert 300 in entities["pr_numbers"]
        assert "good" in entities["authors"]
        assert "alsogood" in entities["authors"]


class TestGetExtractor:
    """Tests for get_extractor factory function"""

    def test_get_gmail_extractor(self):
        """Test getting Gmail extractor"""
        extractor = get_extractor("GMAIL")
        assert extractor is not None
        assert isinstance(extractor, GmailExtractor)

    def test_get_slack_extractor(self):
        """Test getting Slack extractor"""
        extractor = get_extractor("SLACK")
        assert extractor is not None
        assert isinstance(extractor, SlackExtractor)

    def test_get_github_extractor(self):
        """Test getting GitHub extractor"""
        extractor = get_extractor("GITHUB")
        assert extractor is not None
        assert isinstance(extractor, GitHubExtractor)

    def test_get_extractor_case_insensitive(self):
        """Test that app name is case-insensitive"""
        extractor = get_extractor("gmail")
        assert extractor is not None
        assert isinstance(extractor, GmailExtractor)

    def test_get_unknown_extractor(self):
        """Test getting extractor for unknown app (falls back to GenericExtractor)"""
        extractor = get_extractor("UNKNOWN_APP")
        assert extractor is not None
        assert isinstance(extractor, GenericExtractor)

    def test_get_extractor_with_none(self):
        """Test getting extractor with None app name (falls back to GenericExtractor)"""
        extractor = get_extractor(None)
        assert extractor is not None
        assert isinstance(extractor, GenericExtractor)
