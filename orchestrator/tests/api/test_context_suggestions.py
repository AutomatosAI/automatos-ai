"""
Unit Tests for Context-Aware Suggestions (PRD-41)
=================================================

Tests the generate_context_suggestions function.
"""

import pytest
from api.tools import generate_context_suggestions


class TestGenerateContextSuggestions:
    """Tests for context-aware suggestion generation"""

    def test_gmail_with_sender(self):
        """Test generating Gmail suggestions with sender entity"""
        entities = {
            "senders": ["Sarah Johnson"],
            "subjects": ["Deadline Update"],
            "labels": ["INBOX"],
            "email_ids": ["msg_123"]
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=2)

        assert len(suggestions) <= 2
        assert any("Sarah Johnson" in s for s in suggestions)

    def test_gmail_with_urgent_label(self):
        """Test generating urgent email suggestion"""
        entities = {
            "senders": ["John Doe"],
            "labels": ["IMPORTANT", "INBOX"],
            "subjects": ["Action Required"]
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=2)

        assert any("urgent" in s.lower() for s in suggestions)

    def test_gmail_with_long_sender_name(self):
        """Test truncation of long sender names"""
        entities = {
            "senders": ["A Very Long Name That Should Be Truncated For Display"],
            "subjects": []
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=2)

        # Should be truncated with ...
        assert any("..." in s for s in suggestions)

    def test_gmail_with_subject(self):
        """Test subject-based suggestion"""
        entities = {
            "senders": [],
            "subjects": ["Important Meeting Tomorrow"],
            "labels": []
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=3)

        assert any("Important Meeting" in s for s in suggestions)

    def test_slack_with_channel(self):
        """Test Slack channel suggestion"""
        entities = {
            "channels": ["#general", "#engineering"],
            "mentions": [],
            "message_ids": []
        }

        suggestions = generate_context_suggestions("SLACK", entities, limit=2)

        assert any("#general" in s for s in suggestions)

    def test_slack_with_mention(self):
        """Test Slack mention suggestion"""
        entities = {
            "channels": [],
            "mentions": ["@john", "@sarah"],
            "message_ids": []
        }

        suggestions = generate_context_suggestions("SLACK", entities, limit=2)

        assert any("@john" in s for s in suggestions)

    def test_github_with_pr(self):
        """Test GitHub PR suggestion"""
        entities = {
            "pr_numbers": [123, 456],
            "issue_numbers": [],
            "repos": [],
            "authors": []
        }

        suggestions = generate_context_suggestions("GITHUB", entities, limit=2)

        assert any("PR #123" in s for s in suggestions)

    def test_github_with_issue(self):
        """Test GitHub issue suggestion"""
        entities = {
            "pr_numbers": [],
            "issue_numbers": [789],
            "repos": ["org/myrepo"],
            "authors": []
        }

        suggestions = generate_context_suggestions("GITHUB", entities, limit=2)

        assert any("issue #789" in s for s in suggestions)

    def test_github_with_repo(self):
        """Test GitHub repo suggestion"""
        entities = {
            "pr_numbers": [],
            "issue_numbers": [],
            "repos": ["organization/repository-name"],
            "authors": ["johndoe"]
        }

        suggestions = generate_context_suggestions("GITHUB", entities, limit=3)

        assert any("organization/repository-name" in s for s in suggestions)

    def test_empty_entities(self):
        """Test with no entities"""
        entities = {
            "senders": [],
            "subjects": [],
            "labels": []
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=2)

        assert suggestions == []

    def test_limit_enforced(self):
        """Test that limit is respected"""
        entities = {
            "senders": ["Alice", "Bob"],
            "subjects": ["Subject 1", "Subject 2"],
            "labels": ["IMPORTANT"]
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=1)

        assert len(suggestions) == 1

    def test_unknown_tool(self):
        """Test with unsupported tool"""
        entities = {"some_field": ["value"]}

        suggestions = generate_context_suggestions("UNKNOWN_TOOL", entities, limit=2)

        # Should return empty for unsupported tools
        assert suggestions == []

    def test_long_subject_truncation(self):
        """Test truncation of long email subjects"""
        entities = {
            "subjects": ["This is a very long email subject line that needs to be truncated"],
            "senders": []
        }

        suggestions = generate_context_suggestions("GMAIL", entities, limit=2)

        # Should have truncated subject
        assert any("..." in s for s in suggestions)
