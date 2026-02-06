"""Tests for core.services.plugin_context_service — the plugin context engine.

Tests cover:
- get_assigned_plugins (DB query)
- build_tier1_summary (markdown generation)
- build_tier2_content (async S3/Redis + manifest loading)
- _select_relevant_plugins (keyword scoring + fallback)
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import _make_plugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(mock_db):
    """Shorthand to create a PluginContextService."""
    from core.services.plugin_context_service import PluginContextService
    return PluginContextService(mock_db)


# ===========================================================================
# get_assigned_plugins
# ===========================================================================

class TestGetAssignedPlugins:
    def test_returns_empty_for_agent_with_no_plugins(self, mock_db):
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = []
        svc = _svc(mock_db)
        result = svc.get_assigned_plugins(agent_id=99)
        assert result == []

    def test_returns_plugin_tuples_ordered_by_priority(self, mock_db, three_plugins):
        sorted_rows = sorted(three_plugins, key=lambda r: r[0].priority)
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = sorted_rows
        svc = _svc(mock_db)
        result = svc.get_assigned_plugins(agent_id=42)
        assert len(result) == 3
        assert result[0][0].priority == 0
        assert result[1][0].priority == 1
        assert result[2][0].priority == 2

    def test_query_joins_and_filters_correctly(self, mock_db):
        svc = _svc(mock_db)
        svc.get_assigned_plugins(agent_id=42)
        mock_db.query.assert_called_once()
        chain = mock_db.query.return_value
        chain.join.assert_called_once()
        chain.join.return_value.filter.assert_called_once()
        chain.join.return_value.filter.return_value.order_by.assert_called_once()


# ===========================================================================
# build_tier1_summary
# ===========================================================================

class TestBuildTier1Summary:
    def test_empty_rows_returns_empty_string(self, mock_db):
        svc = _svc(mock_db)
        assert svc.build_tier1_summary([]) == ""

    def test_single_plugin_formats_correctly(self, mock_db, plugin_rows):
        svc = _svc(mock_db)
        result = svc.build_tier1_summary(plugin_rows)
        assert "Email Automation" in result
        assert "`email-automation`" in result
        assert "Send and manage emails" in result
        assert "email" in result
        assert "Skills: 5" in result
        assert "Commands: 3" in result
        assert "~1500 tokens" in result

    def test_multiple_plugins_all_included(self, mock_db, three_plugins):
        svc = _svc(mock_db)
        result = svc.build_tier1_summary(three_plugins)
        assert "Database Manager" in result
        assert "Email Sender" in result
        assert "Slack Bot" in result

    def test_missing_optional_fields_handled(self, mock_db):
        """Plugin with None tags, description, token_estimate should not crash."""
        aap, plugin = _make_plugin(
            "bare-plugin", "Bare Plugin",
            description=None, tags=None, token_estimate=None,
        )
        svc = _svc(mock_db)
        result = svc.build_tier1_summary([(aap, plugin)])
        assert "Bare Plugin" in result
        assert "None" not in result


# ===========================================================================
# build_tier2_content (async)
# ===========================================================================

class TestBuildTier2Content:
    @pytest.mark.asyncio
    async def test_empty_rows_returns_empty_string(self, mock_db):
        svc = _svc(mock_db)
        result = await svc.build_tier2_content([])
        assert result == ""

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_loads_skill_md_from_cache(self, mock_get_cache, mock_db, plugin_rows):
        cache = MagicMock()
        cache.get_file = AsyncMock(return_value="# Email Plugin\nSend emails with ease.")
        cache.get_manifest = AsyncMock(return_value=None)
        mock_get_cache.return_value = cache

        svc = _svc(mock_db)
        result = await svc.build_tier2_content(plugin_rows, max_plugins=5)
        assert "Email Plugin" in result
        assert "Send emails with ease" in result
        cache.get_file.assert_called_once_with("email-automation", "1.2.0", "SKILL.md")

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_loads_tool_schemas_from_manifest(self, mock_get_cache, mock_db, plugin_rows):
        cache = MagicMock()
        cache.get_file = AsyncMock(return_value=None)
        cache.get_manifest = AsyncMock(return_value={
            "tools": [
                {"name": "send_email", "description": "Send an email message"},
                {"name": "list_inbox", "description": "List inbox messages"},
            ]
        })
        mock_get_cache.return_value = cache

        svc = _svc(mock_db)
        result = await svc.build_tier2_content(plugin_rows, max_plugins=5)
        assert "**send_email**" in result
        assert "Send an email message" in result
        assert "**list_inbox**" in result

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_s3_failure_graceful(self, mock_get_cache, mock_db, plugin_rows):
        """get_file raising an exception should not crash — returns partial content."""
        cache = MagicMock()
        cache.get_file = AsyncMock(side_effect=Exception("S3 timeout"))
        cache.get_manifest = AsyncMock(return_value=None)
        mock_get_cache.return_value = cache

        svc = _svc(mock_db)
        result = await svc.build_tier2_content(plugin_rows, max_plugins=5)
        assert "email-automation" in result

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_max_plugins_limits_tier2_loading(self, mock_get_cache, mock_db):
        """With 5 plugins and max_plugins=2, only 2 get SKILL.md loaded."""
        rows = [
            _make_plugin(f"plugin-{i}", f"Plugin {i}", f"Description {i}", priority=i)
            for i in range(5)
        ]
        cache = MagicMock()
        cache.get_file = AsyncMock(return_value="skill content")
        cache.get_manifest = AsyncMock(return_value=None)
        mock_get_cache.return_value = cache

        svc = _svc(mock_db)
        await svc.build_tier2_content(rows, max_plugins=2)
        assert cache.get_file.call_count == 2


# ===========================================================================
# _select_relevant_plugins
# ===========================================================================

class TestSelectRelevantPlugins:
    def test_returns_all_when_fewer_than_max(self, mock_db, plugin_rows):
        svc = _svc(mock_db)
        result = svc._select_relevant_plugins(plugin_rows, task_context=None, max_plugins=5)
        assert len(result) == 1

    def test_no_context_uses_priority_order(self, mock_db, three_plugins):
        sorted_rows = sorted(three_plugins, key=lambda r: r[0].priority)
        svc = _svc(mock_db)
        result = svc._select_relevant_plugins(sorted_rows, task_context=None, max_plugins=2)
        assert len(result) == 2
        assert result[0][1].slug == "email-sender"
        assert result[1][1].slug == "slack-bot"

    def test_keyword_matching_scores_correctly(self, mock_db, three_plugins):
        sorted_rows = sorted(three_plugins, key=lambda r: r[0].priority)
        svc = _svc(mock_db)
        result = svc._select_relevant_plugins(
            sorted_rows, task_context="send email notification", max_plugins=1
        )
        assert len(result) == 1
        assert result[0][1].slug == "email-sender"

    def test_slug_and_name_contribute_to_score(self, mock_db):
        rows = [
            _make_plugin("data-export", "Data Export", "Export data to CSV", priority=0),
            _make_plugin("email-sender", "Email Sender", "Send emails", priority=1),
        ]
        svc = _svc(mock_db)
        result = svc._select_relevant_plugins(
            rows, task_context="I need to send an email", max_plugins=1
        )
        assert result[0][1].slug == "email-sender"

    def test_no_matches_falls_back_to_priority(self, mock_db, three_plugins):
        sorted_rows = sorted(three_plugins, key=lambda r: r[0].priority)
        svc = _svc(mock_db)
        result = svc._select_relevant_plugins(
            sorted_rows, task_context="quantum physics research", max_plugins=2
        )
        assert len(result) == 2
        assert result[0][1].slug == "email-sender"
        assert result[1][1].slug == "slack-bot"
