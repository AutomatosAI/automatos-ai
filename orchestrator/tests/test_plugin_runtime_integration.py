"""Integration-style tests for the full plugin runtime chain.

Still uses mocked DB/S3/Redis but tests the interactions between:
- PluginContextService (tier 1 + tier 2)
- PluginContentCache (Redis + S3 fallback)
- System prompt assembly
- Skills-skip-when-plugins-present logic
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import _make_plugin


# ===========================================================================
# Plugins skip skills
# ===========================================================================

class TestPluginsSkipSkills:
    """Verify that agents with plugins skip legacy skill loading."""

    def test_agent_with_plugins_skips_skill_loading(self, mock_db, plugin_rows, mock_agent):
        """When an agent has assigned plugins, skills should NOT appear in prompt."""
        from core.services.plugin_context_service import PluginContextService

        mock_agent.assigned_plugins = [plugin_rows[0][0]]
        mock_agent.skills = [MagicMock(name="legacy_skill")]

        svc = PluginContextService(mock_db)
        tier1 = svc.build_tier1_summary(plugin_rows)

        assert "Email Automation" in tier1
        assert "legacy_skill" not in tier1

    def test_agent_without_plugins_loads_skills(self, mock_db, mock_agent):
        """When no plugins are assigned, tier1 returns empty (caller loads skills)."""
        from core.services.plugin_context_service import PluginContextService

        mock_agent.assigned_plugins = []

        svc = PluginContextService(mock_db)
        tier1 = svc.build_tier1_summary([])
        assert tier1 == ""


# ===========================================================================
# Plugin context in system prompt
# ===========================================================================

class TestPluginContextInSystemPrompt:
    def test_tier1_in_prompt_always(self, mock_db, three_plugins):
        """All assigned plugins appear in Tier 1 summary section."""
        from core.services.plugin_context_service import PluginContextService

        svc = PluginContextService(mock_db)
        tier1 = svc.build_tier1_summary(three_plugins)

        assert "## Available Plugins" in tier1
        for _aap, plugin in three_plugins:
            assert plugin.name in tier1

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_tier2_loaded_for_relevant_plugins(self, mock_get_cache, mock_db):
        """When task context matches a plugin, its SKILL.md content appears."""
        rows = [
            _make_plugin("email-sender", "Email Sender", "Send emails", ["email"], priority=0),
            _make_plugin("db-tool", "DB Tool", "Query databases", ["database"], priority=1),
        ]

        cache = MagicMock()
        cache.get_file = AsyncMock(side_effect=lambda slug, _ver, _path: {
            "email-sender": "# Email Skills\nCompose and send.",
            "db-tool": "# DB Skills\nRun queries.",
        }.get(slug))
        cache.get_manifest = AsyncMock(return_value=None)
        mock_get_cache.return_value = cache

        from core.services.plugin_context_service import PluginContextService
        svc = PluginContextService(mock_db)
        result = await svc.build_tier2_content(
            rows, task_context="send email to user", max_plugins=1
        )
        assert "Email Skills" in result
        assert "Compose and send" in result

    @pytest.mark.asyncio
    @patch("core.services.plugin_cache.get_plugin_content_cache")
    async def test_tool_schemas_from_manifest_in_prompt(self, mock_get_cache, mock_db, plugin_rows):
        """Manifest tools appear as 'Available Tools' section."""
        cache = MagicMock()
        cache.get_file = AsyncMock(return_value="# Plugin Content")
        cache.get_manifest = AsyncMock(return_value={
            "tools": [
                {"name": "compose_email", "description": "Compose a new email"},
                {"name": "search_inbox", "description": "Search inbox by query"},
            ]
        })
        mock_get_cache.return_value = cache

        from core.services.plugin_context_service import PluginContextService
        svc = PluginContextService(mock_db)
        result = await svc.build_tier2_content(plugin_rows, max_plugins=5)
        assert "### Available Tools" in result
        assert "**compose_email**" in result
        assert "**search_inbox**" in result


# ===========================================================================
# PluginContentCache
# ===========================================================================

class TestPluginCacheService:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_s3(self):
        """Pre-populated Redis → S3 never called."""
        from core.services.plugin_cache import PluginContentCache

        mock_s3 = MagicMock()
        cache = PluginContentCache(s3_service=mock_s3)
        mock_redis = MagicMock()
        mock_redis.get.return_value = "# Cached SKILL.md content"
        cache._redis = mock_redis
        cache._redis_available = True

        result = await cache.get_file("email-auto", "1.0.0", "SKILL.md")
        assert result == "# Cached SKILL.md content"
        mock_s3.get_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_s3_and_populates(self):
        """Empty Redis → S3 called → Redis SET called."""
        from core.services.plugin_cache import PluginContentCache

        mock_s3 = MagicMock()
        mock_s3.get_file = AsyncMock(return_value="# S3 Content")
        cache = PluginContentCache(s3_service=mock_s3)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        cache._redis = mock_redis
        cache._redis_available = True

        result = await cache.get_file("email-auto", "1.0.0", "SKILL.md")
        assert result == "# S3 Content"
        mock_s3.get_file.assert_called_once()
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_unavailable_falls_back_to_s3(self):
        """Redis raises → S3 still works."""
        from core.services.plugin_cache import PluginContentCache
        from redis.exceptions import RedisError

        mock_s3 = MagicMock()
        mock_s3.get_file = AsyncMock(return_value="# S3 Fallback")
        cache = PluginContentCache(s3_service=mock_s3)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = RedisError("Connection refused")
        cache._redis = mock_redis
        cache._redis_available = True

        result = await cache.get_file("email-auto", "1.0.0", "SKILL.md")
        assert result == "# S3 Fallback"

    def test_invalidate_clears_all_keys(self):
        """invalidate_plugin() deletes content, manifest, and file keys."""
        from core.services.plugin_cache import PluginContentCache

        mock_s3 = MagicMock()
        cache = PluginContentCache(s3_service=mock_s3)
        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = [
            "plugin_files:email-auto:1.0.0:SKILL.md",
            "plugin_files:email-auto:1.0.0:README.md",
        ]
        cache._redis = mock_redis
        cache._redis_available = True

        cache.invalidate_plugin("email-auto", "1.0.0")

        mock_redis.delete.assert_any_call(
            "plugin_content:email-auto:1.0.0",
            "plugin_manifest:email-auto:1.0.0",
        )
        assert mock_redis.delete.call_count == 2
