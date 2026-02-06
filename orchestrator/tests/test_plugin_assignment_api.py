"""Tests for api.agent_plugins — CRUD endpoints for agent plugin assignments.

Tests call the endpoint functions directly (not via HTTP) to avoid
needing a running FastAPI server. DB, auth, and models are mocked.

Module-level imports that require heavy deps (jwt, clerk) are pre-patched
via sys.modules to avoid ModuleNotFoundError in CI/local without full venv.
"""
import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Stub out transitive imports that need jwt/clerk before importing the API module
# ---------------------------------------------------------------------------
_stubs = {}
for mod_name in [
    "jwt", "jwt.algorithms", "jwt.exceptions",
    "core.auth.clerk", "core.auth.hybrid",
]:
    if mod_name not in sys.modules:
        stub = ModuleType(mod_name)
        # Provide the symbols that downstream modules expect
        stub.get_request_context_hybrid = MagicMock()
        stub.get_clerk_auth = MagicMock()
        stub.decode = MagicMock()
        stub.DecodeError = Exception
        stub.ExpiredSignatureError = Exception
        sys.modules[mod_name] = stub
        _stubs[mod_name] = stub

from api.agent_plugins import (  # noqa: E402
    list_agent_plugins,
    update_agent_plugins,
    UpdateAgentPluginsBody,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(agent_id=42, workspace_id=None):
    agent = MagicMock()
    agent.id = agent_id
    agent.workspace_id = workspace_id or uuid4()
    return agent


def _make_ctx(workspace_id):
    ctx = MagicMock()
    ctx.workspace_id = workspace_id
    return ctx


def _make_aap_row(plugin_slug, plugin_name, priority=0, **kwargs):
    """Return (AgentAssignedPlugin-mock, MarketplacePlugin-mock)."""
    plugin = MagicMock()
    plugin.id = kwargs.get("plugin_id", uuid4())
    plugin.slug = plugin_slug
    plugin.name = plugin_name
    plugin.version = kwargs.get("version", "1.0.0")
    plugin.description = kwargs.get("description", f"{plugin_name} description")
    plugin.skills_count = kwargs.get("skills_count", 2)
    plugin.commands_count = kwargs.get("commands_count", 1)
    plugin.token_estimate = kwargs.get("token_estimate", 500)

    aap = MagicMock()
    aap.agent_id = kwargs.get("agent_id", 42)
    aap.plugin_id = plugin.id
    aap.priority = priority
    aap.assigned_at = datetime(2025, 1, 15, 12, 0, 0)
    return aap, plugin


# ===========================================================================
# list_agent_plugins
# ===========================================================================

class TestListAgentPlugins:
    @pytest.mark.asyncio
    async def test_returns_assigned_plugins_with_details(self):
        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)
        row1 = _make_aap_row("email-auto", "Email Automation", priority=0)
        row2 = _make_aap_row("slack-bot", "Slack Bot", priority=1)

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        rows_q = MagicMock()
        rows_q.join.return_value.filter.return_value.order_by.return_value.all.return_value = [row1, row2]
        db.query.side_effect = [agent_q, rows_q]

        result = await list_agent_plugins(agent_id=42, ctx=ctx, db=db)
        items = result["items"]
        assert len(items) == 2
        assert items[0]["slug"] == "email-auto"
        assert items[1]["slug"] == "slack-bot"
        assert items[0]["priority"] == 0

    @pytest.mark.asyncio
    async def test_empty_when_no_plugins_assigned(self):
        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        rows_q = MagicMock()
        rows_q.join.return_value.filter.return_value.order_by.return_value.all.return_value = []
        db.query.side_effect = [agent_q, rows_q]

        result = await list_agent_plugins(agent_id=42, ctx=ctx, db=db)
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_404_when_agent_not_found(self):
        from fastapi import HTTPException

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = None
        db.query.return_value = agent_q

        ctx = _make_ctx(uuid4())
        with pytest.raises(HTTPException) as exc_info:
            await list_agent_plugins(agent_id=999, ctx=ctx, db=db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_403_on_workspace_mismatch(self):
        from fastapi import HTTPException

        agent = _make_agent(workspace_id=uuid4())
        different_ws = uuid4()
        ctx = _make_ctx(different_ws)

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        db.query.return_value = agent_q

        with pytest.raises(HTTPException) as exc_info:
            await list_agent_plugins(agent_id=42, ctx=ctx, db=db)
        assert exc_info.value.status_code == 403


# ===========================================================================
# update_agent_plugins
# ===========================================================================

class TestUpdateAgentPlugins:
    @pytest.mark.asyncio
    async def test_replaces_all_assignments(self):
        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)
        plugin_c_id = uuid4()
        body = UpdateAgentPluginsBody(plugin_ids=[plugin_c_id])

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        wep_row = MagicMock()
        wep_row.plugin_id = plugin_c_id
        wep_q = MagicMock()
        wep_q.filter.return_value.all.return_value = [wep_row]
        delete_q = MagicMock()
        delete_q.filter.return_value.delete.return_value = 2
        db.query.side_effect = [agent_q, wep_q, delete_q]

        result = await update_agent_plugins(agent_id=42, body=body, ctx=ctx, db=db)
        assert result["success"] is True
        assert len(result["plugin_ids"]) == 1
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplicates_plugin_ids(self):
        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)
        pid_a = uuid4()
        pid_b = uuid4()
        body = UpdateAgentPluginsBody(plugin_ids=[pid_a, pid_a, pid_b])

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        wep_a = MagicMock(); wep_a.plugin_id = pid_a
        wep_b = MagicMock(); wep_b.plugin_id = pid_b
        wep_q = MagicMock()
        wep_q.filter.return_value.all.return_value = [wep_a, wep_b]
        delete_q = MagicMock()
        delete_q.filter.return_value.delete.return_value = 0
        db.query.side_effect = [agent_q, wep_q, delete_q]

        result = await update_agent_plugins(agent_id=42, body=body, ctx=ctx, db=db)
        assert len(result["plugin_ids"]) == 2
        assert db.add.call_count == 2

    @pytest.mark.asyncio
    async def test_validates_workspace_enabled(self):
        from fastapi import HTTPException

        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)
        unenabled_plugin = uuid4()
        body = UpdateAgentPluginsBody(plugin_ids=[unenabled_plugin])

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        wep_q = MagicMock()
        wep_q.filter.return_value.all.return_value = []
        db.query.side_effect = [agent_q, wep_q]

        with pytest.raises(HTTPException) as exc_info:
            await update_agent_plugins(agent_id=42, body=body, ctx=ctx, db=db)
        assert exc_info.value.status_code == 400
        assert "not enabled" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_priority_matches_list_order(self):
        ws_id = uuid4()
        agent = _make_agent(workspace_id=ws_id)
        ctx = _make_ctx(ws_id)
        pid_c, pid_a, pid_b = uuid4(), uuid4(), uuid4()
        body = UpdateAgentPluginsBody(plugin_ids=[pid_c, pid_a, pid_b])

        db = MagicMock()
        agent_q = MagicMock()
        agent_q.filter.return_value.first.return_value = agent
        weps = [MagicMock(plugin_id=pid_c), MagicMock(plugin_id=pid_a), MagicMock(plugin_id=pid_b)]
        wep_q = MagicMock()
        wep_q.filter.return_value.all.return_value = weps
        delete_q = MagicMock()
        delete_q.filter.return_value.delete.return_value = 0
        db.query.side_effect = [agent_q, wep_q, delete_q]

        await update_agent_plugins(agent_id=42, body=body, ctx=ctx, db=db)

        added_objects = [call.args[0] for call in db.add.call_args_list]
        assert len(added_objects) == 3
        assert added_objects[0].priority == 0
        assert added_objects[0].plugin_id == pid_c
        assert added_objects[1].priority == 1
        assert added_objects[1].plugin_id == pid_a
        assert added_objects[2].priority == 2
        assert added_objects[2].plugin_id == pid_b
