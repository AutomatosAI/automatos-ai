"""Tests for api.agents._build_agent_response() — plugin fields in agent responses.

Tests verify that the AgentResponse model includes plugins and that
_build_agent_response correctly populates the plugins list.

Module-level imports that require heavy deps (jwt, clerk) are pre-patched
via sys.modules to avoid ModuleNotFoundError in CI/local without full venv.
"""
import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Stub out transitive imports that need jwt/clerk
# ---------------------------------------------------------------------------
for mod_name in [
    "jwt", "jwt.algorithms", "jwt.exceptions",
    "core.auth.clerk", "core.auth.hybrid",
]:
    if mod_name not in sys.modules:
        stub = ModuleType(mod_name)
        stub.get_request_context_hybrid = MagicMock()
        stub.get_clerk_auth = MagicMock()
        stub.decode = MagicMock()
        stub.DecodeError = Exception
        stub.ExpiredSignatureError = Exception
        sys.modules[mod_name] = stub

from api.agents import _build_agent_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_agent(assigned_plugins=None):
    """Create an Agent mock with all fields _build_agent_response needs."""
    agent = MagicMock()
    agent.id = 42
    agent.name = "Test Agent"
    agent.description = "A test agent"
    agent.agent_type = "code_architect"
    agent.status = "active"
    agent.configuration = {}
    agent.priority_level = "medium"
    agent.max_concurrent_tasks = 5
    agent.auto_start = False
    agent.tags = ["test"]
    agent.created_at = datetime(2025, 1, 1)
    agent.updated_at = datetime(2025, 1, 2)
    agent.created_by = "user_123"
    agent.performance_metrics = {}
    agent.model_config = {"model_id": "gpt-4"}
    agent.skills = []
    agent.assigned_plugins = assigned_plugins if assigned_plugins is not None else []
    agent.workspace_id = uuid4()
    return agent


def _make_marketplace_plugin(plugin_id=None, slug="test-plugin", name="Test Plugin",
                              version="1.0.0", description="A test plugin",
                              skills_count=3, commands_count=2):
    mp = MagicMock()
    mp.id = plugin_id or uuid4()
    mp.slug = slug
    mp.name = name
    mp.version = version
    mp.description = description
    mp.skills_count = skills_count
    mp.commands_count = commands_count
    return mp


def _make_assignment(plugin_id, priority=0, plugin=None):
    aap = MagicMock()
    aap.plugin_id = plugin_id
    aap.priority = priority
    aap.plugin = plugin
    return aap


# ===========================================================================
# _build_agent_response plugins
# ===========================================================================

class TestBuildAgentResponsePlugins:
    def test_plugins_included_in_response(self):
        mp = _make_marketplace_plugin(slug="email-auto", name="Email Automation")
        aap = _make_assignment(mp.id, plugin=mp)
        agent = _make_full_agent(assigned_plugins=[aap])

        db = MagicMock()
        tools_q = MagicMock()
        tools_q.filter.return_value.all.return_value = []
        db.query.side_effect = [tools_q]

        resp = _build_agent_response(agent, db)
        assert len(resp.plugins) == 1
        assert resp.plugins[0]["slug"] == "email-auto"
        assert resp.plugins[0]["name"] == "Email Automation"
        assert "plugin_id" in resp.plugins[0]

    def test_empty_plugins_when_none_assigned(self):
        agent = _make_full_agent(assigned_plugins=[])
        db = MagicMock()
        tools_q = MagicMock()
        tools_q.filter.return_value.all.return_value = []
        db.query.side_effect = [tools_q]

        resp = _build_agent_response(agent, db)
        assert resp.plugins == []

    def test_plugin_fields_populated_correctly(self):
        pid = uuid4()
        mp = _make_marketplace_plugin(
            plugin_id=pid, slug="db-tool", name="DB Tool",
            version="2.1.0", description="Database management",
            skills_count=7, commands_count=4,
        )
        aap = _make_assignment(pid, plugin=mp)
        agent = _make_full_agent(assigned_plugins=[aap])

        db = MagicMock()
        tools_q = MagicMock()
        tools_q.filter.return_value.all.return_value = []
        db.query.side_effect = [tools_q]

        resp = _build_agent_response(agent, db)
        p = resp.plugins[0]
        assert p["plugin_id"] == str(pid)
        assert p["version"] == "2.1.0"
        assert p["description"] == "Database management"
        assert p["skills_count"] == 7
        assert p["commands_count"] == 4

    def test_enhanced_model_includes_plugins_field(self):
        """AgentResponse Pydantic model must have a 'plugins' field."""
        from core.models.core import AgentResponse

        fields = AgentResponse.model_fields
        assert "plugins" in fields
        default = fields["plugins"].default
        assert default == []
