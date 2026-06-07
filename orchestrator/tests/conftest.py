"""Shared fixtures for plugin system tests."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


@pytest.fixture(autouse=True)
def _repair_stubbed_consumers_chatbot():
    """Rebind ``consumers.chatbot`` onto the ``consumers`` parent if a sibling
    leaked a hand-built stub into ``sys.modules``.

    Several unit tests (test_memory_single_write_path, test_tool_loop_*, etc.)
    inject ``types.ModuleType`` stubs for ``consumers`` and ``consumers.chatbot``
    directly into ``sys.modules`` to dodge the heavy package __init__. A real
    ``import`` sets the child as an attribute on the parent module; a manual
    sys.modules insert does NOT. pytest's ``monkeypatch.setattr("a.b.c", ...)``
    walks attributes from the top package, so a later sibling that monkeypatches
    ``consumers.chatbot.<x>`` trips on ``'module' object ... has no attribute
    'chatbot'``. Rebinding here is idempotent and a no-op once the real package
    is loaded.
    """
    parent = sys.modules.get("consumers")
    child = sys.modules.get("consumers.chatbot")
    if parent is not None and child is not None and getattr(parent, "chatbot", None) is not child:
        parent.chatbot = child
    yield


# ---- Database mock ----

@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session with chainable query API."""
    db = MagicMock()
    # Make .query().join().filter().order_by().all() chainable
    q = MagicMock()
    q.join.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = []
    q.first.return_value = None
    q.delete.return_value = 0
    db.query.return_value = q
    return db


# ---- Plugin model mocks ----

@pytest.fixture
def sample_plugin():
    """A MarketplacePlugin-like object with realistic data."""
    plugin = MagicMock()
    plugin.id = uuid4()
    plugin.slug = "email-automation"
    plugin.name = "Email Automation"
    plugin.version = "1.2.0"
    plugin.description = "Send and manage emails across providers"
    plugin.tags = ["email", "automation", "notification"]
    plugin.skills_count = 5
    plugin.commands_count = 3
    plugin.token_estimate = 1500
    plugin.s3_bucket = "automatos-marketplace"
    plugin.s3_path = "plugins/email-automation/1.2.0/"
    return plugin


@pytest.fixture
def sample_assignment(sample_plugin):
    """An AgentAssignedPlugin-like object."""
    aap = MagicMock()
    aap.agent_id = 42
    aap.plugin_id = sample_plugin.id
    aap.priority = 0
    aap.assigned_at = datetime(2025, 1, 15, 12, 0, 0)
    return aap


@pytest.fixture
def plugin_rows(sample_assignment, sample_plugin):
    """List of (assignment, plugin) tuples — single plugin."""
    return [(sample_assignment, sample_plugin)]


def _make_plugin(slug, name, description=None, tags=None, priority=0, **overrides):
    """Helper to create (assignment, plugin) tuples for multi-plugin tests."""
    plugin = MagicMock()
    plugin.id = uuid4()
    plugin.slug = slug
    plugin.name = name
    plugin.version = overrides.get("version", "1.0.0")
    plugin.description = description
    plugin.tags = tags
    plugin.skills_count = overrides.get("skills_count", 2)
    plugin.commands_count = overrides.get("commands_count", 1)
    plugin.token_estimate = overrides.get("token_estimate", 500)
    plugin.s3_bucket = "automatos-marketplace"
    plugin.s3_path = f"plugins/{slug}/1.0.0/"

    aap = MagicMock()
    aap.agent_id = overrides.get("agent_id", 42)
    aap.plugin_id = plugin.id
    aap.priority = priority
    aap.assigned_at = datetime(2025, 1, 15, 12, 0, 0)
    return aap, plugin


@pytest.fixture
def three_plugins():
    """Three plugins with different priorities and themes."""
    return [
        _make_plugin("db-manager", "Database Manager", "Query and manage databases", ["database", "sql"], priority=2),
        _make_plugin("email-sender", "Email Sender", "Send email notifications", ["email", "notification"], priority=0),
        _make_plugin("slack-bot", "Slack Bot", "Post messages to Slack channels", ["slack", "messaging"], priority=1),
    ]


# ---- Agent mock ----

@pytest.fixture
def mock_agent():
    """A realistic Agent-like mock."""
    agent = MagicMock()
    agent.id = 42
    agent.name = "Test Agent"
    agent.workspace_id = uuid4()
    agent.skills = []
    agent.assigned_plugins = []
    agent.model_config = {"model_id": "gpt-4", "temperature": 0.7}
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
    agent.use_custom_persona = False
    agent.custom_persona_prompt = None
    agent.persona_id = None
    return agent


# ---- Playbook mocks ----

@pytest.fixture
def mock_playbook():
    """A WorkflowTemplate (Playbook)-like mock with cron schedule_config."""
    playbook = MagicMock()
    playbook.id = 42
    playbook.template_id = "test-cron-playbook"
    playbook.name = "Test Cron Playbook"
    playbook.workspace_id = uuid4()
    playbook.steps = [
        {"step_id": "s1", "order": 1, "agent_id": 101, "prompt_template": "Do the thing"},
    ]
    playbook.schedule_config = {
        "type": "cron",
        "cron_expression": "0 9 * * *",
    }
    playbook.owner_type = "workspace"
    playbook.is_system = False
    return playbook


@pytest.fixture
def mock_playbook_manual():
    """A WorkflowTemplate (Playbook)-like mock with manual schedule (no cron)."""
    playbook = MagicMock()
    playbook.id = 99
    playbook.template_id = "test-manual-playbook"
    playbook.name = "Test Manual Playbook"
    playbook.workspace_id = uuid4()
    playbook.steps = [
        {"step_id": "s1", "order": 1, "agent_id": 101, "prompt_template": "Manual task"},
    ]
    playbook.schedule_config = {
        "type": "manual",
    }
    playbook.owner_type = "workspace"
    playbook.is_system = False
    return playbook


# ---- Request context mock ----

@pytest.fixture
def mock_ctx(mock_agent):
    """A RequestContext-like mock matching the agent's workspace."""
    ctx = MagicMock()
    ctx.workspace_id = mock_agent.workspace_id
    ctx.user = MagicMock()
    ctx.user.id = "user_123"
    ctx.auth_type = "clerk"
    return ctx
