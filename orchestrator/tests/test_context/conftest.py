"""Shared fixtures for context module tests."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


@pytest.fixture
def mock_agent():
    """A realistic Agent-like mock for context tests."""
    agent = MagicMock()
    agent.id = 42
    agent.name = "Test Agent"
    agent.agent_type = "assistant"
    agent.description = "A helpful test agent"
    agent.use_custom_persona = False
    agent.custom_persona_prompt = None
    agent.persona = None
    return agent


@pytest.fixture
def simple_agent():
    """Minimal agent using SimpleNamespace (no MagicMock magic)."""
    return SimpleNamespace(
        id=1,
        name="Simple Agent",
        agent_type="researcher",
        description=None,
        use_custom_persona=False,
        custom_persona_prompt=None,
        persona=None,
    )


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session with chainable query API."""
    db = MagicMock()
    q = MagicMock()
    q.join.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = []
    q.first.return_value = None
    db.query.return_value = q
    return db


@pytest.fixture
def section_ctx(mock_agent, mock_db):
    """Pre-built SectionContext for section render tests."""
    from modules.context.sections.base import SectionContext

    return SectionContext(
        agent=mock_agent,
        workspace_id="ws_test_123",
        workspace_name="Test Workspace",
        db_session=mock_db,
        messages=[
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ],
        kwargs={},
    )
