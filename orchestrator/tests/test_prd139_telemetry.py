"""PRD-139 US-001: Unit tests for universal tool execution telemetry.

Tests:
1. Mocked tool execution writes one row with correct fields
2. Telemetry write failure does not propagate to caller
3. Composio call produces exactly one log row (no double-write)
"""
import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Ensure orchestrator root is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


# ---- Direct import of telemetry module (avoids triggering modules/tools/__init__.py) ----
_telemetry_path = Path(_orchestrator_root) / "modules" / "tools" / "execution" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("telemetry_mod", _telemetry_path)
telemetry_mod = importlib.util.module_from_spec(_spec)


# Create a mock ToolExecutionLog class that behaves like a SQLAlchemy model
class MockToolExecutionLog:
    """Mock of ToolExecutionLog that captures constructor kwargs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# telemetry.py imports ToolExecutionLog lazily *inside* write_telemetry(), so
# the fake model only needs to be live while this file's tests run — not at
# import. Installing it in setup_module (and restoring in teardown_module)
# keeps the fake out of sibling modules' collection/runtime. (PRD-142 W2-S2b.)
_mock_composio_cache = MagicMock()
_mock_composio_cache.ToolExecutionLog = MockToolExecutionLog

_spec.loader.exec_module(telemetry_mod)

write_telemetry = telemetry_mod.write_telemetry
fire_telemetry = telemetry_mod.fire_telemetry

_saved_composio_cache = None


def setup_module(module):
    global _saved_composio_cache
    _saved_composio_cache = sys.modules.get("core.models.composio_cache")
    sys.modules["core.models.composio_cache"] = _mock_composio_cache


def teardown_module(module):
    if _saved_composio_cache is None:
        sys.modules.pop("core.models.composio_cache", None)
    else:
        sys.modules["core.models.composio_cache"] = _saved_composio_cache


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    return db


@pytest.fixture
def sample_workspace_id():
    return uuid4()


class TestTelemetryWrite:
    """Test that write_telemetry creates a correct log row."""

    @pytest.mark.asyncio
    async def test_writes_one_row_with_correct_fields(self, mock_db, sample_workspace_id):
        """AC-9: Mocked tool execution writes one row with correct fields."""
        await write_telemetry(
            mock_db,
            tool_name="platform_list_agents",
            parameters={"workspace_id": str(sample_workspace_id)},
            agent_id=42,
            workspace_id=sample_workspace_id,
            result={"success": True, "data": []},
            execution_time_ms=150,
            caller_context={
                "user_query": "list all agents",
                "conversation_id": "conv-123",
                "turn_id": "turn-1",
                "routing_source": "keyword",
                "routing_candidates": ["platform_list_agents", "platform_get_agent"],
                "routing_chain_hints": ["agent_management"],
            },
        )

        # Verify db.add was called with a MockToolExecutionLog instance
        assert mock_db.add.call_count == 1
        log_entry = mock_db.add.call_args[0][0]

        # Verify fields
        assert log_entry.agent_id == 42
        assert log_entry.app_name == "PLATFORM"
        assert log_entry.action_name == "platform_list_agents"
        assert log_entry.workspace_id == sample_workspace_id
        assert log_entry.status == "success"
        assert log_entry.execution_time_ms == 150
        assert log_entry.user_query == "list all agents"
        assert log_entry.routing_source == "keyword"
        assert log_entry.telemetry_source == "production"
        assert log_entry.router_decision["candidates"] == ["platform_list_agents", "platform_get_agent"]
        assert log_entry.router_decision["conversation_id"] == "conv-123"
        assert log_entry.router_decision["turn_id"] == "turn-1"
        assert log_entry.router_decision["chain_hints"] == ["agent_management"]
        assert log_entry.error_message is None

        # Verify commit was called
        assert mock_db.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_id_none_for_zero(self, mock_db, sample_workspace_id):
        """agent_id=0 should be stored as None (nullable FK)."""
        await write_telemetry(
            mock_db,
            tool_name="workspace_read_file",
            parameters={"path": "/tmp/test.txt"},
            agent_id=0,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=50,
        )

        log_entry = mock_db.add.call_args[0][0]
        assert log_entry.agent_id is None

    @pytest.mark.asyncio
    async def test_error_result_captured(self, mock_db, sample_workspace_id):
        """Error results should capture error_message and status='error'."""
        await write_telemetry(
            mock_db,
            tool_name="platform_execute",
            parameters={"action": "unknown"},
            agent_id=5,
            workspace_id=sample_workspace_id,
            result={"success": False, "error": "Unknown action: unknown"},
            execution_time_ms=10,
        )

        log_entry = mock_db.add.call_args[0][0]
        assert log_entry.status == "error"
        assert log_entry.error_message == "Unknown action: unknown"

    @pytest.mark.asyncio
    async def test_composio_app_name_detection(self, mock_db, sample_workspace_id):
        """Composio tools should derive app_name correctly."""
        await write_telemetry(
            mock_db,
            tool_name="COMPOSIO_SEARCH_WEB",
            parameters={"query": "test"},
            agent_id=10,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=200,
        )

        log_entry = mock_db.add.call_args[0][0]
        assert log_entry.app_name == "COMPOSIO"

    @pytest.mark.asyncio
    async def test_workspace_app_name_detection(self, mock_db, sample_workspace_id):
        """Workspace tools should derive app_name as WORKSPACE."""
        await write_telemetry(
            mock_db,
            tool_name="workspace_read_file",
            parameters={"path": "/tmp/test.txt"},
            agent_id=3,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=30,
        )

        log_entry = mock_db.add.call_args[0][0]
        assert log_entry.app_name == "WORKSPACE"


class TestTelemetryFailureIsolation:
    """Test that telemetry failures do not propagate."""

    @pytest.mark.asyncio
    async def test_db_commit_failure_does_not_raise(self, mock_db, sample_workspace_id):
        """AC-10: Telemetry write failure does not propagate to caller."""
        mock_db.commit.side_effect = Exception("DB connection lost")

        # Should NOT raise
        await write_telemetry(
            mock_db,
            tool_name="platform_list_agents",
            parameters={},
            agent_id=1,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=100,
        )

        # Verify rollback was attempted
        assert mock_db.rollback.call_count == 1

    @pytest.mark.asyncio
    async def test_db_add_failure_does_not_raise(self, mock_db, sample_workspace_id):
        """If db.add throws, telemetry should still not propagate."""
        mock_db.add.side_effect = Exception("Session closed")

        # Should NOT raise
        await write_telemetry(
            mock_db,
            tool_name="test_tool",
            parameters={},
            agent_id=1,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=50,
        )

        # rollback should be called
        assert mock_db.rollback.call_count == 1

    @pytest.mark.asyncio
    async def test_fire_telemetry_non_blocking(self, mock_db, sample_workspace_id):
        """fire_telemetry uses create_task and does not block."""
        # fire_telemetry should schedule a task without blocking
        fire_telemetry(
            mock_db,
            tool_name="platform_list_agents",
            parameters={},
            agent_id=1,
            workspace_id=sample_workspace_id,
            result={"success": True},
            execution_time_ms=100,
        )

        # Allow the event loop to process the task
        await asyncio.sleep(0.05)

        # Verify the write happened via the background task
        assert mock_db.add.call_count == 1


class TestNoDoubleWrite:
    """Test that Composio calls produce exactly one log row."""

    @pytest.mark.asyncio
    async def test_composio_single_log_row(self, mock_db, sample_workspace_id):
        """AC-11: Composio call produces exactly one log row (no double-write).

        The old exec_composio.py wrote its own log. Now only the unified
        telemetry hook in execute_tool writes. Verify that exec_composio's
        execute_composio_execute no longer writes to the DB.
        """
        # Import exec_composio directly to avoid triggering full import chain
        _exec_composio_path = Path(_orchestrator_root) / "modules" / "tools" / "execution" / "exec_composio.py"
        _ec_spec = importlib.util.spec_from_file_location("exec_composio_mod", _exec_composio_path)
        # Need to mock the registry import
        mock_registry_mod = MagicMock()
        mock_registry_mod.ToolSpec = MagicMock()
        sys.modules["modules.tools.registry.tool_registry"] = mock_registry_mod
        exec_composio_mod = importlib.util.module_from_spec(_ec_spec)
        _ec_spec.loader.exec_module(exec_composio_mod)

        # Create a mock executor with a composio_executor that returns success
        mock_executor = MagicMock()
        mock_executor.db = mock_db
        mock_executor.composio_executor = AsyncMock()
        mock_executor.composio_executor.execute = AsyncMock(
            return_value={"success": True, "data": {"result": "ok"}}
        )

        result = await exec_composio_mod.execute_composio_execute(
            mock_executor,
            tool_name="GMAIL_SEND_EMAIL",
            parameters={
                "action": "GMAIL_SEND_EMAIL",
                "params": {"to": "test@example.com"},
                "app_name": "GMAIL",
            },
            agent_id=10,
            workspace_id=sample_workspace_id,
            trace_id="test-trace",
        )

        # The function should NOT have written to DB (no db.add call from exec_composio)
        assert mock_db.add.call_count == 0, (
            f"exec_composio wrote {mock_db.add.call_count} rows — expected 0 "
            f"(unified telemetry hook handles logging)"
        )
        assert mock_db.commit.call_count == 0

        # The result should still be successful
        assert result.get("success") is True

        # Cleanup
        del sys.modules["modules.tools.registry.tool_registry"]
