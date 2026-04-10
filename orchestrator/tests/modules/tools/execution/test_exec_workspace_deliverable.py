"""Tests for PRD-129 US-004: auto-register deliverables on workspace_write_file.

These tests mock the WorkspaceClient + DB so they stay pure unit tests.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---- Stub out heavy imports BEFORE loading exec_workspace ------------------
# exec_workspace lazy-imports WorkspaceClient, DeliverableService, SessionLocal,
# and the Agent model inside _auto_register_deliverable. We don't want those to
# pull in pgvector / the real database config, so we pre-register stubs in
# sys.modules. Each test patches the specific attributes it cares about.

_STUB_WORKSPACE_CLIENT = MagicMock(name="WorkspaceClientStub")
_STUB_DELIVERABLE_SERVICE = MagicMock(name="DeliverableServiceStub")
_STUB_SESSION_LOCAL = MagicMock(name="SessionLocalStub")


def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# core.workspace_client
_make_stub_module("core", __path__=[])
_make_stub_module("core.workspace_client", WorkspaceClient=_STUB_WORKSPACE_CLIENT)
_make_stub_module("core.database", __path__=[])
_make_stub_module("core.database.database", SessionLocal=_STUB_SESSION_LOCAL)
_make_stub_module("core.models", __path__=[])
_make_stub_module("core.models.core", Agent=MagicMock(name="AgentStub"))
_make_stub_module("services", __path__=[])
_make_stub_module(
    "services.deliverable_service",
    DeliverableService=_STUB_DELIVERABLE_SERVICE,
    AGENT_REGISTERABLE_ARTIFACT_TYPES=frozenset({
        "report", "image", "document", "slide", "spreadsheet", "code",
    }),
    _infer_artifact_type=lambda fp: (
        "report" if fp.endswith(".md")
        else "archive" if fp.endswith(".zip")
        else "document"
    ),
    _humanize_basename=lambda fp: fp.split("/")[-1],
)

# Load exec_workspace directly to avoid pulling the entire modules.tools package.
_THIS = Path(__file__).resolve()
_EXEC_WORKSPACE = _THIS.parents[4] / "modules" / "tools" / "execution" / "exec_workspace.py"
_spec = importlib.util.spec_from_file_location("exec_workspace_under_test", _EXEC_WORKSPACE)
exec_workspace = importlib.util.module_from_spec(_spec)
sys.modules["exec_workspace_under_test"] = exec_workspace
_spec.loader.exec_module(exec_workspace)

_derive_source = exec_workspace._derive_source
execute_workspace_action = exec_workspace.execute_workspace_action


WORKSPACE_ID = uuid4()


class TestDeriveSource:
    def test_default_is_chat(self):
        assert _derive_source(None) == ("chat", None)
        assert _derive_source({}) == ("chat", None)

    def test_explicit_source_type_wins(self):
        ctx = {"source_type": "playbook", "source_id": "pb-42"}
        assert _derive_source(ctx) == ("playbook", "pb-42")

    def test_heartbeat_id_maps_to_heartbeat(self):
        assert _derive_source({"heartbeat_id": 7}) == ("heartbeat", "7")

    def test_mission_id_maps_to_mission(self):
        assert _derive_source({"mission_id": "m-1"}) == ("mission", "m-1")

    def test_task_id_maps_to_task(self):
        assert _derive_source({"task_id": "t-9"}) == ("task", "t-9")

    def test_precedence_heartbeat_over_task(self):
        # heartbeat_id comes first in the precedence tuple
        ctx = {"heartbeat_id": 1, "task_id": 2}
        assert _derive_source(ctx) == ("heartbeat", "1")


@pytest.fixture(autouse=True)
def _reset_stubs():
    """Reset stub mocks before each test."""
    _STUB_WORKSPACE_CLIENT.reset_mock()
    _STUB_DELIVERABLE_SERVICE.reset_mock()
    _STUB_SESSION_LOCAL.reset_mock()

    # Default: SessionLocal() returns a context manager
    fake_session = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(return_value=False)
    _STUB_SESSION_LOCAL.return_value = fake_ctx

    # Default register() returns success
    fake_service_inst = MagicMock()
    fake_service_inst.register = MagicMock(return_value={
        "success": True, "deliverable_id": "d-1", "created": True,
    })
    _STUB_DELIVERABLE_SERVICE.return_value = fake_service_inst
    yield fake_service_inst


def _make_client(**method_returns):
    client = MagicMock()
    client.list_dir = AsyncMock(return_value={"entries": []})
    for method, val in method_returns.items():
        setattr(client, method, AsyncMock(return_value=val))
    _STUB_WORKSPACE_CLIENT.return_value = client
    return client


@pytest.mark.asyncio
async def test_write_file_auto_registers_deliverable(_reset_stubs):
    """A successful workspace_write_file should create a deliverables row."""
    _make_client(write_file={
        "success": True,
        "path": "reports/scout/weekly.md",
        "size": 2048,
    })

    result = await execute_workspace_action(
        executor=MagicMock(),
        tool_name="workspace_write_file",
        parameters={"path": "reports/scout/weekly.md", "content": "# Weekly"},
        workspace_id=WORKSPACE_ID,
        agent_id=42,
        caller_context={"heartbeat_id": 99},
    )

    assert result["success"] is True
    service_inst = _STUB_DELIVERABLE_SERVICE.return_value
    assert service_inst.register.call_count == 1
    call_kwargs = service_inst.register.call_args.kwargs
    assert call_kwargs["file_path"] == "reports/scout/weekly.md"
    assert call_kwargs["artifact_type"] == "report"
    assert call_kwargs["source_type"] == "heartbeat"
    assert call_kwargs["source_id"] == "99"
    assert call_kwargs["agent_id"] == 42
    assert call_kwargs["file_size_bytes"] == 2048


@pytest.mark.asyncio
async def test_write_file_skips_non_registerable_extension(_reset_stubs):
    """Archive (.zip) is not in AGENT_REGISTERABLE_ARTIFACT_TYPES → skipped."""
    _make_client(write_file={"success": True, "size": 100})

    result = await execute_workspace_action(
        executor=MagicMock(),
        tool_name="workspace_write_file",
        parameters={"path": "artifacts/bundle.zip", "content": "binary"},
        workspace_id=WORKSPACE_ID,
    )

    assert result["success"] is True
    _STUB_DELIVERABLE_SERVICE.return_value.register.assert_not_called()


@pytest.mark.asyncio
async def test_register_failure_does_not_break_write(_reset_stubs):
    """If DeliverableService.register raises, the write still returns success."""
    _make_client(write_file={"success": True, "size": 1})
    _STUB_DELIVERABLE_SERVICE.return_value.register.side_effect = RuntimeError("db down")

    result = await execute_workspace_action(
        executor=MagicMock(),
        tool_name="workspace_write_file",
        parameters={"path": "reports/x/today.md", "content": "hi"},
        workspace_id=WORKSPACE_ID,
    )

    assert result["success"] is True


@pytest.mark.asyncio
async def test_read_file_does_not_register(_reset_stubs):
    """Reads should never create deliverable rows."""
    _make_client(read_file={"success": True, "content": "data"})

    result = await execute_workspace_action(
        executor=MagicMock(),
        tool_name="workspace_read_file",
        parameters={"path": "reports/x/today.md"},
        workspace_id=WORKSPACE_ID,
    )

    assert result["success"] is True
    _STUB_DELIVERABLE_SERVICE.return_value.register.assert_not_called()


@pytest.mark.asyncio
async def test_write_file_defaults_source_type_to_chat(_reset_stubs):
    """With no caller_context, source_type defaults to 'chat'."""
    _make_client(write_file={"success": True, "size": 50})

    result = await execute_workspace_action(
        executor=MagicMock(),
        tool_name="workspace_write_file",
        parameters={"path": "reports/scout/x.md", "content": "hi"},
        workspace_id=WORKSPACE_ID,
    )

    assert result["success"] is True
    kwargs = _STUB_DELIVERABLE_SERVICE.return_value.register.call_args.kwargs
    assert kwargs["source_type"] == "chat"
    assert kwargs["source_id"] is None
