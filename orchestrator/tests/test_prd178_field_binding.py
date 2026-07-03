"""PRD-178 S1 (F020) — field auto-injection binds to the CALLING task's run.

Before this wave, ``platform_executor`` auto-injected ``field_id`` by taking
``.first()`` of any ``state=='running'`` OrchestrationRun in the workspace —
no ordering, no link to the calling task. With two concurrent missions running
in one workspace, agents in mission A could write into and read from mission
B's field (F020), and a running mission would shadow workspace recall (F021).

The fix threads the calling task's field context (``field_id`` + ``mission_id``
/``run_id``) down through ``caller_context`` and drops the ``.first()`` lookup.
These tests pin that behaviour with the DB mocked at the boundary — a
``.first()`` regression would make ``test_field_binding_to_task`` fail because
the injected id would no longer track the threaded context.
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern — see
# tests/test_prd143_su_executor_gate.py). The port points at nothing so the
# fail-soft import-time DB connect refuses instantly. Nothing here touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from modules.tools.discovery.action_registry import ActionDefinition
from modules.tools.discovery.platform_executor import PlatformActionExecutor

pytestmark = pytest.mark.asyncio

_FIELD_ACTION = "platform_field_inject"


def _action(name: str) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description="PRD-178 S1 field-binding probe",
        category="field",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
    )


class _FakeRegistry:
    def __init__(self, *defs: ActionDefinition):
        self._defs = {d.name: d for d in defs}

    def get(self, name: str):
        return self._defs.get(name)

    def get_all(self):
        return list(self._defs.values())


def _executor(workspace_id: uuid.UUID) -> PlatformActionExecutor:
    return PlatformActionExecutor(MagicMock(), workspace_id)


def _capture_handler(ex: PlatformActionExecutor, action: str) -> dict:
    """Replace the action's handler with one that captures the params it
    received, so the test can assert which ``field_id`` was injected."""
    captured: dict = {}

    async def _handler(db, workspace_id, params):
        captured["params"] = dict(params)
        return {"success": True}

    ex._handlers[action] = _handler
    return captured


async def _run(ex, captured_registry, action, params, caller_context):
    with patch(
        "modules.tools.discovery.get_action_registry",
        return_value=captured_registry,
    ):
        return await ex.execute(action, params, caller_context=caller_context)


async def test_field_binding_to_task():
    """The field_id injected for a field tool is the one threaded from the
    CALLING task's run via caller_context — NOT an arbitrary running run."""
    ws = uuid.uuid4()
    ex = _executor(ws)
    captured = _capture_handler(ex, _FIELD_ACTION)
    registry = _FakeRegistry(_action(_FIELD_ACTION))

    task_a_field = str(uuid.uuid4())

    # A stray running run exists in the workspace — the OLD .first() code would
    # have injected ITS field_id. If the DB is consulted at all here the test
    # fails, because we assert the threaded id wins.
    stray_run = MagicMock()
    stray_run.config = {"field_id": "STRAY-RUN-FIELD"}
    ex.db.query.return_value.filter.return_value.first.return_value = stray_run

    result = await _run(
        ex, registry, _FIELD_ACTION, {},
        caller_context={"field_context": {"field_id": task_a_field,
                                          "mission_id": str(uuid.uuid4())}},
    )

    assert result["success"] is True
    assert captured["params"].get("field_id") == task_a_field, (
        "field_id must come from the calling task's threaded context"
    )
    assert captured["params"].get("field_id") != "STRAY-RUN-FIELD"


async def test_concurrent_tasks_bind_to_own_field():
    """Two concurrent tasks in one workspace each bind to their OWN field —
    task B never sees task A's field_id."""
    ws = uuid.uuid4()
    registry = _FakeRegistry(_action(_FIELD_ACTION))

    field_a = str(uuid.uuid4())
    field_b = str(uuid.uuid4())

    ex_a = _executor(ws)
    cap_a = _capture_handler(ex_a, _FIELD_ACTION)
    await _run(ex_a, registry, _FIELD_ACTION, {},
               caller_context={"field_context": {"field_id": field_a}})

    ex_b = _executor(ws)
    cap_b = _capture_handler(ex_b, _FIELD_ACTION)
    await _run(ex_b, registry, _FIELD_ACTION, {},
               caller_context={"field_context": {"field_id": field_b}})

    assert cap_a["params"].get("field_id") == field_a
    assert cap_b["params"].get("field_id") == field_b
    assert cap_a["params"]["field_id"] != cap_b["params"]["field_id"]


async def test_no_field_context_no_injection_no_db_scan():
    """Without threaded field context, NO field_id is injected and the DB is
    NOT scanned for a running run (the .first() ambient-guess bug is gone).

    An explicitly-passed field_id is left untouched."""
    ws = uuid.uuid4()
    ex = _executor(ws)
    captured = _capture_handler(ex, _FIELD_ACTION)
    registry = _FakeRegistry(_action(_FIELD_ACTION))

    # A stray running run is present. The OLD .first() code would inject its
    # field_id here; the fix must NOT — proving the ambient DB guess is gone
    # (if the DB were consulted for binding, field_id would be the mock value).
    stray_run = MagicMock()
    stray_run.config = {"field_id": "STRAY-RUN-FIELD"}
    ex.db.query.return_value.filter.return_value.first.return_value = stray_run

    # No caller_context field_context, no field_id in params.
    await _run(ex, registry, _FIELD_ACTION, {}, caller_context=None)
    assert "field_id" not in captured["params"], (
        "no ambient .first() guess when nothing is threaded"
    )

    # An explicit field_id passed by the caller survives untouched.
    explicit = str(uuid.uuid4())
    await _run(ex, registry, _FIELD_ACTION, {"field_id": explicit},
               caller_context=None)
    assert captured["params"]["field_id"] == explicit
