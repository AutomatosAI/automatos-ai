"""F139 (night 4, B7) — the Composio intent gate checks the action the call names.

A playbook step's Dropbox read failed with "Tool composio_execute failed: Action
not allowed for this intent: Unknown action: execute". The gate took the action
id from the tool name before the arguments, and the generic wrapper's name,
composio_execute, reads as the action "execute", which no metadata row knows.
The spine always dispatches that way (tool_name="composio_execute",
tool_args={"action": ...}), so every intent-checked Composio call from a step
was refused the same way. It is unrelated to PRD-251's deny list.
"""
from __future__ import annotations

import asyncio

import pytest

import core.composio.client as cc
import modules.tools.tool_router as tr
from config import config


class _SDK:
    """Stands in for ``composio.Composio``: the import succeeded."""


@pytest.fixture
def gate(monkeypatch):
    """Composio up; the gate refuses whatever it is asked, and records the action id."""
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(cc, "_get_composio", lambda: _SDK)
    cc.reset_composio_availability()
    asked = []

    def _validate(action_id, intent, allow_destructive=False):
        asked.append(action_id)
        return False, "stopped by the test"

    monkeypatch.setattr(tr, "validate_action_for_intent", _validate)
    yield asked
    cc.reset_composio_availability()


def _run(tool_name, tool_args):
    return asyncio.run(tr.execute_tool_with_validation(tool_name, tool_args, "read my roast log", agent_id=1))


def test_the_wrapper_is_checked_as_the_action_it_names(gate):
    result = _run("composio_execute", {"action": "DROPBOX_READ_FILE", "params": {"path": "/roast-log.md"}})
    assert gate == ["DROPBOX_READ_FILE"]
    assert result["action_id"] == "DROPBOX_READ_FILE"


def test_a_per_action_tool_name_is_still_read_from_the_name(gate):
    _run("composio_SLACK_SEND_MESSAGE", {"channel": "#roastery", "text": "Batch 14 is out"})
    assert gate == ["SLACK_SEND_MESSAGE"]
