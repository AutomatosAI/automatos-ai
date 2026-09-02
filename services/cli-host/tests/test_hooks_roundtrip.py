"""The hook shim ↔ hook server round trip, exactly as Claude Code drives it:
a command with JSON on stdin, one JSON line back on stdout."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from automatos_cli_host.hook_server import HookServer

ROOT = Path(__file__).resolve().parents[1]


def _run_shim(payload: dict, env_extra: dict) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
    proc = subprocess.run([sys.executable, "-m", "automatos_cli_host.hook_shim"],
                          input=json.dumps(payload), capture_output=True, text=True, timeout=20, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_shim_forwards_to_the_registered_session_and_prints_its_answer(short_tmp):
    server = HookServer(short_tmp / "h.sock")
    server.start()
    seen = []

    def handler(payload):
        seen.append(payload)
        if payload.get("hook_event_name") == "PreToolUse":
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": "test says no"}}
        return {}

    server.register("42", handler)
    try:
        env = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "h.sock"), "AUTOMATOS_TASK_ID": "42"}
        out = _run_shim({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}}, env)
        assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert seen[-1]["automatos_task_id"] == "42" and seen[-1]["tool_name"] == "Bash"
        # A non-gated event with an empty answer prints nothing (exit 0).
        assert _run_shim({"hook_event_name": "Notification", "message": "idle"}, env) == ""
        # An unknown session: gated events are denied, others are silent.
        env_unknown = {**env, "AUTOMATOS_TASK_ID": "999"}
        assert json.loads(_run_shim({"hook_event_name": "PermissionRequest", "tool_name": "Bash"}, env_unknown))[
            "hookSpecificOutput"]["decision"]["behavior"] == "deny"
        assert _run_shim({"hook_event_name": "Stop"}, env_unknown) == ""
    finally:
        server.stop()


def test_shim_fails_closed_when_the_host_is_unreachable(short_tmp):
    env = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "missing.sock"), "AUTOMATOS_TASK_ID": "1"}
    out = json.loads(_run_shim({"hook_event_name": "PreToolUse", "tool_name": "Bash"}, env))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "unreachable" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert _run_shim({"hook_event_name": "PostToolUse"}, env) == ""
    # No socket configured at all → same posture.
    out = json.loads(_run_shim({"hook_event_name": "PreToolUse"}, {"AUTOMATOS_HOST_SOCK": ""}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
