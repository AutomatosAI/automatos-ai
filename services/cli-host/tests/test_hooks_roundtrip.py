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


def _run_shim(payload: dict, env_extra: dict, args: tuple = ()) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
    proc = subprocess.run([sys.executable, "-m", "automatos_cli_host.hook_shim", *args],
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


def test_the_shim_talks_to_the_host_process_only(short_tmp):
    """F234: under Copilot's sandbox the socket path is granted read-write so the
    hooks can reach the host — a sandboxed command could then bind its own socket
    there and answer its own calls. The shim checks the peer is the host's PID."""
    server = HookServer(short_tmp / "h.sock")
    server.start()
    seen = []

    def handler(payload):
        seen.append(payload)
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}

    server.register("42", handler)
    try:
        env = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "h.sock"), "AUTOMATOS_TASK_ID": "42"}
        call = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}}
        host = {**env, "AUTOMATOS_HOST_PID": str(os.getpid())}       # the server runs in this process
        assert json.loads(_run_shim(call, host))["hookSpecificOutput"]["permissionDecision"] == "allow"
        impostor = {**env, "AUTOMATOS_HOST_PID": str(os.getpid() + 100000)}
        out = json.loads(_run_shim(call, impostor))["hookSpecificOutput"]
        assert out["permissionDecision"] == "deny" and "not the host's" in out["permissionDecisionReason"]
        assert _run_shim({"hook_event_name": "Stop"}, impostor) == ""
        assert len(seen) == 1                                          # nothing was sent to the impostor
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


def test_an_event_named_on_the_command_line_reaches_the_host(short_tmp):
    """PRD-253 S0.4 — design §5 ``event_name_source: argv``: a CLI whose payload
    carries no event name (Copilot's camelCase-only ``permissionRequest`` and
    ``notification``) gets it from the hook entry's own command line."""
    server = HookServer(short_tmp / "h.sock")
    server.start()
    seen = []

    def handler(payload):
        seen.append(payload)
        return {}

    server.register("42", handler)
    try:
        env = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "h.sock"), "AUTOMATOS_TASK_ID": "42"}
        _run_shim({"toolName": "bash", "toolArgs": {"command": "ls"}}, env, ("--event", "PermissionRequest"))
        assert seen[-1]["hook_event_name"] == "PermissionRequest" and seen[-1]["toolName"] == "bash"
        # A name in the payload wins over the command line.
        _run_shim({"hook_event_name": "Stop"}, env, ("--event", "PermissionRequest"))
        assert seen[-1]["hook_event_name"] == "Stop"
    finally:
        server.stop()
    # With the host gone, an event named only on the command line is still a gated one: denied.
    gone = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "missing.sock"), "AUTOMATOS_TASK_ID": "1"}
    out = json.loads(_run_shim({"toolName": "bash"}, gone, ("--event", "PermissionRequest")))
    assert out["hookSpecificOutput"]["decision"]["behavior"] == "deny"


def test_the_operators_own_terminal_is_not_gated_by_the_shim(short_tmp):
    """PRD-253: the Canvas terminal opens a per-agent CLI (Codex, Copilot) in the
    agent's own home, where these hooks are installed. Someone is at the keyboard
    there, so with no host socket the shim stands aside and the CLI's own prompts
    apply; a supervised session whose host is unreachable still fails closed."""
    terminal = {"AUTOMATOS_HOST_SOCK": "", "AUTOMATOS_TERMINAL": "1"}
    assert _run_shim({"hook_event_name": "PreToolUse", "tool_name": "Bash"}, terminal) == ""
    assert _run_shim({"hook_event_name": "PermissionRequest"}, terminal) == ""
    gone = {"AUTOMATOS_HOST_SOCK": str(short_tmp / "missing.sock"), "AUTOMATOS_TERMINAL": "1"}
    out = json.loads(_run_shim({"hook_event_name": "PreToolUse", "tool_name": "Bash"}, gone))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
