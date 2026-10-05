"""#818: the hook channel Windows uses — a named pipe both ends authenticate.

On Windows (the ``cli-host-windows`` lane) these run over a real named pipe. On
Linux and macOS the same server runs over a Unix socket (``family="AF_UNIX"``), so
the key check and the host-process check are proven on every platform.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from automatos_cli_host.hook_pipe import PIPE_PREFIX, PipeHookServer
from automatos_cli_host.hook_server import HookServer, hook_server_for

ROOT = Path(__file__).resolve().parents[1]
GATED = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}}
ALLOW = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}


@pytest.fixture
def make_server():
    started = []

    def _make(key=None):
        if sys.platform == "win32":
            server = PipeHookServer(key=key)
        else:   # AF_UNIX paths are capped (~104 bytes on macOS): a short directory
            folder = tempfile.mkdtemp(prefix="acli-", dir="/tmp")
            server = PipeHookServer(address=os.path.join(folder, "k.sock"), family="AF_UNIX", key=key)
        server.start()
        started.append(server)
        return server

    yield _make
    for server in started:
        server.stop()


def _run_shim(payload: dict, env_extra: dict) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTOMATOS_")}
    env.update({"PYTHONPATH": str(ROOT), **env_extra})
    proc = subprocess.run([sys.executable, "-m", "automatos_cli_host.hook_shim"], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=30, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _session_env(server: PipeHookServer, **over) -> dict:
    return {**server.session_env(), "AUTOMATOS_TASK_ID": "42", "AUTOMATOS_HOST_PID": str(os.getpid()), **over}


def _recording(server: PipeHookServer, answer: dict) -> list:
    seen: list = []
    server.register("42", lambda payload: seen.append(payload) or answer)
    return seen


def test_a_hook_call_reaches_its_session_and_the_answer_comes_back(make_server):
    server = make_server()
    seen = _recording(server, ALLOW)
    env = _session_env(server)
    assert json.loads(_run_shim(GATED, env)) == ALLOW
    assert seen[-1]["automatos_task_id"] == "42" and seen[-1]["tool_name"] == "Bash"
    # An empty answer prints nothing; a session nobody owns is denied on a gated event.
    server.register("42", lambda payload: {})
    assert _run_shim({"hook_event_name": "Notification", "message": "idle"}, env) == ""
    unknown = json.loads(_run_shim(GATED, {**env, "AUTOMATOS_TASK_ID": "999"}))
    assert unknown["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_a_shim_with_the_wrong_key_sends_nothing_and_denies(make_server):
    server = make_server()
    seen = _recording(server, ALLOW)
    out = json.loads(_run_shim(GATED, _session_env(server, AUTOMATOS_HOST_KEY="00" * 32)))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "not the host's" in out["permissionDecisionReason"]
    assert seen == []


def test_the_shim_talks_to_the_host_process_only(make_server):
    # A command the CLI runs inherits the key; serving the pipe from another process
    # must still not answer the policy gate (F234's check, on the pipe).
    server = make_server()
    seen = _recording(server, ALLOW)
    impostor = _session_env(server, AUTOMATOS_HOST_PID=str(os.getpid() + 100000))
    out = json.loads(_run_shim(GATED, impostor))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "not the host's" in out["permissionDecisionReason"]
    assert _run_shim({"hook_event_name": "Stop"}, impostor) == ""
    assert seen == []


def test_an_unreachable_host_denies_the_gated_events(make_server):
    server = make_server()
    env = _session_env(server)
    server.stop()
    out = json.loads(_run_shim(GATED, env))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "unreachable" in out["permissionDecisionReason"]
    assert _run_shim({"hook_event_name": "Stop"}, env) == ""


def test_the_host_refuses_a_caller_without_its_key(make_server):
    from multiprocessing import AuthenticationError
    from multiprocessing.connection import Client

    server = make_server()
    seen = _recording(server, ALLOW)
    with pytest.raises(AuthenticationError):
        Client(server.address, family=server.family, authkey=b"not the key").close()
    assert seen == []
    assert json.loads(_run_shim(GATED, _session_env(server))) == ALLOW   # still serving


def test_each_host_start_gets_its_own_pipe_and_key():
    first, second = PipeHookServer(), PipeHookServer()
    assert first.address.startswith(PIPE_PREFIX) and first.address != second.address
    assert len(first.key) == 32 and first.key != second.key
    assert first.session_env() == {"AUTOMATOS_HOST_SOCK": first.address, "AUTOMATOS_HOST_KEY": first.key.hex()}


def test_windows_gets_the_pipe_and_everything_else_the_unix_socket(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    assert isinstance(hook_server_for(tmp_path / "hooks.sock"), PipeHookServer)
    monkeypatch.setattr(sys, "platform", "linux")
    unix = hook_server_for(tmp_path / "hooks.sock")
    assert isinstance(unix, HookServer) and unix.session_env() == {"AUTOMATOS_HOST_SOCK": str(tmp_path / "hooks.sock")}


def test_a_caller_that_stalls_the_handshake_does_not_hold_up_the_next_call(make_server):
    # The handshake used to run in the accept loop: one silent caller blocked every
    # hook call after it until the CLI timed them out and let the tools run.
    from multiprocessing.connection import Client

    server = make_server()
    _recording(server, ALLOW)
    staller = Client(server.address, family=server.family)      # connects, never answers the challenge
    try:
        started = time.monotonic()
        assert json.loads(_run_shim(GATED, _session_env(server))) == ALLOW
        assert time.monotonic() - started < 15
    finally:
        staller.close()


def test_the_shim_denies_rather_than_waits_on_a_host_that_never_completes_the_handshake(tmp_path):
    # A hook the CLI times out lets the tool run, so the shim gives up first.
    from multiprocessing.connection import Listener

    if sys.platform == "win32":
        address, family = f"{PIPE_PREFIX}automatos-test-{os.getpid()}-stall", "AF_PIPE"
    else:
        address, family = os.path.join(tempfile.mkdtemp(prefix="acli-", dir="/tmp"), "s.sock"), "AF_UNIX"
    listener = Listener(address, family=family)                 # accepts, then says nothing
    held = []
    threading.Thread(target=lambda: held.append(listener.accept()), daemon=True).start()
    env = {"AUTOMATOS_HOST_SOCK": address, "AUTOMATOS_HOST_KEY": "11" * 32,
           "AUTOMATOS_TASK_ID": "42", "AUTOMATOS_HOST_PID": str(os.getpid())}
    try:
        started = time.monotonic()
        out = json.loads(_run_shim(GATED, env))["hookSpecificOutput"]
        assert out["permissionDecision"] == "deny" and "unreachable" in out["permissionDecisionReason"]
        assert time.monotonic() - started < 15
    finally:
        listener.close()
