"""One ticket through a supervised interactive session, against the fake ``claude``.

Proves the turn the PRD describes: files written under the host's state dir
(never in the repo), the trust decision recorded, the CLI spawned under a pty
with the invariant-safe argv, hooks gating tools (an in-directory Write allowed,
``git push`` denied, a TUI permission prompt denied), the contract injected as
additionalContext, ``Stop`` ending the turn, usage read from the transcript, the
process terminated, and an honest outcome for early exit, bad cwd and cancel.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.session import Session

from conftest import FAKE_CLAUDE


def _cfg(short_tmp, **over) -> HostConfig:
    base = dict(state_dir=short_tmp / "state", claude_binary=str(FAKE_CLAUDE), use_worktrees=False,
                session_timeout_seconds=120)
    base.update(over)
    return HostConfig(**base)


def _ticket(workdir: Path, **over) -> dict:
    t = {"task_id": 42, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_name": "Dwight",
         "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt\nOUTPUT: the file\nTOOLS: Write\nBOUNDARIES: this dir",
         "cwd": str(workdir), "model": "sonnet", "allowed_tools": [], "provider": "claude"}
    t.update(over)
    return t


def _run(short_tmp, ticket, cfg=None, allow=None):
    cfg = cfg or _cfg(short_tmp)
    hooks = HookServer(cfg.socket_path)
    hooks.start()
    allow = allow or [str(short_tmp / "ws")]
    s = Session(ticket, cfg, allow, cfg.socket_path, default_root=allow[0])
    hooks.register(str(ticket["task_id"]), s.handle_hook)
    try:
        return s, s.run()
    finally:
        hooks.unregister(str(ticket["task_id"]))
        hooks.stop()


def test_happy_turn(short_tmp, fake_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    ticket = _ticket(workdir)
    s, out = _run(short_tmp, ticket)

    assert out.status == "success", out
    assert out.exit_reason == "completed"
    assert "Done. Wrote hello.txt" in out.result_text
    assert "Push was denied by policy" in out.result_text        # the fake saw the deny
    assert "Contract seen" in out.result_text                    # additionalContext delivered
    assert (workdir / "hello.txt").read_text() == "hi\n"
    assert out.files_touched == [str((workdir / "hello.txt").resolve())]  # the fake reports its real cwd
    stages = sorted(d["stage"] for d in out.permission_denials)
    assert stages == ["PermissionRequest", "PreToolUse"]
    assert any("git push" in json.dumps(d) for d in out.permission_denials)
    assert out.usage["total_tokens"] == 150 and out.usage["model"] == "sonnet"
    assert "usd" not in json.dumps(out.usage)
    assert out.session_id == ticket["session_id"]
    assert out.transcript_path and Path(out.transcript_path).exists()
    assert s.proc is not None and s.proc.poll() is not None      # terminated after Stop

    # Session files live under the host's state dir — nothing in the repo.
    sdir = short_tmp / "state" / "sessions" / "42"
    assert (sdir / "ticket.md").exists() and (sdir / "settings.json").exists() and (sdir / "system_prompt.md").exists()
    assert not (workdir / ".automatos").exists()
    # The trust decision for the registered directory was recorded where Claude reads it.
    state = json.loads((fake_home / ".claude.json").read_text())
    assert state["projects"][str(workdir.resolve())]["hasTrustDialogAccepted"] is True
    # Events streamed the lifecycle.
    events = []
    while not s.events.empty():
        events.append(s.events.get_nowait()["event"])
    assert events[0] == "SessionStart" and "Stop" in events and "PreToolUse" in events


def test_cwd_outside_the_allowlist_never_starts_a_process(short_tmp, fake_home, env_clean):
    outside = short_tmp / "elsewhere"
    outside.mkdir()
    s, out = _run(short_tmp, _ticket(outside))
    assert out.status == "error" and out.exit_reason == "cwd_not_allowed"
    assert s.proc is None and "register it with --allow" in (out.error or "")


def test_claude_exiting_before_stop_is_an_error_with_diagnostics(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "exit-early")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "exited_before_stop"
    assert "code 3" in (out.error or "")


def test_cancel_terminates_a_running_session(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SLOW_SECONDS", "60")
    cfg = _cfg(short_tmp)
    hooks = HookServer(cfg.socket_path)
    hooks.start()
    ticket = _ticket(workdir)
    s = Session(ticket, cfg, [str(short_tmp / "ws")], cfg.socket_path, default_root=str(short_tmp / "ws"))
    hooks.register("42", s.handle_hook)
    holder = {}
    t = threading.Thread(target=lambda: holder.setdefault("out", s.run()))
    t.start()
    try:
        deadline = time.time() + 30
        while s.transcript_path is None and time.time() < deadline:
            time.sleep(0.1)
        assert s.transcript_path is not None, "SessionStart never arrived"
        s.request_cancel()
        t.join(timeout=30)
        assert not t.is_alive()
        out = holder["out"]
        assert out.status == "cancelled" and out.exit_reason == "cancelled"
        assert s.proc.poll() is not None
    finally:
        hooks.stop()


def test_not_onboarded_claude_is_refused_before_spawn(short_tmp, env_clean, monkeypatch):
    home = short_tmp / "home2"
    home.mkdir()
    (home / ".claude.json").write_text("{}")
    monkeypatch.setenv("HOME", str(home))
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    s, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "claude_not_onboarded" and s.proc is None


def test_a_session_that_never_starts_is_an_error_not_a_four_hour_wait(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "no-start")
    cfg = _cfg(short_tmp, startup_timeout_seconds=3)
    s, out = _run(short_tmp, _ticket(workdir), cfg=cfg)
    assert out.status == "error" and out.exit_reason == "no_session_start"
    assert "login screen" in (out.error or "") and s.proc.poll() is not None
