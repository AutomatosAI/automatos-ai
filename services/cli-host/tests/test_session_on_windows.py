"""#818: one whole ticket through a session, on whatever platform runs the suite.

On Windows (the ``cli-host-windows`` lane) this is the end-to-end proof:
- the fake ``claude`` runs as an ``.exe`` (pip's own launcher for console scripts);
- it runs on a ConPTY pseudo console;
- its hooks reach the host over the authenticated named pipe;
- the gate denies the push, the contract arrives, and Stop ends the turn.

On Linux and macOS the same test runs on a pty and the Unix socket.
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import hook_server_for
from automatos_cli_host.sandbox import SessionSandbox
from automatos_cli_host.session import Session

from conftest import FAKE_CLAUDE


def _claude_binary(bin_dir: Path) -> str:
    """The fake ``claude`` as this platform runs a CLI: the script itself, or on
    Windows an .exe launcher (a .cmd one the host would refuse)."""
    if sys.platform != "win32":
        return str(FAKE_CLAUDE)
    from pip._vendor.distlib.scripts import ScriptMaker

    maker = ScriptMaker(str(FAKE_CLAUDE.parent), str(bin_dir))
    maker.executable = sys.executable
    return next(path for path in maker.make(FAKE_CLAUDE.name) if path.endswith(".exe"))


@pytest.fixture
def root():
    # A Unix socket path is capped (~104 bytes on macOS); Windows has no /tmp.
    return Path(tempfile.mkdtemp(prefix="acli-", dir=None if sys.platform == "win32" else "/tmp"))


@pytest.fixture
def home(root, monkeypatch):
    """A home with a Claude Code that has completed onboarding. Windows reads USERPROFILE."""
    folder = root / "home"
    folder.mkdir()
    (folder / ".claude.json").write_text('{"hasCompletedOnboarding": true, "projects": {}}')
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(folder))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return folder


def test_a_whole_ticket_runs_in_a_session_on_this_platform(root, home, env_clean):
    (root / "bin").mkdir()
    workdir = root / "ws" / "repo"
    workdir.mkdir(parents=True)
    cfg = HostConfig(state_dir=root / "state", cli_binaries={"claude": _claude_binary(root / "bin")},
                     use_worktrees=False, session_timeout_seconds=120,
                     session_sandbox=SessionSandbox(enabled=False))   # none exists on Windows
    ticket = {"task_id": 42, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_name": "Dwight",
              "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt\nOUTPUT: the file\nTOOLS: Write\nBOUNDARIES: this dir",
              "cwd": str(workdir), "model": "sonnet", "allowed_tools": [], "provider": "claude"}
    hooks = hook_server_for(cfg.socket_path)
    hooks.start()
    s = Session(ticket, cfg, [str(root / "ws")], cfg.socket_path, default_root=str(root / "ws"))
    s.hook_env = hooks.session_env()
    hooks.register("42", s.handle_hook)
    try:
        out = s.run()
    finally:
        hooks.unregister("42")
        hooks.stop()

    assert out.status == "success", out
    assert "Push was denied by policy" in out.result_text and "Contract seen" in out.result_text
    assert (workdir / "hello.txt").read_text() == "hi\n"
    assert s.proc is not None and s.proc.poll() is not None
