"""#838 — sessions that start together all keep their workspace trust.

Each session records trust with a read-modify-write of ``~/.claude.json`` on a
thread of its own. Unserialised, two starting together both read the old file
and the last replace won: the other folder lost its trust and its
``claude --worktree`` refused to start. The read here is slowed down to hold the
race window open, so the lost update is certain without the lock.
"""
from __future__ import annotations

import json
import stat
import threading
import time

from automatos_cli_host.adapters import claude as claude_adapter

SESSIONS = 12


def _slow_reads(monkeypatch):
    real = claude_adapter.read_claude_state

    def slow(home=None):
        state = real(home)
        time.sleep(0.02)  # a read-modify-write stretched wide enough to interleave
        return state

    monkeypatch.setattr(claude_adapter, "read_claude_state", slow)


def test_sessions_starting_together_all_keep_their_trust(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "theme": "dark"}))
    folders = [tmp_path / f"repo-{i}" for i in range(SESSIONS)]
    _slow_reads(monkeypatch)
    start = threading.Barrier(SESSIONS)

    def session(folder):
        start.wait()
        claude_adapter.record_directory_trust(folder, home)

    threads = [threading.Thread(target=session, args=(f,)) for f in folders]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    state = json.loads((home / ".claude.json").read_text())
    lost = [str(f) for f in folders if not (state["projects"].get(str(f)) or {}).get("hasTrustDialogAccepted")]
    assert lost == [], f"{len(lost)} of {SESSIONS} sessions lost their trust"
    assert state["theme"] == "dark" and state["hasCompletedOnboarding"] is True  # nothing else touched


def test_the_lock_file_sits_beside_the_state_and_is_private(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert claude_adapter.record_directory_trust(tmp_path / "repo", home) is True
    lock = home / ".claude.json.automatos-lock"
    assert lock.exists() and stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_a_write_lost_to_another_writer_is_made_again(tmp_path, monkeypatch):
    """A running ``claude`` saving its own state can replace the file between
    the write and the read-back; the flag is then written again."""
    home = tmp_path / "home"
    home.mkdir()
    real = claude_adapter._write_trust
    calls = []

    def first_write_lost(path, cwd, h):
        calls.append(cwd)
        if len(calls) > 1:
            real(path, cwd, h)

    monkeypatch.setattr(claude_adapter, "_write_trust", first_write_lost)
    assert claude_adapter.record_directory_trust(tmp_path / "repo", home) is True
    assert len(calls) == 2 and claude_adapter.is_directory_trusted(tmp_path / "repo", home)
