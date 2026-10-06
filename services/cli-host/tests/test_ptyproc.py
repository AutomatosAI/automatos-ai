"""#818: the host's pseudo-terminal child behaves the same on every platform.

Each test starts a real Python child on a real terminal: a Unix pty on Linux and
macOS, a ConPTY pseudo console on Windows (the ``cli-host-windows`` CI lane runs
this file on windows-latest). The last tests cover the Windows command line and
environment block, which need no Windows to check.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from automatos_cli_host.ptyproc import KILL, TERMINATE, environment_block, spawn, windows_command_line

ROWS, COLS = 30, 100
WAIT_SECONDS = 20.0


def _script(tmp_path: Path, name: str, code: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    return path


def _start(tmp_path: Path, code: str):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return spawn([sys.executable, str(_script(tmp_path, "child.py", code))], cwd=tmp_path, env=env, rows=ROWS, cols=COLS)


class _Output:
    """Drains the child in the background, as a session does."""

    def __init__(self, child):
        self.data = bytearray()
        self.ended = threading.Event()
        threading.Thread(target=self._drain, args=(child,), daemon=True).start()

    def _drain(self, child) -> None:
        try:
            while True:
                try:
                    chunk = child.read(65536)
                except OSError:
                    break
                if not chunk:
                    break
                self.data.extend(chunk)
        finally:
            self.ended.set()

    def shows(self, text: str, timeout: float = WAIT_SECONDS) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if text.encode() in bytes(self.data):
                return True
            time.sleep(0.05)
        return False


def test_the_child_writes_to_the_terminal_and_its_exit_code_comes_back(tmp_path):
    child = _start(tmp_path, """
        import sys
        print("hello from the child")
        sys.exit(3)
    """)
    out = _Output(child)
    try:
        assert out.shows("hello from the child")
        assert child.wait(WAIT_SECONDS) == 3 and child.poll() == 3 and child.returncode == 3
        assert out.ended.wait(WAIT_SECONDS), "the output never ended after the child exited"
    finally:
        child.close()


def test_what_is_written_reaches_the_child(tmp_path):
    child = _start(tmp_path, """
        print("ready")
        print("got:" + input())
    """)
    out = _Output(child)
    try:
        assert out.shows("ready")
        child.write(b"ping\r")
        assert out.shows("got:ping")
        assert child.wait(WAIT_SECONDS) == 0
    finally:
        child.close()


def test_the_terminal_has_the_size_given_and_follows_a_resize(tmp_path):
    child = _start(tmp_path, """
        import os, sys
        def size():
            s = os.get_terminal_size(sys.stdout.fileno())
            print(f"size={s.columns}x{s.lines}")
        size()
        input()
        size()
    """)
    out = _Output(child)
    try:
        assert out.shows(f"size={COLS}x{ROWS}")
        child.resize(40, 120)
        child.write(b"\r")
        assert out.shows("size=120x40")
    finally:
        child.stop(((KILL, WAIT_SECONDS),))
        child.close()


def test_stop_ends_the_child_and_what_it_started(tmp_path):
    beat = tmp_path / "beat"
    grandchild = _script(tmp_path, "grandchild.py", f"""
        import time
        while True:
            with open({str(beat)!r}, "a") as f:
                f.write(".")
            time.sleep(0.1)
    """)
    child = _start(tmp_path, f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, {str(grandchild)!r}])
        print("started")
        while True:
            time.sleep(1)
    """)
    out = _Output(child)
    try:
        assert out.shows("started")
        deadline = time.monotonic() + WAIT_SECONDS
        while not (beat.exists() and beat.stat().st_size > 2) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert beat.exists(), "the grandchild never ran"

        assert child.stop(((TERMINATE, 2.0), (KILL, WAIT_SECONDS))) is True
        time.sleep(1.0)                      # anything still running beats in this second
        before = beat.stat().st_size
        time.sleep(1.0)
        assert beat.stat().st_size == before, "the grandchild outlived the stop"
    finally:
        child.close()


def test_a_child_that_ignores_the_polite_stage_is_killed(tmp_path):
    child = _start(tmp_path, """
        import signal, time
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_IGN)
        print("stubborn")
        while True:
            time.sleep(1)
    """)
    out = _Output(child)
    try:
        assert out.shows("stubborn")
        assert child.stop(((TERMINATE, 0.5), (KILL, WAIT_SECONDS))) is True
        assert child.returncode is not None
    finally:
        child.close()


def test_a_closed_terminal_reads_as_the_end(tmp_path):
    child = _start(tmp_path, "print('bye')")
    child.wait(WAIT_SECONDS)
    child.close()
    child.close()                            # twice is fine
    assert child.read(1024) == b""


# ── the Windows command line and environment, checked anywhere ─────────────────


def test_the_windows_command_line_quotes_each_argument():
    args = [r"C:\Program Files\Claude\claude.exe", "a b", 'say "hi"', ""]
    assert windows_command_line(args) == subprocess.list2cmdline(args)


@pytest.mark.parametrize("launcher", [r"C:\npm\copilot.cmd", r"C:\tools\claude.BAT"])
def test_a_batch_file_launcher_is_refused(launcher):
    # cmd.exe would re-read the session's own arguments (its prompt) as commands.
    with pytest.raises(OSError, match="batch file"):
        windows_command_line([launcher, "--resume", "x & calc"])


def test_the_environment_block_is_sorted_without_regard_to_case_and_ends_twice():
    block = environment_block({"b": "2", "Path": r"C:\bin", "A": "1"})
    assert block.decode("utf-16-le") == "A=1\0b=2\0Path=C:\\bin\0\0"


def test_names_that_differ_only_in_case_are_one_variable():
    block = environment_block({"PATH": "old", "Path": "new"}).decode("utf-16-le")
    assert block == "Path=new\0\0"


@pytest.mark.parametrize("bad", [{"": "x"}, {"A=B": "x"}, {"A": "x\0y"}])
def test_a_malformed_variable_is_refused(bad):
    with pytest.raises(ValueError):
        environment_block(bad)
