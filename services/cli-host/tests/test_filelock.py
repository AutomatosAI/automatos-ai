"""#818: the host's file lock holds between processes on every platform.

Claude Code's trust file is rewritten under it (#838), so a second process must
wait for the first to let go. Runs on Linux and macOS (flock) and on Windows
(msvcrt, the ``cli-host-windows`` CI lane).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from automatos_cli_host.filelock import lock_exclusive

ROOT = Path(__file__).resolve().parents[1]
HOLD_SECONDS = 1.5

_HOLDER = """
import os, sys, time
from automatos_cli_host.filelock import lock_exclusive
fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)
lock_exclusive(fd)
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
os.close(fd)
"""


def test_a_second_process_waits_for_the_lock(tmp_path):
    path = tmp_path / "trust.automatos-lock"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(path), str(HOLD_SECONDS)],
                              stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    try:
        assert holder.stdout.readline().strip() == "locked"
        started = time.monotonic()
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            lock_exclusive(fd)
            waited = time.monotonic() - started
        finally:
            os.close(fd)
        assert waited >= HOLD_SECONDS / 2, f"took the lock after {waited:.2f}s while another process held it"
    finally:
        holder.wait(10)


def test_the_lock_is_free_again_once_closed(tmp_path):
    path = tmp_path / "trust.automatos-lock"
    for _ in range(2):                       # the second take finds it released by the close
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        lock_exclusive(fd)
        os.close(fd)
