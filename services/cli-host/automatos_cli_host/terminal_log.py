"""A bounded on-disk copy of what the session's terminal showed.

The host never parses the TUI, but when a session ends without a result the
operator needs to see what was on screen (a login prompt, a trust dialog, a
rate-limit banner, a model that never answered). ``terminal.log`` in the session
folder keeps the most recent ``max_bytes`` of raw terminal output — appended as
it arrives, trimmed to the newest half whenever it grows past the cap.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_MAX_BYTES = 2 * 1024 * 1024
FILENAME = "terminal.log"


class BoundedLog:
    def __init__(self, path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.path = Path(path)
        self.max_bytes = max(4096, int(max_bytes))
        self._lock = threading.Lock()
        self._size = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh session starts a fresh log; a resumed one appends.
        self._fh = open(self.path, "ab")
        os.chmod(self.path, 0o600)
        self._size = self.path.stat().st_size

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(chunk)
            self._fh.flush()
            self._size += len(chunk)
            if self._size > self.max_bytes:
                self._trim_locked()

    def _trim_locked(self) -> None:
        keep = self.max_bytes // 2
        self._fh.close()
        data = self.path.read_bytes()[-keep:]
        self.path.write_bytes(data)
        self._fh = open(self.path, "ab")
        self._size = len(data)

    def tail(self, n: int = 1500) -> str:
        with self._lock:
            try:
                self._fh.flush()
            except ValueError:
                pass
            try:
                data = self.path.read_bytes()[-max(n * 4, 4096):]
            except OSError:
                return ""
        return data.decode("utf-8", "replace")[-n:]

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
