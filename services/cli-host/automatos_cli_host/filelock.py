"""An exclusive lock on an open file, on every platform the host runs on (#818).

POSIX takes ``flock``. Windows locks the file's first byte with ``msvcrt``,
retrying until it is free: its own blocking mode gives up after ten tries.
Closing the descriptor releases the lock on both.
"""
from __future__ import annotations

import os
import sys
import time

_RETRY_SECONDS = 0.05


def lock_exclusive(fd: int) -> None:
    """Block until this process holds the only lock on ``fd``'s file."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(_RETRY_SECONDS)
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX)


__all__ = ["lock_exclusive"]
