"""Whether a process is still running, on every platform the host runs on (#818).

On Unix that is ``os.kill(pid, 0)``. On Windows ``os.kill`` with any signal but
the two console events calls ``TerminateProcess``: asking that way would end the
process, or another one that reused its PID. So Windows opens the process for
waiting and asks whether it has finished.
"""
from __future__ import annotations

import os
import sys

_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WAIT_TIMEOUT = 0x102
_ERROR_ACCESS_DENIED = 5


def pid_alive(pid: int) -> bool:
    """True while ``pid`` is a running process. Never sends it anything."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED   # running, just not ours to open
    try:
        return kernel32.WaitForSingleObject(wintypes.HANDLE(handle), 0) == _WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


__all__ = ["pid_alive"]
