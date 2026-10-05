"""Windows job objects: a process tree that ends when its owner lets go (#818).

A job created here has ``KILL_ON_JOB_CLOSE``: when its last handle closes, by
``close`` or because the process holding it died, every process in it ends. The
host puts each session's CLI in one (``conpty.py``), and the login task's
supervisor puts the host in one (``winservice.py``), so nothing outlives the
process that started it. Standard library only (``ctypes``).
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
STOPPED_EXIT_CODE = 1


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _fn(name: str, restype, *argtypes):
    fn = getattr(_kernel32, name)
    fn.restype, fn.argtypes = restype, list(argtypes)
    return fn


_CreateJobObjectW = _fn("CreateJobObjectW", wintypes.HANDLE, ctypes.c_void_p, wintypes.LPCWSTR)
_SetInformationJobObject = _fn("SetInformationJobObject", wintypes.BOOL, wintypes.HANDLE, ctypes.c_int,
                               ctypes.c_void_p, wintypes.DWORD)
_AssignProcessToJobObject = _fn("AssignProcessToJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE)
_TerminateJobObject = _fn("TerminateJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.UINT)
_OpenProcess = _fn("OpenProcess", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_CloseHandle = _fn("CloseHandle", wintypes.BOOL, wintypes.HANDLE)


def _check(ok) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def close_handle(handle) -> None:
    if handle:
        _CloseHandle(handle)


def kill_on_close_job():
    """A new job whose processes all end when its last handle closes."""
    job = _CreateJobObjectW(None, None)
    _check(job)
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        close_handle(job)
        raise ctypes.WinError(error)
    return job


def assign(job, process_handle) -> None:
    _check(_AssignProcessToJobObject(job, process_handle))


def assign_pid(job, pid: int) -> None:
    """Put the running process ``pid`` in ``job``."""
    process = _OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    _check(process)
    try:
        assign(job, process)
    finally:
        close_handle(process)


def terminate(job) -> bool:
    """End every process in ``job`` now."""
    return bool(_TerminateJobObject(job, STOPPED_EXIT_CODE))


__all__ = ["assign", "assign_pid", "close_handle", "kill_on_close_job", "terminate"]
