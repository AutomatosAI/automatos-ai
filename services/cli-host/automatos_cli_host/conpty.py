"""ConPTY: a Windows pseudo console for the host's child processes (#818).

Windows 10 1809 or later. Standard library only (``ctypes``), like the rest of the
host: Python's ``subprocess`` cannot attach a pseudo console, so the process is
created with ``CreateProcessW`` and ``PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE``.

- The child starts suspended, goes into a job object, and only then runs, so
  nothing it starts can leave the job first. The job ends the whole tree when the
  host stops the session, and when the host itself dies (``KILL_ON_JOB_CLOSE``).
- A pseudo console keeps its output pipe open after the child exits, so a waiter
  closes the console when the child ends; the reader then sees the end of output.
- ``STARTF_USESTDHANDLES`` with no handles keeps the child on the pseudo console
  even when the host's own stdio is redirected (a service, a test runner).
"""
from __future__ import annotations

import ctypes
import subprocess
import threading
from ctypes import wintypes
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .ptyproc import HANGUP, KILL, TERMINATE, PtyChild, environment_block, windows_command_line

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_SUSPENDED = 0x00000004
STARTF_USESTDHANDLES = 0x00000100
PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x0
WAIT_TIMEOUT = 0x102
ERROR_BROKEN_PIPE = 109
STOPPED_EXIT_CODE = 1


class COORD(ctypes.Structure):
    _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD), ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE),
    ]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


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


_P = ctypes.POINTER
_CreatePipe = _fn("CreatePipe", wintypes.BOOL, _P(wintypes.HANDLE), _P(wintypes.HANDLE), ctypes.c_void_p, wintypes.DWORD)
_CreatePseudoConsole = _fn("CreatePseudoConsole", ctypes.c_long, COORD, wintypes.HANDLE, wintypes.HANDLE,
                           wintypes.DWORD, _P(wintypes.HANDLE))
_ResizePseudoConsole = _fn("ResizePseudoConsole", ctypes.c_long, wintypes.HANDLE, COORD)
_ClosePseudoConsole = _fn("ClosePseudoConsole", None, wintypes.HANDLE)
_InitializeProcThreadAttributeList = _fn("InitializeProcThreadAttributeList", wintypes.BOOL, ctypes.c_void_p,
                                         wintypes.DWORD, wintypes.DWORD, _P(ctypes.c_size_t))
_UpdateProcThreadAttribute = _fn("UpdateProcThreadAttribute", wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD,
                                 ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
_DeleteProcThreadAttributeList = _fn("DeleteProcThreadAttributeList", None, ctypes.c_void_p)
_CreateProcessW = _fn("CreateProcessW", wintypes.BOOL, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                      ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
                      _P(STARTUPINFOEXW), _P(PROCESS_INFORMATION))
_CreateJobObjectW = _fn("CreateJobObjectW", wintypes.HANDLE, ctypes.c_void_p, wintypes.LPCWSTR)
_SetInformationJobObject = _fn("SetInformationJobObject", wintypes.BOOL, wintypes.HANDLE, ctypes.c_int,
                               ctypes.c_void_p, wintypes.DWORD)
_AssignProcessToJobObject = _fn("AssignProcessToJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE)
_TerminateJobObject = _fn("TerminateJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.UINT)
_TerminateProcess = _fn("TerminateProcess", wintypes.BOOL, wintypes.HANDLE, wintypes.UINT)
_ResumeThread = _fn("ResumeThread", wintypes.DWORD, wintypes.HANDLE)
_ReadFile = _fn("ReadFile", wintypes.BOOL, wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                _P(wintypes.DWORD), ctypes.c_void_p)
_WriteFile = _fn("WriteFile", wintypes.BOOL, wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                 _P(wintypes.DWORD), ctypes.c_void_p)
_WaitForSingleObject = _fn("WaitForSingleObject", wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD)
_GetExitCodeProcess = _fn("GetExitCodeProcess", wintypes.BOOL, wintypes.HANDLE, _P(wintypes.DWORD))
_CloseHandle = _fn("CloseHandle", wintypes.BOOL, wintypes.HANDLE)


def _check(ok) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def _close(handle) -> None:
    if handle:
        _CloseHandle(handle)


def _pipe():
    read, write = wintypes.HANDLE(), wintypes.HANDLE()
    _check(_CreatePipe(ctypes.byref(read), ctypes.byref(write), None, 0))
    return read, write


def _kill_on_close_job():
    job = _CreateJobObjectW(None, None)
    _check(job)
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        _close(job)
        raise ctypes.WinError(error)
    return job


def _attributes(console):
    """A thread attribute list that attaches the child to ``console``."""
    size = ctypes.c_size_t(0)
    _InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))  # asks for the size; "fails" by design
    attributes = ctypes.create_string_buffer(size.value)
    _check(_InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)))
    if not _UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, console.value,
                                      ctypes.sizeof(wintypes.HANDLE), None, None):
        error = ctypes.get_last_error()
        _DeleteProcThreadAttributeList(attributes)
        raise ctypes.WinError(error)
    return attributes


def _create_suspended(args: Sequence[str], cwd: Path, env: Optional[Mapping[str, str]], console):
    attributes = _attributes(console)
    try:
        startup = STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        startup.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
        info = PROCESS_INFORMATION()
        command = ctypes.create_unicode_buffer(windows_command_line(args))
        block = ctypes.create_string_buffer(environment_block(env)) if env is not None else None
        flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED
        _check(_CreateProcessW(None, command, None, None, False, flags, block, str(cwd),
                               ctypes.byref(startup), ctypes.byref(info)))
        return info
    finally:
        _DeleteProcThreadAttributeList(attributes)


class ConPtyChild(PtyChild):
    """The child on a pseudo console, inside a job object that holds its whole tree."""

    def __init__(self, args: Sequence[str], *, cwd: Path, env: Optional[Mapping[str, str]], rows: int, cols: int):
        self.args: List[str] = [str(a) for a in args]
        self.returncode: Optional[int] = None
        self._lock = threading.Lock()
        self._console = wintypes.HANDLE()
        self._process = self._job = self._input = self._output = None
        console_input, self._input = _pipe()
        self._output, console_output = _pipe()
        try:
            hr = _CreatePseudoConsole(COORD(cols, rows), console_input, console_output, 0, ctypes.byref(self._console))
            if hr != 0:
                raise OSError(f"CreatePseudoConsole failed (0x{hr & 0xFFFFFFFF:08x}); ConPTY needs Windows 10 1809 or later")
            self._job = _kill_on_close_job()
            self._start(args, cwd, env)
        except BaseException:
            self.close()
            self._release()
            raise
        finally:
            _close(console_input)      # the pseudo console holds its own copies
            _close(console_output)
        threading.Thread(target=self._close_console_on_exit, daemon=True, name=f"conpty-exit-{self.pid}").start()

    def _start(self, args: Sequence[str], cwd: Path, env: Optional[Mapping[str, str]]) -> None:
        info = _create_suspended(args, cwd, env, self._console)
        self._process, self.pid = info.hProcess, int(info.dwProcessId)
        try:
            _check(_AssignProcessToJobObject(self._job, self._process))
            if _ResumeThread(info.hThread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            _TerminateProcess(self._process, STOPPED_EXIT_CODE)
            raise
        finally:
            _close(info.hThread)

    def _close_console_on_exit(self) -> None:
        _WaitForSingleObject(self._process, INFINITE)
        self._close_console()

    def _close_console(self) -> bool:
        with self._lock:
            console, self._console = self._console, wintypes.HANDLE()
        if not console:
            return False
        _ClosePseudoConsole(console)   # attached processes get CTRL_CLOSE_EVENT; output ends
        return True

    def read(self, size: int) -> bytes:
        output = self._output
        if not output:
            return b""
        buffer, got = ctypes.create_string_buffer(size), wintypes.DWORD(0)
        if not _ReadFile(output, buffer, size, ctypes.byref(got), None):
            error = ctypes.get_last_error()
            if error == ERROR_BROKEN_PIPE:
                return b""
            raise ctypes.WinError(error)
        return buffer.raw[:got.value]

    def write(self, data: bytes) -> None:
        pending = bytes(data)
        while pending:
            if not self._input:
                raise OSError("the terminal is closed")
            sent = wintypes.DWORD(0)
            _check(_WriteFile(self._input, pending, len(pending), ctypes.byref(sent), None))
            pending = pending[sent.value:]

    def resize(self, rows: int, cols: int) -> None:
        with self._lock:
            if self._console:
                _ResizePseudoConsole(self._console, COORD(cols, rows))

    def poll(self) -> Optional[int]:
        if self.returncode is None and self._process and _WaitForSingleObject(self._process, 0) == WAIT_OBJECT_0:
            code = wintypes.DWORD(0)
            _check(_GetExitCodeProcess(self._process, ctypes.byref(code)))
            self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        waited = INFINITE if timeout is None else max(0, int(timeout * 1000))
        if _WaitForSingleObject(self._process, waited) == WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.poll()  # type: ignore[return-value]

    def _signal(self, stage: str) -> bool:
        if stage in (HANGUP, TERMINATE):
            self._close_console()      # a second request finds it closed and simply waits
            return True
        if stage == KILL and self._job:
            return bool(_TerminateJobObject(self._job, STOPPED_EXIT_CODE))
        return False

    def close(self) -> None:
        """End the terminal: the console, both pipes, and whatever is still in the job."""
        self._close_console()
        for name in ("_input", "_output", "_job"):
            handle = getattr(self, name)
            setattr(self, name, None)
            _close(handle)

    def _release(self) -> None:
        process, self._process = self._process, None
        _close(process)

    def __del__(self) -> None:
        try:
            self.close()
            self._release()
        except Exception:  # noqa: BLE001 — interpreter shutdown; handles go with the process
            pass


__all__ = ["ConPtyChild"]
