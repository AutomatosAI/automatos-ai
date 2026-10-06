"""A child process on a pseudo-terminal, on every platform the host runs on (#818).

The host runs each CLI interactively under a terminal it only drains
(``session.py``), and bridges the operator's own terminal (``terminal_server.py``).
On macOS and Linux that terminal is a Unix pty; on Windows it is a ConPTY pseudo
console (``conpty.py``). Both offer the same small surface: read, write, resize,
poll, wait, and stop the child together with everything it started.

``args``, ``pid``, ``poll()``, ``wait()`` and ``returncode`` read like
``subprocess.Popen``'s, so a caller that only watches the process is unchanged.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# How a child is stopped, gentlest first, each stage given its grace in seconds.
# POSIX sends SIGHUP, SIGTERM, then SIGKILL to the child's process group. Windows
# closes the pseudo console for the first two (every attached process gets
# CTRL_CLOSE_EVENT) and ends the job object, the whole tree, for the last.
HANGUP, TERMINATE, KILL = "hangup", "terminate", "kill"
Stage = Tuple[str, float]
_POLL_SECONDS = 0.05

# cmd.exe re-parses a batch file's arguments (``&``, ``|``, ``%``…), and a session's
# arguments carry text the host did not write, so a batch launcher is never run.
_BATCH_SUFFIXES = (".bat", ".cmd")
BATCH_LAUNCHER_REFUSED = (
    "{binary} is a batch file, and cmd.exe would re-read the session's arguments as "
    "commands. Install the CLI's standalone .exe and point the host at it (--cli-binary)."
)


class PtyChild:
    """What both platforms implement. ``read`` returns ``b""`` at the end of output."""

    args: List[str]
    pid: int
    returncode: Optional[int]

    def read(self, size: int) -> bytes:
        raise NotImplementedError

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def resize(self, rows: int, cols: int) -> None:
        raise NotImplementedError

    def poll(self) -> Optional[int]:
        raise NotImplementedError

    def wait(self, timeout: Optional[float] = None) -> int:
        raise NotImplementedError

    def close(self) -> None:
        """Release the terminal. Safe to call more than once."""
        raise NotImplementedError

    def _signal(self, stage: str) -> bool:
        """Deliver one stop stage; False when there is nothing left to deliver it to."""
        raise NotImplementedError

    def stop(self, stages: Sequence[Stage]) -> bool:
        """Stop the child and what it started, one stage at a time, each given its
        grace. True when the child has exited."""
        for stage, grace in stages:
            if self.poll() is not None:
                return True
            if not self._signal(stage):
                break
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if self.poll() is not None:
                    return True
                time.sleep(_POLL_SECONDS)
        return self.poll() is not None


class PosixPty(PtyChild):
    """The child on a Unix pty, as the session leader of its own process group."""

    _SIGNALS = {HANGUP: "SIGHUP", TERMINATE: "SIGTERM", KILL: "SIGKILL"}

    def __init__(self, args: Sequence[str], *, cwd: Path, env: Optional[Mapping[str, str]], rows: int, cols: int):
        import pty

        master, slave = pty.openpty()
        self._master: Optional[int] = master
        _set_window(master, rows, cols)

        def _child_setup() -> None:  # runs in the child after setsid()
            import fcntl
            import termios

            try:
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            except OSError:
                pass

        try:
            self._popen = subprocess.Popen(
                list(args), stdin=slave, stdout=slave, stderr=slave, cwd=str(cwd),
                env=dict(env) if env is not None else None,
                start_new_session=True, preexec_fn=_child_setup, close_fds=True,
            )
        except BaseException:
            self.close()
            raise
        finally:
            os.close(slave)

    @property
    def args(self) -> List[str]:  # type: ignore[override]
        return list(self._popen.args)

    @property
    def pid(self) -> int:  # type: ignore[override]
        return self._popen.pid

    @property
    def returncode(self) -> Optional[int]:  # type: ignore[override]
        return self._popen.returncode

    def read(self, size: int) -> bytes:
        if self._master is None:
            return b""
        return os.read(self._master, size)

    def write(self, data: bytes) -> None:
        if self._master is None:
            raise OSError("the terminal is closed")
        os.write(self._master, data)

    def resize(self, rows: int, cols: int) -> None:
        if self._master is not None:
            _set_window(self._master, rows, cols)

    def poll(self) -> Optional[int]:
        return self._popen.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._popen.wait(timeout)

    def close(self) -> None:
        master, self._master = self._master, None
        if master is not None:
            try:
                os.close(master)
            except OSError:
                pass

    def _signal(self, stage: str) -> bool:
        try:
            os.killpg(self.pid, getattr(signal, self._SIGNALS[stage]))
            return True
        except OSError:
            return False


def _set_window(fd: int, rows: int, cols: int) -> None:
    import fcntl
    import struct
    import termios

    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass


def spawn(args: Sequence[str], *, cwd: Path, env: Optional[Mapping[str, str]], rows: int, cols: int) -> PtyChild:
    """Start ``args`` on a new pseudo-terminal of ``rows`` x ``cols``."""
    if sys.platform == "win32":
        from .conpty import ConPtyChild

        return ConPtyChild(args, cwd=cwd, env=env, rows=rows, cols=cols)
    return PosixPty(args, cwd=cwd, env=env, rows=rows, cols=cols)


# ── Windows parts that need no Windows to test ─────────────────────────────────

def windows_command_line(args: Sequence[str]) -> str:
    """The command line CreateProcessW takes, refusing a batch-file launcher."""
    if not args:
        raise OSError("nothing to run")
    if Path(args[0]).suffix.lower() in _BATCH_SUFFIXES:
        raise OSError(BATCH_LAUNCHER_REFUSED.format(binary=args[0]))
    return subprocess.list2cmdline([str(a) for a in args])


def environment_block(env: Mapping[str, str]) -> bytes:
    """A CREATE_UNICODE_ENVIRONMENT block: ``NAME=value`` entries sorted by name,
    case-insensitively (as Windows expects), each ending in NUL, then a final NUL."""
    entries: Dict[str, Tuple[str, str]] = {}
    for name, value in env.items():
        if not name or "=" in name or "\0" in name or "\0" in str(value):
            raise ValueError(f"not an environment variable: {name!r}")
        entries[name.upper()] = (name, str(value))   # Windows names ignore case: the last one wins
    text = "".join(f"{name}={value}\0" for _key, (name, value) in sorted(entries.items())) + "\0"
    return text.encode("utf-16-le")


__all__ = [
    "BATCH_LAUNCHER_REFUSED", "HANGUP", "KILL", "TERMINATE", "PosixPty", "PtyChild", "Stage",
    "environment_block", "spawn", "windows_command_line",
]
