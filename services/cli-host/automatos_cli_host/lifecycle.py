"""Stopping and restarting the host without Unix-only signals (#818).

- ``install_signal_handlers``: Ctrl+C stops the host everywhere; SIGTERM where
  another process can send it (Unix), Ctrl+Break on Windows. SIGHUP, the restart
  request an operator sends by hand, where it exists.
- A restart request (``--nudge``, which ``make up`` runs after a rebuild) is SIGHUP
  where there is one, which a host on older code understands too. Windows has no
  SIGHUP, and its ``os.kill`` with any other signal would end the host outright, so
  there the request is a file the host watches.
- A stop request (the Windows login task's ``--uninstall``) is a file the same way:
  the host stops as it would on Ctrl+C, letting go of its tickets, and exits 0, so
  its supervisor leaves it down.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
from pathlib import Path
from typing import Any, Optional

from .procs import pid_alive

log = logging.getLogger("automatos.cli_host.lifecycle")

RESTART_REQUEST_FILE = "restart.request"
STOP_REQUEST_FILE = "stop.request"
WATCH_SECONDS = 1.0
NUDGE_REASON = "restart requested (make up / --nudge)"


def restart_request_path(state_dir: Path) -> Path:
    return Path(state_dir) / RESTART_REQUEST_FILE


def stop_request_path(state_dir: Path) -> Path:
    return Path(state_dir) / STOP_REQUEST_FILE


def install_signal_handlers(host: Any) -> None:
    """Stop on Ctrl+C / SIGTERM / Ctrl+Break; restart on SIGHUP where there is one."""
    def _stop(signum: int, _frame: Any) -> None:
        host.stopping = f"the CLI host on {socket.gethostname()} stopped ({signal.Signals(signum).name})"
        host.stop.set()

    def _restart(_signum: int, _frame: Any) -> None:
        host.request_restart("SIGHUP")

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _restart)


def watch_restart_requests(host: Any, state_dir: Path) -> Optional[threading.Thread]:
    """Where there is no SIGHUP, turn the request files into ``host.request_restart``
    and a stop, until the host stops. Requests left from before this host started
    are dropped: they were for the host that is already gone."""
    if hasattr(signal, "SIGHUP"):
        return None
    restart, stop = restart_request_path(state_dir), stop_request_path(state_dir)
    restart.unlink(missing_ok=True)
    stop.unlink(missing_ok=True)

    def _watch() -> None:
        while not host.stop.wait(WATCH_SECONDS):
            if stop.exists():
                stop.unlink(missing_ok=True)
                host.stopping = f"the CLI host on {socket.gethostname()} stopped (stop requested)"
                host.stop.set()
            elif restart.exists():
                restart.unlink(missing_ok=True)
                host.request_restart(NUDGE_REASON)

    thread = threading.Thread(target=_watch, name="automatos-restart-requests", daemon=True)
    thread.start()
    return thread


def running_host_pid(pid_path: Path) -> Optional[int]:
    """The pid the running host wrote, while it is still running."""
    try:
        pid = int(Path(pid_path).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def request_stop(pid_path: Path, state_dir: Path) -> Optional[int]:
    """Ask the running host to stop; its pid, or None when no host is running."""
    pid = running_host_pid(pid_path)
    if pid is not None:
        stop_request_path(state_dir).write_text(str(pid), encoding="utf-8")
    return pid


def request_restart(pid_path: Path, state_dir: Path) -> bool:
    """Ask the running host to drain and restart. False when no host is running."""
    pid = running_host_pid(pid_path)
    if pid is None:
        return False
    if hasattr(signal, "SIGHUP"):
        try:
            os.kill(pid, signal.SIGHUP)
        except OSError:
            return False
    else:
        restart_request_path(state_dir).write_text(str(pid), encoding="utf-8")
    log.info("asked host %s to restart", pid)
    return True


__all__ = [
    "NUDGE_REASON", "RESTART_REQUEST_FILE", "STOP_REQUEST_FILE", "install_signal_handlers", "request_restart",
    "request_stop", "restart_request_path", "running_host_pid", "stop_request_path", "watch_restart_requests",
]
