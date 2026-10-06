"""#818: the host stops, restarts and checks on processes without Unix-only calls.

Runs on Linux and macOS and on Windows (the ``cli-host-windows`` lane). Where a
test needs the other platform's behaviour it says so and patches ``sys.platform``
or ``signal``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from automatos_cli_host import lifecycle, terminal_server
from automatos_cli_host.procs import pid_alive

WAIT_SECONDS = 10.0


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_a_running_process_is_alive_and_asking_never_ends_it():
    # On Windows os.kill(pid, 0) would have called TerminateProcess.
    child = _sleeper()
    try:
        for _ in range(3):
            assert pid_alive(child.pid) is True
        assert child.poll() is None, "asking whether it was alive ended it"
    finally:
        child.kill()
        child.wait(WAIT_SECONDS)


def test_a_finished_process_and_no_pid_are_not_alive():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(WAIT_SECONDS)
    assert pid_alive(child.pid) is False
    assert pid_alive(0) is False and pid_alive(-1) is False
    assert pid_alive(os.getpid()) is True


def _host():
    seen = []
    return SimpleNamespace(stop=threading.Event(), stopping=None, request_restart=seen.append, seen=seen)


def test_without_sighup_a_restart_request_is_a_file_the_host_watches(tmp_path, monkeypatch):
    monkeypatch.delattr(signal, "SIGHUP", raising=False)        # as on Windows
    pid_path = tmp_path / "host.pid"
    assert lifecycle.request_restart(pid_path, tmp_path) is False          # no host has written its pid
    lifecycle.restart_request_path(tmp_path).write_text("stale")
    host = _host()
    lifecycle.watch_restart_requests(host, tmp_path)
    try:
        assert not lifecycle.restart_request_path(tmp_path).exists()      # one for a host already gone
        pid_path.write_text(f"{os.getpid()}\n")
        assert lifecycle.request_restart(pid_path, tmp_path) is True
        deadline = time.monotonic() + WAIT_SECONDS
        while not host.seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert host.seen == [lifecycle.NUDGE_REASON]
    finally:
        host.stop.set()


def test_a_request_for_a_host_that_is_not_running_is_refused(tmp_path, monkeypatch):
    monkeypatch.delattr(signal, "SIGHUP", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(WAIT_SECONDS)
    (tmp_path / "host.pid").write_text(f"{child.pid}\n")
    assert lifecycle.request_restart(tmp_path / "host.pid", tmp_path) is False
    assert not lifecycle.restart_request_path(tmp_path).exists()


def test_with_sighup_the_request_is_the_signal_and_nothing_is_watched(tmp_path, monkeypatch):
    if not hasattr(signal, "SIGHUP"):
        monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    sent = []
    monkeypatch.setattr(lifecycle, "pid_alive", lambda pid: True)
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    (tmp_path / "host.pid").write_text("4242\n")
    assert lifecycle.request_restart(tmp_path / "host.pid", tmp_path) is True
    assert sent == [(4242, signal.SIGHUP)]                      # a host on older code understands it too
    assert not lifecycle.restart_request_path(tmp_path).exists()
    assert lifecycle.watch_restart_requests(_host(), tmp_path) is None


def test_the_stop_signals_that_exist_stop_the_host(monkeypatch):
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))
    host = _host()
    lifecycle.install_signal_handlers(host)
    assert signal.SIGINT in installed
    for name in ("SIGTERM", "SIGBREAK", "SIGHUP"):     # each platform has only some of these
        exists = hasattr(signal, name)
        assert (exists and getattr(signal, name) in installed) == exists, name
    installed[signal.SIGINT](signal.SIGINT, None)
    assert host.stop.is_set() and "SIGINT" in host.stopping


def test_windows_opens_powershell_without_a_login_flag(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(terminal_server.shutil, "which", lambda name: r"C:\pwsh\pwsh.exe" if name == "pwsh" else None)
    assert terminal_server.default_shell() == r"C:\pwsh\pwsh.exe"
    assert terminal_server.shell_argv(r"C:\pwsh\pwsh.exe") == [r"C:\pwsh\pwsh.exe"]
    monkeypatch.setattr(terminal_server.shutil, "which", lambda name: None)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    assert terminal_server.default_shell() == r"C:\Windows\System32\cmd.exe"


def test_unix_opens_the_login_shell(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert terminal_server.default_shell() == "/bin/zsh"
    assert terminal_server.shell_argv("/bin/zsh") == ["/bin/zsh", "-l"]


def test_on_windows_a_previous_runs_sessions_are_not_killed_by_pid(tmp_path, monkeypatch):
    # Their job objects ended with the host that started them; the pid may be another process now.
    from automatos_cli_host import state
    from automatos_cli_host.host import Host

    killed = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a), raising=False)
    host = Host.__new__(Host)
    host.cfg = SimpleNamespace(process_table_path=tmp_path / "sessions.json")
    state.save_process_table(host.cfg.process_table_path, {"7": {"pid": os.getpid(), "pgid": os.getpid()}})
    host._reap_previous_run()
    assert killed == []
    assert state.load_process_table(host.cfg.process_table_path) == {}
