"""#818: the CLI host as a Windows login task — Task Scheduler and a supervisor.

The task definition, the supervisor's restart rules and the service actions are
checked on every platform with ``schtasks`` stubbed. The last tests run only on
Windows (the ``cli-host-windows`` lane): a real job object and a real supervised
host process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from automatos_cli_host import lifecycle, service, winservice

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="a real job object needs Windows")


def _find(root, path):
    return root.find(path, NS).text


def test_the_task_starts_at_logon_as_the_user_with_no_limit_and_never_twice(tmp_path):
    argv = [r"C:\Python\pythonw.exe", "-m", "automatos_cli_host.winservice", "--log", r"C:\s & t\host.log", "--",
            r"C:\Python\python.exe", "-m", "automatos_cli_host", "--allow", r"C:\work\repo"]
    root = ET.fromstring(winservice.task_xml(argv, Path(r"C:\host"), r"CONTOSO\ada"))
    assert _find(root, "t:Triggers/t:LogonTrigger/t:UserId") == r"CONTOSO\ada"
    assert _find(root, "t:Principals/t:Principal/t:RunLevel") == "LeastPrivilege"
    assert _find(root, "t:Principals/t:Principal/t:LogonType") == "InteractiveToken"
    assert _find(root, "t:Settings/t:ExecutionTimeLimit") == "PT0S"          # the default stops it after 72 h
    assert _find(root, "t:Settings/t:MultipleInstancesPolicy") == "IgnoreNew"
    assert _find(root, "t:Settings/t:StopIfGoingOnBatteries") == "false"
    assert _find(root, "t:Actions/t:Exec/t:Command") == argv[0]
    assert _find(root, "t:Actions/t:Exec/t:Arguments") == subprocess.list2cmdline(argv[1:])
    assert _find(root, "t:Actions/t:Exec/t:WorkingDirectory") == r"C:\host"


def test_the_supervisor_restarts_the_host_after_a_non_zero_exit_and_stops_after_zero(tmp_path):
    exits = [75, 1, 0]                    # a drift restart, a crash, then "I need you"
    runs = []

    def run(argv, log):
        runs.append(list(argv))
        log.write(b"run\n")
        return exits[len(runs) - 1]

    assert winservice.supervise(["host"], tmp_path / "host.log", throttle=0, run=run) == 0
    assert runs == [["host"]] * 3
    assert (tmp_path / "host.log").read_bytes() == b"run\n" * 3       # appended, never truncated


def test_the_supervisor_command_line_is_refused_when_malformed():
    assert winservice.main(["--log"]) == 2
    assert winservice.main(["host.log", "--", "x", "y"]) == 2


def _schtasks(monkeypatch, *, query=0):
    calls = []

    def fake(*args):
        calls.append(args)
        return SimpleNamespace(returncode=query if args[0] == "/Query" else 0, stdout="", stderr="")

    monkeypatch.setattr(winservice, "_schtasks", fake)
    return calls


def test_install_registers_the_task_from_utf16_xml_and_starts_it(tmp_path, monkeypatch):
    calls = _schtasks(monkeypatch)
    path = winservice.install(["python", "-m", "automatos_cli_host"], tmp_path / "state", tmp_path)
    assert path.read_bytes()[:2] == b"\xff\xfe"                       # schtasks reads its XML as UTF-16
    assert "automatos_cli_host.winservice" in path.read_text(encoding="utf-16")
    assert calls == [("/Create", "/TN", winservice.TASK_NAME, "/XML", str(path), "/F"), ("/Run", "/TN", winservice.TASK_NAME)]


def test_uninstall_asks_the_host_to_stop_before_removing_the_task(tmp_path, monkeypatch):
    calls = _schtasks(monkeypatch)
    monkeypatch.setattr(winservice, "STOP_WAIT_SECONDS", 0.1)
    monkeypatch.setattr(lifecycle, "pid_alive", lambda pid: True)
    (tmp_path / "host.pid").write_text("4242\n")
    assert winservice.uninstall(tmp_path / "host.pid", tmp_path) is True
    assert lifecycle.stop_request_path(tmp_path).read_text() == "4242"   # the host lets go of its tickets
    assert [c[0] for c in calls] == ["/Query", "/End", "/Delete"]


def test_uninstall_without_a_task_changes_nothing(tmp_path, monkeypatch):
    calls = _schtasks(monkeypatch, query=1)
    assert winservice.uninstall(tmp_path / "host.pid", tmp_path) is False
    assert [c[0] for c in calls] == ["/Query"] and not lifecycle.stop_request_path(tmp_path).exists()


def test_status_reads_running_from_the_hosts_pid_not_schtasks_words(tmp_path, monkeypatch):
    _schtasks(monkeypatch)
    (tmp_path / "host.pid").write_text(f"{os.getpid()}\n")
    st = winservice.status(tmp_path / "host.pid", tmp_path)
    assert st == {"manager": "taskscheduler", "installed": True, "running": True, "pid": os.getpid(),
                  "unit": str(tmp_path / winservice.TASK_FILE)}


def test_restart_drains_a_running_host_and_starts_the_task_otherwise(tmp_path, monkeypatch):
    calls = _schtasks(monkeypatch)
    monkeypatch.setattr(lifecycle, "request_restart", lambda pid_path, state_dir: True)
    assert winservice.restart(tmp_path / "host.pid", tmp_path) is True and calls == []
    monkeypatch.setattr(lifecycle, "request_restart", lambda pid_path, state_dir: False)
    assert winservice.restart(tmp_path / "host.pid", tmp_path) is True
    assert calls == [("/Run", "/TN", winservice.TASK_NAME)]


def test_windows_uses_the_login_task(tmp_path, monkeypatch):
    _schtasks(monkeypatch, query=1)
    monkeypatch.setattr(sys, "platform", "win32")
    cfg = SimpleNamespace(state_dir=tmp_path)
    assert service.status(cfg)["manager"] == "taskscheduler"
    with pytest.raises(RuntimeError, match="state directory"):
        service.uninstall()


@WINDOWS_ONLY
def test_closing_the_job_ends_the_process_in_it():
    from automatos_cli_host import winjob

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job = winjob.kill_on_close_job()
    try:
        winjob.assign_pid(job, child.pid)
    finally:
        winjob.close_handle(job)
    assert child.wait(10) is not None


@WINDOWS_ONLY
def test_the_supervisor_runs_a_real_host_process_in_its_job(tmp_path):
    log = tmp_path / "host.log"
    host = [sys.executable, "-c", "print('host ran')"]
    assert winservice.supervise(host, log, throttle=0) == 0
    assert "host ran" in log.read_text()
