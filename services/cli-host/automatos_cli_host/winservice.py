"""The CLI host as a Windows login task: Task Scheduler and a supervisor (#818).

launchd's ``KeepAlive`` and systemd's ``Restart=on-failure`` bring the host back
when it exits non-zero (a drift restart, exit 75; a crash; the backend not up yet
at login) and leave it down after a clean exit (0: the host needs you). Task
Scheduler's own restart only covers a task that fails to start, so the task runs
``supervise``:

- it starts the host and appends the host's output to ``host.log``;
- it starts it again 15 s after a non-zero exit, and stops after exit 0;
- it holds the host in a kill-on-close job object (``winjob``), so ending the task
  never leaves a host running that the next start would duplicate.

The task starts at the user's logon, runs as that user without elevation, has no
time limit (Task Scheduler's default stops a task after 72 hours) and never runs
twice. ``--uninstall`` and ``--restart-service`` ask the host to stop or restart
(``lifecycle``), so running sessions finish before anything is ended.
Standard library only; ``schtasks`` ships with Windows.
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

TASK_NAME = "Automatos CLI host"
THROTTLE_SECONDS = 15
STOP_WAIT_SECONDS = 30.0
TASK_FILE = "cli-host-task.xml"
_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
CREATE_NO_WINDOW = 0x08000000


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True, check=False)


def _windowless_python() -> str:
    """``pythonw.exe`` beside this interpreter, so the task opens no console window."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.exists() else sys.executable


def supervisor_argv(host_argv: Sequence[str], log_path: Path) -> List[str]:
    return [_windowless_python(), "-m", "automatos_cli_host.winservice", "--log", str(log_path), "--", *host_argv]


def _element(parent: ET.Element, tag: str, text: Optional[str] = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, f"{{{_NS}}}{tag}", attrs)
    if text is not None:
        node.text = text
    return node


def task_xml(argv: Sequence[str], working_dir: Path, user: str) -> str:
    """The task definition ``schtasks /Create /XML`` takes."""
    ET.register_namespace("", _NS)
    task = ET.Element(f"{{{_NS}}}Task", {"version": "1.2"})
    info = _element(task, "RegistrationInfo")
    _element(info, "Description", "Automatos CLI host: runs board tickets as your own CLI sessions")
    logon = _element(_element(task, "Triggers"), "LogonTrigger")
    _element(logon, "Enabled", "true")
    _element(logon, "UserId", user)
    principal = _element(_element(task, "Principals"), "Principal", id="Author")
    _element(principal, "UserId", user)
    _element(principal, "LogonType", "InteractiveToken")
    _element(principal, "RunLevel", "LeastPrivilege")
    settings = _element(task, "Settings")
    for tag, value in (("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                       ("StopIfGoingOnBatteries", "false"), ("ExecutionTimeLimit", "PT0S"),
                       ("AllowStartOnDemand", "true"), ("Enabled", "true")):
        _element(settings, tag, value)
    run = _element(_element(task, "Actions", Context="Author"), "Exec")
    _element(run, "Command", argv[0])
    _element(run, "Arguments", subprocess.list2cmdline(list(argv[1:])))
    _element(run, "WorkingDirectory", str(working_dir))
    return ET.tostring(task, encoding="unicode")


def _user() -> str:
    domain = os.environ.get("USERDOMAIN")
    return f"{domain}\\{getpass.getuser()}" if domain else getpass.getuser()


def install(host_argv: Sequence[str], state_dir: Path, working_dir: Path) -> Path:
    """Register (or replace) the login task and start it now."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / TASK_FILE
    xml = task_xml(supervisor_argv(host_argv, state_dir / "host.log"), working_dir, _user())
    path.write_text(xml, encoding="utf-16")             # schtasks reads its XML as UTF-16
    out = _schtasks("/Create", "/TN", TASK_NAME, "/XML", str(path), "/F")
    if out.returncode != 0:
        raise RuntimeError(f"schtasks /Create failed: {(out.stderr or out.stdout).strip()}")
    _schtasks("/Run", "/TN", TASK_NAME)
    return path


def _wait_until_gone(pid: int, wait: float, alive: Callable[[int], bool]) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.5)
    return not alive(pid)


def uninstall(pid_path: Path, state_dir: Path) -> bool:
    """Stop the host (it lets go of its tickets), then remove the task."""
    from .lifecycle import request_stop
    from .procs import pid_alive

    if _schtasks("/Query", "/TN", TASK_NAME).returncode != 0:
        return False
    pid = request_stop(pid_path, state_dir)
    if pid is not None:
        _wait_until_gone(pid, STOP_WAIT_SECONDS, pid_alive)
    _schtasks("/End", "/TN", TASK_NAME)                 # the supervisor, and with it anything left in its job
    _schtasks("/Delete", "/TN", TASK_NAME, "/F")
    (state_dir / TASK_FILE).unlink(missing_ok=True)
    return True


def status(pid_path: Path, state_dir: Path) -> Dict[str, Any]:
    """Whether the task exists, and whether its host is running. ``schtasks`` prints
    its state in the machine's language, so running comes from the host's pid."""
    from .lifecycle import running_host_pid

    installed = _schtasks("/Query", "/TN", TASK_NAME).returncode == 0
    pid = running_host_pid(pid_path)
    return {"manager": "taskscheduler", "installed": installed, "running": pid is not None, "pid": pid,
            "unit": str(state_dir / TASK_FILE)}


def restart(pid_path: Path, state_dir: Path) -> bool:
    """Drain and restart a running host; start the task when no host is running."""
    from .lifecycle import request_restart

    if request_restart(pid_path, state_dir):
        return True
    return _schtasks("/Run", "/TN", TASK_NAME).returncode == 0


# ── the supervisor the task runs ───────────────────────────────────────────────

def _run_in_job(host_argv: Sequence[str], log) -> int:
    from . import winjob

    child = subprocess.Popen(list(host_argv), stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
    job = winjob.kill_on_close_job()
    try:
        winjob.assign_pid(job, child.pid)
        return child.wait()
    finally:
        winjob.close_handle(job)


def supervise(host_argv: Sequence[str], log_path: Path, *, throttle: float = THROTTLE_SECONDS,
              run: Optional[Callable[[Sequence[str], Any], int]] = None) -> int:
    """Run the host until it exits 0; after any other exit, wait ``throttle`` and run it again."""
    run = run or _run_in_job
    while True:
        with Path(log_path).open("ab") as log:
            code = run(host_argv, log)
        if code == 0:
            return 0
        time.sleep(throttle)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "--log" or args[2] != "--":
        sys.stderr.write("usage: python -m automatos_cli_host.winservice --log <host.log> -- <host command>\n")
        return 2
    return supervise(args[3:], Path(args[1]))


if __name__ == "__main__":
    sys.exit(main())
