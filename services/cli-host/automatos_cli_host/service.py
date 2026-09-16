"""Run the CLI host as a login service (PRD-235 W3, 2026-09-07).

One host per machine serves every ``runtime: cli`` agent of the workspace, so
"agents are always available" means "this process is always running". A
terminal window is not that. ``--install`` writes a launchd LaunchAgent (macOS)
or a systemd user unit (Linux) that starts the host at login, restarts it when
it exits, and logs to the state directory. The host itself exits on purpose
when its code or the backend's contract changed (see ``host.py``), and the
service manager brings it back on the new code.

Everything here is standard library; nothing is installed system-wide.
"""
from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import HostConfig

LAUNCHD_LABEL = "app.automatos.cli-host"
SYSTEMD_UNIT = "automatos-cli-host.service"


def _package_root() -> Path:
    """The directory that holds ``automatos_cli_host/`` (the service's cwd)."""
    return Path(__file__).resolve().parent.parent


def _log_path(cfg: HostConfig) -> Path:
    return cfg.state_dir / "host.log"


def service_argv(cfg: HostConfig, passthrough: Optional[List[str]] = None) -> List[str]:
    """The exact command the service runs — the same host, same directories."""
    argv = [sys.executable, "-m", "automatos_cli_host", "--url", cfg.url, "--dir", str(cfg.state_dir), "--name", cfg.name]
    for d in cfg.allow_dirs:
        argv += ["--allow", str(Path(d).expanduser().resolve())]
    if cfg.default_root:
        argv += ["--default-root", str(Path(cfg.default_root).expanduser().resolve())]
    if cfg.max_sessions > 0:
        argv += ["--max-sessions", str(cfg.max_sessions)]
    for cli_id, path in sorted(cfg.cli_binaries.items()):
        argv += ["--cli-binary", f"{cli_id}={path}"]
    if not cfg.use_worktrees:
        argv.append("--no-worktrees")
    if not cfg.terminal_enabled:
        argv.append("--no-terminal")
    if cfg.terminal_port:
        argv += ["--terminal-port", str(cfg.terminal_port)]
    argv += list(passthrough or [])
    return argv


# ── macOS: launchd ─────────────────────────────────────────────────────────────

def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def _gui_domain() -> str:
    return f"gui/{os.getuid()}"


def install_launchd(cfg: HostConfig, passthrough: Optional[List[str]] = None) -> Path:
    cfg.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": service_argv(cfg, passthrough),
        "WorkingDirectory": str(_package_root()),
        "RunAtLoad": True,
        # Restart whenever the host exits non-zero: a deliberate drift restart
        # (exit 75), a crash, or the backend not being up yet at login. A clean
        # exit (0) stays down — that is the host telling you it needs you
        # (not paired, nothing allowed).
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 15,
        "StandardOutPath": str(_log_path(cfg)),
        "StandardErrorPath": str(_log_path(cfg)),
        # launchd gives agents a bare PATH; the host must find YOUR CLIs.
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                                 "HOME": str(Path.home())},
    }
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", _gui_domain(), str(path))  # replace a previous install, ignore failures
    with path.open("wb") as fh:
        plistlib.dump(plist, fh)
    out = _launchctl("bootstrap", _gui_domain(), str(path))
    if out.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {(out.stderr or out.stdout).strip()}")
    return path


def uninstall_launchd() -> bool:
    path = _plist_path()
    if not path.exists():
        return False
    _launchctl("bootout", _gui_domain(), str(path))
    path.unlink(missing_ok=True)
    return True


def status_launchd() -> Dict[str, Any]:
    path = _plist_path()
    out = _launchctl("print", f"{_gui_domain()}/{LAUNCHD_LABEL}")
    running = out.returncode == 0 and "state = running" in out.stdout
    pid = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("pid = "):
            try:
                pid = int(line.split("=", 1)[1].strip())
            except ValueError:
                pid = None
    return {"manager": "launchd", "installed": path.exists(), "running": running, "pid": pid, "unit": str(path)}


def restart_launchd() -> bool:
    return _launchctl("kickstart", "-k", f"{_gui_domain()}/{LAUNCHD_LABEL}").returncode == 0


# ── Linux: systemd --user ──────────────────────────────────────────────────────

def _unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=False)


def install_systemd(cfg: HostConfig, passthrough: Optional[List[str]] = None) -> Path:
    cfg.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    exec_start = " ".join(shlex.quote(a) for a in service_argv(cfg, passthrough))
    unit = "\n".join([
        "[Unit]",
        "Description=Automatos CLI host — runs board tickets as your own Claude Code sessions",
        "After=network-online.target",
        "",
        "[Service]",
        f"WorkingDirectory={_package_root()}",
        f"ExecStart={exec_start}",
        "Restart=on-failure",
        "RestartSec=15",
        f"Environment=PATH={os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        f"StandardOutput=append:{_log_path(cfg)}",
        f"StandardError=append:{_log_path(cfg)}",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)
    _systemctl("daemon-reload")
    out = _systemctl("enable", "--now", SYSTEMD_UNIT)
    if out.returncode != 0:
        raise RuntimeError(f"systemctl enable --now failed: {(out.stderr or out.stdout).strip()}")
    return path


def uninstall_systemd() -> bool:
    path = _unit_path()
    if not path.exists():
        return False
    _systemctl("disable", "--now", SYSTEMD_UNIT)
    path.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    return True


def status_systemd() -> Dict[str, Any]:
    path = _unit_path()
    active = _systemctl("is-active", SYSTEMD_UNIT).stdout.strip()
    pid = None
    out = _systemctl("show", "-p", "MainPID", "--value", SYSTEMD_UNIT)
    try:
        pid = int(out.stdout.strip()) or None
    except ValueError:
        pid = None
    return {"manager": "systemd", "installed": path.exists(), "running": active == "active", "pid": pid, "unit": str(path)}


def restart_systemd() -> bool:
    return _systemctl("restart", SYSTEMD_UNIT).returncode == 0


# ── dispatch by platform ───────────────────────────────────────────────────────

def _manager() -> str:
    if sys.platform == "darwin":
        return "launchd"
    if shutil.which("systemctl"):
        return "systemd"
    raise RuntimeError("no supported service manager here (launchd on macOS, systemd --user on Linux)")


def install(cfg: HostConfig, passthrough: Optional[List[str]] = None) -> Path:
    return install_launchd(cfg, passthrough) if _manager() == "launchd" else install_systemd(cfg, passthrough)


def uninstall() -> bool:
    return uninstall_launchd() if _manager() == "launchd" else uninstall_systemd()


def status() -> Dict[str, Any]:
    return status_launchd() if _manager() == "launchd" else status_systemd()


def restart() -> bool:
    return restart_launchd() if _manager() == "launchd" else restart_systemd()


def nudge(cfg: HostConfig) -> bool:
    """Ask a running host (service or terminal) to drain and restart: SIGHUP to
    the pid it wrote. Used by ``make up`` after the app rebuilt."""
    pid_path = cfg.state_dir / "host.pid"
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, signal.SIGHUP)
        return True
    except OSError:
        return False
