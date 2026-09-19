"""Schedule the night on macOS with launchd (the CLI host's own pattern).

    python3 -m tests.sim.schedule install --hour 1 --minute 30 [--pack auto]
    python3 -m tests.sim.schedule status | uninstall | run-now

Writes ``~/Library/LaunchAgents/app.automatos.sim.plist``: at the given time
launchd runs ``caffeinate -i <this python> -m tests.sim.night run --pack <pack>``
from the repo root, with stdout/stderr in ``~/.automatos-sim/logs/launchd.log``.
The interpreter and the docker directory are resolved at install time and
written in absolute form because launchd's PATH is minimal.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from .config import LOGS_DIR, REPO_ROOT, ensure_dirs

LABEL = "app.automatos.sim"
CAFFEINATE = "/usr/bin/caffeinate"
BASE_PATH = ("/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin")


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _gui_domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def _path_env() -> str:
    docker = shutil.which("docker")
    dirs = ([str(Path(docker).parent)] if docker else []) + list(BASE_PATH)
    return ":".join(dict.fromkeys(dirs))


def build_plist(pack: str, hour: int, minute: int, python: str = sys.executable, repo_root: Path = REPO_ROOT,
                extra: tuple[str, ...] = ()) -> dict:
    program = [python, "-m", "tests.sim.night", "run", "--pack", pack, *extra]
    if Path(CAFFEINATE).exists():
        program = [CAFFEINATE, "-i", *program]
    return {
        "Label": LABEL,
        "ProgramArguments": program,
        "WorkingDirectory": str(repo_root),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "RunAtLoad": False,
        "StandardOutPath": str(LOGS_DIR / "launchd.log"),
        "StandardErrorPath": str(LOGS_DIR / "launchd.log"),
        "EnvironmentVariables": {"PATH": _path_env(), "PYTHONUNBUFFERED": "1"},
    }


def install(pack: str, hour: int, minute: int, extra: tuple[str, ...] = ()) -> Path:
    ensure_dirs()
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", _gui_domain(), str(path))  # replace a previous install; failure is fine
    with path.open("wb") as fh:
        plistlib.dump(build_plist(pack, hour, minute, extra=extra), fh)
    out = _launchctl("bootstrap", _gui_domain(), str(path))
    if out.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {(out.stderr or out.stdout).strip()}")
    return path


def uninstall() -> bool:
    path = plist_path()
    if not path.exists():
        return False
    _launchctl("bootout", _gui_domain(), str(path))
    path.unlink()
    return True


def status() -> str:
    out = _launchctl("print", f"{_gui_domain()}/{LABEL}")
    if out.returncode != 0:
        return "not installed"
    keep = [line.strip() for line in out.stdout.splitlines()
            if any(k in line for k in ("state =", "last exit code", "program =", "run interval", "runs ="))]
    return "\n".join(keep) or out.stdout[:400]


def run_now() -> bool:
    return _launchctl("kickstart", "-k", f"{_gui_domain()}/{LABEL}").returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tests.sim.schedule", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="install or replace the nightly job")
    inst.add_argument("--hour", type=int, default=1)
    inst.add_argument("--minute", type=int, default=30)
    inst.add_argument("--pack", default="auto")
    inst.add_argument("--budget", type=float, help="passed through to night run --budget")
    sub.add_parser("uninstall")
    sub.add_parser("status")
    sub.add_parser("run-now", help="kick the job off immediately (same as the schedule would)")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("launchd scheduling is macOS only; use cron elsewhere: python3 -m tests.sim.night run --pack auto", file=sys.stderr)
        return 2
    if args.command == "install":
        extra = ("--budget", str(args.budget)) if args.budget else ()
        path = install(args.pack, args.hour, args.minute, extra)
        print(f"installed {path}\nfires daily at {args.hour:02d}:{args.minute:02d} · pack {args.pack} · log {LOGS_DIR / 'launchd.log'}")
    elif args.command == "uninstall":
        print("removed" if uninstall() else "not installed")
    elif args.command == "status":
        print(status())
    elif args.command == "run-now":
        print("kicked" if run_now() else "kickstart failed (is it installed?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
