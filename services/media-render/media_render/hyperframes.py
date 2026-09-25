"""The Hyperframes CLI (pinned in the image): the commands the service runs.

Every run takes the composition's own scratch directory as its working
directory: the CLI loads a ``.env`` from its cwd, so it must never start in a
directory the service does not control. Each run is its own process group, so
a timeout takes the CLI's Chrome processes down with it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .config import Settings

# The last part of a failed run's output kept for the job report.
LOG_TAIL_CHARS = 4000


@dataclass(frozen=True)
class CliRun:
    argv: tuple
    returncode: int
    seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()[-LOG_TAIL_CHARS:]


def render_argv(settings: Settings, project_dir: Path, output: Path) -> List[str]:
    return [
        settings.hyperframes_bin,
        "render",
        str(project_dir),
        "--output",
        str(output),
        "--quality",
        settings.render_quality,
        "--fps",
        str(settings.render_fps),
        "--workers",
        settings.render_workers,
        # No GPU in the container: skip the probe launch and use software capture.
        "--no-browser-gpu",
    ]


def snapshot_argv(settings: Settings, project_dir: Path, output_dir: Path, at: Sequence[float]) -> List[str]:
    """PNG frames of the composition at exactly these moments (no end frame, no vision model)."""
    return [
        settings.hyperframes_bin,
        "snapshot",
        str(project_dir),
        "--output",
        str(output_dir),
        "--at",
        ",".join(f"{t:g}" for t in at),
        "--no-end",
        "--describe",
        "false",
        "--no-browser-gpu",
    ]


def check_argv(settings: Settings, project_dir: Path) -> List[str]:
    # Software capture here too: the contrast pass samples the same pixels the render will.
    return [settings.hyperframes_bin, "check", str(project_dir), "--json", "--no-browser-gpu"]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_cli(argv: Sequence[str], cwd: Path, timeout_seconds: int, *, capture: bool) -> CliRun:
    """Run one CLI command. ``capture=False`` streams its output to ours."""
    started = time.monotonic()
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen(
        list(argv), cwd=str(cwd), stdout=pipe, stderr=pipe, text=True, errors="replace", start_new_session=True
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        stdout, stderr = proc.communicate()
    except BaseException:
        _kill_group(proc)
        proc.wait()
        raise
    return CliRun(
        argv=tuple(argv),
        returncode=-1 if timed_out else proc.returncode,
        seconds=round(time.monotonic() - started, 3),
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
    )
