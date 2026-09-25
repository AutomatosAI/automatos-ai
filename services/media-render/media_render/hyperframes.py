"""The Hyperframes CLI (pinned in the image): the commands the service runs.

Every run takes the composition's own scratch directory as its working
directory: the CLI loads a ``.env`` from its cwd, so it must never start in a
directory the service does not control.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .config import Settings


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


def check_argv(settings: Settings, project_dir: Path) -> List[str]:
    return [settings.hyperframes_bin, "check", str(project_dir), "--json"]


def run_cli(argv: Sequence[str], cwd: Path, timeout_seconds: int, *, capture: bool) -> CliRun:
    """Run one CLI command. ``capture=False`` streams its output to ours."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=capture,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CliRun(
            argv=tuple(argv),
            returncode=-1,
            seconds=round(time.monotonic() - started, 3),
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            timed_out=True,
        )
    return CliRun(
        argv=tuple(argv),
        returncode=proc.returncode,
        seconds=round(time.monotonic() - started, 3),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else ""
