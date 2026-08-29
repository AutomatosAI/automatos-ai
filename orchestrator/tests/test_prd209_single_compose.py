"""PRD-209 S9 — exactly one canonical compose file, repo-wide.

The root ``docker-compose.yml`` is the local stack. Six ``infrastructure/docker-compose*.yml``
files (a heavyweight 19-service production-mirror requiring sibling repos; ``.voice``
referenced services decommissioned by #625) predated it and drifted — a second source
of compose truth. They are deleted; this guard asserts exactly one tracked
``docker-compose*.yml`` remains anywhere in the repo.

Pure/static — reads the git index (`git ls-files`); no Docker.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = re.compile(r"(?:^|/)docker-compose[^/]*\.ya?ml$")


def _tracked_compose_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [ln for ln in proc.stdout.splitlines() if _COMPOSE.search(ln)]


def test_exactly_one_tracked_compose_file():
    composes = _tracked_compose_files()
    assert composes == ["docker-compose.yml"], (
        f"expected exactly one canonical compose file (root docker-compose.yml), found: {composes}"
    )
