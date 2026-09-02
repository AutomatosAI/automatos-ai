"""PRD-209 S6 — exactly one frontend lockfile (deterministic fresh-clone builds).

frontend/ tracked package-lock.json + yarn.lock + pnpm-lock.yaml → a fresh clone's
install is nondeterministic (depends which package manager the developer runs). The
frontend-ci lane uses npm against package-lock.json (test.yml: `cache: npm`,
`cache-dependency-path: frontend/package-lock.json`, `npm install`), so that is the
one kept; the other two are deleted. This guard asserts exactly one lockfile remains.

Pure/static — reads the git index (`git ls-files`); no install, no build.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_LOCKFILE = re.compile(r"(?:^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml)$")
# The one the frontend-ci installer consumes.
_KEPT = "frontend/package-lock.json"


def _tracked_frontend_lockfiles() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "frontend/"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [ln for ln in proc.stdout.splitlines() if _LOCKFILE.search(ln)]


def test_exactly_one_frontend_lockfile():
    locks = _tracked_frontend_lockfiles()
    assert locks == [_KEPT], (
        f"expected exactly one frontend lockfile ({_KEPT}, matching the frontend-ci "
        f"npm installer), found: {locks}"
    )
