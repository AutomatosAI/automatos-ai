"""PRD-209 S9 — exactly one canonical compose file (+ one documented override).

The root ``docker-compose.yml`` is the local stack. Six ``infrastructure/docker-compose*.yml``
files (a heavyweight 19-service production-mirror requiring sibling repos; ``.voice``
referenced services decommissioned by #625) predated it and drifted — a second source
of compose truth. They are deleted; this guard asserts one tracked STACK file remains anywhere in
the repo. ``docker-compose.dev.yml`` (PRD-233 slim pass) is allowed alongside it
because it is an OVERRIDE, not a second stack: compose only reads it when it is
passed explicitly (``-f docker-compose.yml -f docker-compose.dev.yml``), and the
guard below proves it declares no service the canonical file does not, and
carries no ``image:`` of its own — so it can never become a rival source of
truth the way the infrastructure/ files did.

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


_ALLOWED_OVERRIDES = {"docker-compose.dev.yml"}


def test_exactly_one_tracked_stack_file():
    composes = _tracked_compose_files()
    stacks = [c for c in composes if c not in _ALLOWED_OVERRIDES]
    assert stacks == ["docker-compose.yml"], (
        f"expected exactly one canonical stack file (root docker-compose.yml), found: {stacks}. "
        f"A second stack is the drift PRD-209 S9 deleted; a documented override belongs in "
        f"{sorted(_ALLOWED_OVERRIDES)}."
    )


def test_the_dev_override_is_only_an_override():
    """It may retarget/mount existing services — never define new ones or pin images."""
    import yaml

    canonical = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
    for name in sorted(_ALLOWED_OVERRIDES):
        path = _REPO_ROOT / name
        if not path.exists():
            continue
        override = yaml.safe_load(path.read_text()) or {}
        assert set(override) <= {"services", "volumes", "networks"}, (
            f"{name} may only override services/volumes/networks, found {sorted(override)}"
        )
        unknown = set(override.get("services") or {}) - set(canonical.get("services") or {})
        assert not unknown, f"{name} defines services absent from the canonical stack: {sorted(unknown)}"
        for svc, body in (override.get("services") or {}).items():
            assert "image" not in (body or {}), (
                f"{name}:{svc} pins an image — overrides must not choose what the stack runs"
            )
