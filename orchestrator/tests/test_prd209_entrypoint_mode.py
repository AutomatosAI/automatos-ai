"""PRD-209 S1 — the container entrypoint carries the executable bit in git.

``docker-entrypoint.sh`` is the image ``ENTRYPOINT`` and compose bind-mounts the
repo copy over the image's ``chmod +x`` stub (``docker-compose.yml:188``), so the
*tracked* mode is what the kernel tries to exec. Tracked ``100644`` → the backend
container dies at start before the wait→migrate→seed lifecycle runs. This guard
reads the git index mode (``git ls-files -s``) and asserts ``100755``; it fails on
``100644``. Contrast anchor: ``services/workspace-worker/entrypoint.sh`` is already
``100755`` — the bit commits fine in this repo, the root script simply never got it.

Pure/static — reads the git index only; no boot, no Docker.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENTRYPOINT = "docker-entrypoint.sh"
_WORKER_ENTRYPOINT = "services/workspace-worker/entrypoint.sh"


def _tracked_mode(path: str) -> str:
    """Return the six-digit git index mode for ``path`` (e.g. ``100755``).

    ``git ls-files -s`` prints ``<mode> <object> <stage>\\t<path>`` — the mode is
    the first whitespace-delimited field.
    """
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", path],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"git ls-files failed for {path}: {proc.stderr}"
    out = proc.stdout.strip()
    assert out, f"{path} is not tracked in git"
    return out.split()[0]


def test_prd209_entrypoint_mode():
    mode = _tracked_mode(_ENTRYPOINT)
    assert mode == "100755", (
        f"{_ENTRYPOINT} tracked mode is {mode}, expected 100755 — a non-executable "
        "entrypoint dies at container exec before the lifecycle runs (PRD-209 S1). "
        "Fix: git update-index --chmod=+x docker-entrypoint.sh (mode-only, no content edit)."
    )


def test_prd209_worker_entrypoint_still_executable():
    # Non-vacuity anchor: the executable bit demonstrably commits in this repo,
    # so a 100755 assertion is meaningful, not a platform artefact.
    assert _tracked_mode(_WORKER_ENTRYPOINT) == "100755"
