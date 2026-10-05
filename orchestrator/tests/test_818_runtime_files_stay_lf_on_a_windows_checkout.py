"""Issue #818: every file the containers run stays LF on a Windows-style checkout.

A clone with Git for Windows' default core.autocrlf=true wrote the shell
scripts with CRLF. The shebang became "#!/bin/bash\\r", and the backend and the
workspace-worker crash-looped with "exec ... no such file or directory". #819
pinned LF in .gitattributes; until now CI checked only the two entrypoints in
the working tree.

This test reads every tracked file of the kinds the containers execute or load
(shell scripts, Python, Dockerfiles, Makefile, YAML, the env defaults) and fails
on any CRLF. The "Checkout line endings" workflow runs it after a checkout with
core.autocrlf=true, so a dropped or narrowed .gitattributes pin fails there. The
file kinds are listed here, not read from .gitattributes, so deleting a pin
cannot also shrink what is checked.
"""

import subprocess
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
RUNTIME_NAME_PATTERNS = (
    "*.sh",
    "*.py",
    "Dockerfile*",
    "Makefile",
    "*.yml",
    "*.yaml",
    ".env.example",
)
# Every file directly inside these folders (the env defaults compose loads).
RUNTIME_DIRS = ("envs",)
MUST_BE_SWEPT = (
    "orchestrator/docker-entrypoint.sh",
    "services/workspace-worker/entrypoint.sh",
    "Dockerfile.backend",
    "envs/api.defaults",
)
MAX_LISTED = 20


def _is_runtime_file(path: str) -> bool:
    posix = PurePosixPath(path)
    if str(posix.parent) in RUNTIME_DIRS:
        return True
    return any(fnmatch(posix.name, pattern) for pattern in RUNTIME_NAME_PATTERNS)


def _runtime_files() -> list:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True
    ).stdout.decode("utf-8")
    return sorted(path for path in listed.split("\0") if path and _is_runtime_file(path))


def test_the_sweep_covers_the_container_entrypoints():
    swept = set(_runtime_files())
    missing = [path for path in MUST_BE_SWEPT if path not in swept]
    assert not missing, f"the CRLF sweep no longer covers {missing}"


def test_no_runtime_file_has_crlf_in_the_working_tree():
    with_crlf = [path for path in _runtime_files() if b"\r\n" in (REPO / path).read_bytes()]
    assert not with_crlf, (
        f"{len(with_crlf)} file(s) the containers run have CRLF line endings, so a "
        f"Windows checkout breaks the stack (issue #818): {with_crlf[:MAX_LISTED]}. "
        "Pin their pattern to 'text eol=lf' in .gitattributes."
    )
