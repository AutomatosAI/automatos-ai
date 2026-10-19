"""PRD-262: the workspace worker's agent toolchain still works after the image's trims.

Agents compile, clone and run Node in the worker (executor.py ALLOWED_COMMANDS), so a
cut that saves bytes must leave each of those working. images.yml runs this inside the
worker image it built, on each architecture, with the image's own Python, as root
(the paths it checks are root's), before anything else in the container runs npm:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_worker_toolchain.py

Each check names the row of the 19 Oct size breakdown it proves:

- row 16, gcc's lto-dump: nothing named lto-dump is left in /usr/bin, and two units
  compiled with ``-flto`` still link (lto-wrapper, lto1 and the linker plugin) into a
  program that runs;
- row 19, the API's git trim: git's server-side and mail binaries are gone, git-core's
  ``git`` is a link to /usr/bin/git, and a repository can still be made, committed to,
  branched, checked out, cloned and pulled; an HTTPS clone works (git-remote-https), and
  git's credential protocol answers.

Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

COMMAND_TIMEOUT_S = 300
STDERR_TAIL_CHARS = 800

# Row 16: the names gcc's lto-dump goes by (aarch64-linux-gnu-lto-dump-14 on arm64).
LTO_DUMP_GLOB = "/usr/bin/*lto-dump*"
LTO_UNITS = {
    "main.c": '#include <stdio.h>\nint answer(void);\nint main(void) { printf("lto %d\\n", answer()); return 0; }\n',
    "answer.c": "int answer(void) { return 42; }\n",
}
LTO_OUTPUT = "lto 42"

# Row 19: what the Dockerfile's git trim deletes, as orchestrator/Dockerfile's US-019 does.
GIT_REMOVED = (
    "/usr/bin/git-shell",
    "/usr/bin/scalar",
    "/usr/lib/git-core/git-daemon",
    "/usr/lib/git-core/git-http-backend",
    "/usr/lib/git-core/git-http-push",
    "/usr/lib/git-core/git-imap-send",
    "/usr/lib/git-core/git-shell",
    "/usr/lib/git-core/scalar",
)
GIT_CORE_GIT = "/usr/lib/git-core/git"
GIT_BINARY = "/usr/bin/git"
GIT_REMOTE_HTTPS = "/usr/lib/git-core/git-remote-https"
GIT_IDENTITY = ("-c", "user.name=probe", "-c", "user.email=probe@example.invalid")
# A public repository of a few KB that GitHub keeps for exactly this kind of check.
HTTPS_REPO = "https://github.com/octocat/Hello-World.git"
HTTPS_REPO_FILE = "README"
CREDENTIAL_INPUT = "protocol=https\nhost=example.invalid\nusername=probe\npassword=probe-secret\n\n"

Check = Callable[[Path], List[str]]


def run(argv: Sequence[str], cwd: Path, stdin: str = "") -> Tuple[int, str]:
    """(exit code, stdout + stderr) of ``argv``, never raising for a failed command."""
    done = subprocess.run(list(argv), cwd=str(cwd), input=stdin, capture_output=True, text=True,
                          timeout=COMMAND_TIMEOUT_S, check=False)
    return done.returncode, (done.stdout + done.stderr).strip()


def must(argv: Sequence[str], cwd: Path, stdin: str = "") -> str:
    """stdout + stderr of ``argv``; a failed command raises with its output's tail."""
    code, output = run(argv, cwd, stdin)
    if code != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {code}: {output[-STDERR_TAIL_CHARS:]}")
    return output


# --------------------------------------------------------------------------- row 16


def check_lto(work: Path) -> List[str]:
    """No lto-dump, and an -flto build still links and runs."""
    failures = [f"{path} is still in the image" for path in sorted(glob.glob(LTO_DUMP_GLOB))]
    for name, source in LTO_UNITS.items():
        (work / name).write_text(source, encoding="utf-8")
    objects = []
    for name in LTO_UNITS:
        objects.append(name.replace(".c", ".o"))
        must(["gcc", "-flto", "-O2", "-c", name, "-o", objects[-1]], work)
    must(["gcc", "-flto", "-O2", *objects, "-o", "lto-probe"], work)
    output = must(["./lto-probe"], work)
    if output != LTO_OUTPUT:
        failures.append(f"the -flto program printed {output!r}, not {LTO_OUTPUT!r}")
    return failures


# --------------------------------------------------------------------------- row 19


def _git(work: Path, *args: str) -> str:
    return must(["git", *GIT_IDENTITY, *args], work)


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", f"add {name}")


def check_git_trim(work: Path) -> List[str]:
    """The server-side binaries are gone, and git-core's git is a link to /usr/bin/git."""
    failures = [f"{path} is still in the image" for path in GIT_REMOVED if os.path.lexists(path)]
    if not os.path.islink(GIT_CORE_GIT) or os.path.realpath(GIT_CORE_GIT) != GIT_BINARY:
        failures.append(f"{GIT_CORE_GIT} is not a link to {GIT_BINARY}")
    if not os.access(GIT_REMOTE_HTTPS, os.X_OK):
        failures.append(f"{GIT_REMOTE_HTTPS} is missing: HTTPS clones need it")
    return failures


def check_git_local(work: Path) -> List[str]:
    """init, commit, clone, branch, checkout and pull on repositories in ``work``."""
    origin, clone = work / "origin", work / "clone"
    must(["git", "init", "-q", "-b", "main", str(origin)], work)
    _commit(origin, "one.txt", "one\n")
    must(["git", "clone", "-q", f"file://{origin}", str(clone)], work)
    _git(clone, "checkout", "-q", "-b", "feature")
    _commit(clone, "two.txt", "two\n")
    _git(clone, "checkout", "-q", "main")
    _commit(origin, "three.txt", "three\n")
    _git(clone, "pull", "-q", "--no-rebase", "origin", "main")
    expected = {"one.txt", "three.txt"}
    found = {path.name for path in clone.iterdir() if path.suffix == ".txt"}
    return [] if found == expected else [f"the clone's main holds {sorted(found)}, not {sorted(expected)}"]


def check_git_https(work: Path) -> List[str]:
    """An HTTPS clone, through git-remote-https, and git's credential protocol."""
    target = work / "https-clone"
    must(["git", "clone", "-q", "--depth", "1", HTTPS_REPO, str(target)], work)
    failures = [] if (target / HTTPS_REPO_FILE).is_file() else [f"{HTTPS_REPO} cloned without its {HTTPS_REPO_FILE}"]
    answer = must(["git", "credential", "fill"], work, stdin=CREDENTIAL_INPUT)
    if "password=probe-secret" not in answer.splitlines():
        failures.append("git credential fill did not hand back the credential it was given")
    return failures


# --------------------------------------------------------------------------- the run


CHECKS: Tuple[Tuple[str, Check], ...] = (
    ("row 16: no lto-dump, and gcc -flto links", check_lto),
    ("row 19: git's server-side binaries are gone", check_git_trim),
    ("row 19: git init, commit, clone, branch, checkout and pull", check_git_local),
    ("row 19: an HTTPS clone and git's credential protocol", check_git_https),
)


def run_check(title: str, check: Check) -> List[str]:
    """The check's failures; one that raises is a failure with its message."""
    with tempfile.TemporaryDirectory(prefix="worker-toolchain-") as work:
        try:
            return check(Path(work))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return [f"{type(exc).__name__}: {exc}"]


def main() -> int:
    failed = 0
    for title, check in CHECKS:
        failures = run_check(title, check)
        print(f"{'FAIL' if failures else 'PASS'}  {title}")
        for failure in failures:
            print(f"      {failure}")
        failed += bool(failures)
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
