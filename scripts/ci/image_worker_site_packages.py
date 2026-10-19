"""PRD-262: the workspace worker's site-packages, as the image's cuts leave it.

images.yml runs this inside the worker image it built, on each architecture, with the
image's own Python, as the worker user:

    docker run --rm -i --user 1000 --entrypoint python IMAGE - < scripts/ci/image_worker_site_packages.py

Each check names the row of the 19 Oct size breakdown it proves:

- row 40, the strip: every ``*.so*`` file under site-packages outside an auditwheel
  ``*.libs`` directory (the files the Dockerfile's ``strip --strip-debug`` touched) has
  no ``.debug_*`` section and still loads, each in a fresh interpreter: one that exports
  ``PyInit_<name>`` is imported by its dotted name, any other is opened with ctypes. The
  extensions named in ``MUST_LOAD`` must be among them, so an empty glob never passes;
- row 47, the tests directories: none named exactly ``tests`` is left.

binutils (readelf) comes with build-essential, which the worker keeps for its agents.
Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Callable, List, Optional, Tuple

SITE = Path(sysconfig.get_paths()["purelib"])
COMMAND_TIMEOUT_S = 120
STDERR_TAIL_CHARS = 600

# Row 40: what the Dockerfile's strip names, and what it leaves alone.
LIBRARY_GLOB = "*.so*"
AUDITWHEEL_SUFFIX = ".libs"
DEBUG_PREFIXES = (".debug_", ".zdebug_")
SECTION_NAME = re.compile(r"^\s*\[\s*\d+\]\s+(\S+)", re.M)
# Extensions the worker's lock always installs: the strip must have reached them.
MUST_LOAD = frozenset({
    "_cffi_backend",
    "cryptography.hazmat.bindings._rust",
    "pydantic_core._pydantic_core",
    "rpds.rpds",
})

# Row 47.
TESTS_DIR = "tests"

Check = Callable[[], List[str]]


def run(argv: List[str]) -> Tuple[int, str]:
    """(exit code, stdout + stderr) of ``argv``, run from /, never raising for a failed command."""
    done = subprocess.run(argv, cwd="/", capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S, check=False)
    return done.returncode, (done.stdout + done.stderr).strip()


def readelf(flag: str, path: Path) -> str:
    code, output = run(["readelf", flag, "-W", str(path)])
    if code != 0:
        raise RuntimeError(f"readelf {flag} {path} exited {code}: {output[-STDERR_TAIL_CHARS:]}")
    return output


# --------------------------------------------------------------------------- row 40


def stripped_files() -> List[Path]:
    """Every file the Dockerfile's strip touched: *.so* outside the *.libs directories."""
    return sorted(
        path for path in SITE.rglob(LIBRARY_GLOB)
        if path.is_file() and not any(part.endswith(AUDITWHEEL_SUFFIX) for part in path.relative_to(SITE).parts[:-1])
    )


def module_name(path: Path) -> str:
    """The dotted name an extension at ``path`` imports as (``rpds/rpds.cpython-312-….so`` → ``rpds.rpds``)."""
    relative = path.relative_to(SITE)
    return ".".join([*relative.parts[:-1], relative.name.split(".", 1)[0]])


def load_error(path: Path) -> Tuple[Optional[str], bool]:
    """(the error loading ``path`` in a fresh interpreter or None, whether it is an extension)."""
    name = module_name(path)
    extension = re.search(rf"\bPyInit_{re.escape(name.rsplit('.', 1)[-1])}\b", readelf("--dyn-syms", path))
    code = (f"import importlib; importlib.import_module({name!r})" if extension
            else f"import ctypes; ctypes.CDLL({str(path)!r})")
    exit_code, output = run([sys.executable, "-c", code])
    return (None if exit_code == 0 else f"exit {exit_code}: {output[-STDERR_TAIL_CHARS:]}"), bool(extension)


def check_strip() -> List[str]:
    """No debug sections, and every stripped file still loads."""
    failures: List[str] = []
    loaded = set()
    for path in stripped_files():
        relative = path.relative_to(SITE)
        debug = [name for name in SECTION_NAME.findall(readelf("-S", path)) if name.startswith(DEBUG_PREFIXES)]
        failures += [f"{relative} still has {', '.join(debug)}"] if debug else []
        error, extension = load_error(path)
        failures += [f"{relative} does not load: {error}"] if error else []
        loaded |= {module_name(path)} if extension and not error else set()
    failures += [f"{name} was not found among the stripped extensions" for name in sorted(MUST_LOAD - loaded)]
    return failures


# --------------------------------------------------------------------------- row 47


def check_tests_dirs() -> List[str]:
    """No directory named exactly ``tests`` under site-packages."""
    return [f"{path.relative_to(SITE)} is in the image" for path in sorted(SITE.rglob(TESTS_DIR)) if path.is_dir()]


# --------------------------------------------------------------------------- the run


CHECKS: Tuple[Tuple[str, Check], ...] = (
    ("row 40: every stripped library loads, with no debug sections", check_strip),
    ("row 47: no tests directories in site-packages", check_tests_dirs),
)


def run_check(check: Check) -> List[str]:
    """The check's failures; one that raises is a failure with its message."""
    try:
        return check()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return [f"{type(exc).__name__}: {exc}"]


def main() -> int:
    failed = 0
    print(f"site-packages: {SITE}, {len(stripped_files())} stripped libraries")
    for title, check in CHECKS:
        failures = run_check(check)
        print(f"{'FAIL' if failures else 'PASS'}  {title}")
        for failure in failures:
            print(f"      {failure}")
        failed += bool(failures)
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
