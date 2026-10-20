"""PRD-262: the workspace worker's site-packages, as the image's cuts leave it.

images.yml runs this inside the worker image it built, on each architecture, with the
image's own Python, as the worker user:

    docker run --rm -i --user 1000 --network NET -e PROBE_DATABASE_URL=postgresql://postgres@db:5432/postgres \
        --entrypoint python IMAGE - < scripts/ci/image_worker_site_packages.py

Each check names the row of the 19 Oct size breakdown it proves:

- row 40, the strip: every ``*.so*`` file under site-packages outside an auditwheel
  ``*.libs`` directory (the files the Dockerfile's ``strip --strip-debug`` touched) has
  no ``.debug_*`` section and still loads, each in a fresh interpreter: one that exports
  ``PyInit_<name>`` is imported by its dotted name, any other is opened with ctypes. The
  extensions named in ``MUST_LOAD`` must be among them, so an empty glob never passes;
- row 47, the tests directories: none named exactly ``tests`` is left;
- row 27, psycopg2: built from source against Debian's libpq. psycopg2-binary and its
  bundled libraries are gone, ``_psycopg`` links Debian's libpq.so.5 (libpq 17 or
  later), and psycopg2 connects to the Postgres at ``PROBE_DATABASE_URL`` (the compose
  stack's image, which images.yml starts) and runs a parameterised query;
- the distributions a cut removed (``REMOVED``, each with its row) are not installed,
  and what replaced them imports: redis.asyncio without hiredis's parser, and the
  worker's HTTP server on Starlette and uvicorn with no aiohttp (row 36): its modules
  import from /app and build the app with every route and /metrics.

binutils (readelf) comes with build-essential, which the worker keeps for its agents.
Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

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

# Row 27.
BUNDLED_LIBS_DIR = "psycopg2_binary.libs"
LIBPQ_MIN_VERSION = 170000  # Debian trixie's libpq5 is 17.x
# Debian's, from /usr/lib/<arch> (ldd may print it through the /lib symlink), never a copy
# a wheel bundles under site-packages.
SYSTEM_LIBPQ = re.compile(r"^\s*libpq\.so\.5 => ((?:/usr)?/lib/\S+)", re.M)
PSYCOPG2_PROBE = ("import psycopg2, psycopg2._psycopg as c, psycopg2.extensions as e; "
                  "print(e.libpq_version(), c.__file__)")
DATABASE_URL_ENV = "PROBE_DATABASE_URL"
CONNECT_ATTEMPTS = 30
CONNECT_PAUSE_S = 1.0
CONNECT_TIMEOUT_S = 5
ECHO = "probe"

# Distributions a cut took out of the lock, by the breakdown row that took it.
REMOVED = {
    "psycopg2-binary": "row 27",
    "aiohttp": "row 36",
    "aiohappyeyeballs": "row 36",
    "aiosignal": "row 36",
    "frozenlist": "row 36",
    "multidict": "row 36",
    "propcache": "row 36",
    "yarl": "row 36",
    "hiredis": "row 47",
    "python-dotenv": "row 47",
}
# What the worker imports in their place, each in a fresh interpreter. The worker's app
# has 15 routes and /metrics (worker_http._routes and automatos_metrics).
WORKER_APP_ROUTES = 16
REPLACEMENT_IMPORTS = (
    "import redis.asyncio, redis.utils; assert not redis.utils.HIREDIS_AVAILABLE",
    # PYTHONPATH=/app (the image's ENV) finds the worker's modules from any directory.
    "import types, starlette, uvicorn, worker_http, worker_otel; "
    "app = worker_http.build_app(types.SimpleNamespace(_worker_id='probe', _active_tasks={}, concurrency=1, "
    "_redis=None, health_port=8081, _running=True)); "
    f"assert len(app.routes) == {WORKER_APP_ROUTES}, [route.path for route in app.routes]",
)

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


# --------------------------------------------------------------------------- row 27


def check_psycopg2_build() -> List[str]:
    """No bundled libraries, and _psycopg links Debian's libpq 17."""
    failures = [f"{BUNDLED_LIBS_DIR} is in the image"] if (SITE / BUNDLED_LIBS_DIR).exists() else []
    exit_code, output = run([sys.executable, "-c", PSYCOPG2_PROBE])
    if exit_code:
        return [*failures, f"import psycopg2 failed, exit {exit_code}: {output[-STDERR_TAIL_CHARS:]}"]
    libpq_version, extension = output.split()
    if int(libpq_version) < LIBPQ_MIN_VERSION:
        failures.append(f"psycopg2 runs on libpq {libpq_version}, older than Debian's {LIBPQ_MIN_VERSION}")
    linked = SYSTEM_LIBPQ.search(run(["ldd", extension])[1])
    return failures if linked else [*failures, f"{extension} does not link Debian's libpq.so.5"]


def _connect(psycopg2: Any, url: str) -> Any:
    """A connection to ``url``, retried while the database starts."""
    for _ in range(CONNECT_ATTEMPTS - 1):
        try:
            return psycopg2.connect(url, connect_timeout=CONNECT_TIMEOUT_S)
        except psycopg2.OperationalError:
            time.sleep(CONNECT_PAUSE_S)
    return psycopg2.connect(url, connect_timeout=CONNECT_TIMEOUT_S)


def _echo(psycopg2: Any, url: str) -> str:
    """ECHO, sent through a bound parameter and read back."""
    connection = _connect(psycopg2, url)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT %(word)s::text", {"word": ECHO})
            return cursor.fetchone()[0]
    finally:
        connection.close()


def check_psycopg2_connects() -> List[str]:
    """psycopg2 reaches the database and runs a parameterised query, as the worker's UPDATE does."""
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        return [f"{DATABASE_URL_ENV} is not set: start a Postgres and point it there"]
    import psycopg2

    try:
        echoed = _echo(psycopg2, url)
    except psycopg2.Error as exc:
        return [f"psycopg2 could not query the database: {type(exc).__name__}: {exc}"]
    return [] if echoed == ECHO else [f"the query echoed {echoed!r}, not {ECHO!r}"]


# --------------------------------------------------------------------------- removed distributions


def installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def check_removed() -> List[str]:
    """Each removed distribution is gone, and its replacement imports."""
    failures = [f"{name} ({row}) is still installed" for name, row in REMOVED.items() if installed(name)]
    for code in REPLACEMENT_IMPORTS:
        exit_code, output = run([sys.executable, "-c", code])
        failures += [f"{code!r} failed, exit {exit_code}: {output[-STDERR_TAIL_CHARS:]}"] if exit_code else []
    return failures


# --------------------------------------------------------------------------- the run


CHECKS: Tuple[Tuple[str, Check], ...] = (
    ("row 40: every stripped library loads, with no debug sections", check_strip),
    ("row 47: no tests directories in site-packages", check_tests_dirs),
    ("row 27: psycopg2 is built against Debian's libpq", check_psycopg2_build),
    ("row 27: psycopg2 connects to Postgres and runs a query", check_psycopg2_connects),
    ("rows 27, 36 and 47: the removed distributions are gone, their replacements import", check_removed),
)


def run_check(check: Check) -> List[str]:
    """The check's failures; one that raises is a failure with its message."""
    try:
        return check()
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
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
