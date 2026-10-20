"""PRD-262 US-032: the API image ships its boot modules' bytecode, and that bytecode is good.

orchestrator/Dockerfile's compile stage compiles the modules
``scripts/boot_modules.py`` lists (the static import closure of ``main``, the routers
``router_manifest.py`` mounts and the entrypoint's seed check) with ``compileall
--invalidation-mode unchecked-hash``, and the production stage copies /app from it as
one layer. CI runs this inside the built image, with the image's own Python and as
the image's user:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_app_bytecode.py

It fails when:

- a ``.pyc`` under /app is not this interpreter's (magic number, cache tag), is not
  unchecked-hash, sits outside a ``__pycache__``, has no source, or carries a source
  hash that is not its source's: Python would run it without comparing, so it must be
  the code it sits beside;
- the compiled modules are not exactly the boot set the image's own
  ``scripts/boot_modules.py`` lists: a module missing from the bytecode, or bytecode
  that came from somewhere else (US-017 keeps a checkout's out of the context);
- ``import main``, in a fresh interpreter in /app, rewrites a shipped ``.pyc``.

The modules ``import main`` compiles at start-up because the static closure does not
reach them (an import inside a function that runs at import time) are reported, not
failed: they still work, and the count shows what a wider boot set would add.

Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

APP_DIR = Path("/app")
BOOT_MODULES = "scripts/boot_modules.py"
APP_IMPORT = "import main"
# Importing the app builds its database engine, which needs the database settings, but
# connects only on a first query: throwaway values with nothing listening, the same as
# scripts/ci/image_site_packages.py gives it. The image itself sets none.
APP_IMPORT_ENV = {
    "POSTGRES_USER": "probe",
    "POSTGRES_PASSWORD": "probe",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "probe",
    "AUTH_EDITION": "local",
    "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000000c1",
}
TIMEOUT_SECONDS = 600
STDERR_TAIL_LINES = 15
SHOWN = 20
HEADER_BYTES = 16
# PEP 552: bit 0 set = hash-based; bit 1 set = check_source. unchecked-hash is 0b01.
UNCHECKED_HASH_FLAGS = 0b01

Snapshot = Dict[Path, Tuple[int, int]]


def pyc_files(app: Path) -> List[Path]:
    return sorted(app.rglob("*.pyc"))


def pyc_problem(pyc: Path, magic: bytes, cache_tag: str) -> Optional[str]:
    """Why one .pyc is not unchecked-hash bytecode of its own source for this interpreter; None when it is."""
    try:
        source = Path(importlib.util.source_from_cache(str(pyc)))
    except ValueError:
        return f"{pyc} is not a __pycache__ entry"
    if f".{cache_tag}." not in pyc.name or not pyc.name.endswith(f".{cache_tag}.pyc"):
        return f"{pyc} is not plain {cache_tag} bytecode"
    if not source.is_file():
        return f"{pyc} has no source ({source})"
    header = pyc.read_bytes()[:HEADER_BYTES]
    if header[:4] != magic:
        return f"{pyc} has another interpreter's magic number"
    if int.from_bytes(header[4:8], "little") != UNCHECKED_HASH_FLAGS:
        return f"{pyc} is not unchecked-hash bytecode (flags {int.from_bytes(header[4:8], 'little'):#04b})"
    if header[8:16] != importlib.util.source_hash(source.read_bytes()):
        return f"{pyc} was compiled from another version of {source.name}"
    return None


def compiled_sources(pycs: Sequence[Path]) -> Set[Path]:
    """The sources the .pyc files belong to (the ones that name one)."""
    found = set()
    for pyc in pycs:
        try:
            found.add(Path(importlib.util.source_from_cache(str(pyc))))
        except ValueError:
            continue
    return found


def boot_set(app: Path) -> Set[Path]:
    """What the image's own boot_modules.py lists. Raises RuntimeError when it fails."""
    done = subprocess.run([sys.executable, str(app / BOOT_MODULES)], cwd=app, capture_output=True,
                          text=True, timeout=TIMEOUT_SECONDS, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{BOOT_MODULES} failed with exit {done.returncode}: {done.stderr.strip()}")
    return {Path(line) for line in done.stdout.splitlines() if line.strip()}


def set_problems(compiled: Set[Path], expected: Set[Path]) -> List[str]:
    """The boot set and the bytecode must be the same files."""
    if not expected:
        return [f"{BOOT_MODULES} listed no module"]
    missing, extra = sorted(expected - compiled), sorted(compiled - expected)
    found = [f"{len(missing)} boot module(s) ship without bytecode: {', '.join(map(str, missing[:SHOWN]))}"] \
        if missing else []
    if extra:
        found.append(f"{len(extra)} .pyc file(s) are not the compile stage's: {', '.join(map(str, extra[:SHOWN]))}")
    return found


def snapshot(app: Path) -> Snapshot:
    return {pyc: (pyc.stat().st_mtime_ns, pyc.stat().st_size) for pyc in pyc_files(app)}


def import_main(app: Path) -> None:
    """``import main`` in a fresh interpreter in ``app``. Raises RuntimeError when it fails."""
    try:
        done = subprocess.run([sys.executable, "-c", APP_IMPORT], cwd=app, capture_output=True, text=True,
                              env={**os.environ, **APP_IMPORT_ENV}, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"`{APP_IMPORT}` did not finish in {TIMEOUT_SECONDS} s") from error
    if done.returncode != 0:
        tail = " | ".join(done.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        raise RuntimeError(f"`{APP_IMPORT}` failed with exit {done.returncode}: {tail}")


def runtime_changes(before: Snapshot, after: Snapshot) -> Tuple[List[Path], List[Path]]:
    """(shipped .pyc that changed, .pyc written at start-up) between two snapshots."""
    rewritten = sorted(pyc for pyc, stamp in before.items() if after.get(pyc, stamp) != stamp)
    return rewritten, sorted(set(after) - set(before))


def check(app: Path) -> List[str]:
    """Every problem with the image's app bytecode; [] when there is none."""
    pycs = pyc_files(app)
    problems = [found for found in (pyc_problem(pyc, importlib.util.MAGIC_NUMBER, sys.implementation.cache_tag)
                                    for pyc in pycs) if found]
    problems += set_problems(compiled_sources(pycs), boot_set(app))
    before = snapshot(app)
    import_main(app)
    rewritten, written = runtime_changes(before, snapshot(app))
    problems += [f"`{APP_IMPORT}` rewrote shipped bytecode: {pyc}" for pyc in rewritten[:SHOWN]]
    print(f"{len(pycs)} .pyc shipped under {app}; `{APP_IMPORT}` compiled {len(written)} more module(s) at start-up")
    if written:
        names = ", ".join(str(importlib.util.source_from_cache(str(pyc))) for pyc in written[:SHOWN])
        print(f"::notice title=App bytecode (PRD-262 US-032)::`{APP_IMPORT}` compiled {len(written)} module(s) "
              f"the static boot set does not reach: {names}")
    return problems


def main() -> int:
    try:
        problems = check(APP_DIR)
    except RuntimeError as error:
        problems = [str(error)]
    for problem in problems:
        print(f"::error title=App bytecode (PRD-262 US-032)::{problem}")
    if not problems:
        print("the boot modules ship as unchecked-hash bytecode of their own source, and import main uses it as is")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
