"""PRD-262 US-020: the API image's site-packages still loads after pybuild's prune.

orchestrator/Dockerfile's pybuild stage strips the debug sections from every compiled
library it installs (``strip --strip-debug``, leaving the auditwheel ``*.libs``
directories and ``.dynsym`` alone) and deletes every directory named exactly ``tests``
under site-packages. CI runs this inside the built image, with the image's own Python
and as the image's user:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_site_packages.py

It finds every ``*.so*`` file under site-packages outside a ``*.libs`` directory, the
files the strip touched, and reads each one with the ELF parser below (the image has no
binutils):

- one that exports ``PyInit_<name>`` from ``.dynsym`` is an extension module, and is
  imported by its dotted name;
- any other is a plain library that a package opens through ctypes (for example
  ``pypdfium2_raw/libpdfium.so``), and is opened with ``ctypes.CDLL``;
- none may carry a ``.debug_*`` section.

The failures must be exactly the ``EXPECTED_FAILURES`` whose files this image has: any
other ImportError or OSError fails, and so does a listed one that now loads, so the
list stays true. It also checks that no ``tests`` directory is left and that the
``testing`` packages other code imports (numpy, SQLAlchemy, alembic) are still there,
then runs ``import main`` in a fresh interpreter in /app, with throwaway database
settings (nothing connects on import), under ``python -v``: no line may say ``bytecode is
stale`` (PRD-262 US-022: images.yml's reproducible build clamps every file's mtime, and
only unchecked-hash bytecode ignores that). The probe's own imports read site-packages
only, so the app's directories (``alembic/``) shadow nothing.

Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import ctypes
import faulthandler
import gc
import importlib
import mmap
import os
import struct
import subprocess
import sys
import sysconfig
import types
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# What fails to load in this image today, each with its reason. A file the image does
# not have is not expected to fail: Railway's [leiden] build has packages the GHCR
# build lacks. Keys are the dotted module of an extension, or the site-packages path
# of a plain library.
EXPECTED_FAILURES: Dict[str, str] = {
    "PIL._imagingtk": (
        "Pillow's Tk bridge imports tkinter, and python:slim ships _tkinter without the Tcl/Tk "
        "libraries it links. Nothing in the app draws to a Tk window."
    ),
    "matplotlib.backends._tkagg": (
        "Railway's [leiden] build only (graspologic brings matplotlib). matplotlib's Tk backend "
        "loads tkinter's functions when imported, and _tkinter can't load without Tcl/Tk. "
        "Nothing renders through Tk."
    ),
    "numba.np.ufunc.tbbpool": (
        "Railway's [leiden] build only (umap-learn, through graspologic, brings numba). numba's "
        "TBB threading layer links libtbb.so.12, which only the optional tbb package installs; "
        "numba falls back to its OpenMP or workqueue layer."
    ),
}
LIBRARY_GLOB = "*.so*"  # the files orchestrator/Dockerfile's strip names
AUDITWHEEL_SUFFIX = ".libs"  # auditwheel's vendored libraries: not stripped, not probed
DEBUG_PREFIXES = (".debug_", ".zdebug_")
TESTS_DIR = "tests"
KEPT_TESTING = ("numpy/testing", "sqlalchemy/testing", "alembic/testing")
APP_DIR = "/app"
APP_IMPORT = "import main"
# Importing the app builds its database engine (core/database/database.py), which needs
# the database settings, but connects only on a first query. These are throwaway values
# with nothing listening, the local edition's as the backend test job sets them; the
# image itself sets none (compose and Railway do).
APP_IMPORT_ENV = {
    "POSTGRES_USER": "probe",
    "POSTGRES_PASSWORD": "probe",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "probe",
    "AUTH_EDITION": "local",
    "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000000c1",
}
APP_IMPORT_TIMEOUT_S = 600
STDERR_TAIL_LINES = 15
# PRD-262 US-022: the import runs under `python -v`, which reports each .pyc it finds
# out of date with this line. pybuild and the production stage compile with
# unchecked-hash, so the clamped mtimes of images.yml's reproducible build can't make
# one stale; one that is would be recompiled by every process (the automatos user can't
# write /usr/local).
STALE_BYTECODE = "bytecode is stale"
STALE_LISTED = 5
MAPS = Path("/proc/self/maps")

ELF_MAGIC = b"\x7fELF"
EI_CLASS, EI_DATA, EI_NIDENT = 4, 5, 16
SHT_DYNSYM = 11
SHN_UNDEF = 0
SHN_XINDEX = 0xFFFF
EXPORTED_BINDINGS = frozenset({1, 2, 10})  # STB_GLOBAL, STB_WEAK, STB_GNU_UNIQUE
EXPORTED_VISIBILITY = frozenset({0, 3})  # STV_DEFAULT, STV_PROTECTED


class _Format(NamedTuple):
    """struct formats for one ELF class: the header after e_ident, a section header, a symbol."""

    header: str
    section: str
    symbol: str
    symbol_fields: Tuple[int, int, int, int]  # where st_name, st_info, st_other, st_shndx sit


FORMATS = {
    1: _Format("HHIIIIIHHHHHH", "IIIIIIIIII", "IIIBBH", (0, 3, 4, 5)),  # ELFCLASS32
    2: _Format("HHIQQQIHHHHHH", "IIQQQQIIQQ", "IBBHQQ", (0, 1, 2, 3)),  # ELFCLASS64
}
BYTE_ORDER = {1: "<", 2: ">"}  # ELFDATA2LSB, ELFDATA2MSB


class Section(NamedTuple):
    name: str
    kind: int
    offset: int
    size: int
    link: int
    entsize: int


class Elf(NamedTuple):
    format: _Format
    order: str
    sections: Tuple[Section, ...]


class Result(NamedTuple):
    key: str  # the dotted module of an extension, the site-packages path of anything else
    kind: str  # extension, library or unreadable
    error: Optional[str]
    debug: Tuple[str, ...]


# --------------------------------------------------------------------------- ELF


def _unpack(data, fmt: str, offset: int) -> tuple:
    if offset < 0 or offset + struct.calcsize(fmt) > len(data):
        raise ValueError(f"truncated: {struct.calcsize(fmt)} bytes at offset {offset} run past the end")
    return struct.unpack_from(fmt, data, offset)


def _string(data, offset: int) -> str:
    end = data.find(b"\0", offset) if 0 <= offset < len(data) else -1
    if end < 0:
        raise ValueError(f"no string at offset {offset}")
    return bytes(data[offset:end]).decode("utf-8", "replace")


def parse(data) -> Elf:
    """The section headers of the ELF file in ``data`` (bytes or an mmap), names resolved."""
    if bytes(data[:4]) != ELF_MAGIC or len(data) < EI_NIDENT:
        raise ValueError("not an ELF file")
    fmt, order = FORMATS.get(data[EI_CLASS]), BYTE_ORDER.get(data[EI_DATA])
    if fmt is None or order is None:
        raise ValueError(f"unknown ELF class {data[EI_CLASS]} or byte order {data[EI_DATA]}")
    header = _unpack(data, order + fmt.header, EI_NIDENT)
    shoff, shentsize, count, names_at = header[5], header[10], header[11], header[12]
    if shoff == 0:
        raise ValueError("no section headers")
    first = _unpack(data, order + fmt.section, shoff)
    count = count or first[5]  # more than 0xff00 sections: the count is in section 0's sh_size
    names_at = first[6] if names_at == SHN_XINDEX else names_at
    raw = [_unpack(data, order + fmt.section, shoff + index * shentsize) for index in range(count)]
    if names_at >= len(raw):
        raise ValueError(f"section name table {names_at} out of {len(raw)}")
    table = raw[names_at][4]
    sections = tuple(Section(_string(data, table + row[0]), row[1], row[4], row[5], row[6], row[9])
                     for row in raw)
    return Elf(fmt, order, sections)


def debug_sections(elf: Elf) -> Tuple[str, ...]:
    """The names of ``elf``'s debug sections (``.debug_*``, and the old compressed ``.zdebug_*``)."""
    return tuple(section.name for section in elf.sections if section.name.startswith(DEBUG_PREFIXES))


def _name_offsets(data, strings: Section, symbol: str) -> frozenset:
    """Every offset in a string table at which ``symbol`` sits as a whole string."""
    wanted, start, end = symbol.encode() + b"\0", strings.offset, strings.offset + strings.size
    found, at = [], data.find(wanted, start, end)
    while at >= 0:
        found.append(at - start)
        at = data.find(wanted, at + 1, end)
    return frozenset(found)


def exports(data, elf: Elf, symbol: str) -> bool:
    """True when ``.dynsym`` defines ``symbol`` and exports it: global, weak or unique binding,
    default or protected visibility. That is what the loader's ``dlsym`` finds."""
    width = struct.calcsize(elf.order + elf.format.symbol)
    name_at, info_at, other_at, index_at = elf.format.symbol_fields
    for table in (section for section in elf.sections if section.kind == SHT_DYNSYM):
        step = table.entsize or width
        if table.link >= len(elf.sections) or step < width:
            raise ValueError(f".dynsym's string table {table.link} or entry size {table.entsize} is wrong")
        names = _name_offsets(data, elf.sections[table.link], symbol)
        if not names:
            continue
        rows = (_unpack(data, elf.order + elf.format.symbol, table.offset + index * step)
                for index in range(table.size // step))
        if any(row[name_at] in names and row[index_at] != SHN_UNDEF and row[info_at] >> 4 in EXPORTED_BINDINGS
               and row[other_at] & 3 in EXPORTED_VISIBILITY for row in rows):
            return True
    return False


def module_name(relative: PurePosixPath) -> str:
    """The module a file under site-packages loads as. Pure.

    ``PIL/_imaging.cpython-311-x86_64-linux-gnu.so`` is ``PIL._imaging``. A compiled
    package body (mypyc's ``chardet/models/__init__.cpython-311-….so``) is the package,
    ``chardet.models``, whose init function Python looks up as ``PyInit_models``.
    """
    stem = relative.name.split(".", 1)[0]
    package_body = stem == "__init__" and bool(relative.parent.parts)
    return ".".join([*relative.parent.parts, *([] if package_body else [stem])])


# --------------------------------------------------------------------------- the image


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["platlib"])


def native_files(site: Path) -> List[Path]:
    """Every regular ``*.so*`` file under ``site`` outside a ``*.libs`` directory, sorted."""
    return sorted(path for path in site.rglob(LIBRARY_GLOB)
                  if path.is_file() and not path.is_symlink()
                  and not any(part.endswith(AUDITWHEEL_SUFFIX) for part in path.relative_to(site).parts[:-1]))


def _inspect(path: Path, symbol: str) -> Tuple[bool, Tuple[str, ...]]:
    """(whether the file exports ``symbol``, its debug sections), read without loading it."""
    with path.open("rb") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            raise ValueError("empty file")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            elf = parse(data)
            return exports(data, elf, symbol), debug_sections(elf)


def _mapped(path: Path) -> bool:
    """True when the dynamic loader already has ``path`` open in this process."""
    return f" {os.path.realpath(path)}\n" in MAPS.read_text(encoding="utf-8", errors="replace")


def _module_built_from(path: Path) -> bool:
    """True when a live module object came from ``path``, under whatever name its package gave it."""
    real = os.path.realpath(path)
    return any(isinstance(obj, types.ModuleType) and isinstance(getattr(obj, "__file__", None), str)
               and os.path.realpath(obj.__file__) == real for obj in gc.get_objects())


def _described(error: BaseException) -> str:
    """``Type: message`` on one line, so it fits a workflow annotation. Pure."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


def load_extension(name: str, path: Path) -> Optional[str]:
    """Import ``name``; None when it loads, else the error.

    The parent package comes first. A package that loads its extension under another
    name (opencv's cv2 loads ``cv2/cv2.abi3.so`` as the top-level ``cv2``) has already
    run its init once; a second init under the file's own name is not what the package
    does, so an extension the loader has open, with a live module built from it, counts
    as loaded.
    """
    try:
        parent = name.rpartition(".")[0]
        if parent:
            importlib.import_module(parent)
        if name not in sys.modules and _mapped(path) and _module_built_from(path):
            return None
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 — any import error is this file's failure, reported by name
        return _described(error)
    return None


def load_library(path: Path) -> Optional[str]:
    """Open ``path`` with ctypes, as the package that ships it does; None when it opens, else the error."""
    try:
        ctypes.CDLL(str(path))
    except OSError as error:
        return _described(error)
    return None


def probe(site: Path, path: Path) -> Result:
    """Read ``path``'s ELF headers, then import it or open it as its kind requires."""
    relative = PurePosixPath(path.relative_to(site).as_posix())
    name = module_name(relative)
    try:
        extension, debug = _inspect(path, f"PyInit_{name.rpartition('.')[2]}")
    except (OSError, ValueError) as error:
        return Result(str(relative), "unreadable", _described(error), ())
    if extension:
        return Result(name, "extension", load_extension(name, path), debug)
    return Result(str(relative), "library", load_library(path), debug)


# --------------------------------------------------------------------------- verdict


def problems_with(results: Sequence[Result], expected: Dict[str, str]) -> List[str]:
    """Unexpected failures, listed failures that now load, and debug sections. Pure."""
    if not results:
        return ["no *.so file under site-packages: the walk found nothing to check"]
    failed = {result.key: result.error for result in results if result.error}
    present = {result.key for result in results}
    problems = [f"{key} does not load: {error}" for key, error in sorted(failed.items()) if key not in expected]
    problems += [f"{key} loads now: remove it from EXPECTED_FAILURES ({expected[key]})"
                 for key in sorted(present & expected.keys()) if key not in failed]
    problems += [f"{result.key} still carries {', '.join(result.debug)}" for result in results if result.debug]
    return problems


def tree_problems(site: Path) -> List[str]:
    """A ``tests`` directory left under ``site``, or a kept ``testing`` package gone."""
    left = sorted(path.relative_to(site).as_posix() for path in site.rglob(TESTS_DIR) if path.is_dir())
    problems = [f"site-packages still has {path}/" for path in left]
    problems += [f"site-packages lost {path}/: only directories named exactly {TESTS_DIR} go"
                 for path in KEPT_TESTING if not (site / path / "__init__.py").is_file()]
    return problems


def stale_bytecode(verbose_stderr: str) -> List[str]:
    """The ``bytecode is stale`` lines of a ``python -v`` run's stderr. Pure."""
    return [line.strip() for line in verbose_stderr.splitlines() if STALE_BYTECODE in line]


def app_problems() -> List[str]:
    """``import main`` in a fresh interpreter in /app, so the probe's imports can't help or hurt it,
    under ``-v``: no module it imports may find its .pyc stale (PRD-262 US-022)."""
    try:
        done = subprocess.run([sys.executable, "-v", "-c", APP_IMPORT], cwd=APP_DIR, capture_output=True,
                              text=True, env={**os.environ, **APP_IMPORT_ENV}, timeout=APP_IMPORT_TIMEOUT_S,
                              check=False)
    except subprocess.TimeoutExpired:
        return [f"`{APP_IMPORT}` did not finish in {APP_IMPORT_TIMEOUT_S} s"]
    if done.returncode != 0:
        tail = " | ".join(done.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        return [f"`{APP_IMPORT}` failed with exit {done.returncode}: {tail}"]
    stale = stale_bytecode(done.stderr)
    print(f"`python -v -c '{APP_IMPORT}'`: {len(stale)} lines say `{STALE_BYTECODE}`")
    if stale:
        return [f"{len(stale)} modules found their bytecode stale under `{APP_IMPORT}`: "
                f"{' | '.join(stale[:STALE_LISTED])}"]
    return []


def _probe_all(site: Path, files: Iterable[Path]) -> List[Result]:
    """Each file in turn, its name printed first so a crash shows which one it was."""
    print(f"::group::loading the native libraries under {site}")
    results = []
    for path in files:
        print(f"{path.relative_to(site)}: ", end="", flush=True)
        result = probe(site, path)
        print(f"{result.kind} {result.key}, {result.error or 'ok'}", flush=True)
        results.append(result)
    print("::endgroup::")
    return results


def main() -> int:
    faulthandler.enable()
    sys.path = [entry for entry in sys.path if entry]  # stdin's '' is the working directory
    site = site_packages()
    results = _probe_all(site, native_files(site))
    kinds = [result.kind for result in results]
    print(f"{len(results)} files: {kinds.count('extension')} extensions imported, "
          f"{kinds.count('library')} libraries opened with ctypes")
    for result in results:
        if result.error and result.key in EXPECTED_FAILURES:
            print(f"::notice::{result.key} fails as expected ({EXPECTED_FAILURES[result.key]})")
    problems = problems_with(results, EXPECTED_FAILURES) + tree_problems(site) + app_problems()
    for problem in problems:
        print(f"::error title=site-packages (PRD-262 US-020)::{problem}")
    if not problems:
        print(f"every file loads as expected, none carries debug sections, no {TESTS_DIR}/ is left, "
              f"and `{APP_IMPORT}` works with no stale bytecode")
    return 1 if problems else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skips interpreter shutdown: the probe loaded every extension in the image into one
    # process, and their finalizers were never meant to run side by side.
    os._exit(code)
