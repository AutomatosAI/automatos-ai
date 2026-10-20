#!/usr/bin/env python3
"""The app modules every API process imports at boot, read with no import (PRD-262 US-032).

The production image ships bytecode for these files and no others. The Dockerfile's
``compile`` stage runs:

    python scripts/boot_modules.py > /tmp/boot-modules.txt
    python -m compileall -q --invalidation-mode unchecked-hash -i /tmp/boot-modules.txt

The set is the static import closure, inside this app directory, of the boot roots:

* ``main``, which uvicorn imports in every worker;
* the routers ``router_manifest.py`` mounts by name (``RouterSpec("api.…")``), which
  main imports through importlib;
* ``core.seeds.fingerprint``, which the entrypoint runs with ``python -m`` before
  every server start (PRD-262 US-031).

An import counts when it runs as its module runs: at module level, in a class body,
or under ``if``/``try``/``with`` there, but not inside a function and not under ``if
TYPE_CHECKING:``. ``import a.b`` and ``from a.b import c`` bring in each package on the
way (``a/__init__.py``, ``a/b/__init__.py``); ``from a import b`` brings in ``a/b.py``
when ``b`` is a module. A relative import resolves against the importing module's
package. A name that is no file here (the standard library, site-packages) is not
ours to compile. A package directory wins over a module file of the same name, as
in Python's own finder, and this directory comes first on the import path, as it
does under uvicorn.

It prints the files' absolute paths, one per line, sorted. Exit 1, with the reason on
stderr, when a root has no file or a file on the way does not parse.

Standard library only: it runs in the build before anything is compiled.
"""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

APP_DIR = Path(__file__).resolve().parents[1]
ROOTS = ("main", "core.seeds.fingerprint")
MANIFEST = "router_manifest"
MANIFEST_CALL = "RouterSpec"
# The fields of a statement whose bodies run when the statement does (if/else, try,
# with, for/while, class bodies, except handlers, match cases).
BODY_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


class BootModulesError(Exception):
    """The boot set cannot be read: a root has no file, or a file does not parse."""


def module_file(app: Path, name: str) -> Optional[Path]:
    """The file ``name`` imports from ``app``: a package's ``__init__.py`` first, then ``<name>.py``."""
    base = app.joinpath(*name.split("."))
    for candidate in (base / "__init__.py", base.parent / f"{base.name}.py"):
        if candidate.is_file():
            return candidate
    return None


def with_parents(name: str) -> List[str]:
    """``a.b.c`` -> ``a``, ``a.b``, ``a.b.c``: importing a module imports each package on the way."""
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts) + 1)]


def _is_type_checking(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def runs_on_import(body: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """The statements that run when the module runs: none inside a function or under TYPE_CHECKING."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        if _is_type_checking(node):
            yield from runs_on_import(node.orelse)
            continue
        for field in BODY_FIELDS:
            yield from runs_on_import(getattr(node, field, None) or [])


def _package(name: str, path: Path) -> str:
    """The package a module's relative imports resolve against."""
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def absolute(module: Optional[str], level: int, package: str) -> Optional[str]:
    """The absolute name of ``from <level dots><module> import …`` in ``package``.

    None past the top-level package, where Python raises instead.
    """
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level > len(parts):
        return None
    base = parts[:len(parts) - (level - 1)] + ([module] if module else [])
    return ".".join(base) or None


def imported_names(node: ast.AST, name: str, path: Path) -> List[str]:
    """Every module name one import statement can load, parents included; [] for any other statement."""
    if isinstance(node, ast.Import):
        return [found for alias in node.names for found in with_parents(alias.name)]
    if not isinstance(node, ast.ImportFrom):
        return []
    base = absolute(node.module, node.level, _package(name, path))
    if not base:
        return []
    return with_parents(base) + [f"{base}.{alias.name}" for alias in node.names if alias.name != "*"]


def parse(path: Path) -> ast.Module:
    """``path``'s syntax tree. An invalid escape's warning is left to compileall, which reports it too."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise BootModulesError(f"{path}: {exc}") from exc


def manifest_modules(app: Path) -> List[str]:
    """The modules router_manifest.py mounts by name: each ``RouterSpec("<module>", …)``'s first argument."""
    path = module_file(app, MANIFEST)
    if path is None:
        raise BootModulesError(f"{MANIFEST}.py is not in {app}")
    names = set()
    for node in ast.walk(parse(path)):
        if not (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if called == MANIFEST_CALL and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
    return sorted(names)


def closure(app: Path, roots: Sequence[str]) -> Dict[str, Path]:
    """Module name -> file, for every app module the roots import as they run."""
    found: Dict[str, Path] = {}
    elsewhere: Set[str] = set()
    queue = deque(roots)
    while queue:
        name = queue.popleft()
        if name in found or name in elsewhere:
            continue
        path = module_file(app, name)
        if path is None:
            elsewhere.add(name)
            continue
        found[name] = path
        queue.extend(with_parents(name))
        for node in runs_on_import(parse(path).body):
            queue.extend(imported_names(node, name, path))
    return found


def boot_files(app: Path = APP_DIR, roots: Sequence[str] = ROOTS) -> List[Path]:
    """The boot set's files, sorted. Raises BootModulesError when a root has no file."""
    missing = [root for root in roots if module_file(app, root) is None]
    if missing:
        raise BootModulesError(f"no file in {app} for the boot root(s) {', '.join(missing)}")
    return sorted(set(closure(app, [*roots, *manifest_modules(app)]).values()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The app modules every API process imports at boot.")
    parser.add_argument("--app", type=Path, default=APP_DIR, help="the app directory (default: this script's parent's)")
    args = parser.parse_args(argv)
    try:
        files = boot_files(args.app.resolve())
    except BootModulesError as error:
        print(f"boot_modules: {error}", file=sys.stderr)
        return 1
    for path in files:
        print(path)
    print(f"boot_modules: {len(files)} modules from {', '.join(ROOTS)} and {MANIFEST}'s routers",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
