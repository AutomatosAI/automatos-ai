#!/usr/bin/env python3
"""Render, check and snapshot never load a pruned npm path (PRD-262 #7; the media-render CI job).

Stdlib only: it runs on the runner, not in the image. The job starts the image
with ``NODE_OPTIONS=--import=/ci/trace_modules.mjs``, renders the fixture (a check,
then a render) and its preview (a check, then a snapshot), and passes the log the
hook wrote. It fails unless:

- the trace saw each command in ``--commands`` run as ``hyperframes <command>``;
- it is a real trace: every package in ``REQUIRED_PACKAGES`` loaded files;
- no file Node loaded lies under a path ``--prune`` lists (the js stage deletes
  those after ``npm ci``), and none is a source map, a type declaration or Markdown;
- no specifier that failed to resolve names a pruned path: code that tried one and
  swallowed the error would pass the renders while working differently.

Specifiers that fail for other reasons (optional packages such as ``canvas`` or
``bufferutil``) are listed for the log and pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

NODE_MODULES = "/opt/media-render/js/node_modules"
CLI_NAMES = frozenset({"hyperframes", "hyperframes.mjs"})
# Packages every render loads: a trace without them traced nothing.
REQUIRED_PACKAGES = ("hyperframes", "puppeteer-core", "sharp", "esbuild")
# The js stage deletes these suffixes everywhere (licence and notice files aside).
DROPPED_SUFFIXES = (".map", ".d.ts", ".d.mts", ".d.cts", ".md")

logger = logging.getLogger("assert_module_trace")


@dataclass
class Trace:
    """What the hook logged: each thread's argv, every file resolved, every specifier that failed."""

    argvs: List[List[str]] = field(default_factory=list)
    files: Set[str] = field(default_factory=set)
    failures: List[Tuple[str, Optional[str]]] = field(default_factory=list)


def read_prune(path: Path) -> Tuple[str, ...]:
    """The pruned paths, relative to node_modules: one per line, ``#`` comments dropped."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    entries = tuple(line for line in lines if line)
    if not entries:
        raise ValueError(f"{path} lists no path")
    return entries


def read_trace(path: Path) -> Trace:
    trace = Trace()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number} is not JSON: {exc}") from None
        if "argv" in entry:
            trace.argvs.append([str(arg) for arg in entry["argv"]])
        elif "file" in entry:
            trace.files.add(str(entry["file"]))
        elif "failed" in entry:
            trace.failures.append((str(entry["failed"]), entry.get("parent")))
    return trace


def commands(trace: Trace) -> Set[str]:
    """The ``hyperframes <command>`` runs the trace saw."""
    return {argv[1] for argv in trace.argvs if len(argv) > 1 and posixpath.basename(argv[0]) in CLI_NAMES}


def under_node_modules(path: str, node_modules: str) -> Optional[str]:
    """``path`` relative to ``node_modules``, or None outside it."""
    prefix = node_modules.rstrip("/") + "/"
    return path[len(prefix) :] if path.startswith(prefix) else None


def pruned_entry(relative: str, pruned: Sequence[str]) -> Optional[str]:
    """The pruned path that ``relative`` (a node_modules-relative path) is, or lies under."""
    return next((entry for entry in pruned if relative == entry or relative.startswith(entry + "/")), None)


def failed_target(specifier: str, parent: Optional[str], node_modules: str) -> Optional[str]:
    """The node_modules-relative path a failed specifier was after, when it can be told."""
    if specifier.startswith("file:"):
        return under_node_modules(unquote(urlparse(specifier).path), node_modules)
    if specifier.startswith(("./", "../", "/")):
        if specifier.startswith("/"):
            return under_node_modules(posixpath.normpath(specifier), node_modules)
        if not parent or not parent.startswith("file:"):
            return None
        base = posixpath.dirname(unquote(urlparse(parent).path))
        return under_node_modules(posixpath.normpath(posixpath.join(base, specifier)), node_modules)
    if specifier.startswith("node:"):
        return None
    return specifier


def loaded_problems(trace: Trace, pruned: Sequence[str], node_modules: str) -> List[str]:
    problems = []
    for path in sorted(trace.files):
        relative = under_node_modules(path, node_modules)
        if relative is None:
            continue
        entry = pruned_entry(relative, pruned)
        if entry:
            problems.append(f"loaded {path}, under the pruned {entry}")
        elif relative.endswith(DROPPED_SUFFIXES):
            problems.append(f"loaded {path}, a file type the js stage deletes")
    return problems


def failure_problems(trace: Trace, pruned: Sequence[str], node_modules: str) -> List[str]:
    problems = []
    for specifier, parent in trace.failures:
        target = failed_target(specifier, parent, node_modules)
        entry = pruned_entry(target, pruned) if target else None
        if entry:
            problems.append(f"{parent or '?'} tried {specifier!r}, under the pruned {entry}, and failed")
    return problems


def coverage_problems(trace: Trace, expected: Iterable[str], node_modules: str) -> List[str]:
    problems = [f"no `hyperframes {name}` run in the trace" for name in sorted(set(expected) - commands(trace))]
    packages = {package_of(relative) for relative in _relatives(trace, node_modules)}
    problems += [f"no file of {name} loaded: the trace saw no render" for name in REQUIRED_PACKAGES if name not in packages]
    return problems


def package_of(relative: str) -> str:
    """The package a node_modules-relative path belongs to (``@scope/name`` or ``name``)."""
    parts = relative.split("/")
    return "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]


def _relatives(trace: Trace, node_modules: str) -> List[str]:
    found = (under_node_modules(path, node_modules) for path in trace.files)
    return [relative for relative in found if relative]


def trace_problems(trace: Trace, pruned: Sequence[str], node_modules: str, expected: Iterable[str]) -> List[str]:
    """Every reason the trace fails; empty when it passes."""
    return (
        coverage_problems(trace, expected, node_modules)
        + loaded_problems(trace, pruned, node_modules)
        + failure_problems(trace, pruned, node_modules)
    )


def _report(trace: Trace, node_modules: str) -> None:
    runs = Counter(argv[1] for argv in trace.argvs if len(argv) > 1 and posixpath.basename(argv[0]) in CLI_NAMES)
    logger.info("hyperframes runs: %s", dict(sorted(runs.items())))
    logger.info("threads traced: %d; files resolved: %d", len(trace.argvs), len(trace.files))
    packages = Counter(package_of(relative) for relative in _relatives(trace, node_modules))
    logger.info("packages loaded (files): %s", ", ".join(f"{name} {n}" for name, n in sorted(packages.items())))
    failed = sorted({specifier for specifier, _ in trace.failures})
    logger.info("specifiers that did not resolve (optional ones pass): %s", ", ".join(failed) or "none")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", type=Path, required=True, help="the log ci/trace_modules.mjs wrote")
    parser.add_argument("--prune", type=Path, required=True, help="services/media-render/js/prune.txt")
    parser.add_argument("--node-modules", default=NODE_MODULES, help="the image's node_modules directory")
    parser.add_argument("--commands", default="check,render,snapshot", help="the hyperframes commands that must have run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    trace = read_trace(args.trace)
    pruned = read_prune(args.prune)
    _report(trace, args.node_modules)
    problems = trace_problems(trace, pruned, args.node_modules, [c for c in args.commands.split(",") if c])
    for problem in problems:
        logger.error("FAIL  %s", problem)
    if problems:
        return 1
    logger.info("PASS  %d pruned paths, none loaded or tried by %s", len(pruned), args.commands)
    return 0


if __name__ == "__main__":
    sys.exit(main())
