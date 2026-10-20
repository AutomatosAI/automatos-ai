#!/usr/bin/env python3
"""The code's alembic head(s), read with no import, and whether a database is at them.

PRD-262 US-030. Before a migration the entrypoint asks whether one is needed at all.
alembic's own start imports the app's models (``alembic/env.py`` imports
``core.models``) before it finds there is nothing to do: about 7-9 s on a warm local
boot, 35 s under load (F228), on every boot. This script reads the head(s) from the
migration scripts with ``ast``, importing nothing outside the standard library (no
alembic, no SQLAlchemy, no app module), and compares them with the database's
``alembic_version`` rows, which the entrypoint passes on stdin (``psql -tA``: one
version per line).

* The same set: exit 0. The entrypoint logs ``schema at head — alembic skipped``.
* Anything else: exit 1, and the entrypoint runs ``alembic upgrade heads`` as before.
  That covers an empty ``alembic_version``, a database behind the code, and one ahead
  of it (a version this code does not know, after a rollback), where alembic fails
  loudly.
* A migration script it cannot read statically: exit 2, and alembic decides. That
  covers a syntax error, a revision that is not a literal, a ``down_revision`` that
  names no script, or two scripts declaring the same revision.

The heads are alembic's: a revision that no ``down_revision`` names, where a
``down_revision`` is a revision id, ``None``, or a tuple of ids (a merge revision).
A ``depends_on`` link never makes a head, in alembic or here. The scripts are the
ones alembic loads: every ``*.py`` file directly in ``alembic/versions``, except
``__init__`` and an editor's ``.#`` lock files. Only module-level assignments count,
the last one winning, as when the module is imported.

    printf '%s\\n' "$stamped" | python scripts/alembic_heads.py   # exit 0 at head
    python scripts/alembic_heads.py --print                         # the heads, one per line
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, TextIO, Tuple

VERSIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"
# alembic.script.base._only_source_rev_file: the file names alembic loads as revisions.
REVISION_FILE = re.compile(r"(?!\.\#|__init__)(.*\.py)$")
GRAPH_NAMES = ("revision", "down_revision")
AT_HEAD, NOT_AT_HEAD, UNREADABLE = 0, 1, 2

Graph = Dict[str, Tuple[str, ...]]


class UnreadableHeads(Exception):
    """The revision graph cannot be read from the scripts' source alone."""


def _module_assignments(path: Path) -> Dict[str, ast.expr]:
    """The module-level ``revision`` and ``down_revision`` values; the last one wins."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise UnreadableHeads(f"{path.name}: {exc}") from exc
    found: Dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in GRAPH_NAMES:
                found[target.id] = node.value
    return found


def _literal(path: Path, name: str, node: Optional[ast.expr]):
    if node is None:
        raise UnreadableHeads(f"{path.name}: no module-level {name}")
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise UnreadableHeads(f"{path.name}: {name} is not a literal") from exc


def _parents(path: Path, value) -> Tuple[str, ...]:
    """A ``down_revision`` as the revisions it names: none, one, or a merge's several."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise UnreadableHeads(f"{path.name}: down_revision is {value!r}, not a revision id, a tuple of them or None")


def revision_of(path: Path) -> Tuple[str, Tuple[str, ...]]:
    """One script's revision id and the revisions it revises."""
    found = _module_assignments(path)
    revision = _literal(path, "revision", found.get("revision"))
    if not isinstance(revision, str) or not revision:
        raise UnreadableHeads(f"{path.name}: revision is {revision!r}, not a revision id")
    return revision, _parents(path, _literal(path, "down_revision", found.get("down_revision")))


def read_graph(versions_dir: Path = VERSIONS_DIR) -> Graph:
    """Every revision id and the revisions it revises, from the scripts' source."""
    try:
        paths = sorted(path for path in versions_dir.iterdir()
                       if path.is_file() and REVISION_FILE.match(path.name))
    except OSError as exc:
        raise UnreadableHeads(f"{versions_dir}: {exc}") from exc
    graph: Graph = {}
    for path in paths:
        revision, parents = revision_of(path)
        if revision in graph:
            raise UnreadableHeads(f"{path.name}: revision {revision!r} is declared twice")
        graph[revision] = parents
    return graph


def heads(graph: Graph) -> Tuple[str, ...]:
    """The revisions no revision revises, sorted: what ``alembic heads`` prints."""
    revised = {parent for parents in graph.values() for parent in parents}
    missing = sorted(revised - set(graph))
    if missing:
        raise UnreadableHeads(f"a down_revision names no script: {', '.join(missing)}")
    found = tuple(sorted(set(graph) - revised))
    if not found:
        raise UnreadableHeads("no head: there is no revision, or the revisions form a cycle")
    return found


def stamped_versions(lines: Iterable[str]) -> FrozenSet[str]:
    """``alembic_version`` as ``psql -tA`` prints it: one version per line."""
    return frozenset(line.strip() for line in lines if line.strip())


def at_head(stamped: FrozenSet[str], code_heads: Sequence[str]) -> bool:
    """True only when the database holds exactly the code's heads."""
    return bool(code_heads) and stamped == frozenset(code_heads)


def _names(revisions: Iterable[str]) -> str:
    return ", ".join(sorted(revisions)) or "(none)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Is the database's alembic_version at the code's head(s)?")
    parser.add_argument("--print", dest="print_heads", action="store_true",
                        help="print the code's heads, one per line, and read no stdin")
    parser.add_argument("--versions-dir", type=Path, default=VERSIONS_DIR, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code_heads = heads(read_graph(args.versions_dir))
    except UnreadableHeads as exc:
        print(f"   the code's alembic heads cannot be read statically ({exc}); alembic decides", file=sys.stderr)
        return UNREADABLE
    if args.print_heads:
        print("\n".join(code_heads))
        return 0
    stamped = stamped_versions(stdin if stdin is not None else sys.stdin)
    print(f"   alembic_version: {_names(stamped)}; the code's head(s): {_names(code_heads)}")
    return AT_HEAD if at_head(stamped, code_heads) else NOT_AT_HEAD


if __name__ == "__main__":
    sys.exit(main())
