"""Split the API's lock by package name, one half per pip layer (PRD-262 US-022).

orchestrator/Dockerfile's pybuild stage installs requirements.lock (every build's,
Railway's included, since PRD-262 US-027) in two halves, each into its
own prefix, and the runtime copies each half as its own layer: first STABLE, the
packages that rarely move and weigh the most, then the rest. images.yml builds the API
reproducibly, so a bump that leaves every STABLE pin alone leaves the first layer the
same bytes it was, and an update downloads only the second. Two sets go in neither
half: LEFT_OUT, which no image installs, and DEVELOPMENT_ONLY, which pybuild installs
into a third prefix that only the development stage copies (the 19 Oct breakdown's rows
#11 and #47). The lock keeps pinning both: CI installs it with ``pip install
--require-hashes`` and no ``--no-deps``, which refuses a requirement of a pinned
package that the lock does not pin.

    python split_lock.py LOCK STABLE_OUT REST_OUT DEVELOPMENT_OUT

Each requirement block of LOCK (its ``name==version`` line with the hash and comment
lines under it) goes to at most one of the outputs unchanged, in the lock's order; the
header comments go to none. It refuses a STABLE, LEFT_OUT or DEVELOPMENT_ONLY name the
lock does not pin, so a package leaving the lock cannot empty the first layer or leave
a stale list unnoticed, a pin without a hash, and a line it cannot read, so nothing is
left out silently. scripts/ci/image_pip_layers.py splits the published ``edge``'s lock
with it too.

Standard library only: it runs on pybuild's bare Python.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple

# The stable set PRD-262 US-022 names, as PEP 503 names, but grpcio, which left the lock
# with qdrant-client (the 19 Oct breakdown's row #20).
STABLE = frozenset({
    "numpy", "scipy", "cryptography", "sqlalchemy", "pillow", "lxml", "fonttools",
    "uvloop", "asyncpg", "pydantic-core",
})
# Pinned, never installed in an image (PRD-262 row #11, about 47 MB): graphifyy requires
# all 25 tree-sitter grammars, and only graphify/extract.py imports these 12, a module the
# app never imports (tests/test_prd262_left_out_packages.py). The code graph parses with
# the 13 grammars requirements.in names (modules/codegraph/parsers/treesitter_parser.py).
LEFT_OUT = frozenset({
    "tree-sitter-bash", "tree-sitter-elixir", "tree-sitter-fortran", "tree-sitter-groovy", "tree-sitter-json",
    "tree-sitter-julia", "tree-sitter-lua", "tree-sitter-objc", "tree-sitter-powershell", "tree-sitter-scala",
    "tree-sitter-verilog", "tree-sitter-zig",
})
# The development stage's only (PRD-262 row #47): uvicorn --reload watches the mounted
# source through watchfiles, and the production command never reloads.
DEVELOPMENT_ONLY = frozenset({"watchfiles"})
HASH = "--hash=sha256:"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==\S")
USAGE = "usage: split_lock.py LOCK STABLE_OUT REST_OUT DEVELOPMENT_OUT"


class Block(NamedTuple):
    """One pin of the lock: its PEP 503 name and its lines as the lock wrote them."""

    name: str
    text: str


def canonical(name: str) -> str:
    """``Pydantic_Core`` -> ``pydantic-core`` (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def blocks(text: str) -> List[Block]:
    """Every pin of a uv-compiled lock with the indented lines under it. Pure.

    The header comments before the first pin are dropped. Anything else at the start
    of a line that is not a pin (an option such as ``-e`` or ``--index-url``) raises
    ValueError.
    """
    found: List[Tuple[str, List[str]]] = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if not line.strip() or (line.startswith("#") and not found):
            continue
        pin = _PIN.match(line)
        if pin:
            found.append((canonical(pin.group(1)), [line]))
        elif line[:1].isspace() and found:
            found[-1][1].append(line)
        else:
            raise ValueError(f"line {number} is not a pin this split can read: {line.strip()!r}")
    return [Block(name, "".join(lines)) for name, lines in found]


def _checked(text: str) -> List[Block]:
    """The lock's pins, refused when one has no hash or a named set names a package it does not pin."""
    pins = blocks(text)
    unhashed = sorted({block.name for block in pins if HASH not in block.text})
    if unhashed:
        raise ValueError(f"pins without a hash: {', '.join(unhashed)}")
    pinned = {block.name for block in pins}
    for label, names in (("the stable set", STABLE), ("LEFT_OUT", LEFT_OUT), ("DEVELOPMENT_ONLY", DEVELOPMENT_ONLY)):
        missing = sorted(names - pinned)
        if missing:
            raise ValueError(f"{label} names packages the lock does not pin: {', '.join(missing)}")
    return pins


def split(text: str) -> Tuple[str, str]:
    """(the STABLE pins, every other pin an image installs) of the lock ``text``, each as lock
    text: LEFT_OUT and DEVELOPMENT_ONLY are in neither. Pure."""
    pins = [block for block in _checked(text) if block.name not in LEFT_OUT | DEVELOPMENT_ONLY]
    stable = [block.text for block in pins if block.name in STABLE]
    rest = [block.text for block in pins if block.name not in STABLE]
    if not rest:
        raise ValueError("every pin is in the stable set: the second layer would be empty")
    return "".join(stable), "".join(rest)


def development(text: str) -> str:
    """The DEVELOPMENT_ONLY pins of the lock ``text``, as lock text. Pure."""
    return "".join(block.text for block in _checked(text) if block.name in DEVELOPMENT_ONLY)


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print(USAGE, file=sys.stderr)
        return 2
    lock, *outputs = (Path(arg) for arg in argv)
    try:
        text = lock.read_text(encoding="utf-8")
        parts = (*split(text), development(text))
    except ValueError as error:
        print(f"split_lock.py: {lock}: {error}", file=sys.stderr)
        return 1
    for path, part in zip(outputs, parts):
        path.write_text(part, encoding="utf-8")
    print(f"{lock}: " + ", ".join(f"{len(blocks(part))} pins in {path}" for path, part in zip(outputs, parts))
          + f"; {len(LEFT_OUT)} left out")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
