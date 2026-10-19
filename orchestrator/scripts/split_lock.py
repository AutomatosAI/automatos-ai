"""Split the API's lock in two by package name, one half per pip layer (PRD-262 US-022).

orchestrator/Dockerfile's pybuild stage installs requirements.lock (every build's,
Railway's included, since PRD-262 US-027) in two halves, each into its
own prefix, and the runtime copies each half as its own layer: first STABLE, the
packages that rarely move and weigh the most, then the rest. images.yml builds the API
reproducibly, so a bump that leaves every STABLE pin alone leaves the first layer the
same bytes it was, and an update downloads only the second.

    python split_lock.py LOCK STABLE_OUT REST_OUT

Each requirement block of LOCK (its ``name==version`` line with the hash and comment
lines under it) goes to one of the two outputs unchanged, in the lock's order; the
header comments go to neither. It refuses a STABLE name the lock does not pin, so a
package leaving the lock cannot empty the first layer unnoticed, a pin without a
hash, and a line it cannot read, so nothing is left out silently. scripts/ci/
image_pip_layers.py splits the published ``edge``'s lock with it too.

Standard library only: it runs on pybuild's bare Python.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple

# The stable set PRD-262 US-022 names, as PEP 503 names.
STABLE = frozenset({
    "numpy", "scipy", "grpcio", "cryptography", "sqlalchemy", "pillow", "lxml", "fonttools",
    "uvloop", "asyncpg", "pydantic-core",
})
HASH = "--hash=sha256:"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==\S")
USAGE = "usage: split_lock.py LOCK STABLE_OUT REST_OUT"


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


def split(text: str) -> Tuple[str, str]:
    """(the STABLE pins, every other pin) of the lock ``text``, each as lock text. Pure."""
    pins = blocks(text)
    unhashed = sorted({block.name for block in pins if HASH not in block.text})
    if unhashed:
        raise ValueError(f"pins without a hash: {', '.join(unhashed)}")
    missing = sorted(STABLE - {block.name for block in pins})
    if missing:
        raise ValueError(f"the stable set names packages the lock does not pin: {', '.join(missing)}")
    stable = [block.text for block in pins if block.name in STABLE]
    rest = [block.text for block in pins if block.name not in STABLE]
    if not rest:
        raise ValueError("every pin is in the stable set: the second layer would be empty")
    return "".join(stable), "".join(rest)


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return 2
    lock, stable_out, rest_out = (Path(arg) for arg in argv)
    try:
        stable, rest = split(lock.read_text(encoding="utf-8"))
    except ValueError as error:
        print(f"split_lock.py: {lock}: {error}", file=sys.stderr)
        return 1
    stable_out.write_text(stable, encoding="utf-8")
    rest_out.write_text(rest, encoding="utf-8")
    print(f"{lock}: {len(blocks(stable))} pins in {stable_out}, {len(blocks(rest))} in {rest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
