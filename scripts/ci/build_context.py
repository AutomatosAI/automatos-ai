#!/usr/bin/env python3
"""PRD-262 US-017: the API's build context holds only code.

A self-hoster's checkout collects files the API image must never carry: the generated
documents, thumbnails and brand files under ``orchestrator/documents`` (the default
``DOCUMENT_STORAGE_DIR`` is relative to /app, and the compose bind mount put it in the
checkout), nested ``__pycache__``, and the eval and import-linter caches.
``orchestrator/.dockerignore`` keeps them out of the build context. This script proves it
on a real build in CI:

    build_context.py seed DIR
        Writes a file of random bytes, as large as the budget, at each place those files
        collect (``SEEDS``, relative to the context DIR), so any one of them that reaches
        the context breaks the budget on its own. A file already there is left as it is.
        For a throwaway checkout only: it writes about 240 MB.

    build_context.py check LOG [--service NAME]
        Reads a build log in BuildKit's plain progress, which a piped ``docker build`` or
        ``docker compose build`` prints, and adds up what the build sent as its context:
        the ``transferring context`` lines of ``[NAME internal] load build context``, or
        of ``[internal] load build context`` for a plain ``docker build``. The
        ``load .dockerignore`` step prints the same line and is not counted. Exits 1 over
        ``BUDGET_BYTES``, or when the log holds no such transfer.

test.yml's prod-image-parity checks the production build, whose last layer copies the
whole context. scripts/ci/smoke-fresh-clone.sh checks the compose build, which builds the
production stage too since PRD-262 US-035. Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

# About twice the code the context carries today (19.8 MB in the 18 Oct build logs).
# BuildKit prints decimal units, so the budget is decimal too.
BUDGET_BYTES = 40_000_000
SEED_BYTES = BUDGET_BYTES
CHUNK_BYTES = 1 << 20
# One file for each place US-017's .dockerignore rules cover, relative to the context.
SEEDS = (
    "documents/prd262-context-probe/generated/upload.docx",  # DOCUMENT_STORAGE_DIR's default
    "modules/documents/__pycache__/prd262_context_probe.cpython-311.pyc",
    "core/prd262_context_probe.pyc",
    "scripts/eval/tool_routing/.embedding_cache.json",
    "scripts/eval/memory_recall/results/prd262-context-probe.json",
    ".import_linter_cache/prd262-context-probe",
)
UNITS = {"B": 1, "kB": 10**3, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
         "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_UNIT = "|".join(sorted(UNITS, key=len, reverse=True))
# `#19 [backend internal] load build context`, `#19 transferring context: 19.81MB 1.1s done`.
# A step's lines may follow a prefix, as in a log downloaded from GitHub.
_STEP_LINE = re.compile(r"(?:^|\s)#(\d+) (.+?)\s*$")
_TRANSFER = re.compile(rf"transferring context: ([0-9]+(?:\.[0-9]+)?)\s?({_UNIT})(?:\s|$)")


def context_step(service: Optional[str]) -> str:
    """The name BuildKit gives a build's context transfer."""
    return f"[{service} internal] load build context" if service else "[internal] load build context"


def step_lines(log: str) -> Iterator[Tuple[str, str]]:
    """``(step id, text)`` for each of a plain-progress log's step lines, in order.

    A text that starts with ``[`` names its step (``[backend internal] load build
    context``); the others are that step's output and status (``CACHED``, ``DONE 1.2s``).
    scripts/ci/build_cache.py reads the install steps' status the same way (PRD-262 US-039).
    """
    for line in log.splitlines():
        match = _STEP_LINE.search(line)
        if match:
            yield match.group(1), match.group(2)


def transfers(log: str, service: Optional[str]) -> dict:
    """Bytes the build sent as its context, per step id of that build's context transfer.

    BuildKit prints a step's name before its lines, and again when they resume after
    another step's, so each `transferring context` line belongs to the step its id was
    last named as. A step reports growing totals; the largest is what it sent.
    """
    wanted, named, sent = context_step(service), {}, {}
    for step, text in step_lines(log):
        if text.startswith("["):
            named[step] = text
            continue
        size = _TRANSFER.match(text)
        if size and named.get(step) == wanted:
            sent[step] = max(sent.get(step, 0), round(float(size.group(1)) * UNITS[size.group(2)]))
    return sent


def _mb(size: int) -> str:
    return f"{size / 1e6:.2f} MB"


def verdict(log: str, service: Optional[str], budget: int = BUDGET_BYTES) -> tuple:
    """(passed, message) for one build log."""
    sent = transfers(log, service)
    name = context_step(service)
    if not sent:
        return False, (f"no `transferring context` line for `{name}` in the build log: "
                       "is it BuildKit's plain progress, from a piped build?")
    total = sum(sent.values())
    if total > budget:
        return False, (f"`{name}` sent {_mb(total)}, over the {_mb(budget)} budget: "
                       "orchestrator/.dockerignore lets uploads or caches into the build context")
    return True, f"`{name}` sent {_mb(total)}, within the {_mb(budget)} budget"


def _write_random(path: Path, size: int) -> None:
    """Random bytes, so no compression on the way to the builder could shrink them."""
    with path.open("xb") as out:
        left = size
        while left > 0:
            chunk = min(left, CHUNK_BYTES)
            out.write(os.urandom(chunk))
            left -= chunk


def seed(root: Path, size: int = SEED_BYTES) -> list:
    """Write each seed that is not there yet; returns the files written."""
    written = []
    for relative in SEEDS:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_random(path, size)
        written.append(path)
    return written


def _parse(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    seeding = commands.add_parser("seed", help="write the dummy uploads and caches into a context")
    seeding.add_argument("root", type=Path, help="the build context, e.g. orchestrator")
    checking = commands.add_parser("check", help="read a build's context transfer off its log")
    checking.add_argument("log", type=Path, help="the build log, in BuildKit's plain progress")
    checking.add_argument("--service", help="the compose service, for a `docker compose build` log")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse(argv)
    if args.command == "seed":
        for path in seed(args.root):
            print(f"seeded {path} ({_mb(SEED_BYTES)})")
        return 0
    passed, message = verdict(args.log.read_text(encoding="utf-8", errors="replace"), args.service)
    print(message if passed else f"::error::{message}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
