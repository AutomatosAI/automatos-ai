"""Reads the pins of PRD-262 W3's reference locks (scripts/ci/reference/).

Each library swap's reference runs from a ``requirements-<name>.in`` that pins every
package it installs, compiled by uv into a ``.lock`` with hashes. The suites that hold
a swap to its reference check that the lock is its input pinned and hashed
(``test_prd262_pdf_tables.py``, ``test_prd262_text_splitters.py``,
``test_prd262_leiden_adapter.py``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
HASH = "--hash=sha256:"


def pins(path: Path) -> Dict[str, str]:
    """PEP 503 name -> pinned version, for every ``name==version`` line; comments dropped."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {re.sub(r"[-_.]+", "-", m.group(1)).lower(): m.group(2).rstrip(" \\")
            for m in map(PIN.match, lines) if m}


def unhashed(text: str) -> List[str]:
    """The pins in a lock with no --hash line before the next pin."""
    found: List[str] = []
    pin: Optional[str] = None
    hashed = True
    for line in text.splitlines():
        if PIN.match(line):
            found += [] if hashed else [pin]
            pin, hashed = line.split()[0], False
        hashed = hashed or HASH in line
    return found + ([] if hashed else [pin])
