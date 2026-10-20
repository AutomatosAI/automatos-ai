"""Fail when any named distribution is installed (PRD-262 US-014, US-015).

CI runs it inside a built image, with the image's own Python, to prove that packages
left the image:

    docker run --rm -i --entrypoint python IMAGE - NAME... < scripts/ci/image_dists_absent.py

It reads the installed distributions' metadata, so it finds a package whether or not
anything imports it. Standard library only, because the image has nothing else to
lean on. Exit 0 when none is installed, 1 when any is, 2 when no name was given.
"""
from __future__ import annotations

import sys
from importlib import metadata
from typing import List, Optional


def _version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed(names: List[str]) -> List[str]:
    """`name version` for each of ``names`` that is installed, in the order given."""
    versions = ((name, _version(name)) for name in names)
    return [f"{name} {version}" for name, version in versions if version]


def main(argv: List[str]) -> int:
    if not argv:
        print("::error::no distribution names to look for")
        return 2
    found = installed(argv)
    for item in found:
        print(f"::error::the image has {item}")
    if not found:
        print(f"none of these is installed: {' '.join(argv)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
