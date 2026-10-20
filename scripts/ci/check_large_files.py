#!/usr/bin/env python3
"""No change starts tracking a file over 5 MB (PRD-262 US-042).

Every clone downloads the whole tree of the commit it checks out, the self-host
docs' ``git clone --depth 1`` included, so one large binary costs every installer
from then on, and its history stays in every full clone even after it is deleted.
Large media belongs in a release asset or outside the repository.

The check compares two commits and fails on each file that is over the limit at
``--head`` and either is new at its path (added, or the new side of a rename) or was
at most the limit at ``--base``. A file already over the limit at the base may change
without failing: the check is about what a change brings in. Sizes are git blob
sizes, so the working tree does not matter.

    python3 scripts/ci/check_large_files.py --base <rev> [--head <rev>] [--limit-bytes N]

``.github/workflows/large-files.yml`` runs it on every pull request (its merge
commit against the base branch) and on every push to main (the push against the
commit before it). The standard library only. Exit 0 = nothing over the limit came
in, 1 = something did, 2 = git could not answer.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
# 5 MB in decimal megabytes, the unit the PRD's sizes use (44.4 MB = 44,400,103 bytes).
LIMIT_BYTES = 5_000_000
BYTES_PER_MB = 1_000_000
GIT_TIMEOUT_SECONDS = 120
# Added, modified and type-changed paths. With --no-renames a rename is a delete
# plus an add, so the new path is checked as an added file.
DIFF_FILTER = "AMT"
ADDED = "A"


class GitError(RuntimeError):
    """git failed where the check needs an answer."""


@dataclass(frozen=True)
class LargeFile:
    """A file the change brings in over the limit."""

    path: str
    size: int
    added: bool

    def describe(self, limit: int) -> str:
        """One line for the log, sized in MB like the limit."""
        how = "is newly tracked" if self.added else f"grew past {limit / BYTES_PER_MB:g} MB"
        return f"{self.path} {how} at {self.size / BYTES_PER_MB:.1f} MB ({self.size:,} bytes)"


def _git(args: Sequence[str], cwd: Optional[Path]) -> subprocess.CompletedProcess:
    """git in ``cwd``, this repository when None."""
    return subprocess.run(["git", *args], cwd=str(cwd or REPO), capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_SECONDS)


def changed_paths(base: str, head: str, cwd: Optional[Path] = None) -> List[Tuple[str, str]]:
    """``(status, path)`` for each path added, modified or type-changed from base to head."""
    proc = _git(["diff", "--name-status", "-z", "--no-renames", f"--diff-filter={DIFF_FILTER}",
                 base, head, "--"], cwd)
    if proc.returncode != 0:
        raise GitError(f"git diff {base} {head} failed: {proc.stderr.strip()}")
    fields = [field for field in proc.stdout.split("\0") if field]
    return list(zip(fields[0::2], fields[1::2]))


def blob_size(rev: str, path: str, cwd: Optional[Path] = None) -> Optional[int]:
    """The size of ``path`` at ``rev`` in bytes, or None when it is not a blob there."""
    kind = _git(["cat-file", "-t", f"{rev}:{path}"], cwd)
    if kind.returncode != 0 or kind.stdout.strip() != "blob":
        return None  # absent at rev, or a submodule's commit
    proc = _git(["cat-file", "-s", f"{rev}:{path}"], cwd)
    if proc.returncode != 0:
        raise GitError(f"git cat-file -s {rev}:{path} failed: {proc.stderr.strip()}")
    return int(proc.stdout.strip())


def large_files(base: str, head: str, limit: int = LIMIT_BYTES,
                cwd: Optional[Path] = None) -> List[LargeFile]:
    """Each file the change from base to head brings in over ``limit`` bytes."""
    found = []
    for status, path in changed_paths(base, head, cwd):
        size = blob_size(head, path, cwd)
        if size is None or size <= limit:
            continue
        added = status == ADDED
        before = None if added else blob_size(base, path, cwd)
        if added or before is None or before <= limit:
            found.append(LargeFile(path=path, size=size, added=added))
    return found


def _report(found: Sequence[LargeFile], limit: int) -> None:
    for item in found:
        print(f"::error file={item.path}::{item.describe(limit)}; the limit is "
              f"{limit / BYTES_PER_MB:g} MB per file")
    if found:
        print(f"{len(found)} file(s) over {limit / BYTES_PER_MB:g} MB. Every clone downloads them: "
              "put large media in a release asset or outside the repository, and drop them "
              "from this change.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the commit the change starts from")
    parser.add_argument("--head", default="HEAD", help="the commit the change ends at (default HEAD)")
    parser.add_argument("--limit-bytes", type=int, default=LIMIT_BYTES,
                        help=f"the largest size a file may come in at (default {LIMIT_BYTES:,})")
    args = parser.parse_args(argv)
    try:
        found = large_files(args.base, args.head, args.limit_bytes)
    except GitError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 2
    _report(found, args.limit_bytes)
    if not found:
        print(f"No file over {args.limit_bytes / BYTES_PER_MB:g} MB came in between "
              f"{args.base} and {args.head}.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
