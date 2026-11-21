#!/usr/bin/env python3
"""Before a ``v*`` tag builds anything (PRD-279 FR-6, §5.5).

images.yml's ``changes`` job runs this on a ``v*`` tag only, after the version is read
and before any image builds. It refuses the run, so nothing is built or published, when:

* **a version would be replaced.** ``<image>:<version>`` already exists for any image
  the release publishes. The registry would move the tag; a version is never moved.
  A run whose version tags were never applied (it failed before the tag step) can
  run again; once they exist, the next version is cut instead.
* **the commit is on no release line.** The tagged commit must be in the default
  branch, or in ``release/MAJOR.MINOR`` for the tag's own line (the patch branch of
  §5.5). GitHub's compare API answers it with the job's read-only token; nothing is
  fetched.
* **either answer cannot be read.** A registry or API that does not answer stops the
  run: the check never passes on what it could not read.

The images are public, so the registry is read anonymously.

    python3 scripts/ci/release_preflight.py --version 0.7.0 --sha "$GITHUB_SHA" \\
        --repo "$GITHUB_REPOSITORY" --default-branch main \\
        --prefix ghcr.io/automatosai/automatos- --image api --image frontend --image workspace-worker
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence

GITHUB_API = "https://api.github.com"
API_TIMEOUT_SECONDS = 30
NOT_FOUND = 404
# GitHub's compare status of a branch against the commit: the branch holds the commit.
CONTAINS = ("ahead", "identical")


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = _sibling("image_registry")
release_version = _sibling("release_version")

# A ref -> True (exists), False (the registry says it does not), None (not read).
State = Callable[[str], Optional[bool]]
# A branch -> GitHub's compare status of it against the commit, or None when it does not exist.
Compare = Callable[[str], Optional[str]]


def published(ref: str) -> Optional[bool]:
    """Whether ``ref``'s tag is on its registry; None when the registry did not answer."""
    try:
        registry.resolve(ref)
    except urllib.error.HTTPError as error:
        return False if error.code == NOT_FOUND else None
    except registry.READ_ERRORS:
        return None
    return True


def version_problems(refs: Sequence[str], state: State) -> List[str]:
    """Every version tag that exists already, or could not be read."""
    problems = []
    for ref in refs:
        found = state(ref)
        if found is True:
            problems.append(f"{ref} is already published, and a version is never replaced: cut the next version")
        elif found is None:
            problems.append(f"could not tell whether {ref} is published, so nothing was built")
    return problems


def release_lines(version: str, default_branch: str) -> List[str]:
    """The branches a version may be cut from: the default branch, then its own patch branch."""
    return [default_branch, f"release/{release_version.parse_version(version).major_minor}"]


def line_holding(branches: Sequence[str], compare: Compare) -> Optional[str]:
    """The first branch that holds the commit, or None."""
    return next((branch for branch in branches if compare(branch) in CONTAINS), None)


def github_compare(repo: str, sha: str, token: str = "") -> Compare:
    """GitHub's compare of each branch (head) against ``sha`` (base); None for a missing branch."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def compare(branch: str) -> Optional[str]:
        head = urllib.parse.quote(branch, safe="/")
        request = urllib.request.Request(f"{GITHUB_API}/repos/{repo}/compare/{sha}...{head}?per_page=1",
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                return json.load(response)["status"]
        except urllib.error.HTTPError as error:
            if error.code == NOT_FOUND:
                return None
            raise
    return compare


def check(args: argparse.Namespace, state: State, compare: Compare) -> List[str]:
    """Every reason the run must not build; an empty list lets it go on."""
    refs = [f"{args.prefix}{name}:{args.version}" for name in args.image]
    problems = version_problems(refs, state)
    branches = release_lines(args.version, args.default_branch)
    try:
        holder = line_holding(branches, compare)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        return [*problems, f"could not tell whether {args.sha[:12]} is on a release line ({error})"]
    if holder is None:
        problems.append(f"{args.sha[:12]} is on neither {' nor '.join(branches)}: a version is cut from main, "
                        "or from the release branch of its own MAJOR.MINOR")
    return problems


def main(argv: Optional[List[str]] = None, state: State = published, compare: Optional[Compare] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--version", "--sha", "--repo", "--default-branch", "--prefix"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--image", action="append", required=True, help="a published image's name (repeat)")
    parser.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)
    compare = compare or github_compare(args.repo, args.sha, os.environ.get("GH_TOKEN", ""))
    problems = check(args, state, compare)
    for problem in problems:
        print(f"::error title=Release preflight::{problem}")
    if args.summary and not problems:
        with open(args.summary, "a", encoding="utf-8") as out:
            out.write(f"Preflight: no image is published as `{args.version}` yet, and the commit is on a release line.\n")
    print(f"preflight: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
