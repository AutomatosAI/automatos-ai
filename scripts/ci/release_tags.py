#!/usr/bin/env python3
"""Put the version tags on a release's images, last (PRD-279 US-101, US-103, FR-6).

On a ``v*`` tag the merge job publishes each image only under a staging tag
(``release-staging-<run id>-<attempt>``), signs its digest and checks the signature.
images.yml's ``release-images`` job runs this once, after every merge leg has done so:
nothing carries a version tag before it has been scanned, signed and its signature
checked.

It plans every image before it tags any, so the images never part ways over a read:

* ``<image>:<version>`` absent: it is tagged. Already on this run's digest: nothing to
  do (a re-run of this job after it stopped part-way finishes the set). On another
  digest, or unreadable: nothing is tagged, and the job fails. A version is never moved.
* ``<image>:<MAJOR.MINOR>``, for a full release only (never a candidate): moved to the
  digest. A release line is meant to move.

Then each tag is added with ``docker buildx imagetools create -t … <image>@<digest>``,
which writes no new manifest, and read back: it must be that digest.

    python3 scripts/ci/release_tags.py --digests DIR --prefix ghcr.io/o/automatos- \\
        --version 0.7.0 --image api --image frontend --image workspace-worker
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

NOT_FOUND = 404
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = _sibling("image_registry")
release_version = _sibling("release_version")

# A tagged ref -> the digest it points at, None when absent; raises TagError when unread.
Current = Callable[[str], Optional[str]]


class TagError(ValueError):
    """A version tag that cannot be applied."""


@dataclass(frozen=True)
class Plan:
    """One image: its digest and the tags to point at it."""

    image: str
    digest: str
    tags: Tuple[str, ...]

    @property
    def source(self) -> str:
        """The image by digest, what each tag is pointed at."""
        return f"{self.image}@{self.digest}"


def current(ref: str) -> Optional[str]:
    """The digest ``ref`` points at on its registry, or None when the tag is absent."""
    try:
        return registry.resolve(ref)
    except urllib.error.HTTPError as error:
        if error.code == NOT_FOUND:
            return None
        raise TagError(f"could not read {ref}: {error}") from error
    except registry.READ_ERRORS as error:
        raise TagError(f"could not read {ref}: {error}") from error


def read_digests(directory: Path, names: Sequence[str]) -> Dict[str, str]:
    """``{image name: sha256:…}`` from one file per image, as the merge job writes them."""
    digests = {}
    for name in names:
        path = directory / name
        value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not DIGEST.match(value):
            raise TagError(f"{name}: no signed digest from the merge job (got {value[:80]!r})")
        digests[name] = value
    return digests


def _tags_for(image: str, digest: str, version: str, line: str, lookup: Current) -> Tuple[Tuple[str, ...], str]:
    """The tags to apply to one image, and a problem ("" when none)."""
    pinned = f"{image}:{version}"
    try:
        found = lookup(pinned)
    except TagError as error:
        return (), str(error)
    if found not in (None, digest):
        return (), f"{pinned} already points at {found}, and a version is never moved: cut the next version"
    tags = () if found == digest else (pinned,)
    return tags + ((f"{image}:{line}",) if line else ()), ""


def plan(digests: Mapping[str, str], prefix: str, version: str, lookup: Current) -> List[Plan]:
    """Every image's tags, read before any is written; TagError lists every problem."""
    release = release_version.parse_version(version)
    line = "" if release.candidate else release.major_minor
    plans, problems = [], []
    for name, digest in sorted(digests.items()):
        tags, problem = _tags_for(f"{prefix}{name}", digest, version, line, lookup)
        problems += [problem] if problem else []
        plans.append(Plan(image=f"{prefix}{name}", digest=digest, tags=tags))
    if problems:
        raise TagError("; ".join(problems))
    return plans


def apply(plans: Sequence[Plan], lookup: Current, run=subprocess.run) -> List[str]:
    """Point each plan's tags at its digest, then read every one back."""
    applied = []
    for step in (step for step in plans if step.tags):
        tag_args = [arg for tag in step.tags for arg in ("-t", tag)]
        run(["docker", "buildx", "imagetools", "create", *tag_args, step.source], check=True)
        for tag in step.tags:
            landed = lookup(tag)
            if landed != step.digest:
                raise TagError(f"{tag} is {landed} after tagging, not {step.digest}")
            applied.append(tag)
    return applied


def main(argv: Optional[List[str]] = None, lookup: Current = current, run=subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--digests", "--prefix", "--version"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--image", action="append", required=True, help="a published image's name (repeat)")
    parser.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)
    try:
        plans = plan(read_digests(Path(args.digests), args.image), args.prefix, args.version, lookup)
        applied = apply(plans, lookup, run)
    except (TagError, release_version.ReleaseError, subprocess.CalledProcessError) as error:
        print(f"::error title=Version tags::{error}")
        return 1
    lines = [f"- `{tag}`" for tag in applied] or ["- every version tag was already on this run's digests"]
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as out:
            out.write(f"### Version {args.version}: image tags\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
