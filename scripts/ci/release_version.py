#!/usr/bin/env python3
"""The release a ``v*`` tag names (PRD-279 US-101, D-1).

images.yml runs this in its ``changes`` job, on a ``v*`` tag only, before any image
builds. It reads the tag and the chart's ``Chart.yaml`` at the tagged commit and
writes the job outputs the rest of the run reads:

* ``version``: the tag without its ``v``, e.g. ``0.7.0`` or ``0.8.0-rc.1``. The images
  are tagged with it, and the chart is packaged with it as both its ``version`` and its
  ``appVersion``.
* ``candidate``: ``true`` for a tag with a suffix (``v0.8.0-rc.1``). A candidate does
  every step of a release, but its images do not move the ``MAJOR.MINOR`` tag.
* ``major_minor``: ``0.8``, the release line a full release also tags.

It fails, and so the run builds and publishes nothing, on a tag that is not
``vMAJOR.MINOR.PATCH`` with an optional ``-suffix``, and on a version below the
chart's own: a release never goes backwards.

    python3 scripts/ci/release_version.py --ref refs/tags/v0.7.0 --chart-yaml Chart.yaml \\
        --outputs "$GITHUB_OUTPUT" --summary "$GITHUB_STEP_SUMMARY"
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

TAG_PREFIX = "refs/tags/v"
_NUMBER = r"(?:0|[1-9][0-9]*)"
# Semantic versioning's core and pre-release; no build metadata, which an OCI tag cannot hold.
_VERSION = re.compile(
    rf"^(?P<major>{_NUMBER})\.(?P<minor>{_NUMBER})\.(?P<patch>{_NUMBER})"
    r"(?:-(?P<suffix>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_CHART_VERSION = re.compile(r"^version:\s*['\"]?(?P<version>[^'\"\s#]+)", re.M)
# An OCI tag is at most 128 characters.
MAX_TAG_LENGTH = 128


class ReleaseError(ValueError):
    """A tag, or a chart, that a release cannot be cut from."""


@dataclass(frozen=True)
class Release:
    """One version, as the tag names it."""

    version: str
    core: Tuple[int, int, int]
    suffix: str

    @property
    def candidate(self) -> bool:
        """A tag with a suffix is a release candidate."""
        return bool(self.suffix)

    @property
    def major_minor(self) -> str:
        """The release line, ``MAJOR.MINOR``."""
        return f"{self.core[0]}.{self.core[1]}"


def parse_version(text: str) -> Release:
    """``0.8.0-rc.1`` as a Release; ReleaseError when it is not a version."""
    match = _VERSION.fullmatch(text)
    if not match or len(text) > MAX_TAG_LENGTH:
        raise ReleaseError(f"{text!r} is not MAJOR.MINOR.PATCH with an optional -suffix")
    core = (int(match["major"]), int(match["minor"]), int(match["patch"]))
    return Release(version=text, core=core, suffix=match["suffix"] or "")


def parse_ref(ref: str) -> Release:
    """The release a ``refs/tags/v…`` ref names."""
    if not ref.startswith(TAG_PREFIX):
        raise ReleaseError(f"{ref!r} is not a version tag (refs/tags/vMAJOR.MINOR.PATCH)")
    return parse_version(ref[len(TAG_PREFIX):])


def chart_version(chart_yaml: str) -> Tuple[int, int, int]:
    """The core of the top-level ``version:`` in a Chart.yaml."""
    match = _CHART_VERSION.search(chart_yaml)
    if not match:
        raise ReleaseError("Chart.yaml has no top-level `version:`")
    return parse_version(match["version"]).core


def check_not_backwards(release: Release, chart: Tuple[int, int, int]) -> None:
    """ReleaseError when the tag's version is below the chart's own (PRD-279 D-1)."""
    if release.core < chart:
        shown = ".".join(str(part) for part in chart)
        raise ReleaseError(f"v{release.version} is below the chart's version {shown}: a release never goes backwards")


def outputs(release: Release) -> List[str]:
    """The ``name=value`` lines for $GITHUB_OUTPUT."""
    return [f"version={release.version}", f"candidate={str(release.candidate).lower()}",
            f"major_minor={release.major_minor}"]


def summary(release: Release) -> str:
    """One line for the step summary."""
    kind = "a release candidate: the images do not move" if release.candidate else "a release: the images also move"
    return f"Version **{release.version}**, {kind} `{release.major_minor}`.\n"


def _append(path: Optional[str], lines: List[str]) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write("".join(f"{line}\n" for line in lines))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="the run's ref ($GITHUB_REF)")
    parser.add_argument("--chart-yaml", required=True, help="charts/automatos/Chart.yaml at the tagged commit")
    parser.add_argument("--outputs", help="append name=value lines here ($GITHUB_OUTPUT)")
    parser.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)
    try:
        release = parse_ref(args.ref)
        check_not_backwards(release, chart_version(Path(args.chart_yaml).read_text(encoding="utf-8")))
    except (ReleaseError, OSError) as error:
        print(f"::error title=Release version::{error}")
        return 1
    _append(args.outputs, outputs(release))
    _append(args.summary, [summary(release).rstrip("\n")])
    print("\n".join(outputs(release)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
