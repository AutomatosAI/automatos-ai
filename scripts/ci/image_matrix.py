"""Which images an images.yml run builds, as the build job's matrix (PRD-262 US-001).

``images.yml``'s ``changes`` job pipes the pull request's changed files in, one per
line, and appends this script's output to ``$GITHUB_OUTPUT``:

* ``matrix``: ``{"include": [...]}``, one entry per image and architecture. A pull
  request builds only the images whose ``inputs`` it touches, on both arches; a
  change to a ``--shared-inputs`` path (the workflow, the budgets, the scan policy,
  these scripts) builds every image. ``--every`` (a push, a schedule, a manual run,
  or a file list the API could not give) builds every image.
* ``any``: ``true`` when the matrix has an entry; the build job is skipped otherwise.

An entry pushes to GHCR only when the run publishes (main or a ``v*`` tag) and the
image is published at all. Every other build pushes by digest to the job's own
``registry:2`` service on ``localhost:5000``, so it is measured without writing to
GHCR; ``registry`` names that service's image, and is empty when nothing needs it.
``edge`` is the published tag the update delta is measured against; an image that
is never published has none.

The weekly OS rebuild (PRD-262 US-011): with ``--no-cache true`` (a schedule run, or a
dispatch with ``no_cache``), each entry's ``no_cache_filters`` joins the stages its
image definition lists under ``no_cache_filters``: the stages whose OS packages ship.
BuildKit then rebuilds them, and every step after them, from fresh packages instead of
serving the cached apt or apk layer. Otherwise it is empty and the cache serves every
stage. ``--created`` (a schedule run's ``created_at``) adds
``weekly_tag=weekly-<yyyymmdd>``, the tag a schedule run publishes instead of
``sha-<7>``. smoke-images.yml works the tag out from the same timestamp, so the two
agree even when the run is re-run on a later day.

Patterns follow GitHub's path filters: ``*`` stops at ``/``, ``**`` does not.

    image_matrix.py --images JSON --shared-inputs JSON --runners JSON \\
        --owner OWNER --publish true|false --local-registry IMAGE \\
        [--no-cache true|false] [--created ISO-8601] [--every] < files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

LOCAL_REGISTRY_HOST = "localhost:5000"
# GitHub gives these a meaning this picker does not copy (``?`` and ``+`` repeat the
# character before them, ``[]`` is a class, a leading ``!`` negates), so a pattern
# using one is refused: read literally, it would silently never match.
UNSUPPORTED = re.compile(r"[?+\[\]!\\]")
GLOB_PART = re.compile(r"\*\*|\*|\?|[^*?]+")
WEEKLY_TAG_PREFIX = "weekly-"
WEEKLY_DATE_FORMAT = "%Y%m%d"


def glob_regex(pattern: str) -> re.Pattern:
    """A GitHub path-filter glob as an anchored regex: ``*`` and ``**`` only.

    GitHub's other filter syntax (``?``, ``+``, ``[]``, a leading ``!``) raises
    ValueError instead of being misread.
    """
    if UNSUPPORTED.search(pattern):
        raise ValueError(f"path filter {pattern!r} uses syntax image_matrix.py does not support")
    parts = []
    for token in GLOB_PART.findall(pattern):
        if token == "**":
            parts.append(".*")
        elif token == "*":
            parts.append("[^/]*")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts) + r"\Z")


def touches(files: Sequence[str], patterns: Iterable[str]) -> bool:
    """True when any file matches any pattern."""
    regexes = [glob_regex(pattern) for pattern in patterns]
    return any(regex.match(path) for path in files for regex in regexes)


def selected(images: List[dict], shared_inputs: List[str], files: Sequence[str], every: bool) -> List[str]:
    """The names of the images this change builds, in ``images`` order."""
    if every or touches(files, shared_inputs):
        return [image["name"] for image in images]
    return [image["name"] for image in images if touches(files, image["inputs"])]


def entry(image: dict, arch: str, runner: str, publish_run: bool, owner: str, local_registry: str,
          no_cache: bool = False) -> dict:
    """One build job: what to build, where it runs, where it pushes, what it is compared with."""
    name = image["name"]
    published = f"ghcr.io/{owner.lower()}/automatos-{name}"
    to_ghcr = publish_run and image["publish"]
    return {
        "name": name,
        "context": image["context"],
        "target": image["target"],
        "build_args": "\n".join(image["build_args"]),
        "cache_mode": image["cache_mode"],
        "no_cache_filters": ",".join(image["no_cache_filters"]) if no_cache else "",
        "arch": arch,
        "runner": runner,
        "publish": to_ghcr,
        "image": published if to_ghcr else f"{LOCAL_REGISTRY_HOST}/automatos-{name}",
        "edge": f"{published}:edge" if image["publish"] else "",
        "registry": "" if to_ghcr else local_registry,
    }


def matrix(images: List[dict], names: List[str], runners: dict, publish_run: bool,
           owner: str, local_registry: str, no_cache: bool = False) -> dict:
    """The build job's ``strategy.matrix`` for ``names``."""
    return {"include": [
        entry(image, arch, runners[arch], publish_run, owner, local_registry, no_cache)
        for image in images if image["name"] in names
        for arch in image["arches"]
    ]}


def weekly_tag(created: str) -> str:
    """``weekly-<yyyymmdd>``: the UTC day a schedule run was created (GitHub's ``created_at``).

    The run's creation time, not the clock: it never changes, so images.yml and
    smoke-images.yml agree on the tag even when a run is re-run on another day.
    """
    moment = datetime.fromisoformat(created.strip())
    if moment.tzinfo is None:
        raise ValueError(f"created_at carries no time zone: {created!r}")
    return WEEKLY_TAG_PREFIX + moment.astimezone(timezone.utc).strftime(WEEKLY_DATE_FORMAT)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, help="the image definitions, JSON")
    parser.add_argument("--shared-inputs", required=True, help="paths that rebuild every image, JSON")
    parser.add_argument("--runners", required=True, help='{"<arch>": "<runner label>"}, JSON')
    parser.add_argument("--owner", required=True)
    parser.add_argument("--publish", required=True, choices=("true", "false"))
    parser.add_argument("--local-registry", required=True, help="the registry service's image")
    parser.add_argument("--no-cache", default="false", choices=("true", "false"),
                        help="rebuild each image's no_cache_filters stages (the weekly OS rebuild)")
    parser.add_argument("--created", help="a schedule run's created_at: adds weekly_tag")
    parser.add_argument("--every", action="store_true", help="build every image")
    args = parser.parse_args(argv)
    images = json.loads(args.images)
    files = [line.strip() for line in sys.stdin if line.strip()]
    names = selected(images, json.loads(args.shared_inputs), files, args.every)
    built = matrix(images, names, json.loads(args.runners), args.publish == "true",
                   args.owner, args.local_registry, args.no_cache == "true")
    print(f"matrix={json.dumps(built, separators=(',', ':'))}")
    print(f"any={'true' if built['include'] else 'false'}")
    if args.created:
        print(f"weekly_tag={weekly_tag(args.created)}")
    print(f"images building: {', '.join(names) or 'none'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
