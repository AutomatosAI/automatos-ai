#!/usr/bin/env python3
"""Move ``edge`` to a published tag whose images booted (PRD-262 US-003, promote-edge).

images.yml publishes ``sha-<7>`` on every main push and no longer moves ``edge``.
smoke-images pulls and boots that tag on amd64 and arm64. When both pass, its
promote-edge job runs this tool, so an image that cannot boot never reaches ``edge``,
which plain ``docker compose up`` and the Helm chart pull by default.

It plans every image before it moves any, so the three ``edge`` tags never part ways
over a read error. For each image:

1. the tag must resolve on GHCR (its index digest);
2. the commit the current ``edge`` was built from is read from its revision label,
   and GitHub compares this run's commit with it. ``ahead`` or ``identical`` moves
   ``edge``. ``behind`` keeps it: a newer commit already booted and moved it, or this
   is a re-run of an old smoke. Anything else (``diverged``, no label) fails;
3. ``docker buildx imagetools create -t …:edge …:<tag>``. With one source and no
   annotations, imagetools pushes the source index's bytes unchanged, so ``edge``
   becomes the very digest cosign signed;
4. ``edge`` is read back and must be that digest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


registry = _sibling("image_registry")

EDGE_TAG = "edge"
REVISION_LABEL = "org.opencontainers.image.revision"
LABEL_ARCH = "amd64"  # every arch of one index is built from the same commit
MOVE, KEEP = ("ahead", "identical"), ("behind",)
GITHUB_API = "https://api.github.com"
API_TIMEOUT_SECONDS = 60
NOT_FOUND = 404

Run = Callable[..., object]


def edge_revision(ref: str) -> Optional[str]:
    """The commit ``ref`` (an ``edge`` tag) was built from; None when no ``edge`` exists yet."""
    try:
        _, config, _ = registry.image(ref, LABEL_ARCH)
    except urllib.error.HTTPError as error:
        if error.code == NOT_FOUND:
            return None
        raise
    return (config.get("config", {}).get("Labels") or {}).get(REVISION_LABEL) or ""


def compare(repo: str, base: str, head: str) -> str:
    """GitHub's verdict on ``head`` against ``base``: ahead, behind, identical or diverged."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GH_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{GITHUB_API}/repos/{repo}/compare/{base}...{head}?per_page=1",
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
        return json.load(response)["status"]


def decide(previous: Optional[str], status: Optional[str]) -> str:
    """``move`` or ``keep``; raises when the order of the two commits cannot be told."""
    if previous is None:
        return "move"  # no edge yet
    if not previous:
        raise ValueError(f"the current edge carries no {REVISION_LABEL} label")
    if status in MOVE:
        return "move"
    if status in KEEP:
        return "keep"
    raise ValueError(f"edge's commit {previous[:12]} and this one compare as {status!r}")


def plan(name: str, prefix: str, tag: str, sha: str, repo: str) -> dict:
    """What happens to ``name``'s ``edge``, read without writing anything."""
    source, edge = f"{prefix}{name}:{tag}", f"{prefix}{name}:{EDGE_TAG}"
    digest = registry.resolve(source)
    previous = edge_revision(edge)
    status = compare(repo, previous, sha) if previous else None
    return {"image": name, "source": source, "edge": edge, "digest": digest, "edge_was": previous,
            "compare": status, "action": decide(previous, status)}


def move(step: dict, run: Run = subprocess.run) -> dict:
    """Point ``edge`` at the planned digest, then read it back."""
    if step["action"] != "move":
        return {**step, "edge_now": None}
    run(["docker", "buildx", "imagetools", "create", "-t", step["edge"], step["source"]], check=True)
    landed = registry.resolve(step["edge"])
    if landed != step["digest"]:
        raise RuntimeError(f"{step['edge']} is {landed} after the move, not {step['digest']}")
    return {**step, "edge_now": landed}


def markdown(steps: Sequence[dict], sha: str) -> str:
    """The promote-edge step summary."""
    lines = [f"### promote-edge: `{sha[:7]}` (PRD-262 US-003)", "",
             "| Image | Tag | Digest | edge was at | Compared | Action |", "|---|---|---|---|---|---|"]
    for step in steps:
        lines.append(f"| {step['image']} | `{step['source'].rsplit(':', 1)[1]}` | `{step['digest'][:19]}…` "
                     f"| `{(step['edge_was'] or 'no edge')[:12]}` | {step['compare'] or '—'} | {step['action']} |")
    return "\n".join(lines) + "\n\n"


def promote(args: argparse.Namespace, run: Run = subprocess.run) -> int:
    """Plan every image, then move each ``edge`` the plan allows."""
    try:
        steps = [plan(name, args.prefix, args.tag, args.sha, args.repo) for name in args.image]
    except (urllib.error.URLError, LookupError, ValueError, OSError) as error:
        print(f"::error title=promote-edge::edge stays where it is: {error}")
        return 1
    moved = [move(step, run) for step in steps]
    for step in moved:
        print(f"{step['image']}: {step['action']} edge ({step['source']} = {step['digest']})")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as summary:
            summary.write(markdown(moved, args.sha))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The commit, its tag, the images and where they live."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sha", required=True, help="the commit the tag was built from")
    parser.add_argument("--tag", required=True, help="the tag that booted, e.g. sha-1bc4c07")
    parser.add_argument("--repo", required=True, help="owner/name, for GitHub's compare API")
    parser.add_argument("--prefix", required=True, help="ghcr.io/<owner>/automatos-")
    parser.add_argument("--image", action="append", required=True, help="api, frontend, workspace-worker")
    parser.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return promote(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
