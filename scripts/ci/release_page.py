#!/usr/bin/env python3
"""Create a version's release page on GitHub (PRD-279 US-301), once its artifacts are proven.

images.yml's ``release-notes`` job runs this on a ``v*`` tag, after every image is signed,
checked and tagged with the version and the chart is published, signed and checked. It is
the edges around ``release_history`` and ``release_notes``, which hold every decision:

* **Git** (``git`` with list arguments, never a shell; the checkout's full history): the
  version tags reachable from the tagged commit and every version tag, the first-parent log
  between the previous release and this one, and each Alembic revision file at the two tags
  (``git ls-tree`` and ``git show <tag>:<path>``: the files are read as text, never imported).
* **The repository's files at the tag:** the upgrade file, ``scripts/ci/release_upgrade_from.txt``.
  A missing or malformed file, or one naming a version that is not tagged or not below this
  one, stops the job before anything is written.
* **GitHub** (the REST API, with the job's own token, which may write contents and nothing
  else): ``GET /repos/{repo}/releases/tags/{tag}``, and only on a 404 ``POST
  /repos/{repo}/releases``. A release page that exists is never changed: the job stops with
  an error, and any answer but 404 stops it too. A release candidate is created as a
  pre-release; a full release is marked the latest only when no full release has a higher
  version (a patch to an older line is not).

The notes are written to ``--notes`` first, so the job's log and summary show the page even
when GitHub refuses it.
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
from typing import Callable, Dict, List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _sibling("release_history")
notes_md = _sibling("release_notes")
release_tags = _sibling("release_tags")
release_version = _sibling("release_version")

GITHUB_API = "https://api.github.com"
API_TIMEOUT_SECONDS = 30
NOT_FOUND = 404
VERSIONS_DIR = "orchestrator/alembic/versions"
UPGRADE_FILE = Path(__file__).resolve().with_name("release_upgrade_from.txt")
TAG_PATTERN = f"{history.TAG_PREFIX}*"

# git's arguments -> its standard output; raises on a non-zero exit.
Git = Callable[[Sequence[str]], str]
# (method, path, payload) -> (status, parsed answer)
Call = Callable[[str, str, Optional[dict]], tuple]


class PageError(RuntimeError):
    """A release page that must not be created."""


# --------------------------------------------------------------------------- git


def run_git(args: Sequence[str]) -> str:
    """``git <args>`` in the checkout, its output as text."""
    done = subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return done.stdout


def version_tags(git: Git, merged: Optional[str] = None) -> List[str]:
    """Every ``v*`` tag, or only those reachable from the commit ``merged``."""
    args = ["tag", "--list"] + (["--merged", merged] if merged else []) + [TAG_PATTERN]
    return [line.strip() for line in git(args).splitlines() if line.strip()]


def revision_files(git: Git, ref: str) -> Dict[str, str]:
    """``{path: source}`` of each ``*.py`` directly in the versions folder at ``ref``."""
    listing = git(["ls-tree", "-z", ref, "--", f"{VERSIONS_DIR}/"])
    files = {}
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        if meta.split()[1] == "blob" and path.endswith(".py") and Path(path).name != "__init__.py":
            files[path] = git(["show", f"{ref}:{path}"])
    return files


def read_notes(args: argparse.Namespace, git: Git, upgrade_text: str, digests: Dict[str, str]) -> "notes_md.Notes":
    """Everything the page says, from git, the upgrade file and the digests."""
    release_version.parse_version(args.version)
    if not notes_md.COMMIT.fullmatch(args.sha):
        raise PageError(f"{args.sha!r} is not a full commit id")
    tag = f"{history.TAG_PREFIX}{args.version}"
    every_tag = version_tags(git)
    if tag not in every_tag:
        raise PageError(f"there is no tag {tag} in the checkout")
    upgrade_from = history.read_upgrade_from(upgrade_text)
    history.check_upgrade_from(upgrade_from, args.version, every_tag)
    previous = history.previous_release(args.version, version_tags(git, args.sha))
    log = git(["log", "--first-parent", "-z", "--format=%H%n%P%n%s%n%b", f"refs/tags/{previous}..{args.sha}"]) \
        if previous else ""
    before = history.revisions(revision_files(git, f"refs/tags/{previous}")) if previous else None
    database = history.database_change(before, history.revisions(revision_files(git, args.sha)))
    return notes_md.Notes(
        repository=args.repo, version=args.version, sha=args.sha, previous=previous,
        changes=tuple(history.parse_log(log)), database=database, upgrade_from=upgrade_from,
        images=notes_md.images_of(digests, args.image), chart_digest=args.chart_digest)


# --------------------------------------------------------------------------- GitHub


def github(token: str) -> Call:
    """GitHub's REST API with ``token``: (status, parsed body); an HTTP error is an answer too."""
    def call(method: str, path: str, payload: Optional[dict]) -> tuple:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{GITHUB_API}{path}", data=data, method=method, headers={
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                body = response.read()
                return response.status, json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            return error.code, None
    return call


def release_request(notes: "notes_md.Notes", body: str, latest: bool) -> dict:
    """The body of ``POST /repos/{repo}/releases`` for this version."""
    candidate = release_version.parse_version(notes.version).candidate
    return {"tag_name": notes.tag, "name": notes_md.title(notes.version), "body": body, "draft": False,
            "prerelease": candidate, "make_latest": "true" if latest and not candidate else "false"}


def publish(notes: "notes_md.Notes", body: str, latest: bool, call: Call) -> str:
    """Create the release page, never over one that exists; its address."""
    status, _ = call("GET", f"/repos/{notes.repository}/releases/tags/{notes.tag}", None)
    if status != NOT_FOUND:
        reason = "it already has a release page, which is never replaced" if status == 200 else \
            f"GitHub answered {status} when asked whether it has one"
        raise PageError(f"{notes.tag}: no release page was created: {reason}")
    status, answer = call("POST", f"/repos/{notes.repository}/releases", release_request(notes, body, latest))
    if status != 201 or not isinstance(answer, dict):
        raise PageError(f"{notes.tag}: GitHub refused the release page ({status})")
    return str(answer.get("html_url", ""))


# --------------------------------------------------------------------------- the command


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--repo", "--version", "--sha", "--digests", "--chart-digest", "--notes"):
        root.add_argument(flag, required=True)
    root.add_argument("--image", action="append", required=True, help="a published image's name (repeat)")
    root.add_argument("--upgrade-file", default=str(UPGRADE_FILE))
    root.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    root.add_argument("--publish", action="store_true", help="create the release page on GitHub")
    return root


def _upgrade_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PageError(f"the upgrade file {Path(path).name} is missing: every release reads it") from error


def build(args: argparse.Namespace, git: Git) -> tuple:
    """The notes and their Markdown, written to ``--notes``."""
    digests = release_tags.read_digests(Path(args.digests), args.image)
    notes = read_notes(args, git, _upgrade_text(args.upgrade_file), digests)
    body = notes_md.render(notes)
    Path(args.notes).write_text(body, encoding="utf-8")
    return notes, body


ERRORS = (PageError, history.HistoryError, notes_md.NotesError, release_tags.TagError, release_version.ReleaseError,
          subprocess.CalledProcessError, urllib.error.URLError, OSError, ValueError)


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(text)


def main(argv: Optional[List[str]] = None, git: Git = run_git, call: Optional[Call] = None) -> int:
    args = parser().parse_args(argv)
    try:
        notes, body = build(args, git)
        _append(args.summary, f"Release notes for `{notes.tag}`:\n\n{body}\n")
        where = "written, not published (no --publish)"
        if args.publish:
            latest = history.is_latest(args.version, version_tags(git))
            where = publish(notes, body, latest, call or github(os.environ.get("GH_TOKEN", "")))
    except ERRORS as error:
        print(f"::error title=Release page::{error}")
        return 1
    _append(args.summary, f"Release page: {where}\n")
    print(f"release page: {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
