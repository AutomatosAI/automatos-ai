#!/usr/bin/env python3
"""A version's release page on GitHub (PRD-279 US-301): checked before anything builds, created last.

images.yml runs this twice on a ``v*`` tag, with the same functions each time:

* ``check``, in the ``changes`` job beside the release preflight, before any image is built:
  everything the page says that does not need the digests. It reads the upgrade file, finds
  the previous release, reads the history and both sets of migrations, renders the notes with
  placeholder digests, and asks GitHub (with that job's read-only token) whether the tag
  already has a page. Anything the final run would refuse stops the run here, so a mistake
  never leaves images and a chart published with no page.
* ``publish``, in the ``release-notes`` job, after every image is signed, checked and tagged
  and the chart is published and checked: the same reads and checks again, with the real
  digests, then the page.

It is the edges around ``release_history`` and ``release_notes``, which hold every decision:

* **Git** (``git`` with list arguments, never a shell; the checkout must hold the full history,
  and a shallow one is refused): the version tags reachable from the tagged commit and every
  version tag, the first-parent log between the previous release and this one, and each
  Alembic revision file at the two tags (``git ls-tree`` and ``git show <tag>:<path>``: the
  files are read as text, never imported).
* **The upgrade file at the tag,** ``scripts/ci/release_upgrade_from.txt``: missing,
  malformed, naming a version that is not tagged or not below this one, or ``none`` when there
  is a previous release, stops the run.
* **GitHub** (the REST API): ``GET /repos/{repo}/releases/tags/{tag}``; only on a 404, and only
  in ``publish``, ``POST /repos/{repo}/releases`` naming the tagged commit as its target. A
  page that exists is never changed, and any answer but 404 stops the run. A release candidate
  is a pre-release; a full release is marked the latest only when no full release has a higher
  version (a patch to an older line is not).

Error text is printed as a workflow annotation, escaped as GitHub's toolkit escapes it, so a
path or a message holding a line break cannot start a workflow command of its own. The step
summary is never what fails a run: once the page is created, a summary that cannot be written
is a warning.
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
CREATED = 201
VERSIONS_DIR = "orchestrator/alembic/versions"
UPGRADE_FILE = Path(__file__).resolve().with_name("release_upgrade_from.txt")
TAG_PATTERN = f"{history.TAG_PREFIX}*"
# What the early check renders in place of the digests, which do not exist before the build.
PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
TITLE = "Release page"

# git's arguments -> its standard output; raises on a non-zero exit.
Git = Callable[[Sequence[str]], str]
# (method, path, payload) -> (status, parsed answer)
Call = Callable[[str, str, Optional[dict]], tuple]


class PageError(RuntimeError):
    """A release page that must not be created."""


# --------------------------------------------------------------------------- annotations


def escaped(text: str, property_value: bool = False) -> str:
    """``text`` for a workflow command: ``%``, CR and LF encoded (and ``:`` and ``,`` in a property)."""
    text = str(text).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if property_value else text


def annotation(level: str, message: str) -> str:
    """One ``::error``/``::warning`` line: nothing in ``message`` can end it or start another."""
    return f"::{level} title={escaped(TITLE, True)}::{escaped(message)}"


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


def refuse_shallow(git: Git) -> None:
    """PageError for a shallow checkout, whose reachable tags would make this look like the first release."""
    if git(["rev-parse", "--is-shallow-repository"]).strip() != "false":
        raise PageError("the checkout is shallow, so the earlier releases and their history cannot be read: "
                        "check it out with fetch-depth: 0")


def revision_files(git: Git, ref: str) -> Dict[str, str]:
    """``{path: source}`` of each ``*.py`` directly in the versions folder at ``ref``."""
    listing = git(["ls-tree", "-z", ref, "--", f"{VERSIONS_DIR}/"])
    files = {}
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        if meta.split()[1] == "blob" and path.endswith(".py") and Path(path).name != "__init__.py":
            files[path] = git(["show", f"{ref}:{path}"])
    return files


def read_notes(args: argparse.Namespace, git: Git, upgrade_text: str, digests: Dict[str, str],
               chart_digest: str) -> "notes_md.Notes":
    """Everything the page says, from git, the upgrade file and the digests; PageError or HistoryError on a refusal."""
    release_version.parse_version(args.version)
    if not notes_md.COMMIT.fullmatch(args.sha):
        raise PageError(f"{args.sha!r} is not a full commit id")
    refuse_shallow(git)
    tag = f"{history.TAG_PREFIX}{args.version}"
    every_tag = version_tags(git)
    if tag not in every_tag:
        raise PageError(f"there is no tag {tag} in the checkout")
    previous = history.previous_release(args.version, version_tags(git, args.sha))
    upgrade_from = history.read_upgrade_from(upgrade_text)
    history.check_upgrade_from(upgrade_from, args.version, every_tag, previous)
    log = git(["log", "--first-parent", "-z", "--format=%H%n%P%n%s%n%b", f"refs/tags/{previous}..{args.sha}"]) \
        if previous else ""
    before = history.revisions(revision_files(git, f"refs/tags/{previous}")) if previous else None
    database = history.database_change(before, history.revisions(revision_files(git, args.sha)))
    return notes_md.Notes(
        repository=args.repo, version=args.version, sha=args.sha, previous=previous,
        changes=tuple(history.parse_log(log)), database=database, upgrade_from=upgrade_from,
        images=notes_md.images_of(digests, args.image), chart_digest=chart_digest)


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
    """The body of ``POST /repos/{repo}/releases``: the tag, and the tagged commit as its target.

    GitHub uses ``target_commitish`` only when the tag does not exist, to create it there; the tag
    exists when this runs, so naming the commit means a tag removed in the meantime can never be
    made again anywhere but the commit this page was written for.
    """
    candidate = release_version.parse_version(notes.version).candidate
    return {"tag_name": notes.tag, "target_commitish": notes.sha, "name": notes_md.title(notes.version),
            "body": body, "draft": False, "prerelease": candidate,
            "make_latest": "true" if latest and not candidate else "false"}


def refuse_existing(notes: "notes_md.Notes", call: Call) -> None:
    """PageError unless GitHub answers that the tag has no release page (404)."""
    status, _ = call("GET", f"/repos/{notes.repository}/releases/tags/{notes.tag}", None)
    if status != NOT_FOUND:
        reason = "it already has a release page, which is never replaced" if status == 200 else \
            f"GitHub answered {status} when asked whether it has one"
        raise PageError(f"{notes.tag}: no release page is created: {reason}")


def create(notes: "notes_md.Notes", body: str, latest: bool, call: Call) -> str:
    """``POST`` the release page; its address. Call ``refuse_existing`` first."""
    status, answer = call("POST", f"/repos/{notes.repository}/releases", release_request(notes, body, latest))
    if status != CREATED or not isinstance(answer, dict):
        raise PageError(f"{notes.tag}: GitHub refused the release page ({status})")
    return str(answer.get("html_url", ""))


def publish(notes: "notes_md.Notes", body: str, latest: bool, call: Call) -> str:
    """Create the release page, never over one that exists; its address."""
    refuse_existing(notes, call)
    return create(notes, body, latest, call)


# --------------------------------------------------------------------------- the commands


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    early = commands.add_parser("check", help="before the build: refuse anything the page would be refused for")
    final = commands.add_parser("publish", help="after the chart: write the notes and create the page")
    for command in (early, final):
        for flag in ("--repo", "--version", "--sha"):
            command.add_argument(flag, required=True)
        command.add_argument("--image", action="append", required=True, help="a published image's name (repeat)")
        command.add_argument("--upgrade-file", default=str(UPGRADE_FILE))
        command.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    for flag in ("--digests", "--chart-digest", "--notes"):
        final.add_argument(flag, required=True)
    return root


def _upgrade_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PageError(f"the upgrade file {Path(path).name} is missing: every release reads it") from error


def _note(path: Optional[str], text: str) -> None:
    """Append to the step summary; a summary that cannot be written is a warning, never a failure."""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as out:
            out.write(text)
    except OSError as error:
        print(annotation("warning", f"the step summary could not be written ({error})"))


def check(args: argparse.Namespace, git: Git, call: Call) -> str:
    """Every refusal ``publish`` would make that needs no digest; the notes, with placeholder digests."""
    digests = {name: PLACEHOLDER_DIGEST for name in args.image}
    notes = read_notes(args, git, _upgrade_text(args.upgrade_file), digests, PLACEHOLDER_DIGEST)
    body = notes_md.render(notes)
    refuse_existing(notes, call)
    _note(args.summary, f"Release page for `{notes.tag}`: checked before the build. The notes, with the "
                        f"digests still to come:\n\n{body}\n")
    return f"{notes.tag} checked: its release page can be written"


def publish_page(args: argparse.Namespace, git: Git, call: Call) -> str:
    """The same refusals as ``check``; then the notes, written to ``--notes`` and the summary; then the page."""
    digests = release_tags.read_digests(Path(args.digests), args.image)
    notes = read_notes(args, git, _upgrade_text(args.upgrade_file), digests, args.chart_digest)
    body = notes_md.render(notes)
    refuse_existing(notes, call)
    Path(args.notes).write_text(body, encoding="utf-8")
    _note(args.summary, f"Release notes for `{notes.tag}`:\n\n{body}\n")
    where = create(notes, body, history.is_latest(args.version, version_tags(git)), call)
    _note(args.summary, f"Release page: {where}\n")
    return f"{notes.tag} release page created: {where}"


ERRORS = (PageError, history.HistoryError, notes_md.NotesError, release_tags.TagError, release_version.ReleaseError,
          subprocess.CalledProcessError, urllib.error.URLError, OSError, ValueError)


def main(argv: Optional[List[str]] = None, git: Git = run_git, call: Optional[Call] = None) -> int:
    args = parser().parse_args(argv)
    command = check if args.command == "check" else publish_page
    try:
        done = command(args, git, call or github(os.environ.get("GH_TOKEN", "")))
    except ERRORS as error:
        print(annotation("error", str(error)))
        return 1
    print(escaped(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
