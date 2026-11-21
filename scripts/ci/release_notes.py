#!/usr/bin/env python3
"""The release page's Markdown (PRD-279 US-301, US-302, §5.4): pure functions, no reads.

``render`` writes the five sections of §5.4 from what ``release_page.py`` has read:

1. **What changed**: each merged pull request since the previous release, grouped by its
   commit type (``release_history.grouped``), with a link to its pull request.
2. **Database**: whether this release changes the database, and how
   (``release_history.DatabaseChange``), with the PRD's sentence on going back.
3. **Upgrading**: the earliest version an install upgrades from in one step (the upgrade
   file).
4. **What to install**: the chart's address, version and digest, and each image's digest.
5. **Checking it**: the ``cosign verify`` command for each, built by ``release_signing``,
   and a link to the page on checking a release.

**Text from a commit is data.** A commit's subject and body, a pull request's title, a scope
and a revision id or file name come from anyone who could open a pull request, so each is
shown only through ``literal``: one line, control characters dropped, cut to a length,
backticks turned to ``'``, inside a single code span. Inside a code span GitHub renders no
HTML, heading, emphasis, link, image, mention or ``#123`` reference, and with no backtick
left the text cannot close the span, so it cannot leave its list item or table cell. Every
link on the page is one this function builds to this repository on github.com.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _sibling("release_history")
release_signing = _sibling("release_signing")
release_version = _sibling("release_version")

GITHUB = "https://github.com"
CHECKING_PAGE = "docs/deployment-infrastructure/checking-a-release.md"
UPGRADE_FILE = "scripts/ci/release_upgrade_from.txt"
REPOSITORY = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
# Control characters, line and paragraph separators, and the invisible and direction marks.
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁠-⁩﻿]+")
TEXT_LIMIT = 200
MAX_CHANGES = 250
MAX_REVISIONS = 100
# GitHub's limit on a release's body is not in its REST documentation; this stays well below
# the 125,000 characters its web form allows, and the caps above keep a page under it.
MAX_BODY = 120_000
BACKUP = ("A release that changes the database can only be undone by restoring a backup, because "
          "migrations go forward only. Take a backup of the database before you upgrade.")


class NotesError(ValueError):
    """Inputs a release page cannot be written from."""


@dataclass(frozen=True)
class Notes:
    """Everything the release page says, read and checked by release_page.py."""

    repository: str
    version: str
    sha: str
    previous: Optional[str]
    changes: Tuple[history.Change, ...]
    database: history.DatabaseChange
    upgrade_from: Optional[str]
    images: Tuple[Tuple[str, str], ...]
    chart_digest: str

    @property
    def tag(self) -> str:
        return f"{history.TAG_PREFIX}{self.version}"

    @property
    def owner(self) -> str:
        return self.repository.split("/", 1)[0].lower()


IMAGE_NAME = re.compile(r"^[a-z0-9-]{1,64}$")


def _forms(notes: Notes) -> List[Tuple[str, str, re.Pattern]]:
    """``(what, value, form)`` for every value the page prints outside a code span."""
    forms = [("repository", notes.repository, REPOSITORY), ("commit", notes.sha, COMMIT),
             ("chart digest", notes.chart_digest, DIGEST)]
    for name, digest in notes.images:
        forms += [(f"image name {name!r}", name, IMAGE_NAME), (f"{name} digest", digest, DIGEST)]
    return forms


def checked(notes: Notes) -> Notes:
    """``notes``, or NotesError when any value the page prints outside a code span is not of its form."""
    release_version.parse_version(notes.version)
    problems = [what for what, value, form in _forms(notes) if not form.fullmatch(value)]
    if notes.previous is not None and history.previous_release(notes.version, [notes.previous]) != notes.previous:
        problems.append(f"previous release {notes.previous!r}")
    if problems:
        raise NotesError("not of the expected form: " + ", ".join(problems))
    return notes


# --------------------------------------------------------------------------- text from commits


def literal(text: str, limit: int = TEXT_LIMIT) -> str:
    """``text`` as one inline code span that cannot close early or carry Markdown or HTML."""
    flat = " ".join(CONTROL.sub(" ", text).split()).replace("`", "'")
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return f"`{flat or '(no description)'}`"


def _link(notes: Notes, path: str, label: str) -> str:
    """A link to this repository on github.com; ``label`` is the page's own text."""
    return f"[{label}]({GITHUB}/{notes.repository}/{path})"


def change_line(change: history.Change, notes: Notes) -> str:
    """One list item: the change's text as data, then its pull request or commit."""
    text = f"{change.scope}: {change.text}" if change.scope else change.text
    where = (_link(notes, f"pull/{change.number}", f"#{change.number}") if change.number
             else _link(notes, f"commit/{change.commit}", change.commit[:12]) if COMMIT.fullmatch(change.commit) else "")
    breaking = "**Breaking:** " if change.breaking else ""
    return f"- {breaking}{literal(text)}" + (f" ({where})" if where else "")


# --------------------------------------------------------------------------- the sections


def changes_section(notes: Notes) -> List[str]:
    """What changed: the merged pull requests since the previous release, by type."""
    lines = ["## What changed", ""]
    if notes.previous is None:
        return lines + [f"This is the first release, so there is no earlier release to compare it with. "
                        f"Its history is {_link(notes, f'commits/{notes.tag}', 'the commits up to ' + notes.tag)}.", ""]
    compare = _link(notes, f"compare/{notes.previous}...{notes.tag}", f"{notes.previous}...{notes.tag}")
    lines += [f"Since {notes.previous}, by pull request ({compare} has every commit).", ""]
    if not notes.changes:
        return lines + ["No pull request was merged since then.", ""]
    shown = notes.changes[:MAX_CHANGES]
    for heading, changes in history.grouped(shown):
        lines += [f"### {heading}", "", *(change_line(change, notes) for change in changes), ""]
    if len(notes.changes) > len(shown):
        lines += [f"And {len(notes.changes) - len(shown)} more: see {compare}.", ""]
    return lines


def _revision_line(revision: history.Revision) -> str:
    merge = f" (merges {', '.join(literal(parent, 80) for parent in revision.parents)})" \
        if len(revision.parents) > 1 else ""
    return f"- {literal(revision.revision, 120)} in {literal(Path(revision.path).name, 120)}{merge}"


def _revision_list(title: str, revisions: Sequence[history.Revision]) -> List[str]:
    if not revisions:
        return []
    shown = [_revision_line(revision) for revision in revisions[:MAX_REVISIONS]]
    more = [f"- and {len(revisions) - MAX_REVISIONS} more"] if len(revisions) > MAX_REVISIONS else []
    return [title, "", *shown, *more, ""]


def _heads(ids: Sequence[str]) -> str:
    return ", ".join(literal(rev, 120) for rev in ids) or "none"


def _migrations(count: int) -> str:
    return f"{count} migration" + ("" if count == 1 else "s")


def _verdict(notes: Notes) -> List[str]:
    change, previous = notes.database, notes.previous
    if change.verdict == "first":
        return [f"**This is the first release.** A new install creates the database at {_heads(change.heads_after)}, "
                f"through {_migrations(len(change.added))}.", ""]
    if change.verdict == "none":
        return [f"**This release does not change the database.** It has the migrations {previous} had, unchanged, "
                f"at {_heads(change.heads_after)}.", ""]
    moved = f"The database moves from {_heads(change.heads_before)} at {previous} to {_heads(change.heads_after)}."
    if change.verdict == "changes":
        return [f"**This release changes the database.** It adds {_migrations(len(change.added))}, applied in the "
                f"order below when an install upgrades. {moved}", ""]
    return [f"**This release changes the database.** It adds no migration, but migrations that {previous} shipped "
            f"were removed or edited, so an install that ran them may not match this release's. Treat it as a change: "
            f"{moved}", ""]


def database_section(notes: Notes) -> List[str]:
    """Database: what the Alembic revisions at the two tags say."""
    change = notes.database
    lines = ["## Database", "", *_verdict(notes)]
    lines += _revision_list("Added:" if change.verdict != "first" else "Its migrations:",
                            change.added if change.verdict != "first" else ())
    lines += _revision_list(f"Removed since {notes.previous}:", change.removed)
    lines += _revision_list(f"Edited since {notes.previous}:", change.rewritten)
    if change.verdict == "none":
        lines += [f"Going back to {notes.previous} needs no change to the database.", ""]
    return lines + [BACKUP, "", "The migrations run when an install upgrades: with the chart, the migration Job "
                    "runs them before the new API pods start; with Docker Compose, the API runs them as it starts.", ""]


def upgrade_section(notes: Notes) -> List[str]:
    """Upgrading: the earliest version this one upgrades from in one step."""
    lines = ["## Upgrading", ""]
    if notes.upgrade_from is None and notes.previous is None:
        lines.append("This is the first release, so there is nothing to upgrade from: install it new.")
    elif notes.upgrade_from is None:
        lines.append(f"No earlier release upgrades to {notes.version} in one step: install it new.")
    else:
        lines.append(f"An install of {notes.upgrade_from} or any later release upgrades to {notes.version} in one "
                     f"step. From an earlier version, first upgrade to a release that it can be upgraded from.")
    return lines + ["", f"This is read from {_link(notes, f'blob/{notes.tag}/{UPGRADE_FILE}', UPGRADE_FILE)}.", ""]


def _refs(notes: Notes) -> Tuple[str, List[Tuple[str, str]]]:
    chart = f"ghcr.io/{notes.owner}/charts/automatos"
    images = [(f"ghcr.io/{notes.owner}/automatos-{name}", digest) for name, digest in notes.images]
    return chart, images


def install_section(notes: Notes) -> List[str]:
    """What to install: the chart's address, version and digest, and every image's digest."""
    chart, images = _refs(notes)
    lines = ["## What to install", "",
             f"The chart `oci://{chart}`, version `{notes.version}`, digest `{notes.chart_digest}`. Its default values "
             f"name every image below by its digest.", "", "```bash",
             f"helm install automatos oci://{chart} --version {notes.version} --set existingSecret=<your Secret>",
             "```", "", "| Image | Digest |", "|---|---|"]
    lines += [f"| `{image}:{notes.version}` | `{digest}` |" for image, digest in images]
    return lines + [""]


def _command(reference: str, notes: Notes) -> List[str]:
    signer = release_signing.release_signer(notes.repository, f"refs/tags/{notes.tag}", notes.sha)
    args = [shlex.quote(arg) for arg in release_signing.verify_args(reference, signer)]
    pairs = [" ".join(args[:2])] + [" ".join(args[i:i + 2]) for i in range(2, len(args) - 1, 2)] + [args[-1]]
    return ["```bash", " \\\n  ".join(pairs), "```"]


def checking_section(notes: Notes) -> List[str]:
    """Checking it: the exact cosign check for the chart and each image, and the page on it."""
    chart, images = _refs(notes)
    lines = ["## Checking it", "",
             f"Each artifact was signed by this repository's `images.yml` at `refs/tags/{notes.tag}`, commit "
             f"`{notes.sha}`. Check each by its digest before you install it. "
             f"{_link(notes, f'blob/{notes.tag}/{CHECKING_PAGE}', 'Checking a release')} says what these commands "
             f"check, how to read each image's contents list and build record, and what a passing check does not prove.",
             "", "The chart:", "", *_command(f"{chart}@{notes.chart_digest}", notes), ""]
    for image, digest in images:
        lines += [f"`{image}`:", "", *_command(f"{image}@{digest}", notes), ""]
    return lines


def render(notes: Notes) -> str:
    """The release page's body: the five sections of §5.4, in order."""
    notes = checked(notes)
    lines: List[str] = []
    for section in (changes_section, database_section, upgrade_section, install_section, checking_section):
        lines += section(notes)
    body = "\n".join(lines).rstrip() + "\n"
    if len(body) > MAX_BODY:
        raise NotesError(f"the notes are {len(body)} characters, over the {MAX_BODY} a release page takes")
    return body


def title(version: str) -> str:
    """The release page's name."""
    return f"Automatos {release_version.parse_version(version).version}"


def images_of(digests: Mapping[str, str], names: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """``((name, digest), …)`` in ``names``' order."""
    return tuple((name, digests[name]) for name in names)
