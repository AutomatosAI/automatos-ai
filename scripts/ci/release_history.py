#!/usr/bin/env python3
"""What a release is compared with, and what changed since (PRD-279 US-301, §5.4).

Pure functions over what ``release_notes.py`` reads from Git, so each case is tested with
fixtures and no history of its own:

* **The previous release** (``previous_release``): of the version tags reachable from the
  tagged commit, the highest full release (no suffix) below the tag's own version, by version
  order, never by date. So a release candidate is compared with the full release before it,
  never with an earlier candidate; and a patch cut on ``release/MAJOR.MINOR`` with the release
  before it on that branch, because a newer release on main is not reachable from it. None for
  the first release.
* **The latest release** (``is_latest``): a full release is marked the repository's latest
  only when no full release has a higher version, so a patch to an older line is not.
* **The changes** (``parse_log``): ``git log -z --first-parent`` between the two, one entry per
  merged pull request. A merge commit (``Merge pull request #N from …``) is read by its pull
  request's title, the first line of its body; a squash merge by its subject, with its
  trailing ``(#N)``. Each is grouped by its ``type(scope): description`` type (``KINDS``),
  anything else under "Other changes". The text is kept as data: release_notes.py decides how
  it is shown.
* **The database** (``database_change``): the Alembic revisions at each tag, read from each
  ``orchestrator/alembic/versions/*.py`` file's module-level ``revision`` and
  ``down_revision`` (by ``ast``, never by importing code from an old tag). Added revisions,
  in the order they apply; removed ones; and rewritten ones (the same id with other source),
  which an earlier release shipped and which must never read as "no change".
* **The upgrade file** (``read_upgrade_from``): ``release_upgrade_from.txt`` beside this
  script, whose one line that is not a comment is the earliest version this release upgrades
  from in one step (``MAJOR.MINOR.PATCH``), or ``none``. A missing or malformed file fails.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_version = _sibling("release_version")

TAG_PREFIX = "v"
# The commit types the repository's messages use (AGENTS.md), each with its heading, in order.
KINDS: Tuple[Tuple[str, str], ...] = (
    ("feat", "Features"), ("fix", "Fixes"), ("perf", "Performance"), ("refactor", "Refactoring"),
    ("docs", "Documentation"), ("test", "Tests"), ("ci", "CI"), ("build", "Build"),
    ("chore", "Chores"), ("revert", "Reverts"), ("other", "Other changes"),
)
KNOWN_KINDS = frozenset(kind for kind, _ in KINDS)
MERGE_SUBJECT = re.compile(r"^Merge pull request #(?P<number>[0-9]{1,9}) from \S+$")
SQUASH_NUMBER = re.compile(r"^(?P<title>.*\S)\s+\(#(?P<number>[0-9]{1,9})\)$")
CONVENTIONAL = re.compile(r"^(?P<kind>[a-z]+)(?:\((?P<scope>[^()]{1,80})\))?(?P<bang>!)?:\s+(?P<text>\S.*)$")
BREAKING_FOOTER = re.compile(r"^BREAKING[ -]CHANGE:", re.M)
UPGRADE_NONE = "none"


class HistoryError(ValueError):
    """History, a migration or the upgrade file a release page cannot be written from."""


# --------------------------------------------------------------------------- versions


def full_releases(tags: Iterable[str]) -> Dict[Tuple[int, int, int], str]:
    """``{version core: tag}`` for every tag that names a full release (no suffix)."""
    found = {}
    for tag in tags:
        if not tag.startswith(TAG_PREFIX):
            continue
        try:
            release = release_version.parse_version(tag[len(TAG_PREFIX):])
        except release_version.ReleaseError:
            continue
        if not release.candidate:
            found[release.core] = tag
    return found


def previous_release(version: str, reachable_tags: Iterable[str]) -> Optional[str]:
    """The tag of the highest full release below ``version`` among ``reachable_tags``; None if none."""
    current = release_version.parse_version(version).core
    earlier = {core: tag for core, tag in full_releases(reachable_tags).items() if core < current}
    return earlier[max(earlier)] if earlier else None


def is_latest(version: str, all_tags: Iterable[str]) -> bool:
    """Whether ``version`` is a full release that no full release in ``all_tags`` outranks."""
    release = release_version.parse_version(version)
    return not release.candidate and all(core <= release.core for core in full_releases(all_tags))


# --------------------------------------------------------------------------- the changes


@dataclass(frozen=True)
class Change:
    """One merged pull request, or one commit made on the branch directly."""

    kind: str
    scope: str
    text: str
    number: Optional[int]
    breaking: bool
    commit: str


def _title(parents: Sequence[str], subject: str, body: str) -> Tuple[str, Optional[int]]:
    """The change's title and its pull request's number, from a merge commit or a squash merge."""
    merge = MERGE_SUBJECT.match(subject)
    if merge and len(parents) > 1:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        return (lines[0] if lines else subject), int(merge["number"])
    squash = SQUASH_NUMBER.match(subject)
    if squash:
        return squash["title"], int(squash["number"])
    return subject, None


def change_of(commit: str, parents: Sequence[str], subject: str, body: str) -> Change:
    """One commit on the release's first-parent line, read as a change."""
    title, number = _title(parents, subject.strip(), body)
    typed = CONVENTIONAL.match(title)
    if not typed or typed["kind"] not in KNOWN_KINDS:
        return Change("other", "", title, number, bool(BREAKING_FOOTER.search(body)), commit)
    breaking = bool(typed["bang"]) or bool(BREAKING_FOOTER.search(body))
    return Change(typed["kind"], (typed["scope"] or "").strip(), typed["text"].strip(), number, breaking, commit)


def parse_log(raw: str) -> List[Change]:
    """``git log -z --first-parent --format=%H%n%P%n%s%n%b`` output, newest first, as changes."""
    changes = []
    for record in raw.split("\0"):
        lines = record.strip("\n").split("\n")
        if len(lines) < 3 or not lines[0].strip():
            continue
        changes.append(change_of(lines[0].strip(), lines[1].split(), lines[2], "\n".join(lines[3:])))
    return changes


def grouped(changes: Sequence[Change]) -> List[Tuple[str, List[Change]]]:
    """``[(heading, changes)]`` in ``KINDS`` order, each kind's changes in the log's order."""
    return [(heading, [change for change in changes if change.kind == kind])
            for kind, heading in KINDS if any(change.kind == kind for change in changes)]


# --------------------------------------------------------------------------- the database


@dataclass(frozen=True)
class Revision:
    """One Alembic revision file: its id, the revisions it follows, and its source."""

    revision: str
    parents: Tuple[str, ...]
    path: str
    source: str


REVISION_NAMES = ("revision", "down_revision")


def _assigned(node: ast.stmt) -> List[Tuple[str, ast.expr]]:
    """``[(name, value)]`` a module-level ``name = value`` or ``name: type = value`` sets."""
    if isinstance(node, ast.Assign):
        return [(target.id, node.value) for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return [(node.target.id, node.value)]
    return []


def _literal(value: ast.expr) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return None


def _module_values(tree: ast.Module) -> Dict[str, object]:
    """The literal values of the module-level ``revision`` and ``down_revision``."""
    return {name: _literal(value) for node in tree.body for name, value in _assigned(node) if name in REVISION_NAMES}


def parse_revision(path: str, source: str) -> Optional[Revision]:
    """The revision a versions file declares; None for a file that declares none."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise HistoryError(f"{path} is not Python ({error.msg}), so its revision cannot be read") from error
    values = _module_values(tree)
    revision, down = values.get("revision"), values.get("down_revision")
    if not isinstance(revision, str) or not revision:
        return None
    parents = (down,) if isinstance(down, str) else tuple(down) if isinstance(down, (tuple, list)) else ()
    return Revision(revision, tuple(parent for parent in parents if isinstance(parent, str)), path, source)


def revisions(files: Mapping[str, str]) -> Dict[str, Revision]:
    """``{revision id: Revision}`` for ``{path: source}``; two files with one id fail."""
    found: Dict[str, Revision] = {}
    for path in sorted(files):
        revision = parse_revision(path, files[path])
        if revision is None:
            continue
        if revision.revision in found:
            raise HistoryError(f"{path} and {found[revision.revision].path} both declare {revision.revision!r}")
        found[revision.revision] = revision
    return found


def heads(revs: Mapping[str, Revision]) -> Tuple[str, ...]:
    """The revisions no other revision follows, sorted."""
    followed = {parent for revision in revs.values() for parent in revision.parents}
    return tuple(sorted(set(revs) - followed))


def in_order(ids: Iterable[str], revs: Mapping[str, Revision]) -> Tuple[Revision, ...]:
    """``ids`` parents first, as ``alembic upgrade`` applies them; ties by id."""
    remaining, placed = set(ids), []
    while remaining:
        ready = sorted(rev for rev in remaining if not set(revs[rev].parents) & remaining)
        if not ready:
            raise HistoryError(f"the revisions {sorted(remaining)} follow each other in a loop")
        placed += [revs[rev] for rev in ready]
        remaining -= set(ready)
    return tuple(placed)


@dataclass(frozen=True)
class DatabaseChange:
    """How the database at this release differs from the database at the previous one."""

    first_release: bool
    added: Tuple[Revision, ...]
    removed: Tuple[Revision, ...]
    rewritten: Tuple[Revision, ...]
    heads_before: Tuple[str, ...]
    heads_after: Tuple[str, ...]

    @property
    def verdict(self) -> str:
        """``first``, ``changes`` (a migration added), ``edits`` (only shipped ones edited or removed) or ``none``."""
        if self.first_release:
            return "first"
        if self.added:
            return "changes"
        return "edits" if self.removed or self.rewritten else "none"


def database_change(before: Optional[Mapping[str, Revision]], after: Mapping[str, Revision]) -> DatabaseChange:
    """Compare the revisions at the previous release (None: there is none) with this release's."""
    if before is None:
        return DatabaseChange(True, in_order(after, after), (), (), (), heads(after))
    added = in_order(set(after) - set(before), after)
    removed = tuple(before[rev] for rev in sorted(set(before) - set(after)))
    rewritten = tuple(after[rev] for rev in sorted(set(before) & set(after)) if before[rev].source != after[rev].source)
    return DatabaseChange(False, added, removed, rewritten, heads(before), heads(after))


# --------------------------------------------------------------------------- the upgrade file


def read_upgrade_from(text: str) -> Optional[str]:
    """The upgrade file's value: a full release's version, or None for ``none``. HistoryError otherwise."""
    values = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != 1:
        raise HistoryError(f"the upgrade file must hold one value, MAJOR.MINOR.PATCH or {UPGRADE_NONE}; "
                           f"it holds {len(values)}")
    if values[0] == UPGRADE_NONE:
        return None
    try:
        release = release_version.parse_version(values[0])
    except release_version.ReleaseError as error:
        raise HistoryError(f"the upgrade file's value is not a version: {error}") from error
    if release.candidate:
        raise HistoryError(f"the upgrade file names {values[0]}, a release candidate: name a full release")
    return release.version


def check_upgrade_from(value: Optional[str], version: str, all_tags: Iterable[str]) -> None:
    """HistoryError unless ``value`` is a full release, tagged, and below ``version``."""
    if value is None:
        return
    if release_version.parse_version(value).core >= release_version.parse_version(version).core:
        raise HistoryError(f"the upgrade file names {value}, which is not below {version}")
    if f"{TAG_PREFIX}{value}" not in set(all_tags):
        raise HistoryError(f"the upgrade file names {value}, but there is no tag {TAG_PREFIX}{value}")
