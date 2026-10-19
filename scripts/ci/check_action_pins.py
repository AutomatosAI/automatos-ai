"""Every action a workflow runs is pinned to a commit (PRD-262 US-012).

A tag such as ``actions/checkout@v4`` is a pointer the action's owner can move, so
the code a lane runs could change with no change in this repository. This check
fails when:

* a ``uses:`` in ``.github/workflows/*.yml`` (or in an in-repo
  ``.github/actions/**/action.yml``) does not name a 40-hex commit. A local
  ``./path`` needs no ref; a ``docker://`` image needs ``@sha256:<digest>``;
* a pinned action is composite, or is a reusable workflow, and one of its own
  ``uses:`` is not pinned the same way. Its ``action.yml`` (or ``action.yaml``) is
  read from GitHub at the pinned commit, and the composites it pins are followed
  in turn. A commit with no action file there fails too;
* an entry of the exception list has no reason, or no longer matches anything.

Exceptions: ``.github/action-pin-exceptions.json``, ``{"exceptions": [{"uses":
"<the ref exactly as written>", "reason": "<why it cannot be pinned>"}]}``.

Run from anywhere: ``python3 scripts/ci/check_action_pins.py [--offline]``.
``--offline`` checks only this repository's files; the backend suite runs it that
way (orchestrator/tests/test_prd262_action_pins.py). Online, it reads GITHUB_TOKEN
when set. A file GitHub will not serve (rate limit, outage) fails the run: the
check never passes on what it could not read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
EXCEPTIONS_FILE = ".github/action-pin-exceptions.json"
GITHUB_API = "https://api.github.com"
API_TIMEOUT_SECONDS = 20
FETCH_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 2

# `uses:` as a step key or a job key, quoted or not, with any trailing comment.
_USES = re.compile(r"""^\s*(?:-\s+)?uses:\s*(?P<quote>['"]?)(?P<ref>[^\s'"#]+)(?P=quote)""")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DOCKER_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
_COMPOSITE = re.compile(r"""^\s*using:\s*['"]?composite['"]?\s*(?:#.*)?$""", re.M)
_WORKFLOW_FILE = re.compile(r"(?:^|/)\.github/workflows/[^/]+\.ya?ml$")

# (owner/repo, path in that repo, commit) -> the file's text, or None when absent.
Fetch = Callable[[str, str, str], Optional[str]]


@dataclass(frozen=True)
class Use:
    """One ``uses:`` reference and where it was written."""

    where: str
    ref: str


@dataclass(frozen=True)
class Finding:
    """A reference that breaks the rule, or a file the check could not read."""

    use: Use
    problem: str


def uses_in(text: str, where: str) -> List[Use]:
    """Every ``uses:`` reference in ``text``, as ``where:<line>``."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _USES.match(line)
        if match:
            found.append(Use(f"{where}:{number}", match.group("ref")))
    return found


def pin_problem(ref: str) -> Optional[str]:
    """Why ``ref`` is not pinned, or None when it is (or needs no pin)."""
    if ref.startswith("./"):
        return None
    if ref.startswith("docker://"):
        return None if _DOCKER_DIGEST.search(ref) else "a docker:// action needs @sha256:<digest>"
    name, at, version = ref.rpartition("@")
    if not at or "/" not in name:
        return "expected owner/repo[/path]@<40-hex commit>"
    if not _COMMIT.match(version):
        return f"{version!r} is a moving ref, not a 40-hex commit"
    return None


def is_remote_pin(ref: str) -> bool:
    """A pinned action that lives in another repository (so its file can be read there)."""
    return not ref.startswith(("./", "docker://")) and pin_problem(ref) is None


def repo_files(root: Path) -> List[Path]:
    """The workflows and in-repo actions whose ``uses:`` this repository controls."""
    github = root / ".github"
    workflows = [*github.glob("workflows/*.yml"), *github.glob("workflows/*.yaml")]
    actions = [*github.glob("actions/**/action.yml"), *github.glob("actions/**/action.yaml")]
    return sorted(workflows + actions)


def repo_uses(root: Path) -> List[Use]:
    """Every ``uses:`` in this repository's workflows and actions."""
    return [
        use
        for path in repo_files(root)
        for use in uses_in(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix())
    ]


def split_action(ref: str) -> Tuple[str, str, str]:
    """``(owner/repo, path inside it, commit)`` of a pinned remote ref."""
    name, _, commit = ref.rpartition("@")
    owner, repo, *rest = name.split("/")
    return f"{owner}/{repo}", "/".join(rest), commit


def action_file(ref: str, fetch: Fetch) -> Tuple[Optional[str], str]:
    """The text of the file ``ref`` runs (None when absent) and a label naming it."""
    repo, sub, commit = split_action(ref)
    if _WORKFLOW_FILE.search(f"/{sub}"):
        candidates = [sub]
    else:
        prefix = f"{sub}/" if sub else ""
        candidates = [f"{prefix}action.yml", f"{prefix}action.yaml"]
    for path in candidates:
        text = fetch(repo, path, commit)
        if text is not None:
            return text, f"{repo}/{path}@{commit[:12]}"
    return None, f"{repo}/{candidates[0]}@{commit[:12]}"


def runs_other_actions(text: str, label: str) -> bool:
    """A composite action or a reusable workflow: its own ``uses:`` run too."""
    return bool(_COMPOSITE.search(text)) or bool(_WORKFLOW_FILE.search(label.rsplit("@", 1)[0]))


def nested_uses(top: Sequence[Use], fetch: Fetch) -> Tuple[List[Use], List[Finding]]:
    """The ``uses:`` inside every pinned composite, followed to the bottom, and unread files."""
    queue = [use for use in top if is_remote_pin(use.ref)]
    seen: Set[str] = set()
    nested: List[Use] = []
    unreadable: List[Finding] = []
    while queue:
        use = queue.pop(0)
        if use.ref in seen:
            continue
        seen.add(use.ref)
        text, label = action_file(use.ref, fetch)
        if text is None:
            unreadable.append(Finding(use, f"no action.yml or action.yaml at {label}"))
            continue
        if runs_other_actions(text, label):
            inner = uses_in(text, label)
            nested.extend(inner)
            queue.extend(found for found in inner if is_remote_pin(found.ref))
    return nested, unreadable


def load_exceptions(path: Path) -> Dict[str, str]:
    """``{ref: reason}`` from the exception list; ValueError when an entry has no reason."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("exceptions") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{EXCEPTIONS_FILE}: 'exceptions' must be a list")
    allowed: Dict[str, str] = {}
    for entry in entries:
        fields = entry if isinstance(entry, dict) else {}
        ref = str(fields.get("uses", "")).strip()
        reason = str(fields.get("reason", "")).strip()
        if not ref or not reason:
            raise ValueError(f"{EXCEPTIONS_FILE}: every exception needs 'uses' and a 'reason': {entry!r}")
        allowed[ref] = reason
    return allowed


def check(root: Path, fetch: Optional[Fetch], exceptions: Dict[str, str]) -> List[Finding]:
    """Every finding; ``fetch=None`` checks this repository's files only."""
    top = repo_uses(root)
    nested, findings = nested_uses(top, fetch) if fetch else ([], [])
    excused: Set[str] = set()
    for use in top + nested:
        problem = pin_problem(use.ref)
        if problem and use.ref in exceptions:
            excused.add(use.ref)
        elif problem:
            findings.append(Finding(use, problem))
    if fetch:
        stale = sorted(set(exceptions) - excused)
        findings += [Finding(Use(EXCEPTIONS_FILE, ref), "this exception matches no reference") for ref in stale]
    return findings


def github_fetch(repo: str, path: str, commit: str) -> Optional[str]:
    """``path`` of ``repo`` at ``commit`` through GitHub's contents API; None on 404."""
    quoted = urllib.parse.quote(path)
    url = f"{GITHUB_API}/repos/{repo}/contents/{quoted}?ref={commit}"
    headers = {"Accept": "application/vnd.github.raw", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code < 500 or attempt == FETCH_ATTEMPTS:
                raise
        except urllib.error.URLError:
            if attempt == FETCH_ATTEMPTS:
                raise
        time.sleep(RETRY_PAUSE_SECONDS * attempt)
    return None


def annotation(finding: Finding) -> str:
    """A GitHub annotation for ``finding``, on the file and line when it is in this repository."""
    where, _, line = finding.use.where.rpartition(":")
    message = f"{finding.use.ref}: {finding.problem}"
    if line.isdigit() and "@" not in where:
        return f"::error file={where},line={line},title=action-pins::{message}"
    return f"::error title=action-pins::{finding.use.where}: {message}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO), help="the repository to check (default: this one)")
    parser.add_argument("--offline", action="store_true", help="check this repository's files only")
    return parser


def main(argv: Optional[List[str]] = None, fetch: Fetch = github_fetch) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    try:
        exceptions = load_exceptions(root / EXCEPTIONS_FILE)
        findings = check(root, None if args.offline else fetch, exceptions)
    except (ValueError, OSError, urllib.error.URLError) as error:
        print(f"::error title=action-pins::the check could not finish: {error}")
        return 1
    for finding in findings:
        print(annotation(finding))
    scope = "this repository's files" if args.offline else "this repository and every composite it pins"
    if findings:
        print(f"{len(findings)} action reference(s) not pinned to a commit in {scope}")
        return 1
    print(f"every uses: in {scope} is pinned ({len(repo_uses(root))} in this repository)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
