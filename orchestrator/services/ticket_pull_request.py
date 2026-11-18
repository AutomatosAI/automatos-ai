"""Ticket PR fields: the pull request a coding session reports on its ticket.

A Claude Code session that finishes its work commits, pushes a branch and opens a pull request
itself, then says so through ``update_ticket`` (``services/session_tools.py``) with four optional
fields: ``pull_request_url``, ``branch``, ``commits`` and ``tested``. This module is the one
definition of them:

* **checked at the boundary** (:func:`pull_request_fields`): the address is a pull request's own
  https page, ``/<owner>/<repo>/pull/<n>`` on any host, or GitLab's
  ``/<group>/<project>/-/merge_requests/<n>``; the branch is a plausible git ref; there are at
  most 50 commits, each a 7-40 character hex sha with its subject; ``tested`` is capped text.
  Anything else refuses the whole call, in words the session can act on.
* **kept on the ticket** under ``runtime_ref.pull_request`` (:func:`record_ticket_update`): the row
  locked, the stored object read, the new one written with ONE ``jsonb_set`` of that key, never a
  whole-document write; ``core/models/kept_runtime_keys.py`` carries it through the CLI host's
  whole-document writes. A second call updates the same object. A new address replaces it, the old
  one kept in ``previous_urls`` (the last few), and leaves the note "Pull request: <url>" so the
  ticket's existing views show it.
* **frozen once the ticket leaves In progress**: only a running ticket's session may write it, so
  what the owner reviews is what his Approve acts on. A re-run (back to In progress) may replace it.
* **read back through the same checks** (:func:`pull_request_of`): the board's serializer and the
  merge on approve (``services/ticket_pr_merge.py``) read the stored object this way, so a value
  that does not pass is treated as no pull request at all. The view also says whether a person's
  Approve would merge it (:func:`merges_on_approve`: the setting on, a github.com pull request, and
  its repository in ``TICKET_APPROVE_MERGES_PR_REPOS``).
* **taken off by the owner** (:func:`clear_pull_request`, ``DELETE /api/v1/tasks/{id}/pull-request``):
  one statement removes the key; the listener then keeps it removed (the row's absence wins too).

Recording a pull request never merges it: only a person's Approve does (``ticket_pr_merge``).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlsplit

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.models.kept_runtime_keys import PULL_REQUEST_KEY

logger = logging.getLogger(__name__)

URL_FIELD, BRANCH_FIELD, COMMITS_FIELD, TESTED_FIELD = "pull_request_url", "branch", "commits", "tested"
FIELDS = (URL_FIELD, BRANCH_FIELD, COMMITS_FIELD, TESTED_FIELD)
# Where each field is kept in the stored object.
STORED_AS = {URL_FIELD: "url", BRANCH_FIELD: "branch", COMMITS_FIELD: "commits", TESTED_FIELD: "tested"}
MAX_URL_CHARS = 500
MAX_BRANCH_CHARS = 200
MAX_COMMITS = 50
MAX_SUBJECT_CHARS = 200
MAX_TESTED_CHARS = 1500
PREVIOUS_URLS_KEPT = 5
HTTPS = "https"
HTTPS_PORT = 443
GITHUB_SHAPE, GITLAB_SHAPE = "github", "gitlab"
GITHUB_HOST = "github.com"
SESSION_BY = "the session"
OPEN_FOR_CHANGES = "in_progress"
REPO_SEPARATOR = ","

_HOST = re.compile(r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")
_SEGMENT = r"(?!\.{1,2}(?:/|$))(?!-(?:/|$))[A-Za-z0-9._-]{1,100}"
_PULL_PATH = re.compile(
    rf"^/(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]{{0,99}})/(?P<repo>{_SEGMENT})/pull/(?P<number>[1-9][0-9]{{0,9}})$")
_MERGE_REQUEST_PATH = re.compile(
    rf"^/(?P<project>[A-Za-z0-9][A-Za-z0-9._-]{{0,99}}(?:/{_SEGMENT}){{1,20}})/-/merge_requests/"
    r"(?P<number>[1-9][0-9]{0,9})$")
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")
# git check-ref-format, in short: no space or control character, none of ~^:?*[\, no "..", "//", "@{" or "/.".
_BAD_REF = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]|\.\.|//|@\{|/\.|\.lock/")

# What a session is told when a field does not pass (the whole call is refused, nothing is saved).
NOTHING_SAVED = " Nothing was saved: call update_ticket again with it corrected, or leave it out."
BAD_URL = ("pull_request_url must be the pull request's own https address, like "
           "https://github.com/<owner>/<repo>/pull/<number>, or a GitLab merge request's "
           "(.../-/merge_requests/<number>)." + NOTHING_SAVED)
BAD_BRANCH = (f"branch must be the git branch you pushed, as git names it: at most {MAX_BRANCH_CHARS} characters, "
              "no spaces, and none of ~ ^ : ? * [ \\ or '..'." + NOTHING_SAVED)
BAD_COMMITS = (f"commits must be a list of at most {MAX_COMMITS} objects, each "
               '{"sha": its 7-40 character hex sha, "subject": its subject line}.' + NOTHING_SAVED)
BAD_TESTED = ("tested must be text: what you ran or checked and what you saw, or that nothing could be run."
              + NOTHING_SAVED)
NOT_SAVED = "the pull request could not be saved on the ticket ({kind}); say it in your result instead"
NOT_OPEN_FOR_CHANGES = ("this ticket is no longer open for changes: it has left In progress, and the pull request "
                        "on it is the one the owner is reviewing. Nothing was saved; say anything new in your result.")
# The note a new address leaves among the ticket's notes.
PULL_REQUEST_NOTE = "Pull request: {url}"

# What the session's model reads in update_ticket's schema (services/session_tools.py spreads these in).
PULL_REQUEST_PROPERTIES: Dict[str, Any] = {
    URL_FIELD: {"type": "string", "description": "The pull request you opened: its https address (…/pull/<number>)."},
    BRANCH_FIELD: {"type": "string", "description": "The branch you pushed."},
    COMMITS_FIELD: {"type": "array", "maxItems": MAX_COMMITS,
                    "items": {"type": "object", "required": ["sha", "subject"],
                              "properties": {"sha": {"type": "string"}, "subject": {"type": "string"}}},
                    "description": "Your commits on that branch, newest first: sha and subject line."},
    TESTED_FIELD: {"type": "string",
                   "description": "What you ran or checked and what you saw, or that nothing could be run."},
}


class PullRequestRefused(ValueError):
    """A reported field does not pass; the message is for the session."""


class TicketNotOpen(Exception):
    """The ticket has left In progress, so its pull request is frozen; the message is for the session."""


@dataclass(frozen=True)
class PullRequestAddress:
    """A pull request's address, taken apart: ``repository`` is ``owner/repo`` (a GitLab project's
    path), ``owner`` and ``repo`` are set for the ``/pull/<n>`` shape only."""

    url: str
    host: str
    shape: str
    repository: str
    number: int
    owner: Optional[str] = None
    repo: Optional[str] = None


def parse_pull_request_url(raw: Any) -> Optional[PullRequestAddress]:
    """The address, normalised (no trailing slash), or None when it is not a pull request's own
    https page in one of the two accepted shapes. Pure."""
    if not isinstance(raw, str) or len(raw) > MAX_URL_CHARS:
        return None
    candidate = raw.strip()
    if not candidate or any(ch.isspace() or ord(ch) < 32 for ch in candidate):
        return None
    try:
        parts = urlsplit(candidate)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname or ""
    if (parts.scheme.lower() != HTTPS or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or not _HOST.match(host)):
        return None
    return _address(host, port, parts.path.rstrip("/"))


def _address(host: str, port: Optional[int], path: str) -> Optional[PullRequestAddress]:
    origin = f"https://{host}{f':{port}' if port and port != HTTPS_PORT else ''}"
    pull = _PULL_PATH.match(path)
    if pull:
        return PullRequestAddress(url=origin + path, host=host, shape=GITHUB_SHAPE,
                                  repository=f"{pull['owner']}/{pull['repo']}", number=int(pull["number"]),
                                  owner=pull["owner"], repo=pull["repo"])
    merge_request = _MERGE_REQUEST_PATH.match(path)
    if merge_request:
        return PullRequestAddress(url=origin + path, host=host, shape=GITLAB_SHAPE,
                                  repository=merge_request["project"], number=int(merge_request["number"]))
    return None


def checked_url(raw: Any) -> str:
    address = parse_pull_request_url(raw)
    if address is None:
        raise PullRequestRefused(BAD_URL)
    return address.url


def checked_branch(raw: Any) -> str:
    name = raw.strip() if isinstance(raw, str) else ""
    if (not name or len(name) > MAX_BRANCH_CHARS or _BAD_REF.search(name) or name == "@"
            or name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock"))):
        raise PullRequestRefused(BAD_BRANCH)
    return name


def checked_commits(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list) or len(raw) > MAX_COMMITS:
        raise PullRequestRefused(BAD_COMMITS)
    return [_checked_commit(item) for item in raw]


def _checked_commit(item: Any) -> Dict[str, str]:
    sha = item.get("sha") if isinstance(item, Mapping) else None
    subject = item.get("subject", "") if isinstance(item, Mapping) else None
    if not isinstance(sha, str) or not _SHA.match(sha.strip()) or not isinstance(subject, str):
        raise PullRequestRefused(BAD_COMMITS)
    return {"sha": sha.strip().lower(), "subject": " ".join(subject.split())[:MAX_SUBJECT_CHARS]}


def checked_tested(raw: Any) -> str:
    if not isinstance(raw, str):
        raise PullRequestRefused(BAD_TESTED)
    return raw.strip()[:MAX_TESTED_CHARS]


_CHECKS: Dict[str, Callable[[Any], Any]] = {
    URL_FIELD: checked_url, BRANCH_FIELD: checked_branch, COMMITS_FIELD: checked_commits, TESTED_FIELD: checked_tested,
}


def pull_request_fields(params: Mapping[str, Any]) -> Dict[str, Any]:
    """The four fields the call gave (a field left out, null or empty text is not given), each
    checked and normalised. Raises :class:`PullRequestRefused` on the first that does not pass."""
    given = {name: params.get(name) for name in FIELDS if params.get(name) not in (None, "")}
    return {name: _CHECKS[name](value) for name, value in given.items()}


def updated_pull_request(stored: Any, fields: Mapping[str, Any], *, by: str, at: str) -> Dict[str, Any]:
    """The ticket's pull request after this call: the fields it gave over the stored ones. A new
    address starts a fresh object, the stored address first in its ``previous_urls``. Pure."""
    current = dict(stored) if isinstance(stored, Mapping) else {}
    url = fields.get(URL_FIELD)
    if url and current.get("url") and url != current["url"]:
        earlier = [current["url"], *[u for u in current.get("previous_urls") or [] if isinstance(u, str)]]
        current = {"previous_urls": [u for u in dict.fromkeys(earlier) if u != url][:PREVIOUS_URLS_KEPT]}
    given = {STORED_AS[name]: value for name, value in fields.items()}
    return {**current, **given, "by": by, "at": at}


def _quietly(check: Callable[[Any], Any], value: Any) -> Any:
    try:
        return check(value)
    except PullRequestRefused:
        return None


def pull_request_of(runtime_ref: Any) -> Optional[Dict[str, Any]]:
    """The ticket's pull request as the board shows it, re-checked, with the stamp of its last write
    (``at``: the web sends it back with an Approve) and whether that Approve would merge it; None
    when it has none."""
    stored = runtime_ref.get(PULL_REQUEST_KEY) if isinstance(runtime_ref, Mapping) else None
    if not isinstance(stored, Mapping):
        return None
    address = parse_pull_request_url(stored.get("url"))
    earlier = stored.get("previous_urls") if isinstance(stored.get("previous_urls"), list) else []
    shown = {
        "url": address.url if address else None,
        "host": address.host if address else None,
        "repository": address.repository if address else None,
        "number": address.number if address else None,
        "branch": _quietly(checked_branch, stored.get("branch")) if stored.get("branch") else None,
        "commits": _quietly(checked_commits, stored.get("commits")) or [],
        "tested": _quietly(checked_tested, stored.get("tested")) or None,
        "previous_urls": [a.url for a in map(parse_pull_request_url, earlier) if a][:PREVIOUS_URLS_KEPT],
        "at": stored.get("at") if isinstance(stored.get("at"), str) else None,
        "merges_on_approve": merges_on_approve(address),
    }
    if not (shown["url"] or shown["branch"] or shown["commits"] or shown["tested"]):
        return None
    return shown


def merge_allowed_repos() -> frozenset:
    """``TICKET_APPROVE_MERGES_PR_REPOS`` as lower-case ``owner/repo`` names; empty merges nothing."""
    names = (name.strip().lower() for name in str(config.TICKET_APPROVE_MERGES_PR_REPOS or "").split(REPO_SEPARATOR))
    return frozenset(name for name in names if name.count("/") == 1 and all(name.split("/")))


def merges_on_approve(address: Optional[PullRequestAddress]) -> bool:
    """Whether an owner's or admin's Approve would try to merge this pull request: the setting on, a
    github.com pull request, and its repository listed. The board's view cannot tell who is looking,
    so the viewer's line says it needs an owner or admin; GitHub still has to say it can."""
    return bool(config.TICKET_APPROVE_MERGES_PR and address is not None and address.shape == GITHUB_SHAPE
                and address.host == GITHUB_HOST and address.repository.lower() in merge_allowed_repos())


_STORED_UNDER_LOCK = text(
    "SELECT status, runtime_ref -> CAST(:key AS text) FROM board_tasks WHERE id = :task_id AND workspace_id = :ws "
    "FOR NO KEY UPDATE"
)
_WRITE_ONE_KEY = text(
    """
    UPDATE board_tasks
       SET runtime_ref = jsonb_set(COALESCE(runtime_ref, CAST('{}' AS jsonb)), CAST(:path AS text[]),
                                   CAST(:value AS jsonb), true)
     WHERE id = :task_id AND workspace_id = :ws
    """
)


def write_pull_request(db: Session, *, task_id: Any, workspace_id: Any, by: str, note: str,
                       fields: Mapping[str, Any]) -> List[str]:
    """In the caller's transaction: lock the ticket's row, write its updated pull request with
    one ``jsonb_set`` of that key, and append the session's note and, for a new address,
    "Pull request: <url>". Returns the notes written. Raises ``LookupError`` for a ticket that
    is not in this workspace, and :class:`TicketNotOpen` for one that is not In progress."""
    from services.cli_host_service import append_session_note

    keys = {"task_id": int(task_id), "ws": str(workspace_id)}
    row = db.execute(_STORED_UNDER_LOCK, {**keys, "key": PULL_REQUEST_KEY}).first()
    if row is None:
        raise LookupError(f"ticket {task_id} is not in this workspace")
    if row[0] != OPEN_FOR_CHANGES:
        raise TicketNotOpen(NOT_OPEN_FOR_CHANGES)
    stored = row[1] if isinstance(row[1], Mapping) else {}
    updated = updated_pull_request(stored, fields, by=by, at=datetime.now(timezone.utc).isoformat())
    db.execute(_WRITE_ONE_KEY, {**keys, "path": "{%s}" % PULL_REQUEST_KEY, "value": json.dumps(updated)})
    lines = [note] if note else []
    url = fields.get(URL_FIELD)
    if url and url != stored.get("url"):
        lines.append(PULL_REQUEST_NOTE.format(url=url))
    for line in lines:
        append_session_note(db, task_id=task_id, workspace_id=workspace_id, note=line, by=by)
    return lines


def record_ticket_update(db: Session, *, task_id: Any, workspace_id: Any, agent_name: Optional[str],
                         params: Mapping[str, Any]) -> Dict[str, Any]:
    """``update_ticket``'s runner: the session's note, as it always was
    (``cli_host_service.record_session_note``), and, when the call reported a pull request, that
    and the note in one transaction. Returns an executor-shaped result the session reads."""
    from services.cli_host_service import publish_note_line, record_session_note

    note = str(params.get("note") or "")
    fields = {name: params[name] for name in FIELDS if name in params}
    if not fields:
        return record_session_note(db, task_id=task_id, workspace_id=workspace_id, agent_name=agent_name, note=note)
    try:
        lines = write_pull_request(db, task_id=task_id, workspace_id=workspace_id, by=agent_name or SESSION_BY,
                                   note=note, fields=fields)
        db.commit()
    except TicketNotOpen as closed:
        logger.info("[ticket-pr] ticket #%s left In progress: its pull request was not changed", task_id)
        _roll_back(db)
        return {"success": False, "error": str(closed)}
    except Exception as exc:  # noqa: BLE001 — logged; the session reads the reason
        logger.exception("[ticket-pr] the pull request was not recorded on ticket #%s", task_id)
        _roll_back(db)
        return {"success": False, "error": NOT_SAVED.format(kind=type(exc).__name__)}
    for line in lines:
        publish_note_line(workspace_id, task_id, line)
    return {"success": True, "result": {"recorded": True, "note": note, "pull_request": sorted(fields)}}


_CLEAR_THE_KEY = text(
    """
    UPDATE board_tasks SET runtime_ref = runtime_ref - CAST(:key AS text)
     WHERE id = :task_id AND workspace_id = :ws AND runtime_ref ? :key
 RETURNING id
    """
)
CLEARED_NOTE = "Took the pull request off this ticket: {url}"
CLEARED_BY = "you"


def clear_pull_request(db: Session, *, task_id: Any, workspace_id: Any) -> Optional[str]:
    """In the caller's transaction: remove the ticket's pull request with one statement and note it,
    signed by the owner. Returns the address it had (or '' when it had none), or None when the ticket
    has no pull request in this workspace. Nothing puts it back: the listener carries the row's
    absence into a later whole-document write (``kept_runtime_keys``)."""
    from services.cli_host_service import append_session_note

    keys = {"task_id": int(task_id), "ws": str(workspace_id), "key": PULL_REQUEST_KEY}
    row = db.execute(_STORED_UNDER_LOCK, keys).first()
    if row is None or not isinstance(row[1], Mapping):
        return None
    if db.execute(_CLEAR_THE_KEY, keys).first() is None:
        return None
    url = row[1].get("url") if isinstance(row[1].get("url"), str) else ""
    append_session_note(db, task_id=task_id, workspace_id=workspace_id,
                        note=CLEARED_NOTE.format(url=url or "it had no address"), by=CLEARED_BY)
    return url


def _roll_back(db: Session) -> None:
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 — logged; the failure itself is already the answer
        logger.exception("[ticket-pr] rollback after a failed pull request write failed too")


__all__ = [
    "BAD_BRANCH", "BAD_COMMITS", "BAD_TESTED", "BAD_URL", "CLEARED_NOTE", "FIELDS", "NOT_OPEN_FOR_CHANGES",
    "PULL_REQUEST_NOTE", "PULL_REQUEST_PROPERTIES", "PullRequestAddress", "PullRequestRefused", "TicketNotOpen",
    "clear_pull_request", "merge_allowed_repos", "merges_on_approve", "parse_pull_request_url", "pull_request_fields",
    "pull_request_of", "record_ticket_update", "updated_pull_request", "write_pull_request",
]
