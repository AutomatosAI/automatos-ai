"""Ticket PR fields: a person's Approve merges the ticket's pull request (off by default).

With ``config.TICKET_APPROVE_MERGES_PR`` on, a person's Approve of a ticket whose session reported
a GitHub pull request (``runtime_ref.pull_request``, ``services/ticket_pull_request.py``) merges
it. A person is the Done gate's person (``done_needs_an_artifact.a_persons_request``: a Clerk
sign-in or a paired phone or browser), and only a workspace owner or admin merges
(``MERGE_PERMISSION``, ``workspace:manage``): approving needs only ``missions:update``, which an
editor holds too, and an editor's Approve closes the ticket and says the merge needs an owner. The merge runs after the approval's own action and before
its result and notes are written (``api/board_card_moves.after_its_action``, called once from
``api/board_tasks.approve_task``), and only while the ticket is still that approval's Done.

* **Only what the owner saw, and only this ticket's own work.** The web sends the pull request it
  showed (its address and the ``at`` stamp of its last write) with the Approve; anything else on
  the ticket now is "changed since you looked". The pull request is frozen once the ticket leaves
  In progress (``ticket_pull_request.write_pull_request``). Its repository must be listed in
  ``TICKET_APPROVE_MERGES_PR_REPOS`` (empty: nothing merges), GitHub must say it belongs to that
  repository and comes from a branch of it (no fork), on the branch the session reported.

* **Through Composio, on the workspace's own GitHub connection.** The platform acts for the person
  who approved, not for an agent, so the call is the platform's admin pattern
  (``api/bug_reports.py``, ``modules/socials/recipes/toolkit.call``: agent 0, ``skip_validation``).
  That skips the agent's app assignment, the action cache's mapping (and its fuzzy auto-map, which
  has no place in a merge) and the per-agent action switches, so this module checks the connection
  itself. The deny list runs inside ``ComposioToolExecutor.execute`` whatever ``skip_validation``
  says: a super-admin who lists GITHUB_MERGE_A_PULL_REQUEST stops this too.
* **Only when GitHub says it can merge.** The pull request is open, not a draft, ``mergeable``, and
  its ``mergeable_state`` is ``clean``; every check run on its head commit has completed and passed,
  and there is at least one. Anything unknown is a no. The merge names the head commit it checked
  (``sha``), so a push after the check is not merged, and it squashes (``MERGE_METHOD``, the
  repository's convention).
* **One attempt, never retried.** The answer is a note: "Merged <url>", or "Approved. The pull
  request was not merged: <reason>. Merge it on GitHub." The ticket closes as approved either way.
* **A merge that happened is never lost.** "Merged <url>" and the merge's mark on the pull request
  are committed in a transaction of their own the moment GitHub says it merged
  (``_keep_the_merge``). If a later step of the approval fails, the ticket stays Done
  (``merged_by_this_approval``, read by the approve route's failure path) instead of reopening.

Nothing else reaches it. An agent's ``update_ticket`` only records the pull request. Auto's status
tool and the chat card's click never call it (the click keeps its note with
``ticket_verdict.keep_approval_note``, not ``record_approval``). A call without a person's lane
returns before reading anything.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

from sqlalchemy import text

from config import config
from services.ticket_pull_request import (GITHUB_HOST, GITHUB_SHAPE, PullRequestAddress, merge_allowed_repos,
                                          parse_pull_request_url, pull_request_of)

logger = logging.getLogger(__name__)

GITHUB_APP = "GITHUB"
GET_PULL_REQUEST = "GITHUB_GET_A_PULL_REQUEST"
LIST_CHECK_RUNS = "GITHUB_LIST_CHECK_RUNS_FOR_A_REF"
MERGE_PULL_REQUEST = "GITHUB_MERGE_A_PULL_REQUEST"
MERGE_METHOD = "squash"
# The admin pattern (api/bug_reports.py): the platform calls the workspace's own connection, as no agent.
PLATFORM_AGENT_ID = 0
CHECK_RUNS_PER_PAGE = 100
GREEN_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
OPEN, CLEAN, UNKNOWN, DIRTY, DRAFT, COMPLETED = "open", "clean", "unknown", "dirty", "draft", "completed"
NAMES_SHOWN = 3
REASON_CHARS = 300
HEAD_SHA_CHARS = 40
# The merge's own short transaction waits at most this long for the ticket's row (then the note
# rides the approval's own transaction instead: logged).
NOTE_LOCK_TIMEOUT_MS = 5000
MERGE_MARK = "merged"
# Merging is an owner's or admin's (modules/policy/roles.py): approving is ``missions:update``, an editor's too.
MERGE_PERMISSION = "workspace:manage"
MERGE_NOTE_BY = "you"   # the person who approved (ticket_verdict.OPERATOR_NOTE_BY)

MERGED_NOTE = "Merged {url}"
NOT_MERGED_NOTE = "Approved. The pull request was not merged: {reason}. Merge it on GitHub."
NOT_MERGED_ELSEWHERE_NOTE = "Approved. The pull request was not merged: {reason}. Merge it on {host}."
ALREADY_MERGED_NOTE = "Approved. The pull request {url} was already merged."
# The merge call answered, but not whether it merged: never said as refused, since it may have.
MERGE_UNKNOWN_NOTE = ("Approved. Whether the pull request merged is unknown: GitHub's answer to the merge could "
                      "not be read. Check it on GitHub: {url}")
# Why it was not merged, in words (each completes NOT_MERGED_NOTE's sentence).
NOT_ON_GITHUB = "it is on {host}, and Automatos merges only GitHub pull requests, through the workspace's GitHub connection"
NOT_CONNECTED = "GitHub is not connected for this workspace"
NOT_OPEN = "it is {state} on GitHub"
IS_DRAFT = "it is a draft"
NOT_WORKED_OUT = "GitHub has not worked out yet whether it can be merged"
HAS_CONFLICTS = "it has conflicts with its base branch"
NOT_CLEAN = "GitHub reports it as {state}: a required check or review has not passed, or its branch is behind its base"
NO_CHECKS = "no checks have reported on its latest commit"
CHECKS_RUNNING = "its checks are still running ({names})"
CHECKS_FAILED = "its checks did not pass ({names})"
TOO_MANY_CHECKS = "it has more checks than one read covers, so whether they passed is unknown"
UNREADABLE = "GitHub's answer about it could not be read"
NO_ANSWER = "GitHub could not be read through the workspace's connection ({error})"
MERGE_REFUSED = "the merge was refused ({error})"
COULD_NOT_TRY = "the merge could not be attempted"
CHANGED_SINCE = "it changed since you looked"
NEEDS_OWNER = "merging needs a workspace owner or admin"
NO_REPOS = "no repository is set up to merge on approve (TICKET_APPROVE_MERGES_PR_REPOS is empty)"
NOT_LISTED = "{repository} is not one of the repositories set up to merge on approve"
NOT_ITS_OWN = "GitHub's answer does not say which repository and branch it comes from"
NOT_THIS_REPO = "GitHub says it belongs to {base}, not {repository}"
FROM_A_FORK = "it comes from another repository ({head}), not from a branch of {base}"
OTHER_BRANCH = "its branch is {ref}, not {branch}, the branch the session reported"
NO_REASON = "no reason given"


@dataclass(frozen=True)
class Readiness:
    """Whether the pull request may be merged: its head commit, or why not."""

    head_sha: Optional[str] = None
    reason: Optional[str] = None
    merged_already: bool = False


async def merge_on_approve(db: Any, ctx: Any, task_id: int, *, decided_at: Any,
                           shown: Any = None) -> Optional[str]:
    """The note ``record_approval`` still has to write about the ticket's pull request after a
    person's Approve, or None: nothing to do (the setting off, not a person, no pull request, the
    ticket moved while the approval ran), or "Merged <url>" already committed on its own.
    ``shown`` is the pull request the Approve's page showed (``{"url", "at"}``), None from a
    client that sends none. Never raises: the approval stands whatever happens to the merge."""
    from services.done_needs_an_artifact import a_persons_request

    if not config.TICKET_APPROVE_MERGES_PR or not a_persons_request(ctx):
        return None
    try:
        return await _merge(db, ctx, int(task_id), decided_at, shown)
    except Exception:  # noqa: BLE001 — logged; the approval stands and the note says what happened
        logger.exception("[ticket-pr-merge] ticket #%s: the merge could not be attempted", task_id)
        return NOT_MERGED_NOTE.format(reason=COULD_NOT_TRY)


async def _merge(db: Any, ctx: Any, task_id: int, decided_at: Any, shown: Any) -> Optional[str]:
    workspace_id = ctx.workspace_id
    view = await asyncio.to_thread(approved_pull_request, db, workspace_id, task_id, decided_at)
    if view is None:
        return None
    if not same_as_shown(view, shown):
        return not_merged(CHANGED_SINCE)
    address = parse_pull_request_url(view.get("url"))
    if address is None:
        return None
    refused = refusal_before_github(address)
    if refused:
        return refused
    if not await asyncio.to_thread(may_merge, db, ctx):
        return not_merged(NEEDS_OWNER)
    if not await asyncio.to_thread(github_connected, db, workspace_id):
        return not_merged(NOT_CONNECTED)
    executor = _executor(db)
    ready = await _readiness(executor, workspace_id, address, view.get("branch"))
    if ready.merged_already:
        return ALREADY_MERGED_NOTE.format(url=address.url)
    if ready.reason:
        logger.info("[ticket-pr-merge] ticket #%s: %s not merged: %s", task_id, address.url, ready.reason)
        return not_merged(ready.reason)
    return await _squash_merge(executor, workspace_id, address, ready.head_sha or "", task_id=task_id,
                               decided_at=decided_at)


def not_merged(reason: str) -> str:
    return NOT_MERGED_NOTE.format(reason=reason)


def same_as_shown(view: Mapping, shown: Any) -> bool:
    """The pull request the owner was shown is the one on the ticket now: the same address and the
    same stamp of its last write. Anything that is not exactly the stored identity does not agree.

    A client that sends none (an older one, the phone today) agrees, and that is safe: the pull
    request is frozen once its ticket leaves In progress (``ticket_pull_request.write_pull_request``
    refuses every write after that), and Approve works only on a ticket in Review, so nothing can be
    swapped between the owner viewing the card and approving it. Pure."""
    if shown is None or shown == {}:
        return True
    if not isinstance(shown, Mapping):
        return False
    return shown.get("url") == view.get("url") and shown.get("at") == view.get("at")


def refusal_before_github(address: PullRequestAddress) -> Optional[str]:
    """The note for a pull request this install never merges, before anything is asked of GitHub:
    another host, or a repository not listed in ``TICKET_APPROVE_MERGES_PR_REPOS``. Pure."""
    if address.shape != GITHUB_SHAPE or address.host != GITHUB_HOST:
        return NOT_MERGED_ELSEWHERE_NOTE.format(reason=NOT_ON_GITHUB.format(host=address.host), host=address.host)
    allowed = merge_allowed_repos()
    if not allowed:
        return not_merged(NO_REPOS)
    if address.repository.lower() not in allowed:
        return not_merged(NOT_LISTED.format(repository=address.repository))
    return None


def may_merge(db: Any, ctx: Any) -> bool:
    """The approving person may merge: a workspace owner or admin (``MERGE_PERMISSION``), the
    platform super-admin included (``workspace_permission_granted``'s one bypass)."""
    from core.auth.workspace_permission import workspace_permission_granted

    return workspace_permission_granted(db, ctx, MERGE_PERMISSION)


def approved_pull_request(db: Any, workspace_id: Any, task_id: int, decided_at: Any) -> Optional[Dict[str, Any]]:
    """The ticket's pull request, as the board shows it, while the ticket is still this approval's
    Done (F195's own test: ``ticket_verdict.record_approval``): ``{}`` when it has none (never had
    one, or the owner took it off), None when the ticket moved while the approval ran."""
    from core.models.core import BoardTask

    row = db.query(BoardTask.runtime_ref).filter(
        BoardTask.id == task_id, BoardTask.workspace_id == workspace_id,
        BoardTask.status == "done", BoardTask.completed_at == decided_at,
    ).first()
    return (pull_request_of(row[0]) or {}) if row is not None else None


def github_connected(db: Any, workspace_id: Any) -> bool:
    """The workspace has a live GitHub connection (the check ``skip_validation`` leaves out). Read
    only (``reading_only``): a pending connection is not settled here, so this request takes no row
    lock that it would then hold across the merge's calls to GitHub (the F210 lesson)."""
    from core.composio.entity_manager import EntityManager
    from core.read_only_call import reading_only

    with reading_only():
        connected = EntityManager(db).get_connected_apps(workspace_id)
    return GITHUB_APP in {str(app).upper().strip() for app in connected}


def _executor(db: Any) -> Any:
    from core.composio.tool_executor import ComposioToolExecutor

    return ComposioToolExecutor(db)


async def _call(executor: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> Tuple[Optional[Mapping], str]:
    """One GitHub action on the workspace's connection: (its answer, '') or (None, why not)."""
    result = await executor.execute(action=action, params=params, agent_id=PLATFORM_AGENT_ID,
                                    workspace_id=workspace_id, app_name=GITHUB_APP, skip_validation=True)
    if not isinstance(result, Mapping) or not result.get("success"):
        error = result.get("error") if isinstance(result, Mapping) else None
        return None, said(error)
    answer = payload_of(result)
    return (answer, "") if answer is not None else (None, UNREADABLE)


def payload_of(result: Mapping) -> Optional[Mapping]:
    """GitHub's own answer inside an executor result: out of Composio's ``{"data", "successful",
    "error"}`` envelope (``result.data.data``, as ``api/workspace_github.list_github_repos`` reads
    it), parsed when it came as JSON text, and out of a lone ``details`` wrapper. None when there is
    no object to read."""
    data = result.get("data")
    if isinstance(data, Mapping) and "data" in data and ("successful" in data or "error" in data or len(data) == 1):
        data = data["data"]
    if isinstance(data, str) and data.strip()[:1] == "{":
        try:
            data = json.loads(data)
        except ValueError:
            return None
    if isinstance(data, Mapping) and set(data) == {"details"} and isinstance(data["details"], Mapping):
        data = data["details"]
    return data if isinstance(data, Mapping) else None


def said(error: Any) -> str:
    """An outside error as one short line."""
    return " ".join(str(error or NO_REASON).split())[:REASON_CHARS]


def _pull_params(address: PullRequestAddress) -> Dict[str, Any]:
    return {"owner": address.owner, "repo": address.repo, "pull_number": address.number}


async def _readiness(executor: Any, workspace_id: Any, address: PullRequestAddress,
                     branch: Optional[str]) -> Readiness:
    """The pull request's own repository and branch, its state, then its head commit's checks,
    read through Composio."""
    pull, error = await _call(executor, workspace_id, GET_PULL_REQUEST, _pull_params(address))
    if pull is None:
        return Readiness(reason=NO_ANSWER.format(error=error) if error != UNREADABLE else UNREADABLE)
    not_its_own = ownership_blocker(pull, address, branch)
    if not_its_own:
        return Readiness(reason=not_its_own)
    if pull.get("merged") is True:
        return Readiness(merged_already=True)
    blocker = pull_request_blocker(pull)
    if blocker:
        return Readiness(reason=blocker)
    head = head_sha(pull)
    if head is None:
        return Readiness(reason=UNREADABLE)
    runs, error = await _call(executor, workspace_id, LIST_CHECK_RUNS, {
        "owner": address.owner, "repo": address.repo, "ref": head, "per_page": CHECK_RUNS_PER_PAGE})
    if runs is None:
        return Readiness(reason=NO_ANSWER.format(error=error) if error != UNREADABLE else UNREADABLE)
    blocker = checks_blocker(runs)
    return Readiness(reason=blocker) if blocker else Readiness(head_sha=head)


def _repo_of(side: Any) -> Optional[str]:
    repo = side.get("repo") if isinstance(side, Mapping) else None
    name = repo.get("full_name") if isinstance(repo, Mapping) else None
    return name if isinstance(name, str) and name else None


def ownership_blocker(pull: Mapping, address: PullRequestAddress, branch: Optional[str]) -> Optional[str]:
    """Why the pull request is not this ticket's own work, or None: GitHub says it belongs to the
    address's repository, comes from a branch of that same repository (not a fork), and, when the
    session reported its branch, from that branch. Anything GitHub left out is a no. Pure."""
    head, base = pull.get("head"), pull.get("base")
    head_repo, base_repo = _repo_of(head), _repo_of(base)
    ref = head.get("ref") if isinstance(head, Mapping) else None
    if not head_repo or not base_repo or not isinstance(ref, str):
        return NOT_ITS_OWN
    if base_repo.lower() != address.repository.lower():
        return NOT_THIS_REPO.format(base=said(base_repo), repository=address.repository)
    if head_repo.lower() != base_repo.lower():
        return FROM_A_FORK.format(head=said(head_repo), base=said(base_repo))
    if branch and ref != branch:
        return OTHER_BRANCH.format(ref=said(ref), branch=branch)
    return None


def pull_request_blocker(pull: Mapping) -> Optional[str]:
    """Why GitHub's view of the pull request stops a merge, or None when it is open, not a draft,
    mergeable and clean. A field GitHub left out is unknown, and unknown is a no. Pure."""
    state = pull.get("state")
    if state != OPEN:
        return NOT_OPEN.format(state=said(state or UNKNOWN)) if isinstance(state, str) else UNREADABLE
    if pull.get("draft") is True:
        return IS_DRAFT
    mergeable, merge_state = pull.get("mergeable"), pull.get("mergeable_state")
    if mergeable is False or merge_state == DIRTY:
        return HAS_CONFLICTS
    if mergeable is not True or merge_state in (None, UNKNOWN):
        return NOT_WORKED_OUT
    if merge_state == DRAFT:
        return IS_DRAFT
    return None if merge_state == CLEAN else NOT_CLEAN.format(state=said(merge_state))


def head_sha(pull: Mapping) -> Optional[str]:
    head = pull.get("head")
    sha = head.get("sha") if isinstance(head, Mapping) else None
    if not isinstance(sha, str) or len(sha) != HEAD_SHA_CHARS or any(c not in "0123456789abcdef" for c in sha.lower()):
        return None
    return sha


def checks_blocker(answer: Mapping) -> Optional[str]:
    """Why the head commit's check runs stop a merge, or None when there is at least one and
    every one has completed and passed. Pure."""
    runs, total = answer.get("check_runs"), answer.get("total_count")
    if not isinstance(runs, list) or not isinstance(total, int) or not all(isinstance(r, Mapping) for r in runs):
        return UNREADABLE
    if total > len(runs):
        return TOO_MANY_CHECKS
    if not runs:
        return NO_CHECKS
    running = [r for r in runs if r.get("status") != COMPLETED]
    if running:
        return CHECKS_RUNNING.format(names=_names(running))
    failed = [r for r in runs if r.get("conclusion") not in GREEN_CONCLUSIONS]
    return CHECKS_FAILED.format(names=_names(failed)) if failed else None


def _names(runs: list) -> str:
    names = [said(r.get("name") or "a check") for r in runs]
    shown = ", ".join(names[:NAMES_SHOWN])
    return shown if len(names) <= NAMES_SHOWN else f"{shown} and {len(names) - NAMES_SHOWN} more"


async def _squash_merge(executor: Any, workspace_id: Any, address: PullRequestAddress, head: str, *,
                        task_id: int, decided_at: Any) -> Optional[str]:
    """The one merge call: squash, at the head commit that was checked. A merge GitHub confirms is
    kept at once, in its own transaction (None then); every other answer is the note to write."""
    answer, error = await _call(executor, workspace_id, MERGE_PULL_REQUEST, {
        **_pull_params(address), "merge_method": MERGE_METHOD, "sha": head})
    if answer is not None and answer.get("merged") is True:
        logger.info("[ticket-pr-merge] ticket #%s: merged %s", task_id, address.url)
        note = MERGED_NOTE.format(url=address.url)
        kept = await asyncio.to_thread(_keep_the_merge, workspace_id, task_id, note, decided_at, answer.get("sha"))
        return None if kept else note
    if error == UNREADABLE or (answer is not None and not isinstance(answer.get("merged"), bool)):
        logger.warning("[ticket-pr-merge] ticket #%s: the merge of %s answered unreadably", task_id, address.url)
        return MERGE_UNKNOWN_NOTE.format(url=address.url)
    why = error or said((answer or {}).get("message"))
    logger.warning("[ticket-pr-merge] ticket #%s: %s was not merged: %s", task_id, address.url, why)
    return not_merged(MERGE_REFUSED.format(error=why))


def _stamp(decided_at: Any) -> str:
    """The approval's own time as the merge's mark keeps it."""
    return decided_at.isoformat() if isinstance(decided_at, datetime) else str(decided_at)


_LOCK_TIMEOUT = text("SELECT set_config('lock_timeout', :timeout, true)")
_MARK_MERGED = text(
    """
    UPDATE board_tasks
       SET runtime_ref = jsonb_set(runtime_ref, CAST(:path AS text[]), CAST(:mark AS jsonb), true)
     WHERE id = :task_id AND workspace_id = :ws AND runtime_ref ? :key
    """
)
_MERGE_MARK_OF = text(
    "SELECT runtime_ref -> CAST(:key AS text) -> CAST(:mark AS text) ->> 'approved_at' "
    "FROM board_tasks WHERE id = :task_id AND workspace_id = :ws"
)


def _keep_the_merge(workspace_id: Any, task_id: int, note: str, decided_at: Any, sha: Any) -> bool:
    """The note "Merged <url>" and the merge's mark (``pull_request.merged``: when, its commit, and this
    approval's time) committed in a transaction of their own, so no later failure of the approval
    loses them. It waits at most NOTE_LOCK_TIMEOUT_MS for the row. True when it landed; on any
    failure (logged) the note rides the approval's own transaction instead."""
    from core.database import database
    from core.models.kept_runtime_keys import PULL_REQUEST_KEY
    from services.cli_host_service import append_session_note

    mark = {"at": datetime.now(timezone.utc).isoformat(), "sha": sha if isinstance(sha, str) else None,
            "approved_at": _stamp(decided_at)}
    keys = {"task_id": int(task_id), "ws": str(workspace_id)}
    try:
        with database.get_db_session() as own:
            own.execute(_LOCK_TIMEOUT, {"timeout": f"{NOTE_LOCK_TIMEOUT_MS}ms"})
            own.execute(_MARK_MERGED, {**keys, "key": PULL_REQUEST_KEY, "mark": json.dumps(mark),
                                       "path": "{%s,%s}" % (PULL_REQUEST_KEY, MERGE_MARK)})
            append_session_note(own, task_id=task_id, workspace_id=workspace_id, note=note, by=MERGE_NOTE_BY)
        return True
    except Exception:  # noqa: BLE001 — logged; the note is written with the approval instead
        logger.exception("[ticket-pr-merge] ticket #%s: the merge's note could not be kept on its own", task_id)
        return False


def merged_by_this_approval(db: Any, workspace_id: Any, task_id: int, decided_at: Any) -> bool:
    """After a later step of the approval failed: whether this approval merged the ticket's pull
    request (its mark carries this approval's time). Then the ticket stays Done, as the merge does.
    Rolls the failed transaction back first; False (logged) when it cannot tell."""
    from core.models.kept_runtime_keys import PULL_REQUEST_KEY

    try:
        db.rollback()
        row = db.execute(_MERGE_MARK_OF, {"key": PULL_REQUEST_KEY, "mark": MERGE_MARK, "task_id": int(task_id),
                                          "ws": str(workspace_id)}).first()
    except Exception:  # noqa: BLE001 — logged; the approval's failure path goes on as before
        logger.exception("[ticket-pr-merge] ticket #%s: could not read whether its approval merged", task_id)
        return False
    merged = bool(row is not None and row[0] == _stamp(decided_at))
    if merged:
        logger.warning("[ticket-pr-merge] ticket #%s: a step after its merge failed; it stays Done, as its pull "
                       "request is merged", task_id)
    return merged


__all__ = [
    "ALREADY_MERGED_NOTE", "GET_PULL_REQUEST", "LIST_CHECK_RUNS", "MERGED_NOTE", "MERGE_METHOD", "MERGE_PULL_REQUEST",
    "MERGE_UNKNOWN_NOTE", "NOT_MERGED_ELSEWHERE_NOTE", "NOT_MERGED_NOTE", "Readiness", "checks_blocker",
    "may_merge", "merge_on_approve", "merged_by_this_approval", "ownership_blocker", "payload_of",
    "pull_request_blocker",
]
