"""Ticket PR fields: a person's Approve merges the ticket's pull request, when the owner turned that on.

``TICKET_APPROVE_MERGES_PR`` (off by default) lets the owner approve a coding ticket on his phone
and have that approval merge the pull request its session opened (``services/ticket_pr_merge.py``).
This pins:

* off, nothing is read and nothing merges; on, only a person's Approve merges (a Clerk sign-in or a
  paired device, the Done gate's lanes), never Auto's status tool or an agent's ``update_ticket``,
  and only a workspace owner's or admin's (``workspace:manage``): an editor may approve
  (``missions:update``), and the ticket closes, but nothing merges and the note says why;
* only what the owner was shown (the address and the ``at`` stamp the Approve carries), only a
  repository listed in ``TICKET_APPROVE_MERGES_PR_REPOS`` (empty: nothing), and only the ticket's own
  work: GitHub says it belongs to that repository, comes from a branch of it (no fork), on the
  branch the session reported;
* the merge is ``GITHUB_MERGE_A_PULL_REQUEST`` through Composio on the workspace's connection, the
  squash method, the number from the address, at the head commit that was checked, and "Merged
  <url>" is kept at once in a transaction of its own;
* GitHub must say it can merge: open, not a draft, mergeable and clean, every check run on the head
  commit completed and green. Anything else, or anything unknown, closes the ticket as approved with
  the not-merged note and nothing is retried;
* a pull request on another host is never merged through the GitHub action;
* nothing outside ``api/board_card_moves.py`` imports the merge module or any of its names.
"""
from __future__ import annotations

import ast
import asyncio
import functools
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from config import config
from core.composio.deny_list import BLOCKED_PREFIX, DENIED_REASON, denied_result
from services import ticket_pr_merge as tpm
from services import ticket_pull_request as tpr
from tests.test_prd256_done_needs_artifact import QUESTION, _body, _not_a_person, _notes, _person
from tests.test_prd256_done_needs_artifact import board, quiet  # noqa: F401 — the fixtures, as that suite builds them

WS = UUID("6f0e8a43-0c55-4c2b-9a1e-3f1d2b7c8e90")
REPO = "automatos-ai/automatos-ai"
URL = f"https://github.com/{REPO}/pull/1559"
BRANCH = "feat/ticket-pr-fields"
AT = "2026-11-18T09:30:00+00:00"
HEAD = "9a9cb36d9e1f4c2b7a0d3e5f6a7b8c9d0e1f2a3b"
OPEN_PULL = {"state": "open", "merged": False, "draft": False, "mergeable": True, "mergeable_state": "clean",
             "head": {"sha": HEAD, "ref": BRANCH, "repo": {"full_name": REPO}},
             "base": {"ref": "main", "repo": {"full_name": REPO}}}
GREEN_RUNS = {"total_count": 2, "check_runs": [{"name": "backend", "status": "completed", "conclusion": "success"},
                                               {"name": "e2e", "status": "completed", "conclusion": "skipped"}]}
MERGE_SHA = "c0ffee" * 6 + "abcd"
MERGED = {"merged": True, "sha": MERGE_SHA, "message": "Pull Request successfully merged"}
PERSON = NS(workspace_id=WS, user_id="2", auth_type="clerk")
DECIDED = object()
SHOWN = {"url": URL, "at": AT}


class _GitHub:
    """The executor's shape. Each action answers what the test scripted, in Composio's envelope;
    a scripted executor result (``{"success": False, …}``) is returned as it is."""

    def __init__(self, pull=None, runs=None, merged=None):
        self.calls = []
        self.answers = {tpm.GET_PULL_REQUEST: OPEN_PULL if pull is None else pull,
                        tpm.LIST_CHECK_RUNS: GREEN_RUNS if runs is None else runs,
                        tpm.MERGE_PULL_REQUEST: MERGED if merged is None else merged}

    async def execute(self, *, action, params, agent_id, workspace_id, app_name=None, skip_validation=False):
        self.calls.append({"action": action, "params": params, "agent_id": agent_id, "workspace_id": workspace_id,
                           "app_name": app_name, "skip_validation": skip_validation})
        answer = self.answers[action]
        if isinstance(answer, dict) and answer.get("success") is False:
            return answer
        return {"success": True, "data": {"data": answer, "successful": True, "error": None}}

    def actions(self):
        return [c["action"] for c in self.calls]


def _run(coro):
    return asyncio.run(coro)


def _view(url=URL, branch=BRANCH, at=AT):
    """The ticket's pull request as the board serves it (and the merge reads it)."""
    return tpr.pull_request_of({"pull_request": {"url": url, "branch": branch, "at": at}})


@pytest.fixture
def merging(monkeypatch):
    """The setting on, the repository listed, GitHub connected, the ticket's pull request; ``use(executor)``
    swaps GitHub, ``kept`` records the merge's own short transaction."""
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", True)
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", REPO)
    monkeypatch.setattr(tpm, "github_connected", lambda db, ws: True)
    state = NS(github=_GitHub(), view=_view(), kept=[], keeps=True, may=True)
    monkeypatch.setattr(tpm, "approved_pull_request", lambda db, ws, task_id, decided_at: state.view)
    monkeypatch.setattr(tpm, "may_merge", lambda db, ctx: state.may)
    monkeypatch.setattr(tpm, "_executor", lambda db: state.github)

    def _keep(workspace_id, task_id, note, decided_at, sha):
        state.kept.append((workspace_id, task_id, note, decided_at, sha))
        return state.keeps

    monkeypatch.setattr(tpm, "_keep_the_merge", _keep)

    def use(github):
        state.github = github
        return github

    return NS(state=state, use=use)


def _merge(ctx=PERSON, shown=None):
    return _run(tpm.merge_on_approve(None, ctx, 116, decided_at=DECIDED, shown=shown))


# ── who and when ─────────────────────────────────────────────────────────────

def test_with_the_setting_off_nothing_is_read_and_nothing_merges(monkeypatch):
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", False)
    monkeypatch.setattr(tpm, "approved_pull_request", lambda *a: pytest.fail("read the ticket with the setting off"))
    monkeypatch.setattr(tpm, "_executor", lambda db: pytest.fail("reached Composio with the setting off"))
    assert _merge() is None


def test_the_settings_are_off_and_empty_unless_the_owner_sets_them():
    import json

    assert config.TICKET_APPROVE_MERGES_PR is False                    # CI sets nothing: these are the defaults
    assert config.TICKET_APPROVE_MERGES_PR_REPOS == "" and tpr.merge_allowed_repos() == frozenset()
    surface = Path(__file__).resolve().parents[1] / "reports" / "config-surface.json"
    names = json.loads(surface.read_text())["settings"]
    assert "TICKET_APPROVE_MERGES_PR" in names and "TICKET_APPROVE_MERGES_PR_REPOS" in names


@pytest.mark.parametrize("ctx", [
    NS(workspace_id=WS, user_id="2"),                         # an in-process call for Auto: no lane
    NS(workspace_id=WS, user_id="2", auth_type="anonymous"),  # no credential: an agent's http_request looks like this
    NS(workspace_id=WS, user_id=None, auth_type="api_key"), NS(workspace_id=WS, user_id=None, auth_type="sdk_key"),
    None,
])
def test_only_a_persons_approve_merges(merging, ctx):
    assert _merge(ctx) is None and merging.state.github.calls == []


@pytest.mark.parametrize("lane", ["clerk", "device"])
def test_a_persons_approve_of_a_green_pull_request_merges_it_squashed_at_the_checked_commit(merging, lane):
    github = merging.state.github
    assert _merge(NS(workspace_id=WS, user_id="2", auth_type=lane), shown=SHOWN) is None   # noted on its own
    assert merging.state.kept == [(WS, 116, tpm.MERGED_NOTE.format(url=URL), DECIDED, MERGE_SHA)]
    assert github.actions() == [tpm.GET_PULL_REQUEST, tpm.LIST_CHECK_RUNS, tpm.MERGE_PULL_REQUEST]
    get, runs, merge = github.calls
    assert get["params"] == {"owner": "automatos-ai", "repo": "automatos-ai", "pull_number": 1559}
    assert runs["params"] == {"owner": "automatos-ai", "repo": "automatos-ai", "ref": HEAD,
                              "per_page": tpm.CHECK_RUNS_PER_PAGE}
    assert merge["params"] == {"owner": "automatos-ai", "repo": "automatos-ai", "pull_number": 1559,
                               "merge_method": "squash", "sha": HEAD}
    assert tpm.MERGE_METHOD == "squash"
    for call in github.calls:   # the platform's admin pattern, on the workspace's own connection
        assert (call["agent_id"], call["app_name"], call["skip_validation"], call["workspace_id"]) == (
            0, "GITHUB", True, WS)


def test_a_client_that_sends_no_identity_merges_the_stored_pull_request(merging):
    assert _merge(shown=None) is None and merging.state.kept and merging.state.github.actions()[-1] == \
        tpm.MERGE_PULL_REQUEST


def test_when_its_own_transaction_fails_the_merge_note_rides_the_approval(merging):
    merging.state.keeps = False
    assert _merge() == tpm.MERGED_NOTE.format(url=URL)


def test_a_person_who_is_not_an_owner_or_admin_approves_but_merges_nothing(merging):
    merging.state.may = False
    assert _merge() == "Approved. The pull request was not merged: merging needs a workspace owner or admin. " \
                       "Merge it on GitHub."
    assert merging.state.github.calls == [] and merging.state.kept == []


@pytest.mark.parametrize("role, merges", [("owner", True), ("admin", True), ("editor", False), ("viewer", False),
                                          (None, False)])
def test_merging_is_an_owners_or_admins_and_approving_is_not(monkeypatch, role, merges):
    from core.auth import workspace_permission
    from modules.policy.roles import workspace_has_permission

    monkeypatch.setattr(workspace_permission, "resolve_workspace_role", lambda db, ctx: role)
    person = NS(workspace_id=WS, user_id="2", auth_type="clerk", user=NS(id="2", clerk_user_id="user_2"))
    assert tpm.MERGE_PERMISSION == "workspace:manage"
    assert tpm.may_merge(None, person) is merges
    assert workspace_has_permission("editor", "missions:update") is True      # an editor may approve


def test_a_ticket_moved_while_the_approval_ran_merges_nothing(merging):
    merging.state.view = None
    assert _merge() is None and merging.state.github.calls == []


# ── only what the owner saw, only listed repositories, only the ticket's own work ──

@pytest.mark.parametrize("shown", [
    {"url": "https://github.com/automatos-ai/automatos-ai/pull/1560", "at": AT},   # another address
    {"url": URL, "at": "2026-11-18T09:31:00+00:00"},                               # written again since
    {"url": URL},                                                                   # no stamp
    "https://github.com/automatos-ai/automatos-ai/pull/1559",                       # not an identity
], ids=["url-differs", "at-differs", "no-at", "malformed"])
def test_a_pull_request_that_changed_since_the_owner_looked_is_never_merged(merging, shown):
    assert _merge(shown=shown) == "Approved. The pull request was not merged: it changed since you looked. " \
                                  "Merge it on GitHub."
    assert merging.state.github.calls == []


def test_a_ticket_with_no_pull_request_merges_nothing_and_says_nothing(merging):
    merging.state.view = {}                                     # the reader's answer for a ticket without one
    assert _merge() is None and merging.state.github.calls == []
    assert _merge(shown=SHOWN) == tpm.not_merged(tpm.CHANGED_SINCE)   # the page showed one: it changed


@pytest.mark.parametrize("repos, reason", [
    ("", tpm.NO_REPOS),
    ("automatos-ai/automatos-go, someone/else", tpm.NOT_LISTED.format(repository=REPO)),
], ids=["allowlist-empty", "repo-not-listed"])
def test_a_repository_not_set_up_to_merge_on_approve_is_never_merged(merging, monkeypatch, repos, reason):
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", repos)
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=reason) and merging.state.github.calls == []


def test_the_allowlist_ignores_case_and_spaces(merging, monkeypatch):
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", " Someone/Else ,AUTOMATOS-AI/Automatos-AI ")
    assert _merge() is None and merging.state.github.actions()[-1] == tpm.MERGE_PULL_REQUEST


@pytest.mark.parametrize("pull, reason", [
    ({**OPEN_PULL, "head": {**OPEN_PULL["head"], "repo": {"full_name": "a-stranger/automatos-ai"}}},
     tpm.FROM_A_FORK.format(head="a-stranger/automatos-ai", base=REPO)),
    ({**OPEN_PULL, "head": {**OPEN_PULL["head"], "ref": "main"}},
     tpm.OTHER_BRANCH.format(ref="main", branch=BRANCH)),
    ({**OPEN_PULL, "base": {"ref": "main", "repo": {"full_name": "automatos-ai/renamed"}},
      "head": {**OPEN_PULL["head"], "repo": {"full_name": "automatos-ai/renamed"}}},
     tpm.NOT_THIS_REPO.format(base="automatos-ai/renamed", repository=REPO)),
    ({**OPEN_PULL, "head": {"sha": HEAD, "ref": BRANCH, "repo": None}}, tpm.NOT_ITS_OWN),   # its fork was deleted
    ({k: v for k, v in OPEN_PULL.items() if k != "base"}, tpm.NOT_ITS_OWN),
], ids=["fork", "branch-mismatch", "another-repository", "head-repo-gone", "no-base"])
def test_a_pull_request_that_is_not_the_tickets_own_work_is_never_merged(merging, pull, reason):
    github = merging.use(_GitHub(pull=pull))
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=reason)
    assert github.actions() == [tpm.GET_PULL_REQUEST]


def test_without_a_reported_branch_any_branch_of_the_repository_is_its_own(merging):
    merging.state.view = _view(branch=None)
    merging.use(_GitHub(pull={**OPEN_PULL, "head": {**OPEN_PULL["head"], "ref": "another-branch"}}))
    assert _merge() is None and merging.state.github.actions()[-1] == tpm.MERGE_PULL_REQUEST


# ── GitHub must say it can merge; unknown is a no ────────────────────────────

@pytest.mark.parametrize("pull, reason", [
    ({**OPEN_PULL, "state": "closed"}, tpm.NOT_OPEN.format(state="closed")),
    ({**OPEN_PULL, "draft": True}, tpm.IS_DRAFT),
    ({**OPEN_PULL, "mergeable": None, "mergeable_state": "unknown"}, tpm.NOT_WORKED_OUT),
    ({k: v for k, v in OPEN_PULL.items() if k != "mergeable"}, tpm.NOT_WORKED_OUT),
    ({**OPEN_PULL, "mergeable": False, "mergeable_state": "dirty"}, tpm.HAS_CONFLICTS),
    ({**OPEN_PULL, "mergeable_state": "blocked"}, tpm.NOT_CLEAN.format(state="blocked")),
    ({**OPEN_PULL, "mergeable_state": "unstable"}, tpm.NOT_CLEAN.format(state="unstable")),
    ({**OPEN_PULL, "mergeable_state": "behind"}, tpm.NOT_CLEAN.format(state="behind")),
    ({**OPEN_PULL, "head": {"ref": BRANCH, "repo": {"full_name": REPO}}}, tpm.UNREADABLE),
    ({"message": "Not Found"}, tpm.NOT_ITS_OWN),
], ids=["closed", "draft", "not-worked-out", "mergeable-missing", "conflicts", "blocked", "unstable", "behind",
        "no-head-sha", "not-a-pull-request"])
def test_a_pull_request_github_will_not_merge_closes_the_ticket_with_the_reason(merging, pull, reason):
    github = merging.use(_GitHub(pull=pull))
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=reason)
    assert tpm.MERGE_PULL_REQUEST not in github.actions()


@pytest.mark.parametrize("runs, reason", [
    ({"total_count": 2, "check_runs": [GREEN_RUNS["check_runs"][0],
                                       {"name": "frontend", "status": "completed", "conclusion": "failure"}]},
     tpm.CHECKS_FAILED.format(names="frontend")),
    ({"total_count": 1, "check_runs": [{"name": "backend", "status": "in_progress", "conclusion": None}]},
     tpm.CHECKS_RUNNING.format(names="backend")),
    ({"total_count": 1, "check_runs": [{"name": "backend", "status": "completed", "conclusion": "cancelled"}]},
     tpm.CHECKS_FAILED.format(names="backend")),
    ({"total_count": 0, "check_runs": []}, tpm.NO_CHECKS),
    ({"total_count": 140, "check_runs": GREEN_RUNS["check_runs"]}, tpm.TOO_MANY_CHECKS),
    ({"check_runs": GREEN_RUNS["check_runs"]}, tpm.UNREADABLE),
    ({"total_count": 1}, tpm.UNREADABLE),
], ids=["red", "running", "cancelled", "none", "more-than-a-page", "no-count", "no-runs"])
def test_checks_that_are_not_all_green_or_unknown_never_merge(merging, runs, reason):
    github = merging.use(_GitHub(runs=runs))
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=reason)
    assert github.actions() == [tpm.GET_PULL_REQUEST, tpm.LIST_CHECK_RUNS]


def test_a_pull_request_already_merged_says_so_and_is_not_merged_again(merging):
    github = merging.use(_GitHub(pull={**OPEN_PULL, "state": "closed", "merged": True}))
    assert _merge() == tpm.ALREADY_MERGED_NOTE.format(url=URL) and github.actions() == [tpm.GET_PULL_REQUEST]


def test_github_that_cannot_be_read_closes_it_with_why(merging):
    merging.use(_GitHub(pull={"success": False, "error": "401 Bad credentials", "data": None}))
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=tpm.NO_ANSWER.format(error="401 Bad credentials"))


def test_the_deny_list_still_stops_the_merge_and_the_note_says_so(merging):
    denial = BLOCKED_PREFIX + DENIED_REASON.format(slug=tpm.MERGE_PULL_REQUEST)
    merging.use(_GitHub(merged=denied_result(denial)))
    note = _merge()
    assert note == tpm.NOT_MERGED_NOTE.format(reason=tpm.MERGE_REFUSED.format(error=denial))
    assert note.startswith("Approved. The pull request was not merged: ") and note.endswith("Merge it on GitHub.")
    assert merging.state.kept == []


def test_a_merge_github_refuses_is_said_once_and_not_retried(merging):
    github = merging.use(_GitHub(merged={"merged": False, "message": "Head branch was modified. Review and try again."}))
    assert _merge() == tpm.NOT_MERGED_NOTE.format(
        reason=tpm.MERGE_REFUSED.format(error="Head branch was modified. Review and try again."))
    assert github.actions().count(tpm.MERGE_PULL_REQUEST) == 1


@pytest.mark.parametrize("answer", [{"sha": HEAD, "message": "Pull Request successfully merged"}, "not json"])
def test_a_merge_answer_that_does_not_say_whether_it_merged_is_never_called_a_refusal(merging, answer):
    merging.use(_GitHub(merged=answer))
    assert _merge() == tpm.MERGE_UNKNOWN_NOTE.format(url=URL)


def test_github_not_connected_merges_nothing(merging, monkeypatch):
    monkeypatch.setattr(tpm, "github_connected", lambda db, ws: False)
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=tpm.NOT_CONNECTED) and merging.state.github.calls == []


@pytest.mark.parametrize("url", [
    "https://gitlab.com/automatos/platform/-/merge_requests/42",   # GitLab's shape
    "https://git.example.org/automatos-ai/automatos-ai/pull/7",    # GitHub's shape on another host
])
def test_a_pull_request_on_another_host_is_never_merged_through_the_github_action(merging, url):
    address = tpr.parse_pull_request_url(url)
    merging.state.view = _view(url=url)
    assert _merge() == tpm.NOT_MERGED_ELSEWHERE_NOTE.format(
        reason=tpm.NOT_ON_GITHUB.format(host=address.host), host=address.host)
    assert merging.state.github.calls == []


def test_a_failure_inside_the_merge_never_reaches_the_approval(merging):
    class _Broken(_GitHub):
        async def execute(self, **kwargs):
            raise RuntimeError("the SDK fell over")

    merging.use(_Broken())
    assert _merge() == tpm.NOT_MERGED_NOTE.format(reason=tpm.COULD_NOT_TRY)


# ── reading GitHub's answers (pure) ──────────────────────────────────────────

@pytest.mark.parametrize("result, expected", [
    ({"success": True, "data": {"data": {"state": "open"}, "successful": True, "error": None}}, {"state": "open"}),
    ({"success": True, "data": {"state": "open"}}, {"state": "open"}),
    ({"success": True, "data": '{"state": "open"}'}, {"state": "open"}),
    ({"success": True, "data": {"data": {"details": {"state": "open"}}, "successful": True}}, {"state": "open"}),
    ({"success": True, "data": "not json"}, None),
    ({"success": True, "data": {"data": ["a", "list"], "successful": True}}, None),
])
def test_githubs_own_answer_is_read_out_of_composios_envelope(result, expected):
    assert tpm.payload_of(result) == expected


def test_many_failing_checks_are_named_briefly():
    runs = [{"name": f"job-{n}", "status": "completed", "conclusion": "failure"} for n in range(5)]
    assert tpm.checks_blocker({"total_count": 5, "check_runs": runs}) == tpm.CHECKS_FAILED.format(
        names="job-0, job-1, job-2 and 2 more")


# ── on the real schema: the board's Approve, and the paths that never merge ──

def _report_a_pull_request(db, board, brief=QUESTION):
    """A card whose session reported its pull request while it ran, now in Review."""
    task = board.card(brief, status="in_progress")
    tpr.write_pull_request(db, task_id=task.id, workspace_id=board.ws, by="CODER", note="",
                           fields=tpr.pull_request_fields({"pull_request_url": URL, "branch": BRANCH}))
    db.refresh(task)
    task.status = "review"
    db.flush()
    return task


def _shown(db, task):
    """What the web sends back with its Approve: the pull request it showed."""
    db.expire_all()
    db.refresh(task)
    served = tpr.pull_request_of(task.runtime_ref)
    return {"url": served["url"], "at": served["at"]}


@pytest.fixture
def github_here(monkeypatch):
    """The setting on, the repository listed, GitHub connected, a scripted GitHub, the approver an owner
    (``role``: the real permission check, its role read from here); the ticket's own row is read for real,
    and the merge's own transaction is the test's session (the board fixture's)."""
    from core.auth import workspace_permission

    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", True)
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", REPO)
    monkeypatch.setattr(tpm, "github_connected", lambda db, ws: True)
    state = NS(github=_GitHub(), role="owner")
    monkeypatch.setattr(tpm, "_executor", lambda db: state.github)
    monkeypatch.setattr(workspace_permission, "resolve_workspace_role", lambda db, ctx: state.role)
    return state


def test_a_persons_approve_on_the_board_merges_the_pull_request_and_says_so(db_session, board, quiet, github_here):
    from api.board_tasks import approve_task

    task = _report_a_pull_request(db_session, board)
    out = _run(approve_task(task.id, _body({"pull_request": _shown(db_session, task)}), ctx=_person(board.ws),
                            db=db_session))

    assert out["success"] is True and out["status"] == "done"
    assert github_here.github.actions()[-1] == tpm.MERGE_PULL_REQUEST
    assert _notes(db_session, task) == [("CODER", f"Pull request: {URL}"), ("you", f"Merged {URL}")]
    assert task.runtime_ref["pull_request"]["merged"]["sha"] == MERGE_SHA


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_an_owner_or_admins_approve_merges_it(db_session, board, quiet, github_here, role):
    from api.board_tasks import approve_task

    github_here.role = role
    task = _report_a_pull_request(db_session, board)
    out = _run(approve_task(task.id, _body({}), ctx=_person(board.ws), db=db_session))
    assert out["status"] == "done" and github_here.github.actions()[-1] == tpm.MERGE_PULL_REQUEST
    assert _notes(db_session, task)[-1] == ("you", f"Merged {URL}")


def test_an_editors_approve_closes_the_ticket_and_merges_nothing(db_session, board, quiet, github_here):
    """An editor holds ``missions:update``, so Approve works; merging is an owner's or admin's."""
    from api.board_tasks import approve_task

    github_here.role = "editor"
    task = _report_a_pull_request(db_session, board)
    out = _run(approve_task(task.id, _body({"pull_request": _shown(db_session, task)}), ctx=_person(board.ws),
                            db=db_session))
    db_session.refresh(task)
    assert out["success"] is True and out["status"] == "done" and task.status == "done"
    assert github_here.github.calls == []
    assert _notes(db_session, task)[-1] == (
        "you", "Approved. The pull request was not merged: merging needs a workspace owner or admin. Merge it on GitHub.")


def test_an_approve_that_shows_another_pull_request_closes_the_ticket_and_merges_nothing(db_session, board, quiet,
                                                                                        github_here):
    from api.board_tasks import approve_task

    task = _report_a_pull_request(db_session, board)
    stale = {**_shown(db_session, task), "at": "2026-11-18T08:00:00+00:00"}
    out = _run(approve_task(task.id, _body({"pull_request": stale}), ctx=_person(board.ws), db=db_session))
    assert out["status"] == "done" and github_here.github.calls == []
    assert _notes(db_session, task)[-1] == (
        "you", "Approved. The pull request was not merged: it changed since you looked. Merge it on GitHub.")


def test_a_pull_request_taken_off_after_the_owner_looked_merges_nothing_and_says_it_changed(db_session, board,
                                                                                          quiet, github_here):
    from api.board_tasks import approve_task

    task = _report_a_pull_request(db_session, board)
    shown = _shown(db_session, task)
    assert tpr.clear_pull_request(db_session, task_id=task.id, workspace_id=board.ws) == URL
    out = _run(approve_task(task.id, _body({"pull_request": shown}), ctx=_person(board.ws), db=db_session))
    assert out["status"] == "done" and github_here.github.calls == []
    assert _notes(db_session, task)[-1] == ("you", tpm.not_merged(tpm.CHANGED_SINCE))


def test_a_persons_approve_of_a_red_pull_request_still_closes_the_ticket_and_says_why(db_session, board, quiet,
                                                                                     github_here):
    from api.board_tasks import approve_task

    github_here.github = _GitHub(runs={"total_count": 1, "check_runs": [
        {"name": "backend", "status": "completed", "conclusion": "failure"}]})
    task = _report_a_pull_request(db_session, board)
    out = _run(approve_task(task.id, _body({"note": "Looks right."}), ctx=_person(board.ws), db=db_session))

    assert out["status"] == "done" and tpm.MERGE_PULL_REQUEST not in github_here.github.actions()
    assert _notes(db_session, task)[1:] == [
        ("you", "Approved. The pull request was not merged: its checks did not pass (backend). Merge it on GitHub."),
        ("you", "Approved: Looks right.")]


@pytest.fixture
def tripwire(monkeypatch):
    """The setting on; anything that reaches the merge, or any Composio call at all, is recorded."""
    from core.composio.tool_executor import ComposioToolExecutor

    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", True)
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", REPO)
    seen = []

    async def _reached_the_merge(*args, **kwargs):
        seen.append(("merge", args))

    async def _composio(self, *args, **kwargs):
        seen.append(("composio", kwargs.get("action")))
        return {"success": False, "error": "no Composio in this test"}

    monkeypatch.setattr(tpm, "_merge", _reached_the_merge)
    monkeypatch.setattr(ComposioToolExecutor, "execute", _composio)
    return seen


def test_an_approve_with_no_persons_lane_never_merges(db_session, board, quiet, tripwire):
    from api.board_tasks import approve_task

    task = _report_a_pull_request(db_session, board)
    out = _run(approve_task(task.id, _body({}), ctx=_not_a_person(board.ws), db=db_session))
    assert out["status"] == "done" and tripwire == []
    assert _notes(db_session, task) == [("CODER", f"Pull request: {URL}")]


def test_autos_status_tool_moving_it_to_done_never_merges(db_session, board, quiet, tripwire):
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    task = _report_a_pull_request(db_session, board)
    out = _run(update_board_task_status(db_session, board.ws, {"task_id": task.id, "status": "done", "_user_id": "2"}))
    db_session.refresh(task)
    assert out["success"] is True and task.status == "done" and tripwire == []


def test_an_agents_update_ticket_records_the_pull_request_and_never_merges(db_session, board, quiet, tripwire):
    from services import session_tools as st

    task = board.card(QUESTION, status="in_progress")
    ctx = st.SessionContext(task_id=task.id, agent_id=board.agent.id, agent_name="CODER", workspace_id=board.ws)
    tool = st.get_tool("update_ticket")
    params = st.resolve_parameters(tool, {"note": "Opened it; merge it.", "pull_request_url": URL}, ctx)
    out = _run(st.call_tool(db_session, tool, params, ctx))
    assert out["success"] is True and tripwire == []
    db_session.expire_all()
    assert (task.runtime_ref or {})["pull_request"]["url"] == URL


# ── nothing else reaches the merge module ────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
MODULE = "services.ticket_pr_merge"
ALLOWED = {"api/board_card_moves.py", "services/ticket_pr_merge.py"}
# Its public names. ``Readiness`` is left out: another module has a class of that name.
PUBLIC = frozenset(tpm.__all__) - {"Readiness"}


@functools.lru_cache(maxsize=1)
def _sources():
    """Every orchestrator source file but the tests, as (path, text)."""
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if {"tests", "node_modules", "site-packages"} & set(parts) or any(p.startswith(".") for p in parts):
            continue
        found.append((path.relative_to(ROOT).as_posix(), path.read_text("utf-8")))
    return tuple(found)


def _may_reach_it(source: str) -> bool:
    """Only a file whose text names the module or one of its names can reach it; only those are parsed
    (a parse of every file overran CI's per-test timeout: tests/test_prd251_composio_deny.py)."""
    return "ticket_pr_merge" in source or any(name in source for name in PUBLIC)


def _reaches_the_merge_module(tree: ast.AST) -> list:
    """Each import of the module (any spelling, dynamic or as a string), each of its names taken from it
    or from the one module allowed to use it, and each use of one of its names."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == MODULE or (node.module == "services" and any(a.name == "ticket_pr_merge" for a in node.names)):
                found.append(("import", node.lineno))
            elif node.module == "api.board_card_moves" and {a.name for a in node.names} & PUBLIC:
                found.append(("re-export", node.lineno))
        elif isinstance(node, ast.Import) and any(a.name == MODULE or a.name.startswith(MODULE + ".") for a in node.names):
            found.append(("import", node.lineno))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(MODULE):
            found.append(("by name", node.lineno))
        elif isinstance(node, ast.Name) and node.id in PUBLIC or isinstance(node, ast.Attribute) and node.attr in PUBLIC:
            found.append(("name", node.lineno))
    return found


def test_nothing_outside_the_boards_approve_imports_the_merge_module_or_its_names():
    """No agent path, no tool, no Playbook and no other route reaches ``merge_on_approve``, the merge
    action, or anything else of ``services.ticket_pr_merge``: only ``api/board_card_moves.py``."""
    offenders = {rel: hits for rel, source in _sources() if rel not in ALLOWED and _may_reach_it(source)
                 for hits in [_reaches_the_merge_module(ast.parse(source))] if hits}
    assert offenders == {}
    naming = sorted(rel for rel, source in _sources() if tpm.MERGE_PULL_REQUEST in source)
    assert naming == ["services/ticket_pr_merge.py"]


@pytest.mark.parametrize("source", [
    "from services.ticket_pr_merge import merge_on_approve",
    "from services import ticket_pr_merge",
    "import services.ticket_pr_merge as m",
    "importlib.import_module('services.ticket_pr_merge')",
    "from api.board_card_moves import merge_on_approve",
    "async def f(m):\n    return await m.merge_on_approve(None, None, 1, decided_at=None)",
    "x = MERGE_PULL_REQUEST",
])
def test_the_scan_catches_every_way_in(source):
    assert _reaches_the_merge_module(ast.parse(source))
