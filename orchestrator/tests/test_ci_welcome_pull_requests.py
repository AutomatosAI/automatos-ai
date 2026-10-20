"""Gerard, 15 Oct ("Can we automate the welcome message???"): the welcome bot's other half.

A first pull request and a first merge, the reply templates (``.github/welcome/``),
failures, the GitHub client and the workflow (``.github/workflows/welcome.yml``). The
claim tests are in ``test_ci_welcome_contributor.py``; the bot and the fake GitHub
both use are in ``helpers_welcome_bot.py`` (no network, no database). The ``.github``
paths named here also let ``scripts/ci/backend_test_plan.py`` select this file when
only a template or the workflow changes.
"""
import io
import json
import logging
import urllib.error

import pytest

from tests.helpers_welcome_bot import (
    LEFTOVER,
    MERGED_AT,
    REPO,
    TEMPLATES,
    WORKFLOW,
    FakeGitHub,
    bot,
    bot_comment,
    comment_event,
    dispatch,
    merged_pr,
    only_post,
    pr_event,
    pull_request,
    rules,
)


# --- Pull requests ------------------------------------------------------------------


@pytest.mark.parametrize("association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"])
def test_a_first_time_contributors_first_pull_request_gets_the_first_pr_welcome(association):
    fake = FakeGitHub()

    assert pull_request(pr_event(association=association), fake) == rules.PR_FIRST
    assert fake.posted[0][0] == 77
    assert rules.marker_for(rules.PR_FIRST, "newbie", "opened") in only_post(fake)
    assert fake.filters == [{"creator": "newbie", "state": "all"}]


def test_a_second_pull_request_from_a_first_time_contributor_gets_no_second_welcome():
    earlier = {"number": 70, "pull_request": {"url": "x", "merged_at": None}}
    fake = FakeGitHub(authored=[earlier])

    assert pull_request(pr_event(), fake) is None
    assert fake.posted == []


def test_issues_and_later_pull_requests_by_the_author_do_not_stop_the_first_pr_welcome():
    authored = [{"number": 60}, {"number": 80, "pull_request": {"url": "x"}}, {"number": 77, "pull_request": {"url": "x"}}]
    fake = FakeGitHub(authored=authored)

    assert pull_request(pr_event(), fake) == rules.PR_FIRST


def test_the_first_pr_welcome_states_the_test_rule_and_points_at_agents_md():
    fake = FakeGitHub()

    pull_request(pr_event(), fake)

    body = only_post(fake)
    assert "test that fails without it" in body
    assert f"https://github.com/{REPO}/blob/main/AGENTS.md" in body


@pytest.mark.parametrize("association", ["CONTRIBUTOR", "OWNER", "MEMBER", "COLLABORATOR"])
def test_a_pull_request_from_an_earlier_contributor_gets_no_welcome(association):
    fake = FakeGitHub()

    assert pull_request(pr_event(association=association), fake) is None
    assert fake.posted == [] and fake.filters == []


def test_a_bots_pull_request_gets_no_welcome():
    fake = FakeGitHub()

    assert pull_request(pr_event(login="dependabot[bot]", user_type="Bot"), fake) is None
    assert fake.posted == []


@pytest.mark.parametrize("indexed", [[], [merged_pr(77, MERGED_AT)]], ids=["index-lagging", "index-current"])
def test_a_first_merged_pull_request_gets_the_congratulations(indexed):
    fake = FakeGitHub(authored=indexed, merged_items=indexed)

    event = pr_event("closed", association="CONTRIBUTOR", merged=True)
    assert pull_request(event, fake) == rules.MERGE_FIRST
    body = only_post(fake)
    assert rules.marker_for(rules.MERGE_FIRST, "newbie", "merged") in body
    # A squash merge here is often authored by the maintainer's account, so the reply can't promise the list.
    assert "graphs/contributors" not in body


def test_a_second_merge_gets_no_congratulations_even_while_search_lags():
    fake = FakeGitHub(authored=[merged_pr(70, "2026-10-14T09:00:00Z")], merged_items=[])

    assert pull_request(pr_event("closed", association="CONTRIBUTOR", merged=True), fake) is None
    assert fake.posted == []


def test_an_older_merge_only_search_still_knows_of_counts():
    fake = FakeGitHub(authored=[], merged_items=[merged_pr(77, MERGED_AT), merged_pr(12, "2026-03-01T09:00:00Z")])

    assert pull_request(pr_event("closed", association="CONTRIBUTOR", merged=True), fake) is None


def test_of_two_merges_back_to_back_only_the_earlier_is_the_first():
    both = [merged_pr(77, MERGED_AT), merged_pr(78, "2026-10-15T20:00:40Z")]

    first = FakeGitHub(authored=both)
    assert pull_request(pr_event("closed", association="CONTRIBUTOR", merged=True, number=77), first) == rules.MERGE_FIRST
    second = FakeGitHub(authored=both)
    event = pr_event("closed", association="CONTRIBUTOR", merged=True, number=78, merged_at="2026-10-15T20:00:40Z")
    assert pull_request(event, second) is None


def test_a_pull_request_closed_without_merging_gets_nothing_and_no_lookup():
    fake = FakeGitHub()

    assert pull_request(pr_event("closed", merged=False), fake) is None
    assert fake.posted == [] and fake.queries == [] and fake.filters == []


def test_a_maintainers_merged_pull_request_gets_no_congratulations():
    fake = FakeGitHub()

    assert pull_request(pr_event("closed", association="OWNER", merged=True), fake) is None
    assert fake.posted == [] and fake.queries == [] and fake.filters == []


def test_a_merge_event_without_merged_at_fails_the_run():
    event = pr_event("closed", association="CONTRIBUTOR", merged=True)
    event["pull_request"].pop("merged_at")

    assert bot.run("pull_request_target", event, FakeGitHub(), REPO) == 1


# --- Templates ----------------------------------------------------------------------

SCENARIOS = {
    rules.CLAIM_FIRST_TIMER: lambda: ("issue_comment", comment_event(), FakeGitHub()),
    rules.CLAIM_RETURNING: lambda: ("issue_comment", comment_event(), FakeGitHub(merged=5)),
    rules.CLAIM_AGAIN: lambda: ("issue_comment", comment_event(), FakeGitHub(comments=[bot_comment(rules.CLAIM_RETURNING)])),
    rules.CLAIM_NEEDS_TRIAGE: lambda: ("issue_comment", comment_event(labels=["needs-triage"]), FakeGitHub()),
    rules.CLAIM_NEEDS_MAINTAINER: lambda: ("issue_comment", comment_event(labels=["bug"]), FakeGitHub()),
    rules.CLAIM_NOT_READY: lambda: ("issue_comment", comment_event(labels=["question"]), FakeGitHub()),
    rules.CLAIM_TAKEN: lambda: ("issue_comment", comment_event(assignees=["x"]), FakeGitHub(free=[(101, "Free")])),
    rules.CLAIM_TAKEN_RETURNING: lambda: ("issue_comment", comment_event(assignees=["x"]), FakeGitHub(merged=5)),
    rules.CLAIM_ONE_AT_A_TIME: lambda: ("issue_comment", comment_event(), FakeGitHub(held=[7])),
    rules.PR_FIRST: lambda: ("pull_request_target", pr_event(), FakeGitHub()),
    rules.MERGE_FIRST: lambda: ("pull_request_target", pr_event("closed", merged=True), FakeGitHub()),
}


def test_every_template_the_bot_uses_exists_and_names_only_known_placeholders():
    for name in bot.TEMPLATES:
        assert (TEMPLATES / f"{name}.md").is_file(), name
    bot.check_templates()


def test_every_reply_kind_has_a_scenario_here():
    assert set(SCENARIOS) == set(rules.REPLY_KINDS)


@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_every_reply_is_posted_with_no_placeholder_left_and_says_it_is_automated(kind):
    event_name, event, fake = SCENARIOS[kind]()

    assert dispatch(event_name, event, fake) == kind
    body = only_post(fake)
    assert not LEFTOVER.search(body), LEFTOVER.findall(body)
    assert "automated" in body.lower()
    assert rules.marker_prefix(kind, "newbie") in body


def test_a_template_with_an_unknown_placeholder_fails_before_anything_is_posted(tmp_path):
    for name in bot.TEMPLATES:
        (tmp_path / f"{name}.md").write_text("Hi @{login}", encoding="utf-8")
    (tmp_path / f"{rules.CLAIM_TAKEN}.md").write_text("Hi @{logn}", encoding="utf-8")

    with pytest.raises(ValueError, match="logn"):
        bot.check_templates(tmp_path)


def test_render_refuses_a_placeholder_it_has_no_value_for():
    with pytest.raises(ValueError, match="login"):
        bot.render(rules.CLAIM_FIRST_TIMER, {"repo_url": "https://github.com/x/y"})


# --- Failures surface ---------------------------------------------------------------


class _FailingGitHub(FakeGitHub):
    def post_comment(self, number, body):
        raise bot.github.GitHubError("POST /repos/x/y/issues/42/comments: HTTP 403")


def test_a_github_failure_is_logged_and_the_run_exits_non_zero(caplog):
    event = comment_event()
    with caplog.at_level(logging.ERROR, logger="welcome_contributor"):
        assert bot.run("issue_comment", event, _FailingGitHub(issue=event["issue"]), REPO) == 1
    assert "stopped" in caplog.text


def test_a_malformed_login_in_the_event_fails_the_run():
    event = comment_event(login="not a login")
    assert bot.run("issue_comment", event, FakeGitHub(issue=event["issue"]), REPO) == 1


@pytest.mark.parametrize(
    "event_name, event",
    [
        ("issue_comment", {"action": "created", "issue": {"number": 42, "state": "open"}}),
        ("issue_comment", {"action": "created", "issue": ["junk"], "comment": {"body": "I'll take this"}}),
        ("pull_request_target", {"action": "opened", "pull_request": "junk"}),
    ],
    ids=["no-comment", "issue-not-an-object", "pull-request-not-an-object"],
)
def test_a_payload_missing_an_object_its_handler_reads_fails_the_run_cleanly(event_name, event, caplog):
    fake = FakeGitHub()

    with caplog.at_level(logging.ERROR, logger="welcome_contributor"):
        assert bot.run(event_name, event, fake, REPO) == 1
    assert "has no" in caplog.text
    assert fake.posted == [] and fake.reads == []


def test_main_refuses_to_run_without_a_token(monkeypatch, tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps(comment_event()), encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert bot.main(["--event-path", str(event), "--event-name", "issue_comment", "--repo", REPO]) == 2


@pytest.mark.parametrize(
    "level, prefix", [(logging.WARNING, "::warning::"), (logging.ERROR, "::error::"), (logging.INFO, "")]
)
def test_warnings_and_errors_are_logged_as_actions_annotations(level, prefix):
    record = logging.LogRecord("welcome_contributor", level, __file__, 1, "GitHub did not assign @x", None, None)

    line_out = bot.ActionsFormatter("%(message)s").format(record)

    assert line_out == f"{prefix}GitHub did not assign @x"


# --- The REST client ----------------------------------------------------------------


def _opener(answers, requests):
    def open_(request, timeout):
        requests.append(request)
        answer = answers[len(requests) - 1]
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode("utf-8"))

    return open_


def _client(answers, requests=None):
    return bot.github.GitHubClient("test-token", REPO, opener=_opener(answers, [] if requests is None else requests))


def test_the_client_pages_through_comments_with_the_api_headers():
    requests = []
    pages = [[{"id": n} for n in range(bot.github.PER_PAGE)], [{"id": bot.github.PER_PAGE}]]
    client = _client(pages, requests)

    assert len(client.list_issue_comments(42)) == bot.github.PER_PAGE + 1
    assert [r.full_url.rsplit("page=", 1)[1] for r in requests] == ["1", "2"]
    first = requests[0]
    assert first.get_header("Accept") == "application/vnd.github+json"
    assert first.get_header("X-github-api-version") == "2022-11-28"
    assert first.get_header("Authorization") == "Bearer test-token"


def test_the_client_reads_the_issue_and_lists_issues_by_rest_filters():
    requests = []
    client = _client([{"number": 42, "state": "open"}, [{"number": 7}, "junk"]], requests)

    assert client.get_issue(42) == {"number": 42, "state": "open"}
    assert client.list_issues({"assignee": "newbie", "state": "open"}) == [{"number": 7}]
    assert requests[0].full_url == f"https://api.github.com/repos/{REPO}/issues/42"
    listed = requests[1].full_url
    assert listed.startswith(f"https://api.github.com/repos/{REPO}/issues?")
    assert "assignee=newbie" in listed and "state=open" in listed and "per_page=100" in listed


def test_the_client_refuses_an_issue_answer_that_is_not_an_issue():
    with pytest.raises(bot.github.GitHubError, match="not an issue"):
        _client([[]]).get_issue(42)


def test_the_client_reads_the_assignees_back_from_githubs_answer():
    requests = []
    client = _client([{"assignees": [{"login": "other"}]}], requests)

    assert client.add_assignee(42, "newbie") == ["other"]
    assert requests[0].get_method() == "POST"
    assert json.loads(requests[0].data) == {"assignees": ["newbie"]}


def test_the_client_reads_pinned_issues_from_graphql():
    answer = {"data": {"repository": {"pinnedIssues": {"nodes": [{"issue": {"number": 1105}}]}}}}

    assert _client([answer]).pinned_issue_numbers() == frozenset({1105})


def test_a_graphql_answer_with_errors_raises():
    with pytest.raises(bot.github.GitHubError, match="pinnedIssues"):
        _client([{"errors": [{"message": "nope"}]}]).pinned_issue_numbers()


def test_an_http_error_names_the_call_and_raises(caplog):
    url = f"https://api.github.com/repos/{REPO}/issues/42/comments"
    error = urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b'{"message": "Resource not accessible"}'))

    with caplog.at_level(logging.ERROR, logger="welcome_contributor"):
        with pytest.raises(bot.github.GitHubError, match=r"POST /repos/.+/issues/42/comments: HTTP 403"):
            _client([error]).post_comment(42, "hi")
    assert "Resource not accessible" in caplog.text


# --- The workflow -------------------------------------------------------------------


def test_the_workflow_never_checks_out_or_interpolates_pull_request_content():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions: {}" in text
    assert "ref: ${{ github.event.repository.default_branch }}" in text
    assert "persist-credentials: false" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    for forbidden in ("head.sha", "head.ref", "head_ref", "comment.body", "pull_request.title", "issue.title"):
        assert forbidden not in text, forbidden
    run_lines = [row for row in text.splitlines() if row.strip().startswith("run:")]
    assert run_lines and all("${{" not in row for row in run_lines)


def test_the_workflow_runs_claims_one_at_a_time_and_queues_instead_of_dropping():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "'welcome-claims'" in text
    assert "queue: max" in text
    assert "cancel-in-progress: false" in text
