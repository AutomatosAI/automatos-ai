"""Gerard, 15 Oct ("Can we automate the welcome message???"): the welcome bot's claims.

A comment on an issue: is it a claim, which reply it gets (from the templates in
``.github/welcome/``), whether the bot assigns, and that a reply is never posted
twice. The pull request, merge, template, client and workflow tests are in
``test_ci_welcome_pull_requests.py``; the bot and the fake GitHub both use are in
``helpers_welcome_bot.py`` (no network, no database). The ``.github`` paths named
here also let ``scripts/ci/backend_test_plan.py`` select this file when only a
template or the workflow changes.
"""
import logging
import re

import pytest

from tests.helpers_welcome_bot import (
    COMMENT_ID,
    FakeGitHub,
    bot,
    bot_comment,
    claim,
    comment_event,
    line,
    merged_pr,
    only_post,
    pending_comment,
    person_comment,
    rules,
)


# --- Claim detection ----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Hi! I would like to work on this issue. I plan on converting the class Config blocks.",
        "I'd like to take this docs-only dead-link cleanup.",
        "Hi, I'd like to work on this issue. Could you assign it to me?",
        "Hi! I'd be interested in getting this solved. If you want to assign it to me i'll get started on it ASAP.",
        "I want to work on this.",
        "Can I take this one?",
        "could i please work on this?",
        "May I pick this up?",
        "I'll take this!",
        "I’ll pick it up.",
        "Please assign me.",
        "Assign this to me please",
        "I'm interested in working on this issue.",
        "I'd love to tackle this.",
        "I'd like to fix this. My plan: guard the None case and add a test.",
        "Let me take this one",
        "Can I be assigned to this?",
        "I'd like to **work on** this",
        "I am happy to take this on.",
    ],
)
def test_a_first_person_request_to_take_the_issue_is_a_claim(body):
    assert rules.is_claim(body)


@pytest.mark.parametrize(
    "body",
    [
        "Is anyone working on this?",
        "I think this is a duplicate of #812.",
        "I'd like to understand why this fails on Windows.",
        "This would be nice to have.",
        "Could you take a look at the logs?",
        "Can someone assign this to @alice?",
        "Can I take it that the API is stable?",
        "Can I take this approach instead of a new table?",
        "Can I fix this by upgrading Pydantic?",
        "Please don't assign me, I'm only reporting it.",
        "Please assign me to #1234 instead.",
        "I'll take a look tonight.",
        "Who wants to take this?",
        "Thanks! I can reproduce this on main.",
        "I'd like to see this fixed soon.",
        "I'd like to work on this project more generally.",
    ],
)
def test_questions_and_remarks_are_not_claims(body):
    assert not rules.is_claim(body)


@pytest.mark.parametrize(
    "body",
    [
        "> I'd like to work on this\n\nIs that still the plan?",
        "```\nI'd like to work on this\n```\nSee the log above.",
        "~~~text\nCan I take this?\n~~~",
        "The bot matches `I'd like to work on this`, right?",
        "<!-- I'd like to work on this -->\nJust a +1 from me.",
    ],
)
def test_a_claim_only_in_quoted_text_code_or_html_comments_is_not_a_claim(body):
    assert not rules.is_claim(body)


def test_a_claim_after_a_quoted_line_still_counts():
    assert rules.is_claim("> Is anyone working on this?\n\nNot yet, I'd like to take it.")


def test_an_unclosed_html_comment_hides_the_rest_of_the_comment():
    assert not rules.is_claim("Just a +1 from me.\n<!-- I'd like to work on this")
    assert rules.is_claim("I'd like to work on this. <!-- unfinished note")


def test_only_the_start_of_a_very_long_comment_is_read():
    padding = "x " * rules.MAX_CLAIM_CHARS

    assert not rules.is_claim(padding + "I'd like to work on this.")
    assert not rules.is_claim("<!--" * (rules.MAX_CLAIM_CHARS // 2) + " I'd like to work on this.")
    assert rules.is_claim("I'd like to work on this. " + padding)


@pytest.mark.parametrize(
    "body",
    [
        "I’m stepping back from this one. The required filters touch two legacy paths.",
        "Sorry, please unassign me, I've run out of time.",
        "I'm withdrawing my claim.",
        "I'm no longer able to work on this.",
    ],
)
def test_a_claimer_giving_the_issue_up_is_a_release(body):
    assert rules.is_release(body)


@pytest.mark.parametrize(
    "body",
    [
        "Let me step back and think about the approach.",
        "Please don't unassign me, the PR is nearly ready.",
        "> I'm stepping back from this one\n\nThanks for letting us know!",
        "I was stepping back from it, but I'd like to take this on again.",
    ],
)
def test_remarks_and_new_claims_are_not_releases(body):
    assert not rules.is_release(body)


# --- Claims the bot leaves alone ----------------------------------------------------


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_a_maintainer_comment_gets_no_reply(association):
    fake = FakeGitHub()

    assert claim(comment_event(association=association), fake) is None
    assert fake.posted == [] and fake.assigned == [] and fake.reads == []


@pytest.mark.parametrize("user_type, login", [("Bot", "coderabbitai"), ("User", "renovate[bot]")])
def test_a_bot_comment_gets_no_reply(user_type, login):
    fake = FakeGitHub()

    assert claim(comment_event(login=login, user_type=user_type), fake) is None
    assert fake.posted == []


def test_a_claim_on_a_closed_issue_gets_no_reply():
    fake = FakeGitHub()

    assert claim(comment_event(state="closed"), fake) is None
    assert fake.posted == [] and fake.queries == [] and fake.reads == []


def test_a_claim_like_comment_on_a_pull_request_gets_no_reply():
    fake = FakeGitHub()

    assert claim(comment_event(on_pr=True), fake) is None
    assert fake.posted == []


def test_a_comment_that_is_not_a_claim_gets_no_reply():
    fake = FakeGitHub()

    assert claim(comment_event("Is anyone working on this?"), fake) is None
    assert fake.posted == [] and fake.queries == [] and fake.reads == []


def test_an_issue_closed_since_the_comment_gets_no_reply():
    event = comment_event()
    fake = FakeGitHub(issue={**event["issue"], "state": "closed"})

    assert claim(event, fake) is None
    assert fake.posted == [] and fake.assigned == []


def test_a_claim_on_the_pinned_start_here_issue_gets_no_reply_and_no_assignment():
    fake = FakeGitHub(pinned={1105})

    event = comment_event("Thanks! I'm interested in working on this.", number=1105, labels=["good first issue", "help wanted"])
    assert claim(event, fake) is None
    assert fake.posted == [] and fake.assigned == []


@pytest.mark.parametrize(
    "title, labels",
    [
        ("Tracking: `async def` route handlers that only use the sync DB session should be plain `def`", ["good first issue"]),
        ("Fix the dead link in the README", ["good first issue", "no-auto-assign"]),
    ],
)
def test_a_shared_or_opted_out_issue_is_left_to_the_maintainer(title, labels):
    fake = FakeGitHub()

    assert claim(comment_event("I'll take this one: orchestrator/api/missions.py", title=title, labels=labels), fake) is None
    assert fake.posted == [] and fake.assigned == []


# --- Claims, by the issue -----------------------------------------------------------


def test_a_claim_on_a_needs_triage_issue_gets_the_triage_reply_and_no_assignment():
    fake = FakeGitHub()

    assert claim(comment_event(labels=["bug", "needs-triage"]), fake) == rules.CLAIM_NEEDS_TRIAGE
    assert fake.assigned == []
    body = only_post(fake)
    assert rules.marker_for(rules.CLAIM_NEEDS_TRIAGE, "newbie", f"c{COMMENT_ID}") in body
    # Triage often leaves an issue off the newcomer lists, so the reply can't promise it.
    assert "it's yours" not in body.lower()


@pytest.mark.parametrize("label", sorted(rules.NOT_READY_LABELS))
@pytest.mark.parametrize("merged", [0, 5])
def test_an_open_question_or_decision_is_assigned_to_no_one(label, merged):
    fake = FakeGitHub(merged=merged)

    assert claim(comment_event(labels=[label, "area: backend"]), fake) == rules.CLAIM_NOT_READY
    assert fake.assigned == [] and fake.queries == []
    assert line(bot.ASSIGNED_LINE) not in only_post(fake)


def test_the_assignee_of_a_needs_triage_issue_is_not_told_to_hold_off():
    fake = FakeGitHub()

    event = comment_event("I'll work on it this weekend.", labels=["needs-triage"], assignees=["newbie"])
    assert claim(event, fake) is None
    assert fake.posted == []


def test_a_claim_on_a_taken_issue_lists_free_good_first_issues_without_pinned_ones():
    free = [(1105, "Welcome, start here")] + [(100 + n, f"Free issue {n}") for n in range(8)]
    fake = FakeGitHub(free=free, pinned={1105})

    assert claim(comment_event(assignees=["someone-else"]), fake) == rules.CLAIM_TAKEN
    body = only_post(fake)
    assert re.findall(r"^- #(\d+) ", body, re.M) == ["100", "101", "102", "103", "104"]
    assert "- #100 `Free issue 0`" in body
    assert "#1105" not in body
    assert fake.assigned == []


def test_listed_titles_cannot_mention_link_or_forge_a_marker():
    title = "@victim see [this](https://evil.example) `x` <!-- welcome-bot:claim-first-timer:victim:c1 -->"
    fake = FakeGitHub(free=[(101, title)])

    claim(comment_event(assignees=["someone-else"]), fake)

    listed = [row for row in only_post(fake).splitlines() if row.startswith("- #101")]
    assert listed == ["- #101 `@victim see [this](https://evil.example) x !-- welcome-bot:claim-first-timer:victim:c1 --`"]
    assert "<!-- welcome-bot:claim-first-timer:victim" not in only_post(fake)


def test_the_taken_reply_searches_only_open_triaged_unassigned_good_first_issues():
    fake = FakeGitHub(free=[(101, "Free")])

    claim(comment_event(assignees=["someone-else"]), fake)

    query = fake.queries[-1]
    terms = ("is:issue", "is:open", 'label:"good first issue"', "-label:needs-triage", "-label:no-auto-assign", "no:assignee")
    for term in terms + tuple(f"-label:{name}" for name in rules.NOT_READY_LABELS):
        assert term in query


def test_the_taken_reply_suggests_only_issues_the_bot_would_hand_out():
    free = [
        {"number": 1065, "title": "Tracking: async route handlers", "labels": [{"name": "good first issue"}]},
        {"number": 1066, "title": "Opted out", "labels": [{"name": "good first issue"}, {"name": "no-auto-assign"}]},
        {"number": 1067, "title": "An open question", "labels": [{"name": "good first issue"}, {"name": "question"}]},
        (101, "Free"),
    ]
    fake = FakeGitHub(free=free)

    claim(comment_event(assignees=["someone-else"]), fake)

    assert re.findall(r"^- #(\d+) ", only_post(fake), re.M) == ["101"]


def test_the_taken_reply_leaves_out_issues_held_while_a_maintainer_confirms():
    free = [(100 + n, f"Free issue {n}") for n in range(8)]
    fake = FakeGitHub(free=free, threads={101: [pending_comment("someone")]})

    claim(comment_event(assignees=["someone-else"]), fake)

    assert re.findall(r"^- #(\d+) ", only_post(fake), re.M) == ["100", "102", "103", "104", "105"]
    # Threads are read only until the list is full.
    assert [number for number in fake.comment_reads if number != 42] == [100, 101, 102, 103, 104, 105]


def test_a_taken_issue_with_no_free_issue_left_says_so():
    fake = FakeGitHub()

    claim(comment_event(assignees=["someone-else"]), fake)

    assert bot.render(bot.NO_FREE_ISSUES_LINE, {}) in only_post(fake)


def test_a_claim_by_the_current_assignee_gets_no_reply():
    fake = FakeGitHub()

    assert claim(comment_event(login="Newbie", assignees=["newbie"]), fake) is None
    assert fake.posted == [] and fake.assigned == [] and fake.queries == []


def test_the_issue_is_read_fresh_so_an_assignment_made_while_queued_counts():
    event = comment_event()
    fake = FakeGitHub(issue={**event["issue"], "assignees": [{"login": "first-claimer"}]})

    assert claim(event, fake) == rules.CLAIM_TAKEN
    assert fake.reads == [42] and fake.assigned == []


def test_a_claim_waiting_on_a_maintainer_holds_the_issue_for_seven_days():
    fake = FakeGitHub(comments=[pending_comment(days_ago=2)])

    assert claim(comment_event(), fake) == rules.CLAIM_TAKEN
    assert fake.assigned == []


def test_a_waiting_claim_older_than_seven_days_has_lapsed():
    fake = FakeGitHub(comments=[pending_comment(days_ago=rules.CLAIM_DAYS + 1)])

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]


def test_a_maintainer_reply_after_a_waiting_claim_ends_the_hold():
    reply = person_comment("AutomatosAI", "Thanks! Let me look at why that didn't assign.", "OWNER")
    fake = FakeGitHub(comments=[pending_comment(), reply])

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]


def test_a_waiting_claimer_stepping_back_ends_the_hold():
    step_back = person_comment("First-Claimer", "I’m stepping back from this one. The filters touch two legacy paths.")
    fake = FakeGitHub(comments=[pending_comment(), step_back])

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER


def test_only_the_waiting_claimer_can_step_back_from_their_hold():
    fake = FakeGitHub(comments=[pending_comment(), person_comment("bystander", "I'm stepping back from this one.")])

    assert claim(comment_event(), fake) == rules.CLAIM_TAKEN


def test_a_waiting_claimer_nudging_gets_no_second_reply_or_assignment_attempt():
    fake = FakeGitHub(comments=[pending_comment("newbie")])

    assert claim(comment_event("Thanks! Can I be assigned?"), fake) is None
    assert fake.posted == [] and fake.assigned == [] and fake.queries == []


def test_a_returning_contributor_on_a_taken_issue_is_not_welcomed_as_new_and_is_pointed_at_help_wanted():
    fake = FakeGitHub(merged=rules.HELP_WANTED_THRESHOLD, free=[(101, "Free")])

    assert claim(comment_event(assignees=["someone-else"]), fake) == rules.CLAIM_TAKEN_RETURNING
    body = only_post(fake)
    assert "welcome" not in body.split("\n", 1)[0].lower()
    assert line(bot.HELP_WANTED_LINE) in body
    assert "- #101 `Free`" in body


def test_a_returning_contributor_below_the_threshold_on_a_taken_issue_gets_no_help_wanted_line():
    fake = FakeGitHub(merged=1)

    assert claim(comment_event(assignees=["someone-else"]), fake) == rules.CLAIM_TAKEN_RETURNING
    assert line(bot.HELP_WANTED_LINE) not in only_post(fake)


# --- Claims, by the contributor's history -------------------------------------------


def test_a_first_timer_is_assigned_and_gets_the_first_timer_welcome():
    fake = FakeGitHub()

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]
    body = only_post(fake)
    assert line(bot.ASSIGNED_LINE) in body
    assert body.startswith("Welcome @newbie")
    assert body.rstrip().endswith(rules.marker_for(rules.CLAIM_FIRST_TIMER, "newbie", f"c{COMMENT_ID}"))
    assert rules.pending_marker("newbie") not in body


def test_a_first_timer_already_holding_an_issue_is_told_one_at_a_time_and_not_assigned():
    fake = FakeGitHub(held=[7])

    assert claim(comment_event(), fake) == rules.CLAIM_ONE_AT_A_TIME
    assert fake.assigned == []
    assert "#7" in only_post(fake)


def test_held_issues_come_from_the_rest_list_not_the_lagging_search_index():
    fake = FakeGitHub(held=[7])

    claim(comment_event(), fake)

    assert {"assignee": "newbie", "state": "open"} in fake.filters
    assert not any("assignee:" in query for query in fake.queries)
    # An assignment already held settles it: no thread is read for a claim held elsewhere.
    assert not any("mentioned" in filters for filters in fake.filters)


def test_a_first_timer_whose_claim_is_held_on_another_issue_is_told_one_at_a_time():
    fake = FakeGitHub(mentioned=[7], threads={7: [pending_comment("newbie")]})

    assert claim(comment_event(), fake) == rules.CLAIM_ONE_AT_A_TIME
    assert fake.assigned == []
    assert "#7" in only_post(fake)
    assert {"mentioned": "newbie", "state": "open"} in fake.filters


def test_a_hold_on_another_issue_that_ended_or_lapsed_does_not_count():
    released = [pending_comment("newbie"), person_comment("newbie", "Please unassign me, no time this month.")]
    lapsed = [pending_comment("newbie", days_ago=rules.CLAIM_DAYS + 1)]
    fake = FakeGitHub(mentioned=[7, 8, 9], threads={7: released, 8: lapsed, 9: [pending_comment("someone-else")]})

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]
    assert [number for number in fake.comment_reads if number != 42] == [7, 8, 9]


def test_a_merge_the_search_index_has_not_caught_up_on_already_counts():
    fake = FakeGitHub(merged=0, authored=[merged_pr(1137, "2026-10-15T19:59:23Z")])

    assert claim(comment_event(labels=["bug"]), fake) == rules.CLAIM_RETURNING
    assert fake.assigned == [(42, "newbie")]
    assert {"creator": "newbie", "state": "closed"} in fake.filters


def test_a_closed_unmerged_pull_request_or_an_issue_is_not_a_merge():
    fake = FakeGitHub(authored=[{"number": 70, "pull_request": {"merged_at": None}}, {"number": 60}])

    assert claim(comment_event(labels=["bug"]), fake) == rules.CLAIM_NEEDS_MAINTAINER
    assert fake.assigned == []


@pytest.mark.parametrize(
    "held",
    [
        [42],
        [{"number": 9, "title": "A pull request", "labels": [], "pull_request": {"url": "x"}}],
        [1105],
        [{"number": 1065, "title": "Tracking: async route handlers", "labels": []}],
    ],
    ids=["this-issue", "a-pull-request", "the-pinned-issue", "a-shared-tracking-issue"],
)
def test_what_does_not_count_as_an_issue_already_held(held):
    fake = FakeGitHub(held=held, pinned={1105})

    assert claim(comment_event(number=42), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]


@pytest.mark.parametrize("labels", [(), ("enhancement", "area: backend"), ("bug", "documentation")])
def test_a_first_timer_on_an_issue_outside_the_newcomer_lists_waits_for_a_maintainer(labels):
    fake = FakeGitHub()

    assert claim(comment_event("Can I work on this on Windows?", labels=labels), fake) == rules.CLAIM_NEEDS_MAINTAINER
    assert fake.assigned == []
    assert line(bot.ASSIGNED_LINE) not in only_post(fake)


@pytest.mark.parametrize("labels", [("good first issue",), ("help wanted", "bug")])
def test_a_first_timer_on_a_newcomer_list_issue_is_assigned(labels):
    fake = FakeGitHub()

    assert claim(comment_event(labels=labels), fake) == rules.CLAIM_FIRST_TIMER
    assert fake.assigned == [(42, "newbie")]


def test_a_returning_contributor_can_claim_any_open_triaged_issue():
    fake = FakeGitHub(merged=1)

    assert claim(comment_event(labels=["bug"]), fake) == rules.CLAIM_RETURNING
    assert fake.assigned == [(42, "newbie")]


def test_a_returning_contributor_is_assigned_with_the_short_reply():
    fake = FakeGitHub(merged=1)

    assert claim(comment_event(), fake) == rules.CLAIM_RETURNING
    assert fake.assigned == [(42, "newbie")]
    body = only_post(fake)
    assert line(bot.ASSIGNED_LINE) in body
    assert line(bot.HELP_WANTED_LINE) not in body
    # Merged once: only the fresh merge count is read, never the issues they hold.
    assert fake.filters == [{"creator": "newbie", "state": "closed"}]


def test_a_returning_contributor_at_the_threshold_is_pointed_at_help_wanted():
    fake = FakeGitHub(merged=rules.HELP_WANTED_THRESHOLD)

    assert claim(comment_event(), fake) == rules.CLAIM_RETURNING
    assert line(bot.HELP_WANTED_LINE) in only_post(fake)
    # Past the threshold the search count settles every reply, so the REST list isn't read.
    assert fake.filters == []


def test_a_returning_contributor_claiming_a_help_wanted_issue_is_not_pointed_at_help_wanted():
    fake = FakeGitHub(merged=rules.HELP_WANTED_THRESHOLD)

    assert claim(comment_event(labels=["help wanted"]), fake) == rules.CLAIM_RETURNING
    assert line(bot.HELP_WANTED_LINE) not in only_post(fake)


@pytest.mark.parametrize("merged", [0, 1])
def test_a_dropped_assignment_is_never_claimed_and_holds_the_issue_while_a_maintainer_confirms(merged, caplog):
    fake = FakeGitHub(merged=merged, assign_sticks=False)

    with caplog.at_level(logging.WARNING, logger="welcome_contributor"):
        claim(comment_event(), fake)

    body = only_post(fake)
    assert line(bot.ASSIGNED_LINE) not in body
    assert line(bot.ASSIGN_PENDING_LINE) in body
    assert "It's yours" not in body
    assert rules.pending_marker("newbie") in body
    assert "did not assign @newbie" in caplog.text


# --- Idempotency --------------------------------------------------------------------


def test_the_same_comment_delivered_again_is_not_answered_or_acted_on_twice():
    fake = FakeGitHub(comments=[bot_comment(rules.CLAIM_FIRST_TIMER, ref=f"c{COMMENT_ID}")])

    assert claim(comment_event(), fake) is None
    assert fake.posted == [] and fake.assigned == []


def test_a_marker_typed_by_a_person_or_another_bot_does_not_stop_the_reply():
    marker = rules.marker_for(rules.CLAIM_FIRST_TIMER, "newbie", f"c{COMMENT_ID}")
    comments = [
        {"user": {"login": "prankster", "type": "User"}, "body": marker},
        {"user": {"login": "coderabbitai[bot]", "type": "Bot"}, "body": f"> {marker}"},
    ]
    fake = FakeGitHub(comments=comments)

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER
    assert len(fake.posted) == 1


def test_a_claim_made_again_after_a_lapse_is_assigned_with_the_short_reply():
    fake = FakeGitHub(comments=[bot_comment(rules.CLAIM_FIRST_TIMER, ref="c1")])

    assert claim(comment_event("I'd like to pick this up again."), fake) == rules.CLAIM_AGAIN
    assert fake.assigned == [(42, "newbie")]
    body = only_post(fake)
    assert body.startswith("Welcome back to this one, @newbie")
    assert rules.marker_for(rules.CLAIM_AGAIN, "newbie", f"c{COMMENT_ID}") in body


def test_the_same_advice_is_not_repeated_to_the_same_person_on_a_thread():
    fake = FakeGitHub(comments=[bot_comment(rules.CLAIM_TAKEN, ref="c1")])

    assert claim(comment_event(assignees=["someone-else"]), fake) is None
    assert fake.posted == []


def test_a_different_reply_for_the_same_person_still_posts():
    fake = FakeGitHub(comments=[bot_comment(rules.CLAIM_NEEDS_TRIAGE, ref="c1")])

    assert claim(comment_event(), fake) == rules.CLAIM_FIRST_TIMER

