#!/usr/bin/env python3
"""Welcome new contributors on their issue and pull request threads (.github/workflows/welcome.yml).

Gerard, 15 Oct ("Can we automate the welcome message???"): newcomers fixing small
bugs grow the contributor list, and CI is strict, so letting them in should be easy;
contributors who have proven themselves get pointed at ``help wanted``, the bigger
work. This posts the welcome he typed by hand, from the templates in
``.github/welcome/`` (edit the wording there, not here):

* **A claim** (``issue_comment``): the issue is read fresh (the event is a snapshot),
  then ``welcome_rules.py`` applies CONTRIBUTING.md's claim rules and this file
  assigns and replies. GitHub drops an assignee it can't assign without an error, so
  the assignment is read back, never assumed; a dropped one is a run warning, the
  reply says a maintainer will confirm, and the issue is held for the claimer meanwhile.
* **A first pull request** (``pull_request_target`` opened): the CI, tests, sign-off and
  template pointers, once per contributor.
* **A first merge** (``pull_request_target`` closed and merged): thanks, and the way on.

``welcome_github.py`` is the GitHub client. All three files are stdlib only, and the
workflow runs them from the default branch, never a pull request's code.

    python3 scripts/ci/welcome_contributor.py [--event-path P] [--event-name N] [--repo O/R]
"""
import argparse
import importlib.util
import itertools
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Sequence, Tuple


def _sibling(name: str) -> ModuleType:
    """``scripts/ci/<name>.py``, loaded by path: the tests load this script by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _sibling("welcome_rules")
github = _sibling("welcome_github")
GitHubApi = github.GitHubApi
Decision = rules.Decision

logger = logging.getLogger("welcome_contributor")

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / ".github" / "welcome"
MAX_SUGGESTED_ISSUES = 5
# Over-fetched: pinned, Tracking: and not-ready issues, and claims held while a maintainer
# confirms, are left out after the search.
SUGGESTION_LOOKUP = 20
MERGE_LOOKUP = 10
# Threads read, at most, for a claim the bot holds for a first-timer elsewhere.
MAX_HOLD_LOOKUPS = 5
NO_PINS: FrozenSet[int] = frozenset()
# The payload objects each event's handler reads; a missing or malformed one fails the run.
EVENT_SECTIONS = {"issue_comment": ("issue", "comment"), "pull_request_target": ("pull_request",)}
LOGIN_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# One-line pieces a reply drops into its placeholders.
ASSIGNED_LINE = "assigned-line"
ASSIGN_PENDING_LINE = "assign-pending-line"
HELP_WANTED_LINE = "help-wanted-line"
NO_FREE_ISSUES_LINE = "no-free-issues-line"
TEMPLATES = rules.REPLY_KINDS + (ASSIGNED_LINE, ASSIGN_PENDING_LINE, HELP_WANTED_LINE, NO_FREE_ISSUES_LINE)
LISTING_KINDS = frozenset({rules.CLAIM_TAKEN, rules.CLAIM_TAKEN_RETURNING})
PLACEHOLDERS = frozenset({"login", "repo_url", "issue_ref", "issue_list", "assigned_line", "help_wanted_line"})
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
BLANK_RUN = re.compile(r"\n{3,}")
# A warning or error in the log also shows as an annotation on the run's summary page.
ANNOTATIONS = {logging.WARNING: "::warning::", logging.ERROR: "::error::", logging.CRITICAL: "::error::"}


class EventError(ValueError):
    """The event payload lacks a field the bot needs, or carries a malformed one."""


@dataclass(frozen=True)
class Thread:
    """The issue or pull request the bot replies on, the contributor, and the event (``ref``) it answers."""

    repo: str
    number: int
    login: str
    ref: str


# --- Templates ----------------------------------------------------------------------


def render(name: str, values: Mapping[str, str], directory: Path = TEMPLATE_DIR) -> str:
    """Template ``name`` with every ``{placeholder}`` filled; one without a value raises ValueError."""
    template = (directory / f"{name}.md").read_text(encoding="utf-8")
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(values))
    if missing:
        raise ValueError(f"{name}.md uses {missing}, which this reply doesn't fill")
    filled = PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
    return BLANK_RUN.sub("\n\n", filled).strip()


def check_templates(directory: Path = TEMPLATE_DIR) -> None:
    """Fail before any write when a template is missing or names a placeholder the bot never fills."""
    for name in TEMPLATES:
        named = set(PLACEHOLDER.findall((directory / f"{name}.md").read_text(encoding="utf-8")))
        if named - PLACEHOLDERS:
            raise ValueError(f"{name}.md names unknown placeholders {sorted(named - PLACEHOLDERS)}")


# --- Replies ------------------------------------------------------------------------


def unheld(client: GitHubApi, items: Iterable[Mapping[str, Any]], now: datetime) -> Iterator[Mapping[str, Any]]:
    """The issues no claim is held on (an assignment GitHub dropped); a thread is read only when reached."""
    return (item for item in items if not rules.pending_holds(client.list_issue_comments(item["number"]), now))


def free_issue_list(client: GitHubApi, thread: Thread, pinned: FrozenSet[int], values: Mapping[str, str]) -> str:
    """Up to ``MAX_SUGGESTED_ISSUES`` free good first issues the bot would hand out, this one left out."""
    items = client.search_issues(rules.free_issues_query(thread.repo), SUGGESTION_LOOKUP)
    offered = [
        item
        for item in items
        if isinstance(item.get("number"), int) and item["number"] != thread.number and rules.suggestible(item, pinned)
    ]
    free = itertools.islice(unheld(client, offered, datetime.now(timezone.utc)), MAX_SUGGESTED_ISSUES)
    listed = rules.issue_lines(free, frozenset(), MAX_SUGGESTED_ISSUES)
    return listed or render(NO_FREE_ISSUES_LINE, values)


def reply_values(
    client: GitHubApi, thread: Thread, decision: Decision, assigned: bool, pinned: FrozenSet[int]
) -> Dict[str, str]:
    """Every placeholder's value for ``decision``; the API is asked only for what its reply shows."""
    base = {
        "login": thread.login,
        "repo_url": f"https://github.com/{thread.repo}",
        "issue_ref": f"#{decision.held_issue}" if decision.held_issue else "",
    }
    return {
        **base,
        "issue_list": free_issue_list(client, thread, pinned, base) if decision.kind in LISTING_KINDS else "",
        "assigned_line": render(ASSIGNED_LINE if assigned else ASSIGN_PENDING_LINE, base) if decision.assign else "",
        "help_wanted_line": render(HELP_WANTED_LINE, base) if decision.help_wanted else "",
    }


def assign(client: GitHubApi, thread: Thread) -> bool:
    """Assign the contributor; True only when GitHub's answer lists them (it drops some silently)."""
    if thread.login.lower() in {name.lower() for name in client.add_assignee(thread.number, thread.login)}:
        return True
    logger.warning(
        "GitHub did not assign @%s to #%s. The reply says a maintainer will confirm: assign them by hand.",
        thread.login,
        thread.number,
    )
    return False


def post_once(
    client: GitHubApi, thread: Thread, decision: Decision, comments: Sequence[Mapping[str, Any]], pinned: FrozenSet[int]
) -> Optional[str]:
    """Post ``decision``'s reply once, assigning first when it says so; the kind posted, or None."""
    if not decision.kind:
        logger.info("No reply needed on #%s for @%s.", thread.number, thread.login)
        return None
    if rules.already_posted(comments, decision.kind, thread.login, thread.ref):
        logger.info("#%s already has the %s reply for @%s; not posting it again.", thread.number, decision.kind, thread.login)
        return None
    assigned = assign(client, thread) if decision.assign else False
    body = render(decision.kind, reply_values(client, thread, decision, assigned, pinned))
    held_for_them = (rules.pending_marker(thread.login),) if decision.assign and not assigned else ()
    markers = (rules.marker_for(decision.kind, thread.login, thread.ref),) + held_for_them
    client.post_comment(thread.number, body + "\n\n" + "\n".join(markers) + "\n")
    logger.info("Posted the %s reply for @%s on #%s.", decision.kind, thread.login, thread.number)
    return decision.kind


# --- Events -------------------------------------------------------------------------


def login_of(user: Mapping[str, Any]) -> str:
    """The user's login, validated: it goes into API queries and the reply."""
    login = str(user.get("login") or "")
    if not LOGIN_PATTERN.match(login):
        raise EventError(f"unexpected login {login!r} in the event")
    return login


def positive_int(value: Any, what: str) -> int:
    """An issue, pull request or comment number from the event, validated."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EventError(f"unexpected {what} {value!r} in the event")
    return value


def check_event(event_name: str, event: Mapping[str, Any]) -> None:
    """Fail on a payload without the objects its handler reads (``EVENT_SECTIONS``), before anything reads them."""
    for key in EVENT_SECTIONS.get(event_name, ()):
        if not isinstance(event.get(key), dict):
            raise EventError(f"the {event_name} event has no {key} object")


def comment_thread(event: Mapping[str, Any], repo: str) -> Thread:
    """The issue, the commenter, and the comment the reply answers."""
    comment = event["comment"]
    comment_id = positive_int(comment.get("id"), "comment id")
    number = positive_int(event["issue"].get("number"), "issue number")
    return Thread(repo, number, login_of(comment.get("user") or {}), f"c{comment_id}")


def merged_prs(client: GitHubApi, thread: Thread) -> int:
    """The contributor's merged pull requests here: the search count, or the fresh REST list's when that is higher.

    A claim often comes seconds after a first merge (the merge-first reply invites it), before the search
    index has the merge. The count only matters up to ``HELP_WANTED_THRESHOLD``, so past it REST isn't asked.
    """
    searched = client.search_count(rules.merged_prs_query(thread.repo, thread.login))
    if searched >= rules.HELP_WANTED_THRESHOLD:
        return searched
    recent = client.list_issues({"creator": thread.login, "state": "closed"})
    return max(searched, rules.count_merged(recent))


def pending_elsewhere(client: GitHubApi, thread: Thread, pinned: FrozenSet[int], now: datetime) -> Tuple[int, ...]:
    """Another open issue the bot holds for the contributor while a maintainer confirms (GitHub dropped the assignment).

    Every assigning reply @mentions them, so the REST ``mentioned`` list finds those threads; each is read for a hold.
    """
    mentioned = client.list_issues({"mentioned": thread.login, "state": "open"})
    candidates = rules.held_issues(mentioned, thread.number, pinned)
    if len(candidates) > MAX_HOLD_LOOKUPS:
        logger.warning(
            "@%s is mentioned on %s open issues; only the newest %s were read for a claim held for them.",
            thread.login,
            len(candidates),
            MAX_HOLD_LOOKUPS,
        )
    login = thread.login.lower()
    held = (n for n in candidates[:MAX_HOLD_LOOKUPS] if login in rules.pending_holds(client.list_issue_comments(n), now))
    first = next(held, None)
    return (first,) if first is not None else ()


def issues_held(client: GitHubApi, thread: Thread, pinned: FrozenSet[int], now: datetime) -> Tuple[int, ...]:
    """The issues a first-timer already holds: assigned (the REST list), or else held for them by the bot."""
    assigned = client.list_issues({"assignee": thread.login, "state": "open"})
    return rules.held_issues(assigned, thread.number, pinned) or pending_elsewhere(client, thread, pinned, now)


def claim_decision(
    client: GitHubApi, thread: Thread, issue: Mapping[str, Any], comments: Sequence[Mapping[str, Any]], pinned: FrozenSet[int]
) -> Decision:
    """Ask GitHub only what the claim's reply turns on: merged pull requests, then (none yet) issues held."""
    now = datetime.now(timezone.utc)
    state = rules.issue_state(issue, thread.login, comments, now)
    labels = rules.names(issue.get("labels"), "name")
    if state not in (rules.ISSUE_TAKEN, rules.ISSUE_FREE):
        return rules.decide_claim(state, 0, (), labels)
    merged = merged_prs(client, thread)
    held = issues_held(client, thread, pinned, now) if state == rules.ISSUE_FREE and merged == 0 else ()
    return rules.decide_claim(state, merged, held, labels)


def handle_comment(event: Mapping[str, Any], client: GitHubApi, repo: str) -> Optional[str]:
    """An ``issue_comment``: answer a claim by CONTRIBUTING.md's rules, on the issue's current state."""
    reason = rules.claim_ignore_reason(event)
    if reason:
        logger.info("No reply: %s.", reason)
        return None
    thread = comment_thread(event, repo)
    issue = client.get_issue(thread.number)
    pinned = client.pinned_issue_numbers()
    if issue.get("state") != "open" or rules.left_to_maintainer(issue, pinned):
        logger.info("No reply: #%s is closed now, or a maintainer hands it out (pinned, Tracking:, no-auto-assign).", thread.number)
        return None
    comments = client.list_issue_comments(thread.number)
    decision = claim_decision(client, thread, issue, comments, pinned)
    return post_once(client, thread, rules.as_reclaim(decision, comments, thread.login), comments, pinned)


def pr_opened_decision(client: GitHubApi, pr: Mapping[str, Any], login: str, number: int) -> Decision:
    """The first-PR welcome, for a first-time contributor with no earlier pull request here."""
    if pr.get("author_association") not in rules.FIRST_PR_ASSOCIATIONS:
        return rules.NO_REPLY
    authored = client.list_issues({"creator": login, "state": "all"})
    return Decision(rules.PR_FIRST) if rules.is_first_pr(number, authored) else rules.NO_REPLY


def pr_merged_decision(client: GitHubApi, pr: Mapping[str, Any], repo: str, login: str, number: int) -> Decision:
    """The first-merge thanks, when no other pull request by the author merged before this one."""
    if not pr.get("merged_at"):
        raise EventError(f"pull request #{number} is merged but the event has no merged_at")
    recent = client.list_issues({"creator": login, "state": "closed"})
    older = client.search_issues(rules.merged_prs_query(repo, login), MERGE_LOOKUP)
    first = rules.is_first_merge(number, rules.parse_time(pr["merged_at"]), recent + older)
    return Decision(rules.MERGE_FIRST) if first else rules.NO_REPLY


def handle_pull_request(event: Mapping[str, Any], client: GitHubApi, repo: str) -> Optional[str]:
    """A ``pull_request_target`` event: the first-PR welcome on open, the first-merge thanks on merge."""
    pr = event.get("pull_request") or {}
    user = pr.get("user") or {}
    if rules.is_bot(user):
        logger.info("No reply: a bot's pull request.")
        return None
    login, number, action = login_of(user), positive_int(pr.get("number"), "pull request number"), event.get("action")
    if action == "opened":
        decision, ref = pr_opened_decision(client, pr, login, number), "opened"
    elif action == "closed" and pr.get("merged") and pr.get("author_association") not in rules.MAINTAINER_ASSOCIATIONS:
        decision, ref = pr_merged_decision(client, pr, repo, login, number), "merged"
    else:
        logger.info("No reply: pull request %s by @%s.", action, login)
        return None
    if not decision.kind:
        logger.info("No reply: not @%s's first pull request or first merge here.", login)
        return None
    thread = Thread(repo, number, login, ref)
    return post_once(client, thread, decision, client.list_issue_comments(number), NO_PINS)


def handle_event(event_name: str, event: Mapping[str, Any], client: GitHubApi, repo: str) -> Optional[str]:
    """Route one webhook event; the reply kind posted, or None."""
    check_event(event_name, event)
    if event_name == "issue_comment":
        return handle_comment(event, client, repo)
    if event_name == "pull_request_target":
        return handle_pull_request(event, client, repo)
    logger.info("No reply: the bot doesn't handle %s events.", event_name)
    return None


def run(event_name: str, event: Mapping[str, Any], client: GitHubApi, repo: str) -> int:
    """Handle one event: 0 when done, 1 (logged) when GitHub, the payload or a template failed."""
    try:
        check_templates()
        posted = handle_event(event_name, event, client, repo)
    except (github.GitHubError, ValueError, OSError):
        logger.exception("The welcome bot stopped on a %s event.", event_name)
        return 1
    logger.info("Done: %s.", f"posted {posted}" if posted else "nothing posted")
    return 0


class ActionsFormatter(logging.Formatter):
    """Log lines with warnings and errors as GitHub Actions annotations, seen without opening the log."""

    def format(self, record: logging.LogRecord) -> str:
        """The line, prefixed ``::warning::`` or ``::error::`` by its level."""
        return ANNOTATIONS.get(record.levelno, "") + super().format(record)


def read_event(path: str) -> Dict[str, Any]:
    """The webhook payload the runner saved at ``path``."""
    event = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise EventError(f"{path} does not hold a JSON object")
    return event


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The event and the repository, defaulting to what the Actions runner sets."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH", ""))
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The workflow's entry point: read the event, reply, exit non-zero on any failure."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ActionsFormatter("%(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    args = parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not args.event_path or not REPO_PATTERN.match(args.repo):
        logger.error("Needs GITHUB_TOKEN, the event file (GITHUB_EVENT_PATH) and GITHUB_REPOSITORY as owner/name.")
        return 2
    try:
        event = read_event(args.event_path)
    except (OSError, ValueError):
        logger.exception("Could not read the event file %s.", args.event_path)
        return 1
    return run(args.event_name, event, github.GitHubClient(token, args.repo), args.repo)


if __name__ == "__main__":
    sys.exit(main())
