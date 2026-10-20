"""The welcome bot under test, and the fake GitHub it talks to (tests/test_ci_welcome_*.py).

``scripts/ci/welcome_contributor.py`` is loaded by path, the same way
``test_ci_backend_test_plan.py`` loads its script. It loads ``welcome_rules.py`` and
``welcome_github.py`` itself; the tests reach them through it (``bot.rules``,
``bot.github``), so an exception class the script catches is the one a fake raises.
No network, no database.
"""
import importlib.util
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = _ROOT / ".github" / "welcome"
WORKFLOW = _ROOT / ".github" / "workflows" / "welcome.yml"
REPO = "AutomatosAI/automatos-ai"
LEFTOVER = re.compile(r"\{[a-z_]+\}")
BOT_USER = {"login": "github-actions[bot]", "type": "Bot"}
COMMENT_ID = 9001
MERGED_AT = "2026-10-15T20:00:00Z"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bot = _load("welcome_contributor")
rules = bot.rules


def _held(item):
    return item if isinstance(item, dict) else {"number": item, "title": f"Held {item}", "labels": []}


class FakeGitHub:
    """The GitHub the bot sees here: canned answers, and a record of every call."""

    def __init__(
        self, *, issue=None, comments=(), merged=0, merged_items=(), held=(), authored=(), free=(), pinned=(),
        assign_sticks=True, mentioned=(), threads=None,
    ):
        self.issue = issue
        self.comments = list(comments)
        self.merged = merged
        self.merged_items = list(merged_items)
        self.held = [_held(item) for item in held]
        self.authored = list(authored)
        self.free = list(free)
        self.pinned = frozenset(pinned)
        self.assign_sticks = assign_sticks
        self.mentioned = [_held(item) for item in mentioned]
        self.threads = dict(threads or {})
        self.posted, self.assigned, self.queries, self.filters, self.reads = [], [], [], [], []
        self.comment_reads = []

    def get_issue(self, number):
        self.reads.append(number)
        return dict(self.issue)

    def list_issues(self, filters):
        self.filters.append(dict(filters))
        if "assignee" in filters:
            return list(self.held)
        if "mentioned" in filters:
            return list(self.mentioned)
        return list(self.authored)

    def list_issue_comments(self, number):
        """This thread's comments; another issue's come from ``threads``."""
        self.comment_reads.append(number)
        return list(self.threads.get(number, self.comments))

    def add_assignee(self, number, login):
        self.assigned.append((number, login))
        return [login] if self.assign_sticks else []

    def post_comment(self, number, body):
        self.posted.append((number, body))

    def search_count(self, query):
        self.queries.append(query)
        assert "is:merged" in query
        return self.merged

    def search_issues(self, query, limit):
        self.queries.append(query)
        if "is:merged" in query:
            return self.merged_items[:limit]
        items = [item if isinstance(item, dict) else {"number": item[0], "title": item[1]} for item in self.free]
        return items[:limit]

    def pinned_issue_numbers(self):
        return self.pinned


def comment_event(
    body="Hi! I'd like to work on this issue.",
    *,
    login="newbie",
    association="NONE",
    user_type="User",
    labels=("good first issue",),
    assignees=(),
    state="open",
    on_pr=False,
    number=42,
    title="Fix the dead link in the README",
    comment_id=COMMENT_ID,
):
    issue = {
        "number": number,
        "title": title,
        "state": state,
        "labels": [{"name": name} for name in labels],
        "assignees": [{"login": name} for name in assignees],
    }
    if on_pr:
        issue["pull_request"] = {"url": f"https://api.github.com/repos/{REPO}/pulls/{number}"}
    user = {"login": login, "type": user_type}
    comment = {"id": comment_id, "body": body, "author_association": association, "user": user}
    return {"action": "created", "issue": issue, "comment": comment}


def pr_event(
    action="opened", *, login="newbie", association="FIRST_TIME_CONTRIBUTOR", merged=False, user_type="User",
    number=77, merged_at=MERGED_AT,
):
    user = {"login": login, "type": user_type}
    pr = {"number": number, "merged": merged, "author_association": association, "user": user}
    if merged:
        pr["merged_at"] = merged_at
    return {"action": action, "pull_request": pr}


def dispatch(event_name, event, fake):
    """Hand the event to the bot; on a comment, the fake's current issue defaults to the payload's."""
    if event_name == "issue_comment" and fake.issue is None:
        fake.issue = event["issue"]
    return bot.handle_event(event_name, event, fake, REPO)


def claim(event, fake):
    return dispatch("issue_comment", event, fake)


def pull_request(event, fake):
    return dispatch("pull_request_target", event, fake)


def only_post(fake):
    assert len(fake.posted) == 1, fake.posted
    return fake.posted[0][1]


def line(name, login="newbie"):
    return bot.render(name, {"login": login, "repo_url": f"https://github.com/{REPO}", "issue_ref": ""})


def bot_comment(kind, login="newbie", ref="c1", *, extra="", created_at=None):
    comment = {"user": BOT_USER, "body": f"Earlier reply\n\n{rules.marker_for(kind, login, ref)}{extra}"}
    return {**comment, "created_at": created_at} if created_at else comment


def merged_pr(number, merged_at):
    return {"number": number, "pull_request": {"merged_at": merged_at}}


def pending_comment(login="first-claimer", days_ago=2):
    """The bot's reply to a claim it couldn't assign: the issue is held for ``login``."""
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    extra = f"\n{rules.pending_marker(login)}"
    return bot_comment(rules.CLAIM_FIRST_TIMER, login, extra=extra, created_at=created.isoformat())


def person_comment(login, body, association="NONE"):
    return {"user": {"login": login, "type": "User"}, "author_association": association, "body": body}
