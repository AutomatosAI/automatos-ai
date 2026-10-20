"""The welcome bot's decisions: pure functions, no API calls (scripts/ci/welcome_contributor.py).

``welcome_contributor.py`` reads the event and GitHub's current state and hands them
here; what comes back says which reply to post and whether to assign. The rules are
CONTRIBUTING.md's "Picking up an issue":

* **A claim** is a first-person "I'd like to work on this" (``is_claim``). Detection is
  conservative on purpose: a missed claim falls back to the maintainer, a false one
  would assign the wrong person.
* **Hands off:** a pinned issue (the "start here" one), a shared ``Tracking:`` checklist
  (one file per contributor) and an issue labelled ``no-auto-assign`` are handed out
  by a maintainer, never by the bot. An issue that is still an open question or
  decision (``NOT_READY_LABELS``) is assigned to no one: the reply says to wait.
* **A first claim** (no merged pull request yet) is assigned only on a ``good first
  issue`` or ``help wanted`` issue, and only while the contributor holds no other.
  After a merge: any open, unassigned, triaged issue ("no need to ask"), and from
  ``HELP_WANTED_THRESHOLD`` merges on the reply points at ``help wanted``. The caller
  counts merges from search and from the fresh REST list, so a claim made seconds
  after a first merge already counts it.
* **A held claim.** When GitHub drops the bot's assignment, the reply carries a pending
  marker, and the issue counts as the claimer's (taken, for everyone else) until a
  maintainer replies on the thread, the claimer steps back, or ``CLAIM_DAYS`` pass.
* **Markers.** Every reply carries a hidden marker naming its kind, the contributor and
  the event, so a re-delivered event never posts twice and the same advice isn't
  repeated, while a new claim after a lapse is still answered.
* **First pull request, first merge:** decided from the author's other pull requests,
  by number and by merge time, so the search index's lag can't make a second merge
  look like a first (or hide a first).
"""
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

HELP_WANTED_THRESHOLD = 3
CLAIM_DAYS = 7  # CONTRIBUTING.md: a claim with no pull request after 7 days lapses.
MAX_TITLE_CHARS = 100
# A claim is a short comment. Only its start is read, which also bounds the regex work
# on a body of up to GitHub's 65,536 characters.
MAX_CLAIM_CHARS = 10_000

NEEDS_TRIAGE = "needs-triage"
GOOD_FIRST_ISSUE = "good first issue"
HELP_WANTED = "help wanted"
FIRST_CLAIM_LABELS = frozenset({GOOD_FIRST_ISSUE, HELP_WANTED})
# Still an open question or decision: nobody is assigned until a maintainer settles it.
NOT_READY_LABELS = frozenset({"question", "wontfix", "duplicate", "invalid"})
NO_AUTO_ASSIGN = "no-auto-assign"
SHARED_TITLE_PREFIX = "tracking:"
MAINTAINER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
FIRST_PR_ASSOCIATIONS = frozenset({"FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"})
# The workflow's GITHUB_TOKEN posts as this account. Only its markers count, never a
# copy someone else typed or quoted. A switch to an app token must change this.
BOT_LOGIN = "github-actions[bot]"

# Each reply is .github/welcome/<kind>.md; the kind also names its marker.
CLAIM_FIRST_TIMER = "claim-first-timer"
CLAIM_RETURNING = "claim-returning"
CLAIM_AGAIN = "claim-again"
CLAIM_NEEDS_TRIAGE = "claim-needs-triage"
CLAIM_NEEDS_MAINTAINER = "claim-needs-maintainer"
CLAIM_NOT_READY = "claim-not-ready"
CLAIM_TAKEN = "claim-taken"
CLAIM_TAKEN_RETURNING = "claim-taken-returning"
CLAIM_ONE_AT_A_TIME = "claim-one-at-a-time"
PR_FIRST = "pr-first"
MERGE_FIRST = "merge-first"
REPLY_KINDS = (
    CLAIM_FIRST_TIMER, CLAIM_RETURNING, CLAIM_AGAIN, CLAIM_NEEDS_TRIAGE, CLAIM_NEEDS_MAINTAINER,
    CLAIM_NOT_READY, CLAIM_TAKEN, CLAIM_TAKEN_RETURNING, CLAIM_ONE_AT_A_TIME, PR_FIRST, MERGE_FIRST,
)
# Replies that assign: each claim gets its own. The others are advice, given once a thread.
ASSIGNING_KINDS = frozenset({CLAIM_FIRST_TIMER, CLAIM_RETURNING, CLAIM_AGAIN})

# What the issue's current state says about a claim on it.
ISSUE_MINE = "mine"
ISSUE_UNTRIAGED = "untriaged"
ISSUE_NOT_READY = "not-ready"
ISSUE_TAKEN = "taken"
ISSUE_FREE = "free"

MARKER = "<!-- welcome-bot:{kind}:{login}:{ref} -->"
PENDING_MARKER = "<!-- welcome-bot:pending:{login} -->"
_PENDING = re.compile(r"<!-- welcome-bot:pending:([a-z0-9-]+) -->")
_TITLE_UNSAFE = re.compile(r"[`<>]")

# Claim detection runs on the comment's own words: quotes, code and HTML comments go first.
# An unclosed fence or HTML comment hides the rest, like an unclosed block in Markdown;
# each pattern then matches at most once per opener, so the work stays linear.
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_FENCED_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?(?:^[ \t]*\1|\Z)", re.M | re.S)
_QUOTED_LINE = re.compile(r"^[ \t]*>.*$", re.M)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_EMPHASIS = re.compile(r"[*_~]+")
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

# What is claimed: "this", "it" or "the issue". Not "take it that…", "take this approach",
# "work on this project": the word after it says the comment is about something else.
_NOT_THE_ISSUE = (
    r"as|that|to|into|with|further|back|approach|route|path|way|idea|branch|repo|repository|project"
    r"|codebase|area|file|part|module|component|weekend|week"
)
_TARGET = rf"(?:this|it|the\s+issue)\b(?!\s+(?:{_NOT_THE_ISSUE})\b)"
_TAKE_ON = (
    rf"(?:work(?:ing)?\s+on\s+{_TARGET}"
    rf"|tak(?:e|ing)(?:\s+(?:on|over))?\s+{_TARGET}"
    rf"|tak(?:e|ing)\s+a\s+(?:stab|crack|shot)\s+at\s+{_TARGET}"
    rf"|pick(?:ing)?\s+(?:up\s+{_TARGET}|{_TARGET}\s+up)"
    rf"|(?:tackl|handl)(?:e|ing)\s+{_TARGET}"
    rf"|giv(?:e|ing)\s+{_TARGET}\s+a\s+(?:try|shot|go))"
)
# "Fix this" claims only after a stated wish: "can I fix this by…" is a support question.
_FIX = (
    r"(?:(?:fix|fixing|solve|solving|resolve|resolving|address|addressing|implement|implementing)"
    rf"\s+{_TARGET}|help(?:ing)?\s+(?:out\s+)?with\s+{_TARGET})"
)
_I_WOULD = r"\bi(?:'d|d|\s+would)\s+(?:really\s+)?"
_WISH = (
    rf"(?:{_I_WOULD}(?:like|love|be\s+(?:happy|glad|keen|willing))\s+to"
    rf"|{_I_WOULD}be\s+interested\s+in"
    r"|\bi(?:'m|m|\s+am)\s+(?:happy|glad|keen|willing|eager)\s+to"
    r"|\bi(?:'m|m|\s+am)\s+(?:very\s+|really\s+)?interested\s+in)\s+"
)
_INTENT = (
    r"(?:\bi\s+(?:really\s+)?(?:want\s+to|wanna)|\b(?:can|could|may)\s+i(?:\s+please)?"
    r"|\bi(?:'ll|\s+will|\s+can)|\blet\s+me)\s+"
)
_ASSIGN_ME = (
    r"(?<!n't\s)(?<!not\s)\bassign\s+(?:(?:this|it)(?:\s+(?:one|issue))?\s+)?to\s+me\b"
    r"|(?<!n't\s)(?<!not\s)\bassign\s+me\b(?!\s+(?:to|on)\s+#)"
    rf"|(?:\b(?:can|could|may)\s+i(?:\s+please)?|{_I_WOULD}like\s+to)\s+(?:be|get)\s+assigned\b"
)
CLAIM_PATTERNS = (
    re.compile(rf"{_WISH}(?:{_TAKE_ON}|{_FIX})"),
    re.compile(rf"{_INTENT}{_TAKE_ON}"),
    re.compile(_ASSIGN_ME),
)
# A claimer giving the issue up, in so many words: "I'm stepping back from this one",
# "please unassign me". Conservative like claims: a missed one lapses after CLAIM_DAYS.
_RELEASE = re.compile(
    r"\bstep(?:ping|ped)?\s+(?:back|away|down)\s+from\s+(?:this|it)\b"
    r"|(?<!n't\s)(?<!not\s)\bunassign\s+me\b"
    r"|\bwithdraw(?:ing)?\s+my\s+claim\b"
    r"|\bno\s+longer\s+(?:working\s+on|able\s+to\s+(?:work\s+on|take))\s+(?:this|it)\b"
)


@dataclass(frozen=True)
class Decision:
    """One reply: its template (also its marker's kind), and what goes with it."""

    kind: str
    assign: bool = False
    held_issue: Optional[int] = None
    help_wanted: bool = False


NO_REPLY = Decision(kind="")


# --- Claims -------------------------------------------------------------------------


def visible_text(body: str) -> str:
    """The commenter's own words in the first ``MAX_CLAIM_CHARS``, lower-cased: quotes, code and HTML comments removed."""
    text = _HTML_COMMENT.sub(" ", body[:MAX_CLAIM_CHARS].replace("\r\n", "\n"))
    text = _FENCED_CODE.sub(" ", text)
    text = _QUOTED_LINE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _EMPHASIS.sub("", text).translate(_APOSTROPHES)
    return " ".join(text.lower().split())


def is_claim(body: str) -> bool:
    """True when the comment asks, in the first person, to take the issue on."""
    text = visible_text(body)
    return any(pattern.search(text) for pattern in CLAIM_PATTERNS)


def is_release(body: str) -> bool:
    """True when the comment gives a claim up and doesn't, in the same breath, claim it."""
    text = visible_text(body)
    return bool(_RELEASE.search(text)) and not any(pattern.search(text) for pattern in CLAIM_PATTERNS)


def is_bot(user: Mapping[str, Any]) -> bool:
    """True for a bot account, by its type or its ``[bot]`` login."""
    return user.get("type") == "Bot" or str(user.get("login") or "").endswith("[bot]")


def names(items: Any, key: str) -> FrozenSet[str]:
    """The lower-cased ``key`` of every object in a payload list (labels, assignees)."""
    return frozenset(str(item.get(key) or "").lower() for item in items or () if isinstance(item, Mapping))


def claim_ignore_reason(event: Mapping[str, Any]) -> Optional[str]:
    """Why an ``issue_comment`` event needs no reply, from the payload alone; None for a claim to answer."""
    issue = event.get("issue") or {}
    comment = event.get("comment") or {}
    if event.get("action") != "created":
        return "not a new comment"
    if issue.get("pull_request"):
        return "a comment on a pull request"
    if issue.get("state") != "open":
        return "the issue is not open"
    if is_bot(comment.get("user") or {}):
        return "a bot's comment"
    if comment.get("author_association") in MAINTAINER_ASSOCIATIONS:
        return "a maintainer's comment"
    if not is_claim(str(comment.get("body") or "")):
        return "not a claim"
    return None


def left_to_maintainer(issue: Mapping[str, Any], pinned: FrozenSet[int]) -> bool:
    """True for an issue a maintainer hands out by hand: pinned, a ``Tracking:`` checklist, or ``no-auto-assign``."""
    title = str(issue.get("title") or "").strip().lower()
    labels = names(issue.get("labels"), "name")
    return issue.get("number") in pinned or NO_AUTO_ASSIGN in labels or title.startswith(SHARED_TITLE_PREFIX)


def suggestible(issue: Mapping[str, Any], pinned: FrozenSet[int]) -> bool:
    """True for a free issue the bot would hand out itself, so a reply may suggest it."""
    return not left_to_maintainer(issue, pinned) and not names(issue.get("labels"), "name") & NOT_READY_LABELS


def parse_time(value: Any) -> datetime:
    """A GitHub timestamp (``2026-10-15T19:59:23Z``) as an aware datetime; ValueError when it isn't one."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def is_own(comment: Mapping[str, Any]) -> bool:
    """True for the bot's own comment: a marker anyone else typed never counts."""
    return str((comment.get("user") or {}).get("login") or "") == BOT_LOGIN


def own_comments(comments: Iterable[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], ...]:
    """The bot's own comments on the thread."""
    return tuple(comment for comment in comments if is_own(comment))


def own_bodies(comments: Iterable[Mapping[str, Any]]) -> Tuple[str, ...]:
    """The text of the bot's own comments."""
    return tuple(str(c.get("body") or "") for c in own_comments(comments))


def pending_login(comment: Mapping[str, Any]) -> Optional[str]:
    """Whom a comment of the bot's holds the issue for (GitHub dropped the assignment); None for any other comment."""
    if not is_own(comment):
        return None
    match = _PENDING.search(str(comment.get("body") or ""))
    return match.group(1) if match else None


def ends_hold(comment: Mapping[str, Any], login: str) -> bool:
    """True for a later comment that ends ``login``'s hold: a maintainer's (it's in their hands now) or ``login`` stepping back."""
    user = comment.get("user") or {}
    if is_bot(user):
        return False
    if comment.get("author_association") in MAINTAINER_ASSOCIATIONS:
        return True
    return str(user.get("login") or "").lower() == login and is_release(str(comment.get("body") or ""))


def pending_holds(comments: Sequence[Mapping[str, Any]], now: datetime) -> FrozenSet[str]:
    """Whom the bot holds the issue for: a claim it couldn't assign, under ``CLAIM_DAYS`` old and not ended since.

    The issue looks free, as GitHub dropped the assignment, but it is the claimer's until a maintainer
    replies (the assignee list is the truth again from there), they step back, or the claim lapses.
    """
    since = now - timedelta(days=CLAIM_DAYS)
    placed = ((index, pending_login(comment)) for index, comment in enumerate(comments))
    return frozenset(
        login
        for index, login in placed
        if login
        and parse_time(comments[index].get("created_at")) >= since
        and not any(ends_hold(later, login) for later in comments[index + 1 :])
    )


def issue_state(issue: Mapping[str, Any], login: str, comments: Sequence[Mapping[str, Any]], now: datetime) -> str:
    """What the issue's current state says about ``login``'s claim: theirs already, untriaged, not ready, taken or free.

    Theirs already covers a claim the bot holds for them, so a nudge while a maintainer confirms gets no reply.
    """
    labels = names(issue.get("labels"), "name")
    holders = names(issue.get("assignees"), "login") | pending_holds(comments, now)
    if login.lower() in holders:
        return ISSUE_MINE
    if NEEDS_TRIAGE in labels:
        return ISSUE_UNTRIAGED
    if labels & NOT_READY_LABELS:
        return ISSUE_NOT_READY
    return ISSUE_TAKEN if holders else ISSUE_FREE


def decide_claim(state: str, merged_count: int, held: Sequence[int], labels: FrozenSet[str]) -> Decision:
    """The reply to a claim: the issue's state, the claimer's merged pull requests, what else they hold, its labels."""
    if state == ISSUE_MINE:
        return NO_REPLY
    if state == ISSUE_UNTRIAGED:
        return Decision(CLAIM_NEEDS_TRIAGE)
    if state == ISSUE_NOT_READY:
        return Decision(CLAIM_NOT_READY)
    bigger_work = merged_count >= HELP_WANTED_THRESHOLD
    if state == ISSUE_TAKEN:
        return Decision(CLAIM_TAKEN_RETURNING if merged_count else CLAIM_TAKEN, help_wanted=bigger_work)
    if merged_count:
        return Decision(CLAIM_RETURNING, assign=True, help_wanted=bigger_work and HELP_WANTED not in labels)
    return decide_first_claim(held, labels)


def decide_first_claim(held: Sequence[int], labels: FrozenSet[str]) -> Decision:
    """A free issue claimed by someone with no merged pull request: one at a time, from the newcomer lists only."""
    if held:
        return Decision(CLAIM_ONE_AT_A_TIME, held_issue=held[0])
    if not labels & FIRST_CLAIM_LABELS:
        return Decision(CLAIM_NEEDS_MAINTAINER)
    return Decision(CLAIM_FIRST_TIMER, assign=True)


def as_reclaim(decision: Decision, comments: Sequence[Mapping[str, Any]], login: str) -> Decision:
    """The short "yours again" reply when the bot already assigned this contributor here (a lapsed claim, back).

    A claim still held for them never gets here: ``issue_state`` calls the issue theirs, and there is no reply.
    """
    if not decision.assign:
        return decision
    earlier = tuple(marker_prefix(kind, login) for kind in ASSIGNING_KINDS)
    if any(prefix in body for body in own_bodies(comments) for prefix in earlier):
        return replace(decision, kind=CLAIM_AGAIN)
    return decision


def held_issues(assigned: Iterable[Mapping[str, Any]], number: int, pinned: FrozenSet[int]) -> Tuple[int, ...]:
    """The open issues (not pull requests) the contributor holds, other than this one and those left to a maintainer."""
    return tuple(
        item["number"]
        for item in assigned
        if isinstance(item.get("number"), int)
        and item["number"] != number
        and not item.get("pull_request")
        and not left_to_maintainer(item, pinned)
    )


# --- Pull requests ------------------------------------------------------------------


def is_first_pr(number: int, authored: Iterable[Mapping[str, Any]]) -> bool:
    """True when none of the author's other pull requests here was opened before this one (numbers grow with time)."""
    earlier = (item.get("number") for item in authored if item.get("pull_request"))
    return not any(isinstance(other, int) and other < number for other in earlier)


def merge_time(item: Mapping[str, Any]) -> Optional[datetime]:
    """When an issue-shaped pull request (a REST list or search item) merged; None when it didn't."""
    merged_at = (item.get("pull_request") or {}).get("merged_at")
    return parse_time(merged_at) if merged_at else None


def count_merged(items: Iterable[Mapping[str, Any]]) -> int:
    """How many of the issue-shaped items are merged pull requests."""
    return sum(1 for item in items if merge_time(item) is not None)


def is_first_merge(number: int, merged: datetime, authored: Iterable[Mapping[str, Any]]) -> bool:
    """True when no other pull request by the author merged here before this one did.

    Judged by merge time, not a count: a search index that hasn't caught up on this merge, or on
    one a minute earlier, can't turn a second merge into a first, and two merges back to back
    still give exactly one first.
    """
    this = (merged, number)
    for item in authored:
        other, other_number = merge_time(item), item.get("number")
        if other is not None and isinstance(other_number, int) and other_number != number and (other, other_number) < this:
            return False
    return True


# --- Text the bot posts -------------------------------------------------------------


def safe_title(title: Any) -> str:
    """Someone else's issue title, made inert: one line, cut short, in a code span (no @mention, link or marker)."""
    text = " ".join(_TITLE_UNSAFE.sub("", str(title or "")).split())
    if len(text) > MAX_TITLE_CHARS:
        text = text[: MAX_TITLE_CHARS - 1].rstrip() + "…"
    return f"`{text}`" if text else ""


def issue_lines(items: Iterable[Mapping[str, Any]], skip: FrozenSet[int], limit: int) -> str:
    """Up to ``limit`` issues as ``- #N `title``` lines, leaving out the numbers in ``skip``."""
    lines = [
        f"- #{item['number']} {safe_title(item.get('title'))}".rstrip()
        for item in items
        if isinstance(item.get("number"), int) and item["number"] not in skip
    ]
    return "\n".join(lines[:limit])


def merged_prs_query(repo: str, login: str) -> str:
    """Search: the contributor's merged pull requests in the repository."""
    return f"repo:{repo} is:pr is:merged author:{login}"


def free_issues_query(repo: str) -> str:
    """Search: open, triaged, unassigned ``good first issue`` issues the bot would hand out (labels can say it wouldn't)."""
    excluded = " ".join(f"-label:{name}" for name in (NEEDS_TRIAGE, NO_AUTO_ASSIGN, *sorted(NOT_READY_LABELS)))
    return f'repo:{repo} is:issue is:open label:"{GOOD_FIRST_ISSUE}" {excluded} no:assignee'


def marker_for(kind: str, login: str, ref: str) -> str:
    """The hidden marker a reply carries: its kind, the contributor, and the event it answers."""
    return MARKER.format(kind=kind, login=login.lower(), ref=ref)


def marker_prefix(kind: str, login: str) -> str:
    """The start every ``kind`` marker for ``login`` shares, whatever event it answered."""
    return f"<!-- welcome-bot:{kind}:{login.lower()}:"


def pending_marker(login: str) -> str:
    """The marker on a claim GitHub wouldn't assign: the issue is held for them while a maintainer confirms."""
    return PENDING_MARKER.format(login=login.lower())


def already_posted(comments: Sequence[Mapping[str, Any]], kind: str, login: str, ref: str) -> bool:
    """True when the bot already answered this event, or already gave ``login`` this advice on the thread.

    An assigning reply is per claim, so a claim made again after a lapse is still answered.
    """
    this_event = f":{login.lower()}:{ref} -->"
    same_advice = marker_prefix(kind, login)
    advice = kind not in ASSIGNING_KINDS
    return any(this_event in body or (advice and same_advice in body) for body in own_bodies(comments))
