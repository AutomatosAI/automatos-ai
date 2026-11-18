"""gh in a dev-box session: what runs and what is refused (dev-box mode, ``dev_box_policy.py``).

Runs: pull requests (create, edit, ready, comment, review, view, list, diff, checks,
close, reopen), issues (create, edit, comment, close, reopen, view, list), workflow
runs (view, list, watch, rerun), workflows and repositories (view; clone), every other
read gh knows, and ``gh api`` for reads and for the writes a ticket's pull request needs.

Refused, with the rule in the reason: merging (``gh pr merge``, auto-merge, the API's
merge endpoints) and approving (``gh pr review --approve``, ``-a`` in any short-flag
cluster or ``-a=…``, a cluster the gate cannot read whole, the API's ``event=APPROVE``);
a repository's deletion, settings, name, visibility, ownership or forks; secrets,
variables, keys, the login, codespaces, extensions, aliases and gh's own settings
(reads included: they print or reach what a session never needs); any write to a
repository whose owner is not origin's (``-R``/``--repo``, ``repos/<owner>/…``, a pull
request or issue URL as the target, a target URL off github.com; reads of other
repositories run); and through ``gh api``, DELETE anywhere, PUT, graphql, access, hooks,
keys, protection, rules, environments, commit statuses (one can pass a required check),
and any write outside issues, pulls and the session's own branch refs. An option the
gate does not know fails closed. ``file_values`` names the files a gh command reads
(``-F key=@file``, ``--input``, ``--body-file``) for ``dev_box_policy`` to confine.

Pure functions over the words of one simple command; ``None`` = not refused here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .dev_box_rules import DevBoxRules, refused, repo_owner, unreadable
from .shell_words import gh_writes, strip_globals

GH_ALLOWED: Mapping[str, FrozenSet[str]] = {
    "pr": frozenset({"create", "edit", "ready", "comment", "review", "view", "list", "diff", "checks", "close",
                     "reopen", "status"}),
    "issue": frozenset({"create", "edit", "comment", "close", "reopen", "view", "list", "status"}),
    "run": frozenset({"view", "list", "watch", "rerun"}),
    "workflow": frozenset({"view", "list"}),
    "repo": frozenset({"view", "clone"}),
}
GH_REFUSED_GROUPS: Mapping[str, str] = {
    "secret": "gh secret reads or changes the repository's secrets",
    "variable": "gh variable reads or changes the repository's variables",
    "ssh-key": "gh ssh-key changes the account's SSH keys",
    "gpg-key": "gh gpg-key changes the account's signing keys",
    "auth": "gh auth prints or changes the GitHub login, the token with it",
    "codespace": "gh codespace reaches machines outside this one",
    "extension": "gh extension installs or runs code the gate cannot read",
    "alias": "gh alias defines a command the gate cannot read",
    "config": "gh config changes the GitHub CLI's own settings",
}
REPO_CHANGE = "changes or removes a repository: that is the owner's"
GH_REFUSED_ACTIONS: Mapping[Tuple[str, str], str] = {
    ("pr", "merge"): "the owner reviews and merges; a session never merges (auto-merge included)",
    **{("repo", action): f"gh repo {action} {REPO_CHANGE}"
       for action in ("delete", "edit", "rename", "archive", "create", "fork", "transfer")},
}
OUTSIDE_THE_LIST = ("a gh write outside the dev-box list (pull requests, issues, workflow runs); everything "
                    "else in gh reads only")
# What reads another repository, whatever gh's own read list says.
GH_READS_ANYWHERE = frozenset({("repo", "clone")})
REPO_OPTIONS = frozenset({"-R", "--repo"})
REVIEW = ("pr", "review")
APPROVE_LONG = "--approve"
APPROVE_SHORT = "a"
REVIEW_PLAIN_SHORTS = frozenset("cr")              # gh pr review -c (comment), -r (request changes): bool flags
APPROVING = "approving is the owner's: a session comments or requests changes, never approves"
UNREADABLE_CLUSTER = ("a gh pr review flag cluster the gate cannot read whole: give -b and -F, which take a "
                      "value, on their own")
# A pull request's or an issue's target as a URL: github.com, origin's owner, and nothing else.
TARGET_GROUPS = frozenset({"pr", "issue"})
TARGET_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9-]+)/[A-Za-z0-9._-]+/"
                           r"(?:pull|issues)/\d+(?:[/?#].*)?$", re.IGNORECASE)
LOOKS_LIKE_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.|github\.com/)", re.IGNORECASE)
# Options whose value is text (a body, a title, a closing comment), never the target.
TEXT_OPTIONS = frozenset({"-b", "--body", "-t", "--title", "-F", "--body-file"})
CLOSING_TEXT_OPTIONS = TEXT_OPTIONS | {"-c", "--comment", "-r", "--reason"}     # gh pr|issue close|reopen
CLOSING_ACTIONS = frozenset({"close", "reopen"})
BODY_FILE_OPTIONS = frozenset({"--body-file", "-F"})     # gh pr / gh issue: the body from a file
BODY_FILE_GROUPS = frozenset({"pr", "issue"})

# ── gh api ──────────────────────────────────────────────────────────────────
API_READ_METHODS = frozenset({"GET", "HEAD"})
API_WRITE_METHODS = frozenset({"POST", "PATCH"})
API_VALUE_OPTIONS = frozenset({"-X", "--method", "-H", "--header", "-f", "--raw-field", "-F", "--field", "--input",
                               "-q", "--jq", "-t", "--template", "--cache", "-p", "--preview", "--hostname"})
API_FLAGS = frozenset({"-i", "--include", "--paginate", "--silent", "--slurp", "--verbose"})
API_METHOD_OPTIONS = frozenset({"-X", "--method"})
API_FIELD_OPTIONS = frozenset({"-f", "--raw-field", "-F", "--field"})
API_INPUT_OPTION = "--input"
API_GRAPHQL = "graphql"
API_PLAIN_AREAS = frozenset({"issues", "pulls"})       # POST/PATCH under repos/<o>/<r>/ — not merge (below)
# A path segment that names what a session never writes, whatever the method.
API_REFUSED_SEGMENTS: Mapping[str, str] = {
    "statuses": "sets a commit status, which can pass a required check: only CI reports one",
    "merge": "merges: the owner merges", "merges": "merges: the owner merges",
    "collaborators": "changes who has access", "invitations": "changes who has access",
    "hooks": "changes the repository's webhooks", "keys": "changes deploy keys",
    "secrets": "reads or changes secrets", "variables": "reads or changes variables",
    "protection": "changes branch protection", "rulesets": "changes the repository's rules",
    "environments": "changes deployment environments",
}
API_UNREADABLE_CHARS = ("%", "://")                    # an encoded path, a full URL
BRANCH_REF_PREFIX = "refs/heads/"
API_FILE_VALUE = "@"                                   # ``-F ref=@file`` reads the value from a file
API_OWNER_PLACEHOLDER = "{owner}"                      # gh fills in the current repository's owner
API_REVIEWS = "reviews"
API_APPROVE_EVENT = "APPROVE"


@dataclass(frozen=True)
class ApiCall:
    """A ``gh api`` command as dev-box mode reads it, or why it cannot."""
    methods: Tuple[str, ...] = ()
    endpoints: Tuple[str, ...] = ()
    fields: Tuple[Tuple[str, str], ...] = ()
    has_input: bool = False
    problem: Optional[str] = None


def _subcommand(peeled: Sequence[str]) -> Tuple[str, str]:
    return (peeled[1] if len(peeled) > 1 else "", peeled[2] if len(peeled) > 2 else "")


def _short_cluster_refusal(arg: str) -> Optional[str]:
    """One short-option word of ``gh pr review``: ``-a`` anywhere in it (``-ca``, ``-a=true``)
    approves; a cluster beyond ``-c``/``-r`` (``-cb ok``, ``-bok``) is one the gate cannot read."""
    if not arg.startswith("-") or arg.startswith("--") or len(arg) < 2:
        return None
    letters = arg[1:]
    if APPROVE_SHORT in letters:
        return APPROVING
    return None if len(letters) == 1 or set(letters) <= REVIEW_PLAIN_SHORTS else UNREADABLE_CLUSTER


def _review_flag_refusal(args: Sequence[str]) -> Optional[str]:
    """``gh pr review``: ``--approve`` in any form, or a short cluster that approves or cannot be read."""
    for arg in args:
        why = APPROVING if arg.startswith(APPROVE_LONG) else _short_cluster_refusal(arg)
        if why:
            return why
    return None


def _gh_rule(sub: str, action: str, args: Sequence[str]) -> Optional[str]:
    """A refused group, a refused action, or an approval."""
    approval = _review_flag_refusal(args) if (sub, action) == REVIEW else None
    return GH_REFUSED_GROUPS.get(sub) or GH_REFUSED_ACTIONS.get((sub, action)) or approval


def _target_words(action: str, args: Sequence[str]) -> List[str]:
    """The operands of a ``gh pr``/``gh issue`` command: every word that is not an option or a
    text option's value (a body, a title, a closing comment)."""
    text = CLOSING_TEXT_OPTIONS if action in CLOSING_ACTIONS else TEXT_OPTIONS
    out: List[str] = []
    skip = False
    for word in args:
        if skip or word.startswith("-"):
            skip = not skip and word in text
            continue
        out = [*out, word]
    return out


def _target_refusal(action: str, args: Sequence[str], rules: DevBoxRules) -> Optional[str]:
    """A pull request or an issue named by URL: on github.com, of origin's owner. A number
    or a branch name is this repository's."""
    for word in _target_words(action, args):
        if "://" not in word and not LOOKS_LIKE_URL_RE.match(word):
            continue
        match = TARGET_URL_RE.match(word)
        if match is None:
            return f"{word!r} is not a pull request or an issue on github.com: a session writes only there"
        if not rules.origins_owner(match.group("owner")):
            return f"{word!r} belongs to {match.group('owner')}: a session writes only to origin's owner's repositories"
    return None


def _foreign_write(words: Sequence[str], sub: str, action: str, args: Sequence[str],
                   rules: DevBoxRules) -> Optional[str]:
    """A write aimed at another owner: through ``-R``/``--repo``, or a target URL."""
    repo = _foreign_repo(words, rules)
    if repo is not None:
        return f"writes to {repo!r}: a session writes only to repositories of origin's owner"
    return _target_refusal(action, args, rules) if sub in TARGET_GROUPS else None


def _named_repos(words: Sequence[str]) -> List[str]:
    """Every repository ``-R``/``--repo`` names, anywhere on the line (gh takes it after the subcommand too)."""
    out: List[str] = []
    for index, word in enumerate(words):
        name, glued, value = word.partition("=")
        if word in REPO_OPTIONS:
            out = [*out, words[index + 1] if index + 1 < len(words) else ""]
        elif name == "--repo" and glued:
            out = [*out, value]
        elif word.startswith("-R") and len(word) > 2:
            out = [*out, word[2:]]
    return out


def _foreign_repo(words: Sequence[str], rules: DevBoxRules) -> Optional[str]:
    """The first repository named that is not one of origin's owner's (or cannot be read)."""
    return next((spec for spec in _named_repos(words) if not rules.origins_owner(repo_owner(spec))), None)


def gh_refusal(words: Sequence[str], rules: DevBoxRules) -> Optional[str]:
    """Why this ``gh …`` command is refused in dev-box mode, or None."""
    peeled, _ = strip_globals(words)
    sub, action = _subcommand(peeled)
    if sub == "api":
        return api_refusal(peeled[2:], rules, peeled)
    why = _gh_rule(sub, action, peeled[3:])
    if why:
        return refused(peeled, why)
    writes = gh_writes(peeled) and (sub, action) not in GH_READS_ANYWHERE
    foreign = _foreign_write(words, sub, action, peeled[3:], rules) if writes else None
    if foreign is not None:
        return refused(peeled, foreign)
    if action in GH_ALLOWED.get(sub, frozenset()) or not writes:
        return None
    return refused(peeled, OUTSIDE_THE_LIST)


def publishes(words: Sequence[str]) -> bool:
    """A ``gh`` command line that writes to GitHub (today's rule: anything but a known read)."""
    return gh_writes(strip_globals(words)[0])


def _option_values(args: Sequence[str], names: FrozenSet[str]) -> List[str]:
    """The values of the options ``names`` (``--x v``, ``--x=v``, ``-Xv``)."""
    out: List[str] = []
    for index, word in enumerate(args):
        name, glued, value = word.partition("=")
        if word in names:
            out = [*out, args[index + 1] if index + 1 < len(args) else ""]
        elif glued and name in names:
            out = [*out, value]
        elif not word.startswith("--") and word[:2] in names and len(word) > 2:
            out = [*out, word[2:]]
    return out


def file_values(words: Sequence[str]) -> List[str]:
    """The files a gh command reads by name: ``gh api``'s ``-F key=@file`` (any field form)
    and ``--input``, and a pull request's or an issue's ``--body-file`` / ``-F``."""
    peeled, _ = strip_globals(words)
    sub, _ = _subcommand(peeled)
    if sub == "api":
        named = [value[len(API_FILE_VALUE):] for _, value in read_api(peeled[2:]).fields
                 if value.startswith(API_FILE_VALUE)]
        return [*named, *_option_values(peeled[2:], frozenset({API_INPUT_OPTION}))]
    return _option_values(peeled[3:], BODY_FILE_OPTIONS) if sub in BODY_FILE_GROUPS else []


# ── gh api ──────────────────────────────────────────────────────────────────

def _split_option(word: str, following: Optional[str]) -> Tuple[str, Optional[str], int]:
    """(option, its value, words used) for one option word of ``gh api``."""
    if word.startswith("--"):
        name, glued, value = word.partition("=")
        if glued:
            return name, value, 1
        return (name, following, 2) if name in API_VALUE_OPTIONS else (name, None, 1)
    short = word[:2]
    if len(word) > 2:
        return short, word[2:], 1
    return (short, following, 2) if short in API_VALUE_OPTIONS else (short, None, 1)


def _option_problem(name: str, value: Optional[str]) -> Optional[str]:
    """Why the gate cannot read one option of ``gh api``."""
    if name in API_FLAGS:
        return f"{name} glued to other options, which the gate does not read" if value is not None else None
    if name not in API_VALUE_OPTIONS:
        return f"{name} is an option the gate does not know"
    return None if value is not None else f"{name} with no value"


def _api_parts(args: Sequence[str]) -> Tuple[List[Tuple[str, Optional[str]]], List[str], Optional[str]]:
    """The options (with values) and the operands of ``gh api``, or why it cannot read them."""
    options: List[Tuple[str, Optional[str]]] = []
    operands: List[str] = []
    index = 0
    while index < len(args):
        word = args[index]
        if not word.startswith("-") or word == "-":
            operands, index = [*operands, word], index + 1
            continue
        name, value, used = _split_option(word, args[index + 1] if index + 1 < len(args) else None)
        problem = _option_problem(name, value)
        if problem:
            return [], [], problem
        options, index = [*options, (name, value)], index + used
    return options, operands, None


def _field(value: str) -> Tuple[str, str]:
    """``-f key=value`` → (key, value)."""
    key, _, rest = value.partition("=")
    return key, rest


def read_api(args: Sequence[str]) -> ApiCall:
    """``gh api [options] <endpoint>`` → its methods, endpoint, fields and body."""
    options, operands, problem = _api_parts(list(args))
    if problem:
        return ApiCall(problem=problem)
    return ApiCall(
        methods=tuple(str(value).upper() for name, value in options if name in API_METHOD_OPTIONS),
        endpoints=tuple(operands),
        fields=tuple(_field(str(value)) for name, value in options if name in API_FIELD_OPTIONS),
        has_input=any(name == API_INPUT_OPTION for name, _ in options),
    )


def _method(call: ApiCall) -> Optional[str]:
    """The method gh sends: the one ``-X`` names; POST with a body; GET otherwise.
    None when it is named twice or in a way the gate cannot read (it does not guess)."""
    if len(call.methods) > 1 or any(unreadable(method) for method in call.methods):
        return None
    if call.methods:
        return call.methods[0]
    return "POST" if call.fields or call.has_input else "GET"


def _unreadable_endpoint(text: str) -> bool:
    """A variable, an encoded path, a full URL, a space: the gate cannot say where it goes."""
    return unreadable(text) or any(mark in text for mark in API_UNREADABLE_CHARS) or any(c.isspace() for c in text)


def _endpoint(endpoints: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """The endpoint's path segments, or None when the gate cannot read it: none or many,
    an unreadable one, a ``..``."""
    if len(endpoints) != 1:
        return None
    text = endpoints[0].split("?", 1)[0].split("#", 1)[0]
    if _unreadable_endpoint(text):
        return None
    parts = tuple(part for part in text.split("/") if part)
    return parts if parts and ".." not in parts else None


def api_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``gh api``: reads run; a write runs only where a ticket's pull request needs it."""
    call = read_api(args)
    if call.problem:
        return refused(words, f"gh api: {call.problem}")
    method = _method(call)
    if method is None:
        return refused(words, "gh api names its method twice, or in a way the gate cannot read")
    if method in API_READ_METHODS:
        return None
    if method == "DELETE":
        return refused(words, "gh api DELETE removes something: refused on every endpoint")
    path = _endpoint(call.endpoints)
    if path is None:
        return refused(words, "gh api names no endpoint the gate can read (one path, no variable, no full URL)")
    why = _write_refusal(path, method, call, rules)
    return refused(words, why) if why else None


def _named_refusal(path: Sequence[str], method: str) -> Optional[str]:
    """graphql, or a segment that names what a session never writes."""
    low = [part.lower() for part in path]
    if low == [API_GRAPHQL]:
        return "gh api graphql can merge, delete and change access in one call: use the gh subcommands"
    named = next((API_REFUSED_SEGMENTS[part] for part in low if part in API_REFUSED_SEGMENTS), None)
    return f"gh api {method} {'/'.join(path)} {named}" if named else None


def _not_on_the_list(method: str, path: Sequence[str]) -> str:
    return (f"gh api {method} {'/'.join(path)} is not one of the API writes a dev-box session makes "
            "(POST/PATCH issues and pulls, its own branch refs)")


def _plain_write(low: Sequence[str], method: str) -> bool:
    """POST/PATCH under ``repos/<o>/<r>/issues`` or ``…/pulls``."""
    return low[3] in API_PLAIN_AREAS


def _owner_refusal(path: Sequence[str], rules: DevBoxRules) -> Optional[str]:
    """A write under ``repos/<owner>/…`` goes only to origin's owner (or gh's ``{owner}``)."""
    if len(path) < 2 or path[0].lower() != "repos" or path[1].lower() == API_OWNER_PLACEHOLDER:
        return None
    if rules.origins_owner(path[1]):
        return None
    return f"gh api writes to {path[1]}/…: a session writes only to repositories of origin's owner"


def _review_refusal(path: Sequence[str], call: ApiCall) -> Optional[str]:
    """A pull request review through the API: a comment or a change request, never an approval."""
    if API_REVIEWS not in (part.lower() for part in path):
        return None
    if call.has_input:
        return "gh api --input hides a review's event, and approving is the owner's"
    events = [value for key, value in call.fields if key == "event"]
    return APPROVING if any(_approving_event(event) for event in events) else None


def _approving_event(event: str) -> bool:
    """``event=APPROVE`` in any case, or an event the gate cannot read (a file, a variable)."""
    return event.upper() == API_APPROVE_EVENT or event.startswith(API_FILE_VALUE) or unreadable(event)


def _repo_write(low: Sequence[str], method: str) -> bool:
    """POST or PATCH under ``repos/<owner>/<repo>/<area>``."""
    return method in API_WRITE_METHODS and len(low) >= 4 and low[0] == "repos"


def _write_refusal(path: Sequence[str], method: str, call: ApiCall, rules: DevBoxRules) -> Optional[str]:
    """Why a ``gh api`` write is refused, or None when it is one a ticket needs."""
    named = _named_refusal(path, method) or _owner_refusal(path, rules) or _review_refusal(path, call)
    if named:
        return named
    low = [part.lower() for part in path]
    if not _repo_write(low, method):
        return _not_on_the_list(method, path)
    if low[3] == "git" and low[4:5] == ["refs"]:
        return _git_refs_refusal(path, method, call, rules)
    return None if _plain_write(low, method) else _not_on_the_list(method, path)


def _branch_refusal(name: str, rules: DevBoxRules) -> Optional[str]:
    """A branch ref a session may create or move: named, and not protected."""
    if not name or rules.protected_branch(name):
        return (f"{name or 'a ref with no name'} is a protected branch (the default and release branches are "
                "the owner's)")
    return None


def _created_ref_refusal(call: ApiCall, rules: DevBoxRules) -> Optional[str]:
    """``POST repos/<o>/<r>/git/refs``: only a branch, named in ``-f ref=…``."""
    if call.has_input:
        return "gh api --input hides which ref it creates"
    refs = [value for key, value in call.fields if key == "ref"]
    if len(refs) != 1 or refs[0].startswith(API_FILE_VALUE) or unreadable(refs[0]):
        return "the gate cannot read which ref it creates (give it as -f ref=refs/heads/<branch>)"
    if not refs[0].startswith(BRANCH_REF_PREFIX):
        return "gh api creates branch refs only (ref=refs/heads/<branch>)"
    return _branch_refusal(refs[0][len(BRANCH_REF_PREFIX):], rules)


def _git_refs_refusal(path: Sequence[str], method: str, call: ApiCall, rules: DevBoxRules) -> Optional[str]:
    """``repos/<o>/<r>/git/refs``: create a branch (POST) or move one (PATCH ``…/heads/<branch>``)."""
    if method == "POST" and len(path) == 5:
        return _created_ref_refusal(call, rules)
    if method == "PATCH" and len(path) >= 7 and path[5].lower() == "heads":
        return _branch_refusal("/".join(path[6:]), rules)
    return "gh api writes git/refs only to create a branch (POST git/refs) or move one (PATCH git/refs/heads/<branch>)"


__all__ = [
    "API_REFUSED_SEGMENTS", "ApiCall", "GH_ALLOWED", "GH_REFUSED_ACTIONS", "GH_REFUSED_GROUPS", "api_refusal",
    "file_values", "gh_refusal", "publishes", "read_api",
]
