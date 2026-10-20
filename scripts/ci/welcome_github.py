"""The welcome bot's GitHub calls (scripts/ci/welcome_contributor.py): REST and GraphQL over urllib.

Stdlib only, so the workflow installs nothing. Every failed call is logged with the
method, the path and GitHub's answer, then raised as :class:`GitHubError`; the
entry point turns that into a red run. The tests hand the bot a fake with the same
methods as :class:`GitHubApi`, so nothing here runs without a network.

Reads that decide an assignment use the REST issue endpoints, which answer from
GitHub's primary store. Search answers from an index that can lag by seconds, so it
only feeds lists where a lag is harmless, and counts a REST read backs up.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Protocol

logger = logging.getLogger("welcome_contributor.github")

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "automatos-welcome-bot"
HTTP_TIMEOUT_SECONDS = 20
ERROR_DETAIL_CHARS = 500
PER_PAGE = 100
MAX_COMMENT_PAGES = 10
MAX_PINNED_ISSUES = 3  # GitHub pins at most three issues per repository.

PINNED_ISSUES_QUERY = (
    "query($owner: String!, $name: String!, $first: Int!) {"
    " repository(owner: $owner, name: $name) { pinnedIssues(first: $first) { nodes { issue { number } } } } }"
)


class GitHubError(RuntimeError):
    """A GitHub API call failed; the message names the call."""


class GitHubApi(Protocol):
    """The GitHub calls the bot makes; the tests hand it a fake with the same methods."""

    def get_issue(self, number: int) -> Dict[str, Any]:
        """The issue as it is now: state, labels, assignees, title."""

    def list_issues(self, filters: Mapping[str, str]) -> List[Dict[str, Any]]:
        """The newest ``PER_PAGE`` issues and pull requests matching the REST list ``filters``."""

    def list_issue_comments(self, number: int) -> List[Dict[str, Any]]:
        """Every comment on the issue or pull request, oldest first."""

    def add_assignee(self, number: int, login: str) -> List[str]:
        """Add ``login`` as an assignee; the logins GitHub lists as assigned afterwards."""

    def post_comment(self, number: int, body: str) -> None:
        """Post ``body`` as a comment."""

    def search_count(self, query: str) -> int:
        """How many issues or pull requests match the search ``query``."""

    def search_issues(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Up to ``limit`` matches of the search ``query``, newest first."""

    def pinned_issue_numbers(self) -> FrozenSet[int]:
        """The numbers of the repository's pinned issues."""


class GitHubClient:
    """:class:`GitHubApi` over GitHub's REST and GraphQL APIs, with urllib (nothing to install)."""

    def __init__(
        self, token: str, repo: str, api_root: str = API_ROOT, opener: Callable[..., Any] = urllib.request.urlopen
    ) -> None:
        self._token = token
        self._repo = repo
        self._root = api_root.rstrip("/")
        self._open = opener

    def get_issue(self, number: int) -> Dict[str, Any]:
        """GET the issue, so a decision rests on its state now, not on the event's snapshot."""
        issue = self._call("GET", f"/repos/{self._repo}/issues/{number}")
        if not isinstance(issue, dict):
            raise GitHubError(f"GET /repos/{self._repo}/issues/{number}: the answer is not an issue")
        return issue

    def list_issues(self, filters: Mapping[str, str]) -> List[Dict[str, Any]]:
        """One page of the repository's issue list (pull requests included), newest first."""
        query = {**filters, "per_page": PER_PAGE, "sort": "created", "direction": "desc"}
        items = self._call("GET", f"/repos/{self._repo}/issues", query) or []
        return [item for item in items if isinstance(item, dict)]

    def list_issue_comments(self, number: int) -> List[Dict[str, Any]]:
        """Every comment on the thread, ``PER_PAGE`` a page, at most ``MAX_COMMENT_PAGES`` pages."""
        comments: List[Dict[str, Any]] = []
        for page in range(1, MAX_COMMENT_PAGES + 1):
            path = f"/repos/{self._repo}/issues/{number}/comments"
            batch = list(self._call("GET", path, {"per_page": PER_PAGE, "page": page}) or [])
            comments = comments + batch
            if len(batch) < PER_PAGE:
                return comments
        logger.warning("#%s has over %s comments; the duplicate check read those only.", number, len(comments))
        return comments

    def add_assignee(self, number: int, login: str) -> List[str]:
        """POST the assignee; GitHub answers with the issue, whose assignees say if it took."""
        path = f"/repos/{self._repo}/issues/{number}/assignees"
        issue = self._call("POST", path, body={"assignees": [login]}) or {}
        return [str(person.get("login") or "") for person in issue.get("assignees") or [] if isinstance(person, Mapping)]

    def post_comment(self, number: int, body: str) -> None:
        """POST a comment on the issue or pull request."""
        self._call("POST", f"/repos/{self._repo}/issues/{number}/comments", body={"body": body})

    def search_count(self, query: str) -> int:
        """The search's ``total_count``."""
        return int(self._search(query, 1).get("total_count") or 0)

    def search_issues(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """The search's first ``limit`` items, newest first."""
        return [item for item in self._search(query, limit).get("items") or [] if isinstance(item, Mapping)]

    def pinned_issue_numbers(self) -> FrozenSet[int]:
        """The pinned issues, from GraphQL (REST has no such list)."""
        owner, name = self._repo.split("/", 1)
        variables = {"owner": owner, "name": name, "first": MAX_PINNED_ISSUES}
        result = self._call("POST", "/graphql", body={"query": PINNED_ISSUES_QUERY, "variables": variables}) or {}
        if result.get("errors"):
            logger.error("GitHub GraphQL pinnedIssues answered errors: %s", result["errors"])
            raise GitHubError("POST /graphql (pinnedIssues): the answer carries errors")
        repository = (result.get("data") or {}).get("repository") or {}
        nodes = (repository.get("pinnedIssues") or {}).get("nodes") or []
        return frozenset(int(node["issue"]["number"]) for node in nodes if (node or {}).get("issue"))

    def _search(self, query: str, limit: int) -> Dict[str, Any]:
        params = {"q": query, "per_page": limit, "sort": "created", "order": "desc"}
        result = self._call("GET", "/search/issues", params) or {}
        if result.get("incomplete_results"):
            logger.warning("GitHub search stopped early for %r; the answer may be short.", query)
        return result

    def _headers(self, has_body: bool) -> Dict[str, str]:
        base = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "Authorization": f"Bearer {self._token}",
            "User-Agent": USER_AGENT,
        }
        return {**base, "Content-Type": "application/json"} if has_body else base

    def _call(
        self, method: str, path: str, query: Optional[Mapping[str, Any]] = None, body: Optional[Mapping[str, Any]] = None
    ) -> Any:
        url = f"{self._root}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method, headers=self._headers(data is not None))
        try:
            with self._open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = (error.read() if error.fp else b"").decode("utf-8", "replace")[:ERROR_DETAIL_CHARS]
            logger.error("GitHub %s %s answered HTTP %s: %s", method, path, error.code, detail)
            raise GitHubError(f"{method} {path}: HTTP {error.code}") from error
        except urllib.error.URLError as error:
            logger.error("GitHub %s %s failed: %s", method, path, error.reason)
            raise GitHubError(f"{method} {path}: {error.reason}") from error
        return json.loads(raw) if raw else None
