"""What a dev-box session never touches on GitHub: the protected names (dev-box mode).

Dev-box mode (``--dev-box``) lets a Claude Code session publish its own work on the
owner's development machine: commit, push its branch, open a pull request. GitHub is
the wall: the session pushes with a fine-grained token of the owner's own account, and
the repository's ruleset (admin bypass "pull requests only") stops a direct push to the
default branch and a merge without review. The gate is the second layer, and these are
the names it refuses on its own: the default and release branches, origin's own default
branch, release tags, and repositories of any owner but origin's.

Pure data and pure functions. ``dev_box_git.py`` and ``dev_box_gh.py`` read them;
``dev_box.py`` builds the rules for one repository.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Union

PROTECTED_BRANCHES: FrozenSet[str] = frozenset({
    "main", "master", "mainline", "develop", "dev", "trunk", "production", "prod", "live", "stable", "next",
    "staging", "uat", "qa", "gh-pages",
})
# release, release/1.2, releases, hotfix/x, main.2: every branch whose name starts so.
PROTECTED_BRANCH_PREFIXES = ("release", "hotfix", "main.")
RELEASE_TAG_PATTERNS = ("v[0-9]*", "release*")
_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
# A remote a session may add: a github.com repository (https, scp-style or ssh://).
# No user information in the URL: a URL can carry a credential of its own.
_GITHUB_URL_RE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/[A-Za-z0-9._-]+?/?$",
    re.IGNORECASE,
)
# ``gh -R``'s repository: OWNER/REPO, HOST/OWNER/REPO, or a URL.
_REPO_URL_PREFIX_RE = re.compile(r"^[a-z]+://", re.IGNORECASE)
# A word whose value is only known when it runs: a variable, a substitution (``$(…)``
# arrives as ``$_``, ``shell_text.SUBSTITUTION_MARK``), a backtick — or that the shell
# turns into several words: a brace expansion (``{feat,main}``, ``{1..3}``; gh's own
# ``{owner}`` placeholder is one word and stays readable).
UNREADABLE_CHARS = ("$", "`")
_BRACE_EXPANSION_RE = re.compile(r"\{[^{}]*(?:,|\.\.)[^{}]*\}")
SHOWN_WORDS = 4
SHOWN_CHARS = 80
# The backend files every reason holding these words as a refusal by the gate, never
# as a hold (``orchestrator/services/session_denials.py``): a dev-box refusal must not
# send its ticket to review with nothing for a person to answer.
REFUSAL_PREFIX = "never allowed in a session"


@dataclass(frozen=True)
class DevBoxRules:
    """One repository's rules: the branch names it protects (lower case), and the
    github.com owner every remote and every GitHub write must share with origin (None:
    origin is not on github.com, or git could not say, and none passes)."""
    protected: FrozenSet[str] = PROTECTED_BRANCHES
    origin_owner: Optional[str] = None

    def protected_branch(self, name: str) -> bool:
        """A protected branch name, in any case (APFS and Windows fold it)."""
        low = name.lower()
        return low in self.protected or low.startswith(PROTECTED_BRANCH_PREFIXES)

    def release_tag(self, name: str) -> bool:
        """A name shaped like a release tag: ``v1.2.3``, ``release-7``."""
        low = name.lower()
        return any(fnmatch.fnmatchcase(low, pattern) for pattern in RELEASE_TAG_PATTERNS)

    def origins_owner(self, owner: Optional[str]) -> bool:
        """``owner`` is origin's own (case-insensitive); never when origin's is unknown."""
        return bool(owner) and self.origin_owner is not None and owner.lower() == self.origin_owner


def rules(default_branch: Optional[str] = None, origin_owner: Optional[str] = None) -> DevBoxRules:
    """The rules for one repository: the named list, plus origin's own default branch
    when the checkout names one, and origin's owner when it is on github.com."""
    named = default_branch if default_branch and _BRANCH_NAME_RE.match(default_branch) else None
    extra = frozenset({named.lower()}) if named else frozenset()
    return DevBoxRules(protected=PROTECTED_BRANCHES | extra, origin_owner=origin_owner.lower() if origin_owner else None)


def github_owner(url: str) -> Optional[str]:
    """The owner of a github.com repository URL, lower case; None for anything else."""
    match = _GITHUB_URL_RE.match(url.strip())
    return match.group("owner").lower() if match else None


def repo_owner(spec: str) -> Optional[str]:
    """The owner ``gh -R`` names: OWNER/REPO, HOST/OWNER/REPO or a URL; None when unreadable."""
    parts = [part for part in _REPO_URL_PREFIX_RE.sub("", spec.strip()).split("/") if part]
    if unreadable(spec) or len(parts) not in (2, 3):
        return None
    return parts[-2].lower()


def unreadable(word: str) -> bool:
    """The gate cannot know this word's value, or how many words it becomes, before it runs."""
    return any(char in word for char in UNREADABLE_CHARS) or bool(_BRACE_EXPANSION_RE.search(word))


def shown(words: Sequence[str]) -> str:
    """The start of a command, as a refusal quotes it."""
    return " ".join(words[:SHOWN_WORDS])[:SHOWN_CHARS]


def refused(command: Union[str, Sequence[str]], why: str) -> str:
    """A refusal in the gate's words, naming the dev-box rule that refuses it."""
    what = command if isinstance(command, str) else shown(command)
    return f"{REFUSAL_PREFIX}: {what!r} — {why} (dev-box mode)"


__all__ = [
    "DevBoxRules", "PROTECTED_BRANCHES", "PROTECTED_BRANCH_PREFIXES", "REFUSAL_PREFIX", "RELEASE_TAG_PATTERNS",
    "github_owner", "refused", "repo_owner", "rules", "shown", "unreadable",
]
