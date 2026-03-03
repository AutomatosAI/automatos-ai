"""
Centralized git URL and branch validation — PRD-70 FIX-01.

All git subprocess calls MUST use these functions. This prevents:
  - Command injection via --upload-pack, -c, etc.
  - SSRF via file://, ssh://, or arbitrary-domain URLs
  - Branch-name argument injection (leading dashes)

Usage:
    from core.security.git_sanitizer import (
        validate_git_url, validate_branch, build_git_clone_cmd,
    )
"""

import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

ALLOWED_GIT_DOMAINS: list[str] = [
    "github.com",
    "gitlab.com",
    "bitbucket.org",
]

# Git arguments that enable command execution
_DANGEROUS_GIT_FLAGS = frozenset({
    "--upload-pack",
    "--receive-pack",
    "-c",
    "--config",
    "--exec-path",
    "--template",
})

# Valid git branch/tag characters: alphanum, dot, slash, underscore, hyphen.
# Must NOT start with a dash.
_BRANCH_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/\-]{0,254}$")


def validate_git_url(
    url: str,
    allowed_domains: list[str] | None = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a git URL: HTTPS only, domain allowlist, no argument injection.

    Returns (True, None) on success or (False, reason) on failure.
    """
    if not url or not isinstance(url, str):
        return False, "URL is required"

    if url.startswith("-"):
        return False, "URL must not start with a dash"

    # Check for dangerous flag substrings embedded anywhere
    url_lower = url.lower()
    for flag in _DANGEROUS_GIT_FLAGS:
        if flag in url_lower:
            return False, f"URL contains disallowed git flag: {flag}"

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    if parsed.scheme not in ("https",):
        return False, f"Only HTTPS URLs are allowed (got: {parsed.scheme or 'none'})"

    if parsed.username or parsed.password:
        return False, "URL must not contain embedded credentials"

    hostname = (parsed.hostname or "").lower()
    domains = allowed_domains or ALLOWED_GIT_DOMAINS
    if not any(hostname == d or hostname.endswith("." + d) for d in domains):
        return False, f"Domain not in allowlist: {hostname}"

    return True, None


def validate_branch(branch: str) -> Tuple[bool, Optional[str]]:
    """Validate a git branch/tag name.

    Returns (True, None) on success or (False, reason) on failure.
    """
    if not branch or not isinstance(branch, str):
        return False, "Branch name is required"

    branch = branch.strip()

    if branch.startswith("-"):
        return False, "Branch name must not start with a dash"

    if ".." in branch:
        return False, "Branch name must not contain '..'"

    if "@{" in branch:
        return False, "Branch name must not contain '@{'"

    if not _BRANCH_PATTERN.match(branch):
        return False, "Branch contains invalid characters"

    return True, None


def build_git_clone_cmd(
    url: str,
    target_dir: str,
    branch: str | None = None,
    depth: int = 50,
) -> list[str]:
    """Build a safe git clone command with ``--`` separator.

    Callers MUST validate url and branch before calling this function.
    The ``--`` separator ensures no positional arg is parsed as a flag.
    """
    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.append("--")  # End of options
    cmd.extend([url, target_dir])
    return cmd
