"""The tool policy a session runs under (PRD-234 §Design 4, v1).

Decided here, enforced through the ``PreToolUse`` hook — never left to a TUI
prompt nobody watches. CLI adapter design §4.2: the policy reads a
``ToolIntent`` — what the call DOES (a class plus the paths/command it touches),
never what the CLI calls the tool — so the rules are the same for every CLI:

* file reads and writes — allowed inside the session's working directory (and
  its git worktree), denied outside;
* a shell command — allowed when it matches the ticket's allowlist (agent
  configuration ``allowed_tools``, else the defaults below); ``git push`` and
  friends are always denied (sessions never push — the manager integrates);
  anything else is HELD for the approvals inbox;
* web/search tools and benign bookkeeping — allowed;
* everything else (MCP tools, Task, an unknown tool) — denied by default; the
  operator's own CLI settings are the other half of the surface.

Pure functions: the session hands in the intent and its context, gets a
decision back.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .adapters.base import ToolClass, ToolIntent

# Sessions never publish. The manager (Auto) integrates.
NEVER_ALLOWED_BASH = (
    re.compile(r"(^|[;&|]\s*)git\s+push\b"),
    re.compile(r"(^|[;&|]\s*)git\s+remote\s+(add|set-url)\b"),
    re.compile(r"(^|[;&|]\s*)gh\s+(pr|release)\s+(create|merge|edit)\b"),
    re.compile(r"(^|[;&|]\s*)(sudo|su)\b"),
    re.compile(r"(^|[;&|]\s*)rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$)"),
    re.compile(r"(^|[;&|]\s*)curl\b.*\|\s*(ba|z)?sh\b"),
)

# Read-only git + the usual build/test verbs a code ticket needs.
DEFAULT_BASH_ALLOW = (
    "git status", "git diff", "git log", "git show", "git branch", "git add",
    "git commit", "git stash", "git restore", "git checkout -b", "git switch -c",
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "which", "echo",
    "python -m pytest", "python3 -m pytest", "pytest", "npm test", "npm run", "pnpm test",
    "pnpm run", "yarn test", "make test", "make lint", "cargo test", "go test",
    "ruff", "black --check", "mypy", "tsc", "eslint", "vitest",
)

# A code ticket must be able to RUN what it just wrote — that is "build and
# test", not "publish". An interpreter may run a file inside the session
# directory; inline code (``python -c``, ``node -e``) stays refused so the
# never-allowed list cannot be bypassed inside a string. This is a guardrail
# against accidents on the user's own machine, not a sandbox.
_INTERPRETER_RE = re.compile(r"^(python(\d+(\.\d+)?)?|node)$")
INLINE_CODE_FLAGS = frozenset({"-c", "-e", "--eval", "-p", "--print"})
OWN_CODE_MODULES = frozenset({"doctest", "unittest", "pytest", "py_compile"})


@dataclass
class PolicyContext:
    cwd: Path
    allowed_bash: Sequence[str] = field(default_factory=lambda: DEFAULT_BASH_ALLOW)
    ask_bash: Sequence[str] = ()          # prefixes routed to the approvals inbox
    extra_dirs: Sequence[Path] = ()       # e.g. the git worktree the session runs in


@dataclass
class Decision:
    behavior: str            # allow | deny | ask
    reason: str = ""

    @property
    def allow(self) -> bool:
        return self.behavior == "allow"


def _inside(path_str: str, roots: Iterable[Path]) -> bool:
    try:
        p = Path(path_str).expanduser()
        for root in roots:
            candidate = (p if p.is_absolute() else root / p).resolve()
            try:
                candidate.relative_to(root.resolve())
                return True
            except ValueError:
                continue
    except (OSError, RuntimeError):
        return False
    return False


def _first_words(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return " ".join(parts[:3])


def _matches_prefix(command: str, prefixes: Sequence[str]) -> bool:
    stripped = command.strip()
    for prefix in prefixes:
        if stripped == prefix or stripped.startswith(prefix + " "):
            return True
    return False


def _split_compound(command: str) -> List[str]:
    # A compound command is judged by EVERY segment (munder/Claude's own rule).
    return [seg.strip() for seg in re.split(r"&&|\|\||;|\|", command) if seg.strip()]


def _absolute_args_inside(args: Sequence[str], roots: Iterable[Path]) -> bool:
    """Relative arguments resolve under the session directory by construction
    ('..' is refused before we get here); every ABSOLUTE path must sit inside it."""
    roots = list(roots)
    return all(_inside(a, roots) for a in args if a.startswith("/") or a.startswith("~"))


def _runs_own_code(segment: str, roots: Iterable[Path]) -> bool:
    """``cd`` within the session directory, or an interpreter run on a file inside it."""
    roots = list(roots)
    try:
        words = shlex.split(segment)
    except ValueError:
        return False
    if not words:
        return False
    head = Path(words[0]).name  # tolerate /usr/bin/python3
    if head == "cd":
        return len(words) == 2 and _inside(words[1], roots)
    if not _INTERPRETER_RE.match(head):
        return False
    args = words[1:]
    if not args or any(a in INLINE_CODE_FLAGS for a in args):
        return False
    if args[0] == "-m":
        return len(args) >= 2 and args[1] in OWN_CODE_MODULES and _absolute_args_inside(args[2:], roots)
    if args[0].startswith("-"):
        return False  # unknown interpreter flag: not a plain "run this file"
    return _inside(args[0], roots) and _absolute_args_inside(args[1:], roots)


def decide_bash(command: str, ctx: PolicyContext) -> Decision:
    for pattern in NEVER_ALLOWED_BASH:
        if pattern.search(command):
            return Decision("deny", f"never allowed in a session: {_first_words(command)!r} (sessions do not push or escalate)")
    if ".." in command and re.search(r"(^|[\s'\"=:;|&(])\.\.([/\\]|[\s'\");|&]|$)", command):
        return Decision("deny", "path traversal ('..') in a shell command")
    segments = _split_compound(command)
    roots = [ctx.cwd, *ctx.extra_dirs]
    if all(_matches_prefix(seg, ctx.allowed_bash) or _runs_own_code(seg, roots) for seg in segments):
        return Decision("allow")
    if any(_matches_prefix(seg, ctx.ask_bash) for seg in segments):
        return Decision("ask", f"{_first_words(command)!r} needs the operator's approval")
    # PRD-235 W2 S3: outside the allowlist is a QUESTION for the operator, not a
    # refusal — the session holds the call while a card is shown on the ticket's
    # Canvas; no answer in time is a deny (the ticket lands in review, as before).
    return Decision("ask", f"{_first_words(command)!r} is outside this ticket's Bash allowlist")


def decide(intent: ToolIntent, ctx: PolicyContext) -> Decision:
    roots = [ctx.cwd, *ctx.extra_dirs]
    if intent.cls in (ToolClass.FILE_READ, ToolClass.FILE_WRITE):
        if not intent.paths:
            return Decision("allow")  # a search without a path works in cwd
        for target in intent.paths:
            if not _inside(str(target), roots):
                return Decision("deny", f"{intent.tool} outside the session directory: {target}")
        return Decision("allow")
    if intent.cls is ToolClass.SHELL:
        return decide_bash(str(intent.command or ""), ctx)
    if intent.cls in (ToolClass.WEB, ToolClass.BENIGN):
        return Decision("allow")
    return Decision("deny", f"tool {intent.tool!r} is not enabled for session tickets")


def bash_allowlist_from_config(configured: Optional[Iterable[str]]) -> Sequence[str]:
    """The ticket's Bash allowlist: the agent's configured prefixes on top of the defaults."""
    extra = [c.strip() for c in (configured or []) if isinstance(c, str) and c.strip()]
    return tuple(DEFAULT_BASH_ALLOW) + tuple(extra)
