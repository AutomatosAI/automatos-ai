"""What one simple command really runs, read from its words (PRD-253 S0.3).

Pure and stdlib-only: the gate (``policy.py``) asks, this answers. Three readings
the word-level checks never had:

* **an inline command line** — a shell's ``-c`` and ``eval`` run a line given as an
  argument: ``bash -c 'git push'`` read as the verb ``bash``;
* **gh writes** — ``gh`` is read-only in a session: every subcommand not known to
  be a read is a write (an issue, a pull request, a release, an SSH key, an
  extension, an alias), and ``gh api`` writes with a body or any method but
  GET/HEAD;
* **commands the gate cannot see through** — a command name only known when it
  runs (``git${IFS}push``, ``$CMD``), code an interpreter takes inline
  (``python3 -c``), a shell reading its commands from input (``echo … | sh``).

In Auto mode — the local edition's default — an unlisted verb runs unasked, so a
command the gate cannot read must be a card, not a pass (W0 security review).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence

SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "mksh", "yash", "fish", "tcsh", "csh",
                    "pwsh", "powershell", "nu", "elvish", "xonsh"})
SOURCE_VERBS = frozenset({"source", "."})
_SHELL_COMMAND_FLAG_RE = re.compile(r"^(?:-[A-Za-z]*c[A-Za-z]*|-[Cc]ommand)$")     # -c, -lc, -ec, pwsh -Command
_INLINE_INTERPRETER_RE = re.compile(r"^(python(\d+(\.\d+)?)?|node|nodejs|deno|bun|perl|ruby|php|lua|osascript)$")
INLINE_CODE_WORDS = frozenset({"-c", "-e", "-E", "--eval", "-p", "--print", "-r", "eval"})

# ``gh`` in a session reads. Subcommand → the actions that only read.
GH_READ_ONLY: Dict[str, FrozenSet[str]] = {
    "pr": frozenset({"view", "list", "diff", "checks", "status"}),
    "issue": frozenset({"view", "list", "status"}),
    "repo": frozenset({"view", "list"}),
    "run": frozenset({"view", "list", "watch"}),
    "workflow": frozenset({"view", "list"}),
    "release": frozenset({"view", "list"}),
    "gist": frozenset({"view", "list"}),
    "label": frozenset({"list"}),
    "secret": frozenset({"list"}),
    "variable": frozenset({"list", "get"}),
    "cache": frozenset({"list"}),
    "project": frozenset({"view", "list", "field-list", "item-list"}),
    "ruleset": frozenset({"view", "list", "check"}),
    "codespace": frozenset({"list", "view"}),
    "org": frozenset({"list"}),
    "auth": frozenset({"status"}),
    "config": frozenset({"get", "list"}),
    "extension": frozenset({"list", "search", "browse"}),
    "alias": frozenset({"list"}),
    "attestation": frozenset({"verify"}),
}
GH_READ_ONLY_GROUPS = frozenset({"search", "browse", "status", "help", "version", "completion",
                                 "--version", "--help", "-h"})
GH_API_READ_METHODS = frozenset({"GET", "HEAD"})
_GH_API_BODY_RE = re.compile(r"^(?:-[fF].*|--(?:raw-)?field(?:=.*)?|--input(?:=.*)?)$")
_GH_API_METHOD_RE = re.compile(r"^(?:-X|--method)(?:=?(?P<value>.+))?$")
# A script argument the gate can name: a plain path, not a device, a stream or an expansion.
_PLAIN_SCRIPT_RE = re.compile(r"^[\w./~+@%:,=-]+$")


def inline_commands(words: Sequence[str]) -> List[str]:
    """The command line a shell's ``-c`` (``-Command``) or an ``eval`` runs."""
    if not words:
        return []
    head = Path(words[0]).name
    if head == "eval":
        return [" ".join(words[1:])] if len(words) > 1 else []
    if head not in SHELLS:
        return []
    flag = next((i for i, word in enumerate(words[1:-1], start=1) if _SHELL_COMMAND_FLAG_RE.match(word)), None)
    return [words[flag + 1]] if flag is not None else []


def _gh_api_method(rest: Sequence[str], index: int) -> Optional[str]:
    match = _GH_API_METHOD_RE.match(rest[index])
    if match is None:
        return None
    return match.group("value") or (rest[index + 1] if index + 1 < len(rest) else "")


def gh_api_writes(words: Sequence[str]) -> bool:
    """``gh api`` that sends a body (a field, ``--input``: gh then POSTs) or names a
    method other than GET or HEAD."""
    rest = list(words[2:])
    for index, word in enumerate(rest):
        if _GH_API_BODY_RE.match(word):
            return True
        method = _gh_api_method(rest, index)
        if method is not None and method.upper() not in GH_API_READ_METHODS:
            return True
    return False


def gh_writes(words: Sequence[str]) -> bool:
    """``gh`` that is not a known read: an unknown subcommand, an alias or an
    extension counts as a write — the only safe reading of a GitHub CLI."""
    if not words or Path(words[0]).name != "gh":
        return False
    sub = words[1] if len(words) > 1 else ""
    if sub == "api":
        return gh_api_writes(words)
    if sub in GH_READ_ONLY_GROUPS:
        return False
    action = words[2] if len(words) > 2 else ""
    return action not in GH_READ_ONLY.get(sub, frozenset())


def _names_a_script(args: Sequence[str]) -> bool:
    """The first non-option argument is a file the gate can name."""
    script = next((a for a in args if not a.startswith("-")), None)
    return bool(script) and not script.startswith("/dev/") and bool(_PLAIN_SCRIPT_RE.match(script))


def _dynamic_verb(verb: str) -> bool:
    """A command name only decided when it runs: an expansion, or a whole line in one word."""
    return "$" in verb or "`" in verb or any(c.isspace() for c in verb)


def _hidden_input(head: str, words: Sequence[str], args: Sequence[str]) -> Optional[str]:
    """A shell or ``source`` reading its commands from somewhere the gate cannot read."""
    if head in SHELLS and not inline_commands(words) and not _names_a_script(args):
        return f"{head} reads its commands from input the gate cannot read"
    if head in SOURCE_VERBS and not _names_a_script(args):
        return f"'{head}' reads commands from a stream the gate cannot read"
    return None


def opaque_reason(words: Sequence[str]) -> Optional[str]:
    """Why the gate cannot see what this command runs, or None when it can."""
    if not words:
        return None
    if _dynamic_verb(words[0]):
        return f"the command's own name is only decided when it runs ({words[0]!r})"
    head, args = Path(words[0]).name, list(words[1:])
    if _INLINE_INTERPRETER_RE.match(head) and any(a in INLINE_CODE_WORDS for a in args):
        return f"{head} runs code given inline, which the gate cannot read"
    return _hidden_input(head, words, args)


__all__ = [
    "GH_API_READ_METHODS", "GH_READ_ONLY", "GH_READ_ONLY_GROUPS", "INLINE_CODE_WORDS", "SHELLS", "SOURCE_VERBS",
    "gh_api_writes", "gh_writes", "inline_commands", "opaque_reason",
]
