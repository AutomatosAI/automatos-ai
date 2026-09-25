"""The tool policy a session runs under (PRD-234 §Design 4; PRD-245 S0.1/S0.2).

Decided here, enforced through the ``PreToolUse`` hook — never left to a TUI
prompt nobody watches. CLI adapter design §4.2: the policy reads a
``ToolIntent`` — what the call DOES (a class plus the paths/command it touches),
never what the CLI calls the tool — so the rules are the same for every CLI:

* file reads and writes — allowed inside the session's working directory (and
  its git worktree), denied outside;
* a shell command — read the way a shell reads it: tokenised quote-aware, cut
  into simple commands on unquoted ``&&``, ``||``, ``;``, ``|``, ``&`` and
  newlines, each judged by its verb against the ticket's allowlist (agent
  configuration ``allowed_tools`` on top of the defaults below). Shell keywords
  and leading ``NAME=value`` assignments are peeled off to reach the verb;
  ``git -C <path> <sub…>`` is ``git <sub…>`` run inside the roots; every
  absolute (or ``~``) path an allowed verb names, and every redirection target,
  must sit inside the session's roots — outside is a refusal, never a question.
  ``git push`` and friends are always denied (sessions never push — the manager
  integrates); a verb outside the allowlist is HELD for the operator;
* an Automatos tool over the loopback MCP bridge (PRD-245 W1) — allowed when
  its name is on the ticket's own list, denied otherwise (the backend enforces
  the scope inside each one; the gate enforces the surface);
* web/search tools and benign bookkeeping — allowed;
* everything else (MCP tools, Task, an unknown tool) — denied by default; the
  operator's own CLI settings are the other half of the surface.

Pure functions: the session hands in the intent and its context, gets a
decision back. A guardrail against accidents on the operator's own machine, not
a sandbox: what a command reads through data it fetched at run time (a list of
paths in a file, ``$(cat list)``) is beyond a static gate.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import secret_reach
from .adapters.base import ToolClass, ToolIntent

# Sessions never publish. The manager (Auto) integrates. Matched on the raw
# command first, then on every simple command once ``git -C <path>``, the shell
# keywords and the assignments are peeled off — so the ``-C`` spelling, a
# newline and ``$(git push)`` meet the same wall.
NEVER_ALLOWED_BASH = (
    re.compile(r"(^|[;&|(]\s*)git\s+push\b"),
    re.compile(r"(^|[;&|(]\s*)git\s+remote\s+(add|set-url)\b"),
    re.compile(r"(^|[;&|(]\s*)gh\s+(pr|release)\s+(create|merge|edit)\b"),
    re.compile(r"(^|[;&|(]\s*)(sudo|su)\b"),
    re.compile(r"(^|[;&|(]\s*)rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$)"),
    re.compile(r"(^|[;&|(]\s*)curl\b.*\|\s*(ba|z)?sh\b"),
)

# Always a card, even under ``--unlisted-bash allow``, and even when the verb
# itself is on the allowlist. Night 1 (2026-09-18, F042): an OPS ticket session
# ran ``cd …/automatos-ai && set -a && . ./.env && set +a`` and then
# ``PGPASSWORD=… psql -h 127.0.0.1 … -f …/change-applied-task-3.sql`` — an
# UPDATE against the platform's own database — plus ``docker logs`` and a
# ``redis-cli -a`` attempt, all inside the allow window. ``cat`` and ``.`` are
# perfectly ordinary verbs; what makes these worth a card is WHAT they touch.
# The operator can still say yes; they just get asked.
ALWAYS_ASK_BASH = (
    (re.compile(r"(^|[\s;&|(])(\.|source)\s+\S*\.env\b", re.IGNORECASE), "reads a .env file (it holds this system's secrets)"),
    (re.compile(r"\.env(\.|\s|$|['\"])", re.IGNORECASE), "touches a .env file (it holds this system's secrets)"),
    (re.compile(r"(^|[;&|(]\s*)psql\b"), "runs psql against a database"),
    (re.compile(r"(^|[;&|(]\s*)(redis-cli|mysql|mongosh)\b"), "opens a database shell"),
    (re.compile(r"(^|[;&|(]\s*)docker\b"), "drives Docker (the platform runs in it)"),
    (re.compile(r"(^|[;&|(]\s*)PGPASSWORD="), "passes a database password on the command line"),
    (re.compile(r"(^|[;&|(]\s*)(alembic|flask|django-admin)\b"), "runs a database migration tool"),
    # F042 review: a link gives a file a second name — a hard link to a secret has
    # no secret-shaped name at all. Asked even under ``--unlisted-bash allow``.
    (re.compile(r"(^|[;&|(]\s*)ln\b"), "makes a link (a link can give a secrets file a harmless name)"),
)


# Verbs whose ARGUMENTS are the command that actually runs. None is on the
# allowlist (each runs a command the gate would not otherwise see), so the
# wrapper itself is held for the operator — but a never-allowed command must
# not ride in as its argument and arrive as a question instead of a refusal:
# ``xargs git push`` is ``git push``. The wrapper is peeled for that check only.
# ``find``'s exec options are here for the orphan case: a second ``-exec`` after
# a ``;`` is a simple command of its own whose first word is the option.
COMMAND_WRAPPERS = frozenset({
    "xargs", "env", "command", "builtin", "exec", "time", "timeout", "nice", "ionice",
    "nohup", "stdbuf", "caffeinate", "chronic", "watch", "-exec", "-execdir", "-ok", "-okdir",
})
_WRAPPER_VALUE_RE = re.compile(r"^\d+[smhd]?$")      # ``timeout 5``, ``timeout 30s``, ``nice -n 10``


def _unwrapped(words: Sequence[str]) -> List[str]:
    """The command left once every leading wrapper, its options, its
    assignments (``env A=1``) and its numeric values are peeled."""
    out = list(words)
    while out and out[0] in COMMAND_WRAPPERS:
        out = out[1:]
        while out and (out[0].startswith("-") or _ASSIGNMENT_RE.match(out[0])
                       or _WRAPPER_VALUE_RE.match(out[0]) or out[0] in FIND_PLACEHOLDERS):
            out = out[1:]
    return out


# Read-only git, the verbs that read and shape text, the usual build/test verbs
# a code ticket needs. Never here: xargs, env, sh, bash, eval, sudo — each runs
# a command the gate would not see.
DEFAULT_BASH_ALLOW = (
    "git status", "git diff", "git log", "git show", "git branch", "git add",
    "git commit", "git stash", "git restore", "git checkout -b", "git switch -c",
    "git ls-files", "git rev-parse", "git blame", "git describe", "git shortlog",
    "git remote -v", "git worktree list", "git stash list",
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "which", "echo",
    "sort", "uniq", "cut", "tr", "sed", "awk", "date", "diff", "stat", "basename",
    "dirname", "printf", "jq", "file", "tree", "du", "true", "test", "[", "[[",
    # Read-only coreutils that belong beside sort/uniq/diff. ``comm`` was the one
    # word that held an otherwise-allowed CSS diff on night 1 (2026-09-18) — the
    # session sat on a two-minute hold for a command that reads two sorted files.
    "comm", "join", "paste", "nl", "tac", "rev", "fold", "expand", "unexpand",
    "column", "md5sum", "sha1sum", "sha256sum", "cksum", "realpath", "readlink",
    "seq",
    "python -m pytest", "python3 -m pytest", "pytest", "npm test", "npm run", "pnpm test",
    "pnpm run", "yarn test", "make test", "make lint", "cargo test", "go test",
    "ruff", "black --check", "mypy", "tsc", "eslint", "vitest",
)

# Verbs that shape text and never open a path: a path-shaped word among their
# arguments is a string, not a file (``echo /etc/passwd`` prints a path, it does
# not read one). Their redirections are confined like everyone else's.
# …only where their output goes to the terminal. Feeding one into another
# command (``cat $(echo /etc/hosts)``) makes the word a path again, so the
# exemption holds at the top level of a command line and nowhere deeper.
PATH_FREE_VERBS = frozenset({"echo", "printf", "basename", "dirname", "date", "true"})

# ``find`` runs a command per hit and can delete what it matches: the command is
# judged like any other simple command, the destructive options are the
# operator's call.
FIND_EXEC_OPTIONS = ("-exec", "-execdir", "-ok", "-okdir")
FIND_WRITE_OPTIONS = ("-delete", "-fls", "-fprint", "-fprint0", "-fprintf")
FIND_PLACEHOLDERS = ("{}", "+")

# A verb whose ARGUMENT is a program can run commands of its own. The verb stays
# on the allowlist (a ticket needs ``sed -n`` and ``awk '{print $1}'``); its
# program is read for the constructs that escape, and a program the gate cannot
# read at all (``-f progfile``) is refused.
# Every quantifier here is BOUNDED and no alternation is ambiguous: the program
# is a session's own text, so a pattern that can backtrack exponentially on it
# would hang the hook thread the gate answers from (CodeQL py/redos).
_AWK_ESCAPE_RE = re.compile(r"system\s{0,8}\(|\bgetline\b|\||print(?:f)?[^;}\n]{0,200}>")
# ``e`` as a COMMAND — after an address (``1e cmd``, ``/x/e cmd``, ``$e cmd``)
# and followed by its command; never a letter's neighbour (``line`` ends in one)
# and never followed by a delimiter (``/e/d`` is an ``e`` inside a regex). Then
# ``e`` among the FLAGS of an ``s`` command (``s/a/b/e``), recognised by the
# delimiter that closes it and the command boundary after the flags — matching
# the two delimited halves instead would need an ambiguous alternation, and an
# untrusted program must never be able to make this pattern backtrack.
# …and ``w``/``W`` (write the pattern space to a FILE), ``r``/``R`` (read a
# file in) and the ``w`` flag of ``s`` — a program text is exempt from the path
# check, so a filename inside one would otherwise be a write or read anywhere.
# ``awk``'s ``print > file`` is caught by its own pattern; sed's was not.
_SED_ESCAPE_RE = re.compile(
    r"(?<![a-zA-Z\\])e(?:[ \t;]|$)"
    r"|[^a-zA-Z0-9\s][a-zA-Z0-9]{0,16}e[a-zA-Z0-9]{0,16}(?:[;\n}]|$)"
    r"|(?<![a-zA-Z\\])[wWrR][ \t]+\S"
)

# Global options that may sit between ``git`` and its subcommand. Peeled before
# the verb is judged, so no spelling of a global hides a ``push`` from the
# never-allowed list; the ones that name a path are confined like any argument.
_GIT_PATH_OPTIONS = ("-C", "--git-dir", "--work-tree", "--exec-path")
_GIT_VALUE_OPTIONS = ("-c", "--namespace", "--config-env", "--super-prefix")
_GIT_FLAG_OPTIONS = ("-p", "-P", "--paginate", "--no-pager", "--bare", "--no-replace-objects",
                     "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
                     "--icase-pathspecs", "--no-optional-locks", "--no-lazy-fetch")
# ``gh`` names a repository, never a path.
_GH_VALUE_OPTIONS = ("-R", "--repo")


@dataclass(frozen=True)
class ScriptVerb:
    """How one program-taking verb spells its program (design: one row per verb)."""
    escape: Any                      # the constructs in a program that run a command
    text_options: Tuple[str, ...]    # options whose VALUE is program text
    file_options: Tuple[str, ...]    # options naming a program FILE — refused
    value_options: Tuple[str, ...]   # options with a value that is not a program


SCRIPT_VERBS: Mapping[str, ScriptVerb] = {
    "awk": ScriptVerb(_AWK_ESCAPE_RE, ("-e", "--source"), ("-f", "--file", "--include"),
                      ("-v", "--assign", "-F", "--field-separator")),
    "sed": ScriptVerb(_SED_ESCAPE_RE, ("-e", "--expression"), ("-f", "--file"), ()),
}

# A code ticket must be able to RUN what it just wrote — that is "build and
# test", not "publish". An interpreter may run a file inside the session
# directory; inline code (``python -c``, ``node -e``) stays refused so the
# never-allowed list cannot be bypassed inside a string.
_INTERPRETER_RE = re.compile(r"^(python(\d+(\.\d+)?)?|node)$")
INLINE_CODE_FLAGS = frozenset({"-c", "-e", "--eval", "-p", "--print"})
OWN_CODE_MODULES = frozenset({"doctest", "unittest", "pytest", "py_compile"})

# ── how a command line is read ───────────────────────────────────────────────
# Words that structure a command line but run nothing; ``for`` is read as a loop
# header (``for NAME in WORDS``), the rest are skipped to reach the verb.
SHELL_KEYWORDS = frozenset({"for", "do", "done", "if", "then", "else", "elif", "fi",
                            "while", "until", "in", "{", "}", "!"})
_LEADING_KEYWORDS = SHELL_KEYWORDS - {"for"}
_PUNCTUATION = "();<>|&\n"                 # shlex punctuation, plus the newline
_SEPARATOR_CHARS = frozenset(";&|()\n")    # a run of these ends a simple command
_REDIRECT_CHARS = frozenset("<>")          # a run holding one of these is a redirection
ALWAYS_WRITABLE = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_GLOB_CHARS = "*?["
_LINE_CONTINUATION_RE = re.compile(r"\\\n")
_HEREDOC_RE = re.compile(r"<<-?\s*(?:'([^']*)'|\"([^\"]*)\"|\\([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))")
_FD_TARGET_RE = re.compile(r"^([0-9]+|-)$")            # ``>&1``, ``<&-``
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
_VAR_REF_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
# A reference the line did not define: ``$NAME``, ``${NAME…}`` in any of its
# forms (``${HOME:-/etc}``), ``$(…)``, a positional or special parameter, a bare
# ``$``. Anything the gate cannot resolve must not read as a plain word.
_UNRESOLVED_RE = re.compile(r"\$(?:[{(]|[A-Za-z_0-9@*#?!$])|^\$$")
# …used as part of a path: ``$HOME/x``, ``${D}/x``, ``${HOME:-/etc}/x``, ``$1/x``.
_PATH_REF_RE = re.compile(r"/\$(?:[{(]|[A-Za-z_0-9@*])|\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*|[0-9])/")
SUBSTITUTION_MARK = "$_"     # stands in for a ``$(…)`` body once that body is judged on its own
MAX_EXPANSIONS = 64          # values one word may take across the line's variables
# ``$'…'`` / ``$"…"`` only where the ``$`` begins a word. Night 1 held every
# ``grep -v '^$' | …`` on earth because the ``$`` ending a quoted regex sat next to the quote.
_ANSI_C_RE = re.compile(r"""(?:^|[\s=(|;&])\$['"]""")
MAX_NESTING = 8              # ``$(…)`` inside ``$(…)`` …
_SEVERITY = {"allow": 0, "ask": 1, "deny": 2}

# F167: why a call was ALLOWED, so the ticket says what really happened. A
# ticket said a command "needed your approval, and it went through" when it had
# run under ``--unlisted-bash allow`` with nobody asked.
ALLOWED_BASH = "within this ticket's Bash allowlist"
ALLOWED_UNLISTED_BASH = ("{command!r} is not on this ticket's Bash allowlist; this host runs such commands "
                         "without asking (--unlisted-bash allow)")
ALLOWED_FILES = "inside the session's folders"
ALLOWED_SESSION_TOOL = "an Automatos tool this ticket may call"
ALLOWED_NO_APPROVAL = "a tool that needs no approval"

Bindings = Mapping[str, Tuple[str, ...]]


@dataclass
class PolicyContext:
    cwd: Path
    allowed_bash: Sequence[str] = field(default_factory=lambda: DEFAULT_BASH_ALLOW)
    ask_bash: Sequence[str] = ()          # prefixes routed to the approvals inbox
    extra_dirs: Sequence[Path] = ()       # e.g. the git worktree the session runs in
    # PRD-245 W1: the Automatos tools THIS ticket may call, from the claim. A
    # name on the list is allowed; anything else on our own MCP server is denied
    # (never held — the operator has nothing to decide about a name we did not
    # offer). Empty = the bridge is not in this ticket, so no platform tool is.
    session_tools: Sequence[str] = ()
    unlisted_bash: str = "ask"           # "allow": verbs the allowlist does not name run without a card
    # F042: the platform's own secrets are out of every session's reach — a hard
    # deny no approval lifts. ``secret_roots``: checkouts whose .env family and
    # credential key no session may read or write (the Automatos checkout this
    # host runs from); ``off_limits``: folders no session may touch at all (the
    # host's own state — its token). Empty = no such guard (tests, other hosts).
    secret_roots: Sequence[Path] = ()
    off_limits: Sequence[Path] = ()


@dataclass
class Decision:
    behavior: str            # allow | deny | ask
    reason: str = ""

    @property
    def allow(self) -> bool:
        return self.behavior == "allow"


def _worst(decisions: Iterable[Decision]) -> Decision:
    """deny over ask over allow; the first reason of the worst kind (an allow's
    too: why a call ran is on the ticket, F167)."""
    worst = Decision("allow")
    for decision in decisions:
        if _SEVERITY[decision.behavior] > _SEVERITY[worst.behavior] or (
                decision.behavior == worst.behavior and decision.reason and not worst.reason):
            worst = decision
    return worst


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


# ── F042: the platform's secrets ────────────────────────────────────────────
# Night 1 (2026-09-18): an OPS session started in ~/Development, went into the
# Automatos checkout and sourced its .env — the platform's database password and
# API keys — then ran psql against the platform. ALWAYS_ASK_BASH put a card in
# front of that shape of Bash; the Read tool read the same file with no card at
# all. A session may work in the folder that holds the platform; it may never
# hold the platform's secrets.
# Case-folded throughout (security review 2026-09-22): APFS is case-insensitive,
# so ``.ENV`` and ``~/.Automatos/…`` open the very files these names guard.
_SECRET_NAME_RE = re.compile(r"^(?:\.env(?:\..+)?|\.credential_key)$", re.IGNORECASE)
_EXAMPLE_ENV_RE = re.compile(r"^\.env\.(?:example|sample|template|dist|defaults)$", re.IGNORECASE)
_SECRET_WORD_RE = re.compile(
    r"""(?:^|[\s'"=:;|&(<>])([^\s'";|&()<>]*?(?:\.env(?:\.[\w.-]+)?|\.credential_key))(?=$|[\s'";|&()<>])""",
    re.IGNORECASE)
_ABSOLUTE_WORD_RE = re.compile(r"""(?:^|[\s'"=:;|&(<>])((?:/|~/)[^\s'";|&()<>]*)""")
_CD_TARGET_RE = re.compile(r"""(?:^|[;&|(]|\s)(?:cd|pushd)\s+(['"]?)([^\s'";|&()]+)\1""")
# Shell options under which ``*`` reaches dot-files (or ignores case): set by
# ``shopt -s …`` or a bare ``GLOBIGNORE=…``, from that command on — never by the
# word merely appearing on the line (``echo dotglob`` sets nothing).
_LOOSE_GLOB_OPTIONS = frozenset({"dotglob", "nocaseglob", "nocasematch", "extglob"})


def platform_secret_roots() -> Tuple[Path, ...]:
    """The Automatos checkout this host runs from — its .env family and credential
    key are the platform's own secrets. Empty when the host is not running from
    a checkout (no ``orchestrator/`` beside ``services/``)."""
    root = Path(__file__).resolve().parents[3]
    return (root,) if (root / "orchestrator").is_dir() else ()


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME_RE.match(name)) and not _EXAMPLE_ENV_RE.match(name)


def _within(path: Path, root: Path) -> bool:
    return secret_reach.within_folded(path, root)


def _inventory(ctx: "PolicyContext") -> secret_reach.Inventory:
    return secret_reach.inventory(ctx.secret_roots, ctx.off_limits, ctx.extra_dirs, _is_secret_name)


def secret_protection_summary() -> str:
    """What this host keeps out of every session's reach — logged at start, so a
    host running from a checkout without ``orchestrator/`` says so out loud."""
    roots = platform_secret_roots()
    if not roots:
        return "platform secrets: NOT protected — this host is not running from an Automatos checkout"
    return "platform secrets out of every session's reach under " + ", ".join(str(r) for r in roots)


def _guard_secret(path_str: str, ctx: "PolicyContext", base: Path) -> Optional["Decision"]:
    """``deny`` for the platform's own secrets and the host's state, ``ask`` for any
    other secret-named file, ``None`` for everything else. Judged on the path as
    written AND as resolved, so a symlink to the platform's .env is caught."""
    if not (ctx.secret_roots or ctx.off_limits) and not _is_secret_name(Path(path_str).name):
        return None
    try:
        written = Path(path_str).expanduser()
        written = written if written.is_absolute() else base / written
        resolved = written.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    forms = (written, resolved)
    if ctx.secret_roots:
        linked = secret_reach.same_file_as_secret(resolved, _inventory(ctx))
        if linked is not None:
            return Decision("deny", f"{written.name} is a hard link to {linked.name}, the platform's own "
                                    "secrets — no session may read or write it, and no approval changes that")
    granted = [Path(d).expanduser().resolve() for d in ctx.extra_dirs]
    for folder in ctx.off_limits:
        folder = Path(folder).expanduser().resolve()
        # The session's OWN folder (its ticket file) sits inside the host's state
        # and is granted to it; the token, the process table and every other
        # session's folder are not.
        if any(_within(f, folder) for f in forms) and not any(_within(f, g) for f in forms for g in granted):
            return Decision("deny", "the CLI host's own state — no session may read or write it")
    if not any(_is_secret_name(f.name) for f in forms):
        return None
    for root in ctx.secret_roots:
        root = Path(root).expanduser().resolve()
        if any(_within(f, root) for f in forms):
            return Decision("deny", f"{written.name} holds the platform's own secrets ({root.name}) — "
                                    "no session may read or write it, and no approval changes that")
    return Decision("ask", f"touches {written.name} (a secrets file) — the operator decides")


def _guard_search(tool: str, globs: Sequence[str], paths: Sequence[str], ctx: "PolicyContext") -> Optional["Decision"]:
    """A file-tool search whose pattern picks out a secrets file. Grep READS
    what its glob selects (Claude Code's Grep runs ripgrep with ``--hidden``,
    and a glob overrides .gitignore), so a glob that selects one of the
    platform's secret files under the search folder is refused. A pattern aimed
    at secret names (``.env``, ``**/.e*``, ``*.ENV`` — not ``*`` or ``**/*.py``)
    is refused over the platform's checkout, from inside it or any folder
    above it, and asked elsewhere."""
    if not globs:
        return None
    inv = _inventory(ctx)
    bases = [Path(p).expanduser() for p in paths] or [ctx.cwd]
    bases = [(b if b.is_absolute() else ctx.cwd / b).resolve() for b in bases]
    if tool == "Grep":
        for base in bases:
            for secret in (f for f in inv.reachable_under(base) if f in inv.files):
                glob = next((g for g in globs if secret_reach.tool_glob_matches(g, secret, base)), None)
                if glob is not None:
                    return Decision("deny", f"a search with glob {glob!r} here would read {secret.name}, the "
                                            "platform's own secrets — no approval changes that")
    named = [g for g in globs if secret_reach.aimed_at_secrets(g, inv.names())]
    if not named:
        return None
    for base in bases:
        for root in inv.roots:
            if _within(base, root) or _within(root, base):
                return Decision("deny", f"a search for {named[0]!r} here would read the platform's own secrets "
                                        f"({root.name}) — no approval changes that")
    return Decision("ask", f"searches for {named[0]!r} (secrets files) — the operator decides")


def _secret_on_line(visible: str, ctx: "PolicyContext") -> Optional["Decision"]:
    """The platform's secrets named anywhere on a Bash line — as written, relative
    to the session's folder, or relative to a ``cd`` earlier on the line (night
    1's ``cd …/automatos-ai && . ./.env``). Only a refusal is returned here; a
    card for any other .env is ALWAYS_ASK_BASH's job."""
    if not (ctx.secret_roots or ctx.off_limits):
        return None
    bases = [ctx.cwd]
    for match in _CD_TARGET_RE.finditer(visible):
        target = Path(match.group(2)).expanduser()
        bases.append(target if target.is_absolute() else ctx.cwd / target)
    words = [m.group(1) for m in _SECRET_WORD_RE.finditer(visible)]
    words += [m.group(1) for m in _ABSOLUTE_WORD_RE.finditer(visible)] if ctx.off_limits else []
    for word in words:
        for base in bases:
            guard = _guard_secret(word, ctx, base)
            if guard is not None and guard.behavior == "deny":
                return guard
    return None


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


# ── tokens and simple commands ───────────────────────────────────────────────

def _heredoc_end(text: str, start: int, delimiter: str) -> Optional[int]:
    """The end of the line that terminates a here-document whose body starts at
    ``start`` (``<<-`` lets the terminator be tab-indented); None when there is none."""
    pos = start
    while pos <= len(text):
        newline = text.find("\n", pos)
        line = text[pos:] if newline < 0 else text[pos:newline]
        if line.lstrip("\t") == delimiter:
            return len(text) if newline < 0 else newline
        if newline < 0:
            return None
        pos = newline + 1
    return None


def _heredoc_delimiter(match: Any) -> Tuple[str, bool]:
    """The here-document's terminator, and whether it was QUOTED. A quoted
    delimiter (``<<'EOF'``) makes the body inert data; an unquoted one
    (``<<EOF``) expands the substitutions inside it as the shell reads it."""
    if match.group(1) is not None:
        return match.group(1), True
    if match.group(2) is not None:
        return match.group(2), True
    if match.group(3) is not None:      # ``<<\EOF`` — bash treats it exactly like ``<<'EOF'``
        return match.group(3), True
    return match.group(4), False


def _heredocs(command: str) -> Tuple[str, List[str]]:
    """The command without its here-document bodies, plus the bodies whose
    delimiter was UNQUOTED.

    A body is data, not part of the command line, so it is cut before tokenising
    (the ``<<`` itself stays, so the redirection beside it is still judged). But
    an unquoted delimiter makes the shell RUN the substitutions in that body, so
    those bodies come back for judging. A body without its terminator is left
    where it is."""
    out = command
    expanded: List[str] = []
    pos = 0
    while True:
        match = _HEREDOC_RE.search(out, pos)
        if match is None:
            return out, expanded
        line_end = out.find("\n", match.end())
        if line_end < 0:
            return out, expanded
        delimiter, quoted = _heredoc_delimiter(match)
        body_end = _heredoc_end(out, line_end + 1, delimiter)
        if body_end is None:
            return out, expanded
        if not quoted:
            expanded = [*expanded, out[line_end + 1:body_end]]
        out = out[:line_end] + out[body_end:]
        pos = match.end()


def _split_parens(token: str) -> List[str]:
    """A punctuation run holding a parenthesis becomes its own tokens: ``<(`` →
    ``<``, ``(``. Otherwise process substitution reads as ONE redirection token
    whose "target" is the first word of the command inside it — and that command
    is then never judged (``echo <(git push)`` ran the push)."""
    if "(" not in token and ")" not in token:
        return [token]
    if not _is_run_of(token, _SEPARATOR_CHARS | _REDIRECT_CHARS):
        return [token]
    out: List[str] = []
    run = ""
    for char in token:
        if char in "()":
            out = [*out, run] if run else out
            out, run = [*out, char], ""
        else:
            run += char
    return [*out, run] if run else out


# Where a '#' begins a comment: at the start of a word, which in bash means the
# start of the text or right after whitespace or a separator. Mid-word it is an
# ordinary character — ``a#b``, ``$#``, ``${#arr}``, ``http://x#frag``.
_COMMENT_WORD_START = frozenset(" \t\r\n;&|()")


def _strip_comments(command: str) -> str:
    """The command without its shell comments, the way bash reads it.

    F056 (night 2, grant 355): a perfectly safe ``grep`` was held as
    "could not be parsed (unbalanced quotes)" because a COMMENT line above it
    said "# Exclude files I've already read" — and the apostrophe in "I've" was
    taken for an opening quote. Bash never reads a comment, so the gate must
    not either. Quote-aware, including ``$'…'`` (where a backslash escapes the
    next character), so a '#' inside any string is left alone. The newline that
    ends a comment is kept: it separates commands.
    """
    out: List[str] = []
    i, n = 0, len(command)
    single = double = ansi = False
    while i < n:
        c = command[i]
        if ansi:                                   # inside $'…'
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if c == "'":
                ansi = False
            i += 1
            continue
        if single:
            out.append(c)
            if c == "'":
                single = False
            i += 1
            continue
        if c == "\\" and i + 1 < n:                # an escape outside single quotes
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        if c == "$" and i + 1 < n and command[i + 1] == "'" and not double:
            out.append("$'")
            ansi = True
            i += 2
            continue
        if c == "'" and not double:
            single = True
        elif c == '"':
            double = not double
        elif c == "#" and not double and (i == 0 or command[i - 1] in _COMMENT_WORD_START):
            end = command.find("\n", i)
            if end == -1:
                break                              # the comment runs to the end
            i = end                                # keep the newline itself
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _tokens(command: str) -> List[str]:
    """Quote-aware tokens, the way a POSIX shell reads them; a run of
    punctuation (``&&``, ``2>``'s ``>``, a newline) is its own token, and a
    parenthesis is always its own. Comments are dropped first, as bash drops
    them (F056). Raises ``ValueError`` on an unbalanced quote."""
    text = _strip_comments(_LINE_CONTINUATION_RE.sub(" ", _heredocs(command)[0]))
    lex = shlex.shlex(text, posix=True, punctuation_chars=_PUNCTUATION)
    lex.whitespace = " \t\r"       # a newline separates commands, like ';'
    lex.whitespace_split = True    # words end on whitespace and punctuation only
    lex.commenters = ""            # '#' is a character (``a#b`` is one word)
    return [part for token in lex for part in _split_parens(token)]


def _is_run_of(token: str, chars: Iterable[str]) -> bool:
    allowed = set(chars)
    return bool(token) and all(c in allowed for c in token)


def _split_compound_depths(command: str) -> List[Tuple[int, List[str]]]:
    """Each simple command of a command line with the PARENTHESIS DEPTH it sits
    at. Cut only on unquoted separators (``&&``, ``||``, ``;``, ``|``, ``&``,
    parentheses, a newline) — a ``|`` inside a grep pattern is a character;
    redirections stay with their command. The depth matters because a command
    inside ``$(…)``, ``<(…)`` or ``(…)`` has its output captured and fed onward,
    so a verb that merely PRINTS a path there is still naming one. Raises
    ``ValueError`` on an unbalanced quote."""
    segments: List[Tuple[int, List[str]]] = []
    current: List[str] = []
    depth = 0
    for token in _tokens(command):
        if _is_run_of(token, _SEPARATOR_CHARS):
            segments = [*segments, (depth, current)] if current else segments
            current = []
            depth = max(0, depth + token.count("(") - token.count(")"))
        else:
            current = [*current, token]
    return [*segments, (depth, current)] if current else segments


def _split_compound(command: str) -> List[List[str]]:
    """The simple commands of a command line, each as its tokens."""
    return [tokens for _, tokens in _split_compound_depths(command)]


def _is_redirection(token: str) -> bool:
    return _is_run_of(token, _SEPARATOR_CHARS | _REDIRECT_CHARS) and any(c in _REDIRECT_CHARS for c in token)


def _peel_redirections(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """The words of a simple command and its redirection targets. A file
    descriptor glued to its operator (``2>``) or named by ``>&1`` is neither."""
    words: List[str] = []
    targets: List[str] = []
    i = 0
    while i < len(tokens):
        if not _is_redirection(tokens[i]):
            words = [*words, tokens[i]]
            i += 1
            continue
        if words and words[-1].isdigit():
            words = words[:-1]
        target = tokens[i + 1] if i + 1 < len(tokens) else ""
        if target and not _FD_TARGET_RE.match(target):
            targets = [*targets, target]
        i += 2
    return words, targets


# ── command substitutions inside a word ──────────────────────────────────────

def _matching_paren(text: str, start: int) -> int:
    """Index of the ')' closing the '(' at ``start``; the end of the text when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _substitution_spans(word: str) -> List[Tuple[int, int, str]]:
    """``(start, end, body)`` of every ``$(…)`` and backtick substitution in a
    word — the tokenizer keeps them whole when quoted. Arithmetic ``$((…))`` is
    skipped; a nested substitution is found when its body is judged."""
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(word):
        if word.startswith("$((", i):
            i = _matching_paren(word, i + 1) + 1
        elif word.startswith("$(", i):
            end = _matching_paren(word, i + 1)
            spans = [*spans, (i, end + 1, word[i + 2:end])]
            i = end + 1
        elif word[i] == "`":
            end = word.find("`", i + 1)
            end = len(word) if end < 0 else end
            spans = [*spans, (i, end + 1, word[i + 1:end])]
            i = end + 1
        else:
            i += 1
    return spans


def _substitutions(word: str) -> List[str]:
    return [body for _, _, body in _substitution_spans(word)]


def _backtick_bodies(line: str) -> List[str]:
    """The bodies of the backtick substitutions only — the ones the tokenizer
    cannot keep whole when unquoted."""
    return [body for start, _, body in _substitution_spans(line) if line[start] == "`"]


def _without_substitutions(word: str) -> str:
    """The word with each substitution replaced by ``SUBSTITUTION_MARK`` — an
    unresolved reference wherever the body's output would land."""
    out = ""
    last = 0
    for start, end, _ in _substitution_spans(word):
        out += word[last:start] + SUBSTITUTION_MARK
        last = end
    return out + word[last:]


# ── the line's own variables ─────────────────────────────────────────────────

def _expand(word: str, bindings: Bindings) -> Tuple[str, ...]:
    """Every value the word can take once the line's own assignments and loop
    variables are substituted. A reference nothing on the line defined stays as
    written (and reads as unresolved). Bounded, so a loop over many words with
    several references cannot explode; what is left unexpanded stays unresolved."""
    pending: List[str] = [word]
    done: List[str] = []
    for _ in range(MAX_EXPANSIONS):
        if not pending:
            break
        current, pending = pending[0], pending[1:]
        ref = next((m for m in _VAR_REF_RE.finditer(current) if (m.group(1) or m.group(2)) in bindings), None)
        if ref is None:
            done = [*done, current]
            continue
        values = bindings[ref.group(1) or ref.group(2)]
        pending = [*pending, *(current[:ref.start()] + value + current[ref.end():] for value in values)]
    return tuple([*done, *pending])


def _resolved(values: Sequence[str]) -> bool:
    return bool(values) and not any(_UNRESOLVED_RE.search(v) for v in values)


def _strip_keywords(words: Sequence[str]) -> List[str]:
    rest = list(words)
    while rest and rest[0] in _LEADING_KEYWORDS:
        rest = rest[1:]
    return rest


def _bind_assignments(words: Sequence[str], bindings: Bindings) -> Tuple[List[str], Bindings]:
    """Leading ``NAME=value`` words are remembered for the rest of the line (a
    value the gate cannot resolve makes NAME unknown) and peeled off."""
    bound: Dict[str, Tuple[str, ...]] = dict(bindings)
    rest = list(words)
    while rest:
        match = _ASSIGNMENT_RE.match(rest[0])
        if match is None:
            break
        values = _expand(match.group(2), bound)
        bound.pop(match.group(1), None)
        if _resolved(values):
            bound[match.group(1)] = values
        rest = rest[1:]
    return rest, bound


# ── paths ────────────────────────────────────────────────────────────────────

def _path_argument(arg: str) -> Optional[str]:
    """The absolute or ``~`` path an argument names — also as an option value
    (``--file=/etc/x``); None for anything relative, which resolves under the
    session directory by construction ('..' is refused before we get here)."""
    if arg.startswith(("/", "~")):
        return arg
    if arg.startswith("-") and "=" in arg:
        value = arg.split("=", 1)[1]
        return value if value.startswith(("/", "~")) else None
    return None


def _judge_path(path: str, roots: Sequence[Path]) -> str:
    """``allow`` | ``ask`` | ``deny`` for one absolute (or ``~``) path. A glob is
    judged by the directory of its literal prefix; a path holding an unresolved
    reference by the directory before it — outside the roots is a refusal,
    inside is a question."""
    unresolved = _UNRESOLVED_RE.search(path)
    cuts = [i for i in (path.find(c) for c in _GLOB_CHARS) if i >= 0]
    if unresolved:
        cuts = [*cuts, unresolved.start()]
    if not cuts:
        return "allow" if _inside(path, roots) else "deny"
    literal = path[:min(cuts)]
    directory = literal.rsplit("/", 1)[0] or "/"
    if not _inside(directory, roots):
        return "deny"
    return "ask" if unresolved else "allow"


def _judge_args(verb: str, args: Sequence[str], bindings: Bindings, roots: Sequence[Path]) -> Decision:
    """Every path an allowed verb would read: absolute or ``~`` arguments (and
    option values) must sit inside the roots — outside is a refusal, never a
    question; a path built from a reference nothing on the line defined is a
    question."""
    verdicts: List[Decision] = []
    for value in (v for arg in args for v in _expand(arg, bindings)):
        path = _path_argument(value)
        if path is not None:
            verdict = _judge_path(path, roots)
            if verdict == "deny":
                verdicts = [*verdicts, Decision("deny", f"{verb} outside the session directory: {path}")]
            elif verdict == "ask":
                verdicts = [*verdicts, Decision("ask", f"{verb} names a path the gate cannot resolve: {path}")]
        elif _PATH_REF_RE.search(value):
            verdicts = [*verdicts, Decision("ask", f"{verb} names a path the gate cannot resolve: {value}")]
    return _worst(verdicts)


def _judge_targets(targets: Sequence[str], bindings: Bindings, roots: Sequence[Path]) -> Decision:
    """Redirection targets: the null and standard devices are always fine; any
    other absolute target must sit inside the roots."""
    verdicts: List[Decision] = []
    for value in (v for target in targets for v in _expand(target, bindings)):
        if value in ALWAYS_WRITABLE:
            continue
        if value.startswith(("/", "~")):
            verdict = _judge_path(value, roots)
            if verdict == "deny":
                verdicts = [*verdicts, Decision("deny", f"redirection outside the session directory: {value}")]
            elif verdict == "ask":
                verdicts = [*verdicts, Decision("ask", f"redirection to a path the gate cannot resolve: {value}")]
        elif _PATH_REF_RE.search(value):
            verdicts = [*verdicts, Decision("ask", f"redirection to a path the gate cannot resolve: {value}")]
    return _worst(verdicts)


# ── one simple command ───────────────────────────────────────────────────────

def _runs_own_code(words: Sequence[str]) -> bool:
    """An interpreter run on a file, or on one of its own test modules — never
    inline code, never an unknown flag. The file's path is confined like every
    other argument (``_judge_args``)."""
    head = Path(words[0]).name  # tolerate /usr/bin/python3
    if not _INTERPRETER_RE.match(head):
        return False
    args = words[1:]
    if not args or any(a in INLINE_CODE_FLAGS for a in args):
        return False
    if args[0] == "-m":
        return len(args) >= 2 and args[1] in OWN_CODE_MODULES
    return not args[0].startswith("-")  # an unknown interpreter flag is not a plain "run this file"


def _judge_cd(words: Sequence[str], bindings: Bindings, roots: Sequence[Path]) -> Decision:
    """``cd`` only to a directory the gate can resolve, inside the roots."""
    if len(words) != 2:
        return Decision("ask", f"{_first_words(' '.join(words))!r} is outside this ticket's Bash allowlist")
    if not words[1] or words[1].startswith("-"):
        # ``cd -`` is $OLDPWD, which an earlier cd may have put anywhere
        return Decision("ask", f"cd to a directory the gate cannot resolve: {words[1] or '(none)'}")
    values = _expand(words[1], bindings)
    if not _resolved(values):
        return Decision("ask", f"cd to a directory the gate cannot resolve: {words[1]}")
    outside = [v for v in values if not _inside(v, roots)]
    if outside:
        return Decision("deny", f"cd outside the session directory: {outside[0]}")
    return Decision("allow")


def _peel_globals(words: Sequence[str], bindings: Bindings, roots: Sequence[Path]) -> Tuple[List[str], Decision]:
    """``git -C <path> -c k=v log`` is ``git log`` run in <path>: EVERY global
    option is peeled off before the verb meets the never-allowed list and the
    allowlist, and every path one names is confined. ``--git-dir=…``, a repeated
    ``-C`` and ``gh -R owner/name`` are the same shape."""
    head = Path(words[0]).name if words else ""
    if head == "git":
        path_options, value_options, flag_options = _GIT_PATH_OPTIONS, _GIT_VALUE_OPTIONS, _GIT_FLAG_OPTIONS
    elif head == "gh":
        path_options, value_options, flag_options = (), _GH_VALUE_OPTIONS, ()
    else:
        return list(words), Decision("allow")
    rest = list(words[1:])
    paths: List[str] = []
    while rest:
        name, glued, value = rest[0].partition("=")
        if name in path_options or name in value_options:
            if glued:
                rest = rest[1:]
            elif len(rest) >= 2:
                value, rest = rest[1], rest[2:]
            else:
                break
            if name in path_options:
                paths = [*paths, value]
        elif name in flag_options and not glued:
            rest = rest[1:]
        else:
            break
    return [head, *rest], _judge_args(f"{head} {' '.join(path_options[:1]) or 'global option'}", paths, bindings, roots)


def _script_texts(spec: ScriptVerb, args: Sequence[str]) -> Tuple[List[str], bool]:
    """The program text a script verb was given, and whether it reads its
    program from a FILE (which the gate cannot judge). Everything after the
    program is data."""
    texts: List[str] = []
    explicit = False
    rest = list(args)
    while rest:
        name, glued, value = rest[0].partition("=")
        if name in spec.file_options:
            return texts, True
        if name in spec.text_options:
            explicit = True
            if glued:
                texts, rest = [*texts, value], rest[1:]
            elif len(rest) >= 2:
                texts, rest = [*texts, rest[1]], rest[2:]
            else:
                break
            continue
        if name in spec.value_options:
            rest = rest[1:] if glued else rest[2:]
            continue
        if rest[0].startswith("-") and rest[0] != "-":
            rest = rest[1:]
            continue
        if not explicit and not texts:
            texts = [*texts, rest[0]]      # the first positional IS the program
        break
    return texts, False


def _judge_script(head: str, args: Sequence[str]) -> Tuple[Decision, List[str]]:
    """The program an allowed script verb would run, and the program texts
    themselves — a program is NOT a path (``sed '/foo/d'`` opens with a regex
    address that reads like one), so the caller leaves them out of the path
    check. Refused when the program can run a command of its own
    (``awk 'BEGIN{system(…)}'``, ``sed 's///e'``) or when the gate cannot read
    it at all (``-f progfile``)."""
    spec = SCRIPT_VERBS.get(head)
    if spec is None:
        return Decision("allow"), []
    texts, from_file = _script_texts(spec, args)
    if from_file:
        return Decision("deny", f"{head} reads its program from a file the gate cannot judge"), texts
    for text in texts:
        if spec.escape.search(text):
            return Decision("deny", f"{head} program runs a command of its own: {text[:60]!r}"), texts
    return Decision("allow"), texts


def _exec_split(words: Sequence[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """``find … -exec cmd …`` → the find arguments, the command's own words
    (placeholders dropped), and the option that introduced it."""
    for index, word in enumerate(words):
        if word in FIND_EXEC_OPTIONS:
            inner = [w for w in words[index + 1:] if w not in FIND_PLACEHOLDERS]
            return list(words[:index]), inner, word
    return list(words), [], None


def _judge_simple(words: Sequence[str], targets: Sequence[str], bindings: Bindings,
                  ctx: PolicyContext, roots: Sequence[Path], depth: int = 0) -> Decision:
    """One simple command, keywords and assignments already peeled: the verb
    against the never-allowed list and the allowlist, then every path it names."""
    on_targets = _judge_targets(targets, bindings, roots)
    if not words:
        return on_targets
    if words[0] == "cd":
        return _worst([on_targets, _judge_cd(words, bindings, roots)])
    words, on_globals = _peel_globals(words, bindings, roots)
    if on_globals.behavior == "deny":
        return on_globals
    joined = " ".join(words)
    for pattern in NEVER_ALLOWED_BASH:
        if pattern.search(joined):
            return Decision("deny", f"never allowed in a session: {_first_words(joined)!r} (sessions do not push or escalate)")
    for pattern, why in ALWAYS_ASK_BASH:
        if pattern.search(joined):
            # Not a refusal — the operator decides. It just never happens silently.
            return _worst([on_targets, Decision("ask", f"this command {why}")])
    inner = _unwrapped(words)
    if inner != list(words):
        # ``xargs git push``, ``timeout 5 git push``, ``env X=1 git push``: the
        # wrapper is off the allowlist and would be HELD — an operator can
        # approve a hold, and approving it runs the push. Refuse it here.
        joined_inner = " ".join(inner)
        for pattern in NEVER_ALLOWED_BASH:
            if pattern.search(joined_inner):
                return Decision("deny", f"never allowed in a session: {_first_words(joined_inner)!r} (sessions do not push or escalate)")
        if words[0] in FIND_EXEC_OPTIONS:
            # an orphan ``-exec cmd ;`` (a second exec clause the ``;`` split off)
            # is judged as the command it runs, like the first clause is
            return _judge_simple(inner, targets, bindings, ctx, roots, depth + 1) if inner else _worst(
                [on_targets, Decision("ask", f"'find {words[0]}' with no command — the operator decides")])
    if not (_matches_prefix(joined, ctx.allowed_bash) or _runs_own_code(words)):
        if _matches_prefix(joined, ctx.ask_bash):
            return _worst([on_targets, Decision("ask", f"{_first_words(joined)!r} needs the operator's approval")])
        # PRD-235 W2 S3: outside the allowlist is a QUESTION for the operator, not a
        # refusal — the session holds the call while a card is shown on the ticket's
        # Canvas; no answer in time is a deny (the ticket lands in review).
        if ctx.unlisted_bash == "allow":
            # ``--unlisted-bash allow`` (2026-09-18): the operator chose to run what the
            # list does not name. NEVER_ALLOWED_BASH was refused above, the explicit
            # ask-list still asks — and the ARGUMENTS are still judged as paths.
            #
            # They were not. This returned ``on_targets`` (redirections) alone, so an
            # unlisted verb could read any file on the machine: ``xxd /etc/passwd``,
            # ``od -c ~/.ssh/id_rsa``, ``strings``, ``base64`` — all allowed, while
            # ``cat`` of the same path was refused. The comment here and
            # test_unlisted_bash_allow_runs_unknown_verbs_but_keeps_the_hard_lines both
            # said otherwise; found 2026-09-22 when that test's stand-in verb changed.
            # "Allow what the list does not name" means the VERB, never the path.
            return _worst([Decision("allow", ALLOWED_UNLISTED_BASH.format(command=_first_words(joined))),
                           on_targets, on_globals, _judge_args(words[0], list(words[1:]), bindings, roots)])
        return _worst([on_targets, Decision("ask", f"{_first_words(joined)!r} is outside this ticket's Bash allowlist")])
    head = Path(words[0]).name
    outer, inner, exec_option = _exec_split(words) if head == "find" else (list(words), [], None)
    verdicts = [on_targets, on_globals]
    if head == "find":
        writes = next((w for w in words if w in FIND_WRITE_OPTIONS), None)
        if writes:
            verdicts = [*verdicts, Decision("ask", f"'find {writes}' writes or deletes per hit — the operator decides")]
    if exec_option:
        verdicts = [*verdicts, _judge_simple(inner, [], bindings, ctx, roots, depth) if inner
                    else Decision("ask", f"'find {exec_option}' with no command — the operator decides")]
    on_script, programs = _judge_script(head, outer[1:])
    verdicts = [*verdicts, on_script]
    if depth > 0 or head not in PATH_FREE_VERBS:
        arguments = [a for a in outer[1:] if a not in programs]
        verdicts = [*verdicts, _judge_args(outer[0], arguments, bindings, roots)]
    return _worst(verdicts)


def _bind_loop(words: Sequence[str], bindings: Bindings, roots: Sequence[Path]) -> Tuple[Decision, Bindings]:
    """``for NAME in WORDS``: the words are confined like arguments right here and
    NAME takes their values for the body; ``for NAME`` alone (the positional
    parameters) leaves NAME unknown."""
    if len(words) < 2 or not _NAME_RE.match(words[1]):
        return Decision("ask", f"{_first_words(' '.join(words))!r} is a loop the gate cannot read"), bindings
    loop_words = list(words[3:]) if len(words) > 2 and words[2] == "in" else []
    verdict = _judge_args("for … in", loop_words, bindings, roots)
    values = tuple(v for word in loop_words for v in _expand(word, bindings))
    bound = {name: vals for name, vals in bindings.items() if name != words[1]}
    return verdict, ({**bound, words[1]: values} if values else bound)


def _judge_segment(tokens: Sequence[str], bindings: Bindings, ctx: PolicyContext,
                   roots: Sequence[Path], depth: int) -> Tuple[Decision, Bindings]:
    """One simple command: its substitutions first (each a command line of its
    own), then the loop header, or the assignments and the verb."""
    words, targets = _peel_redirections(tokens)
    nested = [_judge_command(body, bindings, ctx, roots, depth + 1)
              for word in [*words, *targets] for body in _substitutions(word)]
    words = _strip_keywords([_without_substitutions(w) for w in words])
    targets = [_without_substitutions(t) for t in targets]
    if words and words[0] == "for":
        verdict, bound = _bind_loop(words, bindings, roots)
        return _worst([*nested, verdict]), bound
    words, bound = _bind_assignments(words, bindings)
    return _worst([*nested, _judge_simple(words, targets, bound, ctx, roots, depth)]), bound


def _judge_command(command: str, bindings: Bindings, ctx: PolicyContext,
                   roots: Sequence[Path], depth: int = 0) -> Decision:
    """A command line is judged by EVERY simple command in it (munder/Claude's
    own rule); the line's assignments and loop variables carry left to right. An
    unquoted here-document's body is judged too — the shell runs the
    substitutions inside it while it reads the body."""
    if depth > MAX_NESTING:
        return Decision("ask", f"{_first_words(command)!r} nests substitutions deeper than the gate reads")
    try:
        segments = _split_compound_depths(command)
    except ValueError:
        return Decision("ask", f"{_first_words(command)!r} could not be parsed (unbalanced quotes)")
    verdicts: List[Decision] = [
        _judge_command(body, bindings, ctx, roots, depth + 1)
        for text in _heredocs(command)[1] for body in _substitutions(text)
    ]
    bound = bindings
    for paren_depth, tokens in segments:
        verdict, bound = _judge_segment(tokens, bound, ctx, roots, depth + paren_depth)
        verdicts = [*verdicts, verdict]
    return _worst(verdicts)


Bases = Optional[Tuple[Path, ...]]      # the folders a command may run in; None = the gate cannot tell


def _moved(words: Sequence[str], bindings: Bindings, bases: Bases) -> Bases:
    """Where ``cd``/``pushd`` leaves the line: every value its target can take,
    or unknown (a reference nothing defined, a glob, ``cd -``, bare ``cd``)."""
    if bases is None or len(words) != 2 or words[1].startswith("-"):
        return None
    values = _expand(words[1], bindings)
    if not _resolved(values) or any(secret_reach.has_glob(v) for v in values):
        return None
    moved = []
    for base in bases:
        for value in values:
            target = Path(value).expanduser()
            moved.append((target if target.is_absolute() else base / target).resolve())
    return tuple(moved)


def _read_operands(words: Sequence[str]) -> List[str]:
    """The words a command reads as files: options, a grep/rg pattern and an
    awk/sed program are not files."""
    head = Path(words[0]).name
    if head in secret_reach.GREP_VERBS or head == "rg":
        return secret_reach.search_operands(words)
    programs: List[str] = []
    if head in SCRIPT_VERBS:
        programs, _ = _script_texts(SCRIPT_VERBS[head], words[1:])
    return [w for w in words[1:] if not w.startswith("-") and w not in programs]


def _operand_reach(value: str, bases: Bases, inv: secret_reach.Inventory, ctx: PolicyContext,
                   loose_globs: bool) -> Optional[Decision]:
    """One file operand, as the shell will expand it, from every folder the
    command may run in."""
    unresolved = _UNRESOLVED_RE.search(value) or SUBSTITUTION_MARK in value
    if unresolved:
        if bases is None or any(inv.holds_secrets(b) for b in bases):
            shown = "$(…)" if value in ("$", SUBSTITUTION_MARK) else repr(value)
            return Decision("deny", f"{shown} names a file the gate cannot read (a substitution or a variable the "
                                    "line does not define), in a folder that holds the platform's own secrets — "
                                    "spell the path out")
        return None
    alternatives, _ = secret_reach.expand_braces(value)
    for alternative in alternatives:
        name = Path(alternative).name
        if bases is None:
            if _is_secret_name(name) or (secret_reach.has_glob(name) and secret_reach.pattern_may_name(
                    name, inv.names(), leading_dot_literal=not loose_globs)):
                return Decision("deny", f"{alternative!r} after a cd the gate cannot resolve could be the platform's "
                                        "own secrets — no approval changes that")
            continue
        for base in bases:
            if secret_reach.has_glob(alternative):
                hit = secret_reach.glob_reaches(alternative, base, inv, leading_dot_literal=not loose_globs)
                if hit is not None:
                    return Decision("deny", f"{alternative!r} expands to {hit.name}, the platform's own secrets "
                                            "(or the CLI host's state) — no approval changes that")
                continue
            guard = _guard_secret(alternative, ctx, base)
            if guard is not None and guard.behavior == "deny":
                return guard
    return None


def _command_reach(words: Sequence[str], targets: Sequence[str], bindings: Bindings, bases: Bases,
                   inv: secret_reach.Inventory, ctx: PolicyContext, loose_globs: bool) -> Optional[Decision]:
    """One simple command: the files it opens by name (after the shell expands
    them), and the ones a recursive search or ``find -exec`` walks into."""
    head = Path(words[0]).name if words else ""
    operands = [*targets]
    if words and head not in secret_reach.NO_CONTENT_VERBS and head not in FIND_EXEC_OPTIONS and head != "find":
        operands += _read_operands(words)
    for operand in operands:
        for value in _expand(operand, bindings):
            verdict = _operand_reach(value, bases, inv, ctx, loose_globs)
            if verdict is not None:
                return verdict
    if words and secret_reach.reads_whole_tree(words):
        for operand in _read_operands(words):
            for value in _expand(operand, bindings):
                for base in bases or ():
                    tree = Path(value).expanduser()
                    hits = inv.reachable_under(tree if tree.is_absolute() else base / tree)
                    if hits:
                        return Decision("deny", f"{head} {value!r} would read {hits[0].name}, the platform's own "
                                                "secrets (or the CLI host's state) — no approval changes that")
        if bases is None:
            return Decision("deny", f"{head} from a folder the gate cannot resolve could read the platform's secrets")
    search = secret_reach.search_spec(words) if words else None
    if search is not None:
        if bases is None:
            return Decision("deny", f"{head} searches recursively from a folder the gate cannot resolve — it could "
                                    "walk into the platform's own secrets")
        for base in bases:
            hit = secret_reach.search_reads(search, base, inv)
            if hit is not None:
                return Decision("deny", f"{head} over {hit[0]!r} would read {hit[1].name}, the platform's own secrets"
                                        " (or the CLI host's state) — exclude them (grep --exclude='.env*' "
                                        "--exclude='.cred*', rg -g '!.env*' -g '!.cred*') or search a narrower folder")
    return None


def _find_reach(find: secret_reach.FindSpec, inner: Sequence[str], bases: Bases,
                inv: secret_reach.Inventory) -> Optional[Decision]:
    """``find … -exec CMD {}``: CMD opens every hit, so the walk's own filters
    decide whether a secret is among them."""
    if not inner or Path(inner[0]).name in secret_reach.NO_CONTENT_VERBS:
        return None
    if bases is None:
        return Decision("deny", "find -exec from a folder the gate cannot resolve could open the platform's secrets")
    for base in bases:
        hit = secret_reach.find_reads(find, base, inv)
        if hit is not None:
            return Decision("deny", f"find {hit[0]!r} -exec {Path(inner[0]).name} would open {hit[1].name}, the "
                                    "platform's own secrets — filter it out (! -name '.env*' ! -name '.cred*') "
                                    "or walk a narrower folder")
    return None


def _secret_reach_on_line(visible: str, ctx: PolicyContext) -> Optional[Decision]:
    """The platform's secrets reached WITHOUT being named (secret_reach): each
    simple command judged from the folder it runs in — a ``cd``/``pushd``
    earlier on the line moves it, a subshell's move ends with the subshell, and
    one the gate cannot resolve leaves it unknown. Quotes are already spliced
    (``.e""nv`` is ``.env``), braces and globs are expanded against the files
    that exist."""
    if not (ctx.secret_roots or ctx.off_limits):
        return None
    try:
        segments = _split_compound_depths(visible)
    except ValueError:
        return None                                   # the main judge asks on an unparseable line
    inv = _inventory(ctx)
    loose_globs = False
    roots = [ctx.cwd, *ctx.extra_dirs]
    stack: List[Bases] = [(Path(ctx.cwd).expanduser().resolve(),)]
    bindings: Bindings = {}
    last_find: Optional[Tuple[secret_reach.FindSpec, Bases]] = None
    for depth, tokens in segments:
        stack = stack[:depth + 1] if len(stack) > depth else [*stack, *[stack[-1]] * (depth + 1 - len(stack))]
        words, targets = _peel_redirections(tokens)
        words = _strip_keywords([_without_substitutions(w) for w in words])
        targets = [_without_substitutions(t) for t in targets]
        if words and words[0] == "for":
            loop_words = list(words[3:]) if len(words) > 2 and words[2] == "in" else []
            _, bindings = _bind_loop(words, bindings, roots)
            verdict = _command_reach(["for", *loop_words], targets, {}, stack[depth], inv, ctx, loose_globs)
            if verdict is not None:
                return verdict
            continue
        assigned = [w.split("=", 1)[0] for w in words if _ASSIGNMENT_RE.match(w)]
        words, bindings = _bind_assignments(words, bindings)
        if not words and "GLOBIGNORE" in assigned:
            loose_globs = True                      # a bare GLOBIGNORE= turns dotglob on
        words = _unwrapped(words) if words and words[0] in COMMAND_WRAPPERS - set(FIND_EXEC_OPTIONS) else words
        head = Path(words[0]).name if words else ""
        if head == "shopt" and "-s" in words and _LOOSE_GLOB_OPTIONS & set(words):
            loose_globs = True
            continue
        if head in ("cd", "pushd"):
            stack = [*stack[:depth], _moved(words, bindings, stack[depth])]
            continue
        if head == "popd":
            stack = [*stack[:depth], None]
            continue
        if words and words[0] in FIND_EXEC_OPTIONS and last_find is not None:
            # a second ``-exec`` split off by its ``;`` runs per hit of the find before it
            inner = [w for w in words[1:] if w not in FIND_PLACEHOLDERS and w != ";"]
            verdict = _find_reach(last_find[0], inner, last_find[1], inv)
            if verdict is not None:
                return verdict
            continue
        found = secret_reach.find_spec(words) if head == "find" else None
        if found is not None:
            last_find = (found[0], stack[depth])
            verdict = _find_reach(found[0], found[1], stack[depth], inv)
            if verdict is not None:
                return verdict
        verdict = _command_reach(words, targets, bindings, stack[depth], inv, ctx, loose_globs)
        if verdict is not None:
            return verdict
    return None


def decide_bash(command: str, ctx: PolicyContext) -> Decision:
    # The raw nets read the COMMAND LINE. A here-document's body is data the
    # line writes, not a command in it: an unquoted body's substitutions are
    # judged as command lines of their own below (``_judge_command``), and a
    # quoted body runs nothing at all — so a file whose text happens to contain
    # ``$(git push)`` or ``..`` is not refused for what it says.
    visible = _heredocs(command)[0]
    for pattern in NEVER_ALLOWED_BASH:
        if pattern.search(visible):
            return Decision("deny", f"never allowed in a session: {_first_words(visible)!r} (sessions do not push or escalate)")
    secret = _secret_on_line(visible, ctx) or _secret_reach_on_line(visible, ctx)
    if secret is not None:
        return secret
    if ".." in visible and re.search(r"(^|[\s'\"=:;|&(/])\.\.([/\\]|[\s'\");|&]|$)", visible):
        return Decision("deny", "path traversal ('..') in a shell command")
    if _ANSI_C_RE.search(visible):  # a ``$`` that OPENS a word — ``'^$'`` closing a regex is not quoting
        # ``$'…'`` is ANSI-C quoting: ``$'/etc/passwd'`` IS ``/etc/passwd`` and
        # ``$'\x2f'`` is ``/``. The tokenizer strips the quotes and leaves the
        # ``$``, so the path no longer looks like one and escapes the roots.
        return Decision("ask", "ANSI-C ($'…') or locale ($\"…\") quoting hides a word the gate cannot read")
    roots = [ctx.cwd, *ctx.extra_dirs]
    # The BACKTICK substitutions on the line, judged as command lines of their
    # own — quoted or not. An UNQUOTED backtick body splits on its own spaces
    # when tokenized (`` echo `cat X` `` becomes three words), so the path inside
    # it became an argument of ``echo``, which names no paths: a silent read of
    # any file, the host's own credential included. Only backticks, though. A
    # ``$(…)`` is kept whole when quoted and becomes its own depth-1 segment when
    # not, and the segment pass judges it WITH the bindings in scope — ``for d in
    # …`` binds ``$d`` — which a raw-line pass cannot know. Judging those here too
    # held ``for d in a b; do echo "$(ls x/$d)"; done`` for a ``$d`` it could not
    # resolve, and a real ticket sat in review on it.
    inside = [_judge_command(body, {}, ctx, roots, 1) for body in _backtick_bodies(visible)]
    return _worst([*inside, _judge_command(command, {}, ctx, roots)])


def decide(intent: ToolIntent, ctx: PolicyContext) -> Decision:
    roots = [ctx.cwd, *ctx.extra_dirs]
    if intent.cls in (ToolClass.FILE_READ, ToolClass.FILE_WRITE):
        # F042 first: the platform's secrets are refused wherever they sit, and
        # any other secrets file asks — as the same file read through Bash does.
        guards = [_guard_secret(str(target), ctx, ctx.cwd) for target in intent.paths]
        guards.append(_guard_search(intent.tool, getattr(intent, "globs", ()) or (), intent.paths, ctx))
        guards = [g for g in guards if g is not None]
        if any(g.behavior == "deny" for g in guards):
            return _worst(guards)
        if not intent.paths:
            return _worst([*guards, Decision("allow", ALLOWED_FILES)])  # a search without a path works in cwd
        for target in intent.paths:
            if not _inside(str(target), roots):
                return Decision("deny", f"{intent.tool} outside the session directory: {target}")
        return _worst([*guards, Decision("allow", ALLOWED_FILES)])
    if intent.cls is ToolClass.SHELL:
        # An unlisted verb's own reason comes first and wins over the allowlist's.
        return _worst([decide_bash(str(intent.command or ""), ctx), Decision("allow", ALLOWED_BASH)])
    if intent.cls is ToolClass.PLATFORM:
        name = str(intent.command or "")
        if name and name in set(ctx.session_tools or ()):
            return Decision("allow", ALLOWED_SESSION_TOOL)
        offered = ", ".join(ctx.session_tools or ()) or "none in this ticket"
        return Decision("deny", f"Automatos tool {name!r} is not one this ticket may call ({offered})")
    if intent.cls in (ToolClass.WEB, ToolClass.BENIGN):
        return Decision("allow", ALLOWED_NO_APPROVAL)
    return Decision("deny", f"tool {intent.tool!r} is not enabled for session tickets")


def bash_allowlist_from_config(configured: Optional[Iterable[str]]) -> Sequence[str]:
    """The ticket's Bash allowlist: the agent's configured prefixes on top of the defaults."""
    extra = [c.strip() for c in (configured or []) if isinstance(c, str) and c.strip()]
    return tuple(DEFAULT_BASH_ALLOW) + tuple(extra)
