"""The command gate's dev-box lines (dev-box mode): what ``policy.py`` hands over.

In dev-box mode (``--dev-box``, with the session sandbox on and the GitHub token loaded)
a Claude Code session pushes its own branch and opens its own pull request; GitHub's
rules are the wall and this gate is the second layer. ``policy.py`` consults this module
in five places, a line each. With the mode off (``ctx.dev_box`` is None) every one of
them is the identity, so the gate is exactly what it was:

* ``refusal`` — one simple command: git (``dev_box_git``, ``dev_box_config``) and gh
  (``dev_box_gh``) by the dev-box table; a push or a gh write run BY another command
  (``xargs``, ``timeout``, ``find -exec``, ``parallel``: it can be handed arguments the
  gate never sees); a command that prints the environment, the token with it; any
  command but a read-only git verb that names a ``.git`` path; a file gh would read
  (``-F key=@file``, ``--input``, ``--body-file``) outside the session's folders;
* ``nets`` — the raw never-allowed nets without the two publishing ones
  (``PUBLISH_NETS``, marked by identity), which ``refusal`` judges instead;
* ``gh_writes`` — today's gh rule; in the mode ``dev_box_gh`` decides instead;
* ``line_refusal`` — what the whole line says: a mention of ``GH_TOKEN``, an indirect
  expansion, a read of a process's environment, an assignment that swaps the
  credential or how git and gh find their config, programs and repository;
* ``write_refusals`` — the Write/Edit tools inside a ``.git`` folder.

``allowed_bash`` is what the session adds to its Bash allowlist, so what dev-box mode
allows runs in every mode but Plan instead of waiting on a card.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional, Pattern, Sequence, Set, Tuple

from . import dev_box_gh, dev_box_git
from .adapters.base import ToolClass, ToolIntent
from .dev_box_rules import DevBoxRules, refused, rules, shown, unreadable
from .shell_data import data_positions, non_path_positions
from .shell_text import heredocs
from .shell_words import gh_writes as gh_writes_today
from .shell_words import strip_globals

log = logging.getLogger("automatos.cli_host.dev_box")

# policy.NEVER_ALLOWED_BASH's two publishing nets, defined here so the mode lifts
# exactly these, by identity, and nothing else on that list.
PUBLISH_NETS: Tuple[Pattern[str], ...] = (
    re.compile(r"(^|[;&|(]\s*)git\s+push\b"),
    re.compile(r"(^|[;&|(]\s*)git\s+remote\s+(add|set-url)\b"),
)
DEV_BOX_BASH_ALLOW: Tuple[str, ...] = (
    "git push", "git fetch", "git pull", "git rebase", "git merge", "git cherry-pick", "git reset", "git revert",
    "git tag", "git worktree", "git checkout", "git switch", "git mv", "git rm", "git remote", "git config",
    "git ls-remote", "git merge-base", "git rev-list", "git show-ref",
    *(f"gh {sub} {action}" for sub, actions in sorted(dev_box_gh.GH_ALLOWED.items()) for action in sorted(actions)),
    "gh api",
)

# ── what prints the environment, GH_TOKEN with it ───────────────────────────
PRINTS_ENVIRONMENT = "prints the environment, GH_TOKEN with it"
ENV_VALUE_OPTIONS = frozenset({"-u", "--unset", "-C", "--chdir", "-P"})
ENV_COMMAND_OPTIONS = ("-S", "--split-string")            # the value is the command line env runs
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
JQ_ENV_RE = re.compile(r"\$ENV\b|(?<![\w.$])env\b")      # jq's ``env`` and ``$ENV``, not a ``.env`` field
AWK_ENV_WORD = "ENVIRON"

# ── what the whole line says ────────────────────────────────────────────────
TOKEN_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])GH_TOKEN(?![A-Za-z0-9_])")
INDIRECT_EXPANSION_RE = re.compile(r"\$\{!")
PROC_ENVIRON_RE = re.compile(r"/proc/[^\s'\";|&<>]*environ")
# Variables that swap the credential, or where git and gh find their config, their
# programs, their repository and what they print.
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s;&|(])(GITHUB_TOKEN|GH_ENTERPRISE_TOKEN|GITHUB_ENTERPRISE_TOKEN|GH_HOST|GH_CONFIG_DIR|GH_REPO|"
    r"GIT_CONFIG\w*|GIT_ASKPASS|SSH_ASKPASS|GIT_SSH|GIT_SSH_COMMAND|GIT_EXEC_PATH|GIT_DIR|GIT_WORK_TREE|"
    r"GIT_NAMESPACE|GIT_EXTERNAL_DIFF|GIT_PAGER|GIT_EDITOR|GIT_SEQUENCE_EDITOR|GIT_PROXY_COMMAND|"
    r"GIT_ALLOW_PROTOCOL|GIT_TRACE\w*|GIT_CURL_VERBOSE|HOME|XDG_CONFIG_HOME)=")

# ── a .git folder ───────────────────────────────────────────────────────────
GIT_FOLDER = ".git"
INSIDE_GIT_FOLDER = ("a session never touches a repository's .git folder but through git's own read-only "
                     "commands: its config and hooks run code and redirect pushes")
GIT_READ_VERBS = frozenset({"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "ls-tree", "cat-file"})
BRANCH_LIST_FLAGS = frozenset({"--list", "-l"})
GIT_OUTPUT_OPTION = "--output"                             # a read-only verb that still writes a file
# A name filter is a pattern, not a path: ``find . -name .git -prune``, ``grep --exclude-dir=.git``.
FILTER_VERBS = frozenset({"grep", "egrep", "fgrep", "rg", "find", "tree"})
FILTER_OPTIONS = frozenset({"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "--exclude",
                            "--exclude-dir", "--include", "-g", "--glob", "--iglob", "-I"})
FIND_ACTIONS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"})
_PATH_SEPARATORS_RE = re.compile(r"[/\\]")
_PLAIN_PATH_RE = re.compile(r"^[\w./~+@%:,-]+$")

# ── files gh reads ──────────────────────────────────────────────────────────
STDIN_VALUES = frozenset({"-", ""})


def allowed_bash(dev_box: Optional[DevBoxRules]) -> Tuple[str, ...]:
    """What dev-box mode adds to a session's Bash allowlist (nothing with the mode off)."""
    return DEV_BOX_BASH_ALLOW if dev_box is not None else ()


def nets(patterns: Tuple[Pattern[str], ...], dev_box: Optional[DevBoxRules]) -> Tuple[Pattern[str], ...]:
    """``policy.NEVER_ALLOWED_BASH`` as the mode reads it: without ``PUBLISH_NETS``."""
    if dev_box is None:
        return patterns
    return tuple(pattern for pattern in patterns if not any(pattern is net for net in PUBLISH_NETS))


def gh_writes(words: Sequence[str], dev_box: Optional[DevBoxRules]) -> bool:
    """Today's rule (every gh but a known read is refused); in the mode, ``refusal`` decides."""
    return gh_writes_today(words) if dev_box is None else False


def refusal(words: Sequence[str], inner: Sequence[str], targets: Sequence[str], ctx: Any) -> Optional[str]:
    """Why dev-box mode refuses one simple command, or None. ``inner`` is what a wrapper
    runs (``policy._unwrapped``), ``targets`` its redirections, ``ctx`` the session's
    ``policy.PolicyContext`` (its rules, folders and the host's state)."""
    if ctx.dev_box is None:
        return None
    return (_git_folder_refusal(words, inner, targets, ctx.cwd) or _environment_refusal(words, inner)
            or _hidden_publish(words) or _tool_refusal(words, ctx) or _tool_refusal(inner, ctx))


def _tool_refusal(words: Sequence[str], ctx: Any) -> Optional[str]:
    head = Path(words[0]).name if words else ""
    if head == "git":
        return dev_box_git.git_refusal(words, ctx.dev_box)
    if head == "gh":
        return dev_box_gh.gh_refusal(words, ctx.dev_box) or _gh_file_refusal(words, ctx)
    return None


def _hidden_publish(words: Sequence[str]) -> Optional[str]:
    """A push, a remote change or a gh write run BY another command: it can be handed
    arguments the gate never sees (``echo main | xargs git push origin x``), so what the
    mode allows a plain ``git`` or ``gh`` stays refused there."""
    for index, word in enumerate(words[1:], start=1):
        name, tail = Path(word).name, list(words[index:])
        git_publishes = name == "git" and dev_box_git.publishes(tail)
        if git_publishes or (name == "gh" and len(tail) > 1 and dev_box_gh.publishes(tail)):
            return refused(words, f"{name} run by {Path(words[0]).name!r} gets arguments the gate cannot see: "
                                  "run it as a command of its own")
    return None


# ── files gh reads ──────────────────────────────────────────────────────────

def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _file_outside(value: str, ctx: Any) -> Optional[str]:
    """Why gh may not read this file: unreadable, the host's state, or outside the session's folders."""
    if value in STDIN_VALUES:
        return None
    if unreadable(value):
        return f"gh reads a file the gate cannot name ({value!r})"
    path = Path(value).expanduser()
    try:
        resolved = (path if path.is_absolute() else Path(ctx.cwd) / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("dev-box mode: %r cannot be resolved (%s): refused", value, exc)
        return f"gh reads {value}, which the gate cannot resolve"
    if any(_within(resolved, Path(root).expanduser().resolve()) for root in ctx.off_limits):
        return f"gh reads {value} from the CLI host's own state"
    if not any(_within(resolved, Path(root).expanduser().resolve()) for root in (ctx.cwd, *ctx.extra_dirs)):
        return f"gh reads {value}, outside the session's folders"
    return None


def _gh_file_refusal(words: Sequence[str], ctx: Any) -> Optional[str]:
    """``-F key=@file``, ``--input``, ``--body-file``: a file gh sends to GitHub is the session's own."""
    for value in dev_box_gh.file_values(words):
        why = _file_outside(value, ctx)
        if why:
            return refused(words, why)
    return None


# ── the environment ─────────────────────────────────────────────────────────

def _always(_: Sequence[str]) -> bool:
    return True


def _no_arguments(words: Sequence[str]) -> bool:
    return len(words) == 1


def _env_prints(words: Sequence[str]) -> bool:
    """``env`` with no command after its options and assignments prints the environment."""
    index = 1
    while index < len(words):
        word = words[index]
        if word.startswith(ENV_COMMAND_OPTIONS):
            return False
        if word in ENV_VALUE_OPTIONS:
            index += 2
            continue
        if not (word.startswith("-") or _ASSIGNMENT_RE.match(word)):
            return False
        index += 1
    return True


def _declare_prints(words: Sequence[str]) -> bool:
    """``export``, ``declare``, ``typeset`` with no names, or with ``-p``, print variables."""
    args = list(words[1:])
    names = [arg for arg in args if not arg.startswith(("-", "+"))]
    return not names or any(arg.startswith("-") and "p" in arg[1:] for arg in args)


def _ps_prints(words: Sequence[str]) -> bool:
    """``ps e`` / ``ps axe`` (BSD style) and macOS ``ps -E`` print each process's environment."""
    args = list(words[1:])
    if args and not args[0].startswith("-") and "e" in args[0]:
        return True
    return any(arg.startswith("-") and not arg.startswith("--") and "E" in arg for arg in args)


def _jq_prints(words: Sequence[str]) -> bool:
    """jq's ``env`` and ``$ENV`` in its filter; every word when the gate cannot find the filter."""
    positions = data_positions(words) or range(1, len(words))
    return any(JQ_ENV_RE.search(words[index]) for index in positions)


def _awk_prints(words: Sequence[str]) -> bool:
    return any(AWK_ENV_WORD in word for word in words[1:])


ENVIRONMENT_PRINTERS: Dict[str, Tuple[Callable[[Sequence[str]], bool], str]] = {
    "printenv": (_always, "printenv"), "env": (_env_prints, "env with no command"),
    "set": (_no_arguments, "set with no arguments"), "export": (_declare_prints, "export with no names or -p"),
    "declare": (_declare_prints, "declare with no names or -p"), "typeset": (_declare_prints, "typeset with no names"),
    "ps": (_ps_prints, "ps e / ps -E"), "jq": (_jq_prints, "jq's env / $ENV"), "gojq": (_jq_prints, "gojq's env"),
    "awk": (_awk_prints, "awk's ENVIRON"), "gawk": (_awk_prints, "awk's ENVIRON"), "mawk": (_awk_prints, "awk's ENVIRON"),
}


def _prints_environment(words: Sequence[str]) -> Optional[str]:
    entry = ENVIRONMENT_PRINTERS.get(Path(words[0]).name) if words else None
    return f"{entry[1]} {PRINTS_ENVIRONMENT}" if entry and entry[0](words) else None


def _environment_refusal(words: Sequence[str], inner: Sequence[str]) -> Optional[str]:
    for command in (words, inner):
        why = _prints_environment(command)
        if why:
            return refused(command, why)
    if words and not inner and any(Path(word).name == "env" for word in words):    # ``nohup env``
        return refused(words, f"env with no command {PRINTS_ENVIRONMENT}")
    return None


def line_refusal(command: str, dev_box: Optional[DevBoxRules]) -> Optional[str]:
    """What the whole line says, read before its words: the token named (``$GH_TOKEN``,
    ``os.environ['GH_TOKEN']``, ``ENVIRON["GH_TOKEN"]``), an indirect ``${!name}``, a
    process's environment read from ``/proc``, a credential or git variable assigned.
    Read on the line and on the here-document bodies the shell expands, never on a
    quoted body (data)."""
    if dev_box is None:
        return None
    visible, expanded = heredocs(command)
    texts = (visible, *expanded)
    what = shown(visible.split())
    if any(TOKEN_NAME_RE.search(text) for text in texts):
        return refused(what, "names GH_TOKEN, the session's GitHub token: git and gh read it by themselves")
    if any(INDIRECT_EXPANSION_RE.search(text) for text in texts):
        return refused(what, "${!name} expands a variable named only when it runs (GH_TOKEN, say)")
    if any(PROC_ENVIRON_RE.search(text) for text in texts):
        return refused(what, f"a process's environment in /proc {PRINTS_ENVIRONMENT}")
    assigned = CREDENTIAL_ASSIGNMENT_RE.search(visible)
    if assigned:
        return refused(what, f"{assigned.group(1)}= swaps the session's credential, or where git and gh find their "
                             "config, programs or repository")
    return None


# ── a .git folder ───────────────────────────────────────────────────────────

def _names_git_part(word: str) -> bool:
    """A ``.git`` part (any case: APFS folds it) in a word, an option's value
    (``of=.git/config``, ``--output=.git/x``) or a glued short option's (``-o.git/x``)."""
    pieces: Set[str] = {word, word.split("=", 1)[-1]}
    if word.startswith("-") and not word.startswith("--"):
        pieces = {*pieces, word[2:]}
    return any(part.lower() == GIT_FOLDER for piece in pieces for part in _PATH_SEPARATORS_RE.split(piece))


def _links_into_git_folder(word: str, cwd: Path) -> bool:
    """A plain path word that, through a link on disk, lands in a ``.git`` folder."""
    if word.startswith("-") or "://" in word or not _PLAIN_PATH_RE.match(word):
        return False
    path = Path(word).expanduser()
    try:
        real = os.path.realpath(path if path.is_absolute() else Path(cwd) / path)
    except (OSError, ValueError) as exc:
        log.warning("dev-box mode: %r cannot be resolved (%s): treated as a .git path and refused", word, exc)
        return True
    return any(part.lower() == GIT_FOLDER for part in Path(real).parts)


def names_git_folder(word: str, cwd: Path) -> bool:
    """A word naming a path in a ``.git`` folder, as written or through a link."""
    return _names_git_part(word) or _links_into_git_folder(word, cwd)


def _filter_positions(words: Sequence[str]) -> FrozenSet[int]:
    """Which words of a search are name filters (``-name .git``, ``--exclude-dir=.git``)."""
    head = Path(words[0]).name
    if head not in FILTER_VERBS or (head == "find" and FIND_ACTIONS & set(words)):
        return frozenset()
    after = {index + 1 for index, word in enumerate(words) if word in FILTER_OPTIONS}
    glued = {index for index, word in enumerate(words) if word.partition("=")[0] in FILTER_OPTIONS and "=" in word}
    return frozenset(after | glued)


def _git_reads_only(peeled: Sequence[str]) -> bool:
    sub = peeled[1] if len(peeled) > 1 else ""
    return sub in GIT_READ_VERBS or (sub == "branch" and bool(BRANCH_LIST_FLAGS & set(peeled[2:])))


def _git_output_word(peeled: Sequence[str]) -> Tuple[str, ...]:
    """The files a read-only git verb still writes (``--output=<file>``, ``--output <file>``)."""
    glued = tuple(w.partition("=")[2] for w in peeled if w.startswith(f"{GIT_OUTPUT_OPTION}="))
    after = tuple(peeled[i + 1] for i, w in enumerate(peeled[:-1]) if w == GIT_OUTPUT_OPTION)
    return (*glued, *after)


def _first_in_git_folder(words: Sequence[str], cwd: Path) -> Optional[str]:
    return next((word for word in words if names_git_folder(word, cwd)), None)


def _git_folder_word_in_git(words: Sequence[str], cwd: Path) -> Optional[str]:
    """``git -C .git …``, ``--git-dir``/``--work-tree`` into it, or any git but a read naming it."""
    peeled, places = strip_globals(words)
    where = _first_in_git_folder(places, cwd)
    if where is not None:
        return where
    if _git_reads_only(peeled):
        return _first_in_git_folder(_git_output_word(peeled), cwd)
    data = set(non_path_positions(peeled))
    return _first_in_git_folder([word for index, word in enumerate(peeled) if index > 1 and index not in data], cwd)


def _git_folder_word(words: Sequence[str], cwd: Path) -> Optional[str]:
    """The ``.git`` path a command names, its patterns, programs and name filters left out."""
    if not words:
        return None
    if Path(words[0]).name == "git":
        return _git_folder_word_in_git(words, cwd)
    skip = set(non_path_positions(words)) | _filter_positions(words)
    return next((word for index, word in enumerate(words) if index not in skip and names_git_folder(word, cwd)), None)


def _git_folder_refusal(words: Sequence[str], inner: Sequence[str], targets: Sequence[str], cwd: Path) -> Optional[str]:
    target = next((t for t in targets if names_git_folder(t, cwd)), None)
    if target is not None:
        return refused(words if words else f"> {target}", f"a redirection to {target}: {INSIDE_GIT_FOLDER}")
    for command in (words, inner):
        word = _git_folder_word(command, cwd)
        if word is not None:
            return refused(command, f"{Path(command[0]).name} {word}: {INSIDE_GIT_FOLDER}")
    return None


def _path_in_git_folder(path: str, cwd: Path) -> bool:
    """A Write/Edit path inside a ``.git`` folder, as written or as resolved. A path the
    host cannot resolve is logged and refused: the gate cannot say where it lands."""
    written = Path(path).expanduser()
    written = written if written.is_absolute() else Path(cwd) / written
    try:
        resolved = written.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("dev-box mode: %s cannot be resolved (%s): the write is refused", written, exc)
        return True
    return any(part.lower() == GIT_FOLDER for form in (written, resolved) for part in form.parts)


def write_refusals(intent: ToolIntent, cwd: Path, dev_box: Optional[DevBoxRules]) -> Tuple[str, ...]:
    """The Write/Edit tools inside a ``.git`` folder, as written or through a link: refused."""
    if dev_box is None or intent.cls is not ToolClass.FILE_WRITE:
        return ()
    return tuple(refused(f"{intent.tool} {path}", INSIDE_GIT_FOLDER) for path in intent.paths
                 if _path_in_git_folder(str(path), cwd))


__all__ = [
    "DEV_BOX_BASH_ALLOW", "DevBoxRules", "PUBLISH_NETS", "allowed_bash", "gh_writes", "line_refusal",
    "names_git_folder", "nets", "refusal", "rules", "write_refusals",
]
