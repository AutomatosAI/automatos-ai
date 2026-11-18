"""git in a dev-box session: what runs and what is refused (dev-box mode, ``dev_box_policy.py``).

Normal development runs: fetch, pull, rebase, merge, cherry-pick, reset, tag, worktree,
and ``git push`` of branches the session names. Refused, with the rule in the reason:

* a push that could land on a protected name (``dev_box_rules``), in every spelling the
  gate can read (``HEAD:main``, ``:main``, ``+x:refs/heads/main``, ``heads/Main``), and a
  push whose destination it cannot read: no remote or no branch named, ``HEAD`` with no
  destination, a variable, a substitution, a glob;
* a push of more than the refs it names (``--mirror``, ``--all``, ``--tags``, ``--prune``,
  ``--follow-tags``) or one that runs a program (``--receive-pack``, ``--exec``);
* every config key not on the safe list (``dev_box_config``), set through ``git config``,
  ``git -c``, ``--config-env`` or ``git clone -c``;
* what runs a program the gate never sees: ``git rebase -x``/``--exec``, ``git submodule
  foreach``, ``git bisect run``, ``difftool``, ``mergetool``, ``--upload-pack``, a clone's
  or an init's ``--template``, ``git archive --remote``/``--exec``;
* a remote that is not a github.com repository of origin's owner, and a second or
  separate push URL;
* ``git credential`` (it prints the token), ``--exec-path=`` (it swaps git's own
  programs), and a subcommand the gate does not know: an alias or an extension runs
  what the gate cannot read.

git accepts a unique abbreviation of a long option (``--ex`` for ``--exec``), so a
refused long option is matched by every prefix git would take, and an option the
push judge does not know is refused.

Pure functions over the words of one simple command; ``None`` = not refused here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .dev_box_config import UNSAFE_KEY_WHY, config_refusal, safe_key, split_words
from .dev_box_rules import DevBoxRules, github_owner, refused, unreadable
from .shell_words import GIT_FLAG_OPTIONS, GIT_PATH_OPTIONS, GIT_VALUE_OPTIONS

# git's own subcommands. Anything else is an alias or an extension (``git-foo`` on the
# PATH): the gate cannot read what it runs, so dev-box mode refuses it.
KNOWN_GIT = frozenset({
    "add", "am", "apply", "archive", "bisect", "blame", "branch", "bundle", "cat-file", "check-attr",
    "check-ignore", "check-mailmap", "check-ref-format", "checkout", "cherry", "cherry-pick", "clean", "clone",
    "commit", "commit-graph", "commit-tree", "config", "count-objects", "describe", "diff", "diff-files",
    "diff-index", "diff-tree", "difftool", "fetch", "for-each-ref", "format-patch", "fsck", "gc", "grep",
    "hash-object", "help", "init", "interpret-trailers", "log", "ls-files", "ls-remote", "ls-tree",
    "maintenance", "merge", "merge-base", "merge-file", "merge-tree", "mergetool", "mktag", "mktree", "mv",
    "name-rev", "notes", "pull", "push", "range-diff", "read-tree", "rebase", "reflog", "remote", "repack",
    "replace", "rerere", "reset", "restore", "rev-list", "rev-parse", "revert", "rm", "shortlog", "show",
    "show-branch", "show-ref", "sparse-checkout", "stash", "status", "submodule", "switch", "symbolic-ref",
    "tag", "update-index", "update-ref", "var", "verify-commit", "verify-pack", "verify-tag", "version",
    "whatchanged", "worktree", "write-tree", "--version", "--help", "-h",
})
CREDENTIAL_SUBCOMMAND = "credential"          # credential, credential-store, credential-cache …
EXEC_PATH_OPTION = "--exec-path"
CONFIG_GLOBALS = frozenset({"-c", "--config-env"})
VALUED_GLOBALS = frozenset(o for o in (*GIT_PATH_OPTIONS, *GIT_VALUE_OPTIONS) if o != EXEC_PATH_OPTION)

# ── verbs and options that run a program the gate never sees ─────────────────
REBASE_EXEC_NAMES = frozenset({"--ex", "--exe", "--exec"})     # every abbreviation git takes for --exec
REBASE_EXEC_SHORT = "x"
UPLOAD_PACK_PREFIX = "--upl"                                    # --upload-pack and its abbreviations
TEMPLATE_PREFIX = "--templ"                                     # --template and its abbreviations
ARCHIVE_PROGRAM_PREFIXES = ("--e", "--rem")                     # git archive --exec, --remote
CLONE_UPLOAD_PACK_SHORT = "-u"
CLONE_CONFIG_OPTIONS = frozenset({"-c", "--config"})
ALWAYS_RUN_A_PROGRAM: Dict[str, str] = {
    "difftool": "git difftool runs a diff program the gate cannot read",
    "mergetool": "git mergetool runs a merge program the gate cannot read",
}

# ── git push ────────────────────────────────────────────────────────────────
PUSH_REFUSED_OPTIONS: Dict[str, str] = {
    "--mirror": "--mirror pushes every ref", "--all": "--all pushes every branch",
    "--branches": "--branches pushes every branch", "--tags": "--tags pushes every tag (push one tag by name)",
    "--follow-tags": "--follow-tags pushes tags along with the branch", "--prune": "--prune deletes remote branches",
    "--repo": "--repo names the remote as an option (name it as an argument)",
    "--receive-pack": "--receive-pack runs a program of its choosing", "--exec": "--exec runs a program of its choosing",
}
PUSH_FLAGS = frozenset({
    "-u", "--set-upstream", "-f", "--force", "-n", "--dry-run", "-q", "--quiet", "-v", "--verbose", "--progress",
    "--no-progress", "--porcelain", "--atomic", "--no-atomic", "--verify", "--no-verify", "--thin", "--no-thin",
    "--force-with-lease", "--no-force-with-lease", "--force-if-includes", "--no-force-if-includes", "--signed",
    "--no-signed", "-4", "-6", "--ipv4", "--ipv6", "--no-recurse-submodules", "-d", "--delete",
})
PUSH_SHORT_FLAGS = frozenset("ufnqvd46")
PUSH_VALUE_OPTIONS = frozenset({"-o", "--push-option"})
PUSH_GLUED_OPTIONS = frozenset({"--force-with-lease", "--signed", "--push-option", "--recurse-submodules"})
SAFE_SUBMODULE_MODES = frozenset({"check", "no"})
DELETE_OPTIONS = frozenset({"-d", "--delete"})
HEAD_NAMES = frozenset({"HEAD", "@"})
REVISION_MARKS = ("~", "^", "@{")
REFSPEC_GLOB_CHARS = "*?["
TAG_KEYWORD = "tag"                            # ``git push origin tag v1`` is refs/tags/v1
KIND_BRANCH, KIND_TAG, KIND_EITHER = "branch", "tag", "either"
REF_PREFIXES = (("refs/heads/", KIND_BRANCH), ("refs/tags/", KIND_TAG), ("heads/", KIND_BRANCH),
                ("tags/", KIND_TAG))
_REMOTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NAME_THE_BRANCH = "name the remote and the branch: git push -u origin <branch>"
UNREADABLE_DESTINATION = ("the gate cannot read where it lands (a variable, a substitution or a glob): "
                          "spell the branch out")

# ── git remote ──────────────────────────────────────────────────────────────
REMOTE_READS = frozenset({"show", "get-url", "prune", "update"})
REMOTE_VALUE_OPTIONS = frozenset({"-t", "--track", "-m", "--master"})
REMOTE_REFUSED_OPTIONS: Dict[str, str] = {
    "--mirror": "a mirror remote pushes every ref", "--push": "a separate push URL sends pushes elsewhere",
    "--add": "a second URL sends pushes elsewhere too", "--delete": "removes a remote's URL",
}


@dataclass(frozen=True)
class Globals:
    """git's global options as dev-box mode reads them: the config keys ``-c`` and
    ``--config-env`` set, the words from the subcommand on, or why it cannot read them."""
    config_keys: Tuple[str, ...] = ()
    rest: Tuple[str, ...] = ()
    problem: Optional[str] = None


def _global_value(args: Sequence[str], index: int) -> Tuple[Optional[str], int]:
    """The value of the global option at ``index`` (glued with ``=`` or the next word) and
    the index after it; ``None`` when it has none."""
    _, glued, value = args[index].partition("=")
    if glued:
        return value, index + 1
    return (args[index + 1], index + 2) if index + 1 < len(args) else (None, index + 1)


def read_globals(args: Sequence[str]) -> Globals:
    """``git -C x -c k=v --no-pager push …`` → the keys set and ``push …``."""
    keys: Tuple[str, ...] = ()
    index = 0
    while index < len(args):
        name, glued, _ = args[index].partition("=")
        if name == EXEC_PATH_OPTION:      # bare: git prints its own path and exits
            return Globals(keys, (), "--exec-path= swaps git's own programs") if glued else Globals(keys)
        if name in GIT_FLAG_OPTIONS and not glued:
            index += 1
            continue
        if name not in VALUED_GLOBALS:
            break
        value, index = _global_value(args, index)
        if value is None:
            return Globals(problem=f"{name} with no value")
        keys = (*keys, value.partition("=")[0].lower()) if name in CONFIG_GLOBALS else keys
    return Globals(keys, tuple(args[index:]))


def git_refusal(words: Sequence[str], rules: DevBoxRules) -> Optional[str]:
    """Why this ``git …`` command is refused in dev-box mode, or None."""
    reading = read_globals(list(words[1:]))
    if reading.problem:
        return refused(words, reading.problem)
    key = next((k for k in reading.config_keys if not safe_key(k)), None)
    if key is not None:
        return refused(words, f"git -c {key} {UNSAFE_KEY_WHY}")
    if not reading.rest:
        return None
    sub = reading.rest[0]
    if sub.startswith(CREDENTIAL_SUBCOMMAND):
        return refused(words, "git credential hands over the session's GitHub token")
    if sub not in KNOWN_GIT:
        return refused(words, f"{sub!r} is not a git command the gate knows (an alias or an extension runs what "
                              "the gate cannot read)")
    judge = SUBCOMMAND_JUDGES.get(sub)
    return judge(reading.rest[1:], rules, words) if judge is not None else None


# ── what runs a program ─────────────────────────────────────────────────────

def _long_names(args: Sequence[str]) -> List[str]:
    return [arg.partition("=")[0] for arg in args if arg.startswith("--")]


def _short_clusters(args: Sequence[str]) -> List[str]:
    return [arg[1:] for arg in args if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1]


def rebase_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git rebase -x``/``--exec`` runs a command after every commit."""
    if REBASE_EXEC_NAMES & set(_long_names(args)) or any(REBASE_EXEC_SHORT in c for c in _short_clusters(args)):
        return refused(words, "git rebase --exec / -x runs a command the gate cannot read")
    return None


def submodule_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    return refused(words, "git submodule foreach runs a command in every submodule") if "foreach" in args else None


def bisect_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    operands = [arg for arg in args if not arg.startswith("-")]
    return refused(words, "git bisect run runs a command the gate cannot read") if operands[:1] == ["run"] else None


def _upload_pack(args: Sequence[str]) -> Optional[str]:
    if any(name.startswith(UPLOAD_PACK_PREFIX) for name in _long_names(args)):
        return "--upload-pack runs a program of its choosing"
    return None


def upload_pack_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git fetch``, ``pull``, ``ls-remote``: never with ``--upload-pack``."""
    why = _upload_pack(args)
    return refused(words, why) if why else None


def _template(args: Sequence[str]) -> Optional[str]:
    if any(name.startswith(TEMPLATE_PREFIX) for name in _long_names(args)):
        return "--template copies hooks the gate cannot read into the new repository"
    return None


def _clone_config(args: Sequence[str]) -> Optional[str]:
    """``git clone -c key=value``: the key goes into the new repository's config."""
    for index, word in enumerate(args):
        name, glued, value = word.partition("=")
        if name in CLONE_CONFIG_OPTIONS and not glued:
            value = args[index + 1] if index + 1 < len(args) else ""
        elif name != "--config":
            continue
        key = value.partition("=")[0]
        if not safe_key(key):
            return f"git clone -c {key} {UNSAFE_KEY_WHY}"
    return None


def clone_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git clone``: never with ``--upload-pack``/``-u``, a template, or an unsafe ``-c`` key."""
    upload = "-u runs a program of its choosing" if CLONE_UPLOAD_PACK_SHORT in args else None
    why = upload or _upload_pack(args) or _template(args) or _clone_config(args)
    return refused(words, why) if why else None


def init_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    why = _template(args)
    return refused(words, why) if why else None


def archive_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git archive --remote``/``--exec`` runs git-upload-archive, locally for a path."""
    if any(name.startswith(ARCHIVE_PROGRAM_PREFIXES) for name in _long_names(args)):
        return refused(words, "git archive --remote / --exec runs a program of its choosing")
    return None


def tool_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git difftool`` and ``git mergetool`` always run another program."""
    sub = read_globals(list(words[1:])).rest[:1]
    return refused(words, ALWAYS_RUN_A_PROGRAM[sub[0]]) if sub else None


# ── git push ────────────────────────────────────────────────────────────────

def _refused_push_option(name: str, value: str) -> Optional[str]:
    """Why ``git push`` with this option is refused, whatever else it names."""
    if name in PUSH_REFUSED_OPTIONS:
        return PUSH_REFUSED_OPTIONS[name]
    if name == "--recurse-submodules" and value not in SAFE_SUBMODULE_MODES:
        return "--recurse-submodules pushes submodule commits too"
    return None


def _push_option(word: str) -> Tuple[Optional[str], bool, bool]:
    """One option of ``git push``: (why it is refused, takes the next word, deletes)."""
    name, glued, value = word.partition("=")
    refusal = _refused_push_option(name, value)
    if refusal:
        return refusal, False, False
    if glued:
        return (None, False, False) if name in PUSH_GLUED_OPTIONS else (f"{name}= is an option the gate does not know",
                                                                          False, False)
    if word in PUSH_VALUE_OPTIONS or word in PUSH_FLAGS:
        return None, word in PUSH_VALUE_OPTIONS, word in DELETE_OPTIONS
    if word.startswith("--") or not set(word[1:]) <= PUSH_SHORT_FLAGS:
        return f"{word} is an option the gate does not know", False, False
    return None, False, "d" in word[1:]


def _read_push(args: Sequence[str]) -> Tuple[Tuple[str, ...], bool, Optional[str]]:
    """``git push``'s operands, whether it deletes, and why it is refused outright."""
    operands: Tuple[str, ...] = ()
    deletes = False
    index = 0
    while index < len(args):
        word = args[index]
        if word == "--":
            return (*operands, *args[index + 1:]), deletes, None
        if not word.startswith("-") or word == "-":
            operands, index = (*operands, word), index + 1
            continue
        problem, takes_value, deleting = _push_option(word)
        if problem:
            return (), False, problem
        deletes, index = deletes or deleting, index + (2 if takes_value else 1)
    return operands, deletes, None


def _with_tags_spelled(specs: Sequence[str]) -> Tuple[str, ...]:
    """``tag v1`` → ``refs/tags/v1``, as git reads it."""
    out: Tuple[str, ...] = ()
    index = 0
    while index < len(specs):
        if specs[index] == TAG_KEYWORD and index + 1 < len(specs):
            out, index = (*out, f"refs/tags/{specs[index + 1]}"), index + 2
            continue
        out, index = (*out, specs[index]), index + 1
    return out


def _unreadable_spec(spec: str) -> bool:
    """A refspec whose destination is only known when it runs, or that names many refs."""
    return not spec.strip() or unreadable(spec) or any(char in spec for char in REFSPEC_GLOB_CHARS)


def _names_no_branch(source: str) -> bool:
    """``HEAD``, ``@`` or a revision (``x~1``): with no ``:<dst>``, git picks the destination."""
    return source in HEAD_NAMES or any(mark in source for mark in REVISION_MARKS)


def _destination(spec: str, deletes: bool) -> Tuple[str, Optional[str]]:
    """The ref one refspec updates on the remote, or why the gate cannot read it."""
    if _unreadable_spec(spec):
        return "", UNREADABLE_DESTINATION
    body = spec[1:] if spec.startswith("+") else spec
    if deletes:
        return body, None
    source, colon, target = body.partition(":")
    if colon:
        return (target, None) if target else ("", "an empty destination pushes every matching branch")
    if _names_no_branch(source):
        return "", NAME_THE_BRANCH
    return source, None


def _ref_kind(ref: str) -> Tuple[Optional[str], str]:
    """(the name, branch | tag | either) of a destination; (None, "") for any other ref."""
    low = ref.lower()
    for prefix, kind in REF_PREFIXES:
        if low.startswith(prefix):
            return ref[len(prefix):], kind
    return (None, "") if low.startswith("refs/") else (ref, KIND_EITHER)


def destination_refusal(ref: str, rules: DevBoxRules) -> Optional[str]:
    """Why a push may not land on ``ref``: a protected branch, a release tag, not a
    branch or a tag at all."""
    name, kind = _ref_kind(ref)
    if not name:
        return f"{ref} is not a branch or a tag"
    if rules.protected_branch(name):
        return f"{name} is a protected branch (the default and release branches are the owner's)"
    if kind != KIND_BRANCH and rules.release_tag(name):
        return f"{name} looks like a release tag (v…, release…): releases are the owner's"
    return None


def push_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git push``: a named remote and named, unprotected destinations, or a refusal."""
    operands, deletes, problem = _read_push(list(args))
    if problem:
        return refused(words, problem)
    if len(operands) < 2:
        return refused(words, NAME_THE_BRANCH)
    remote, specs = operands[0], _with_tags_spelled(operands[1:])
    if not _REMOTE_NAME_RE.match(remote):
        return refused(words, "push to a named remote (origin), not to a URL or a path")
    for spec in specs:
        ref, unread = _destination(spec, deletes)
        why = unread or destination_refusal(ref, rules)
        if why:
            return refused(words, why)
    return None


# ── git remote ──────────────────────────────────────────────────────────────

def _owner_refusal(url: str, rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """A remote must be a github.com repository of origin's own owner."""
    if rules.origin_owner is None:
        return refused(words, "origin is not a github.com repository the host could read, so no new remote can be "
                              "checked against its owner")
    if github_owner(url) != rules.origin_owner:
        return refused(words, f"a remote must be a github.com repository of {rules.origin_owner}, origin's owner")
    return None


def _remote_option_refusal(options: Sequence[str]) -> Optional[str]:
    """Why an option of ``git remote`` is refused (a mirror, a second or separate URL)."""
    for option in options:
        name = option.partition("=")[0]
        if name in REMOTE_REFUSED_OPTIONS:
            return REMOTE_REFUSED_OPTIONS[name]
    return None


def _new_url(action: str, rest: Sequence[str]) -> Optional[str]:
    """The URL ``git remote add NAME URL`` or ``git remote set-url NAME URL [OLD]`` writes."""
    if action == "add" and len(rest) == 2:
        return rest[1]
    if action == "set-url" and len(rest) in (2, 3):
        return rest[1]
    return None


def remote_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git remote``: reads, and add / set-url to origin's owner on github.com."""
    split = split_words(args, REMOTE_VALUE_OPTIONS)
    if not split.operands or split.operands[0] in REMOTE_READS:
        return None
    why = _remote_option_refusal(split.options)
    if why:
        return refused(words, why)
    url = _new_url(split.operands[0], split.operands[1:])
    if url is not None:
        return _owner_refusal(url, rules, words)
    return refused(words, f"git remote {split.operands[0]} is not one a dev-box session runs (add or set-url to "
                          "origin's owner on github.com; show, get-url, prune, update)")


Judge = Callable[[Sequence[str], DevBoxRules, Sequence[str]], Optional[str]]
SUBCOMMAND_JUDGES: Dict[str, Judge] = {
    "push": push_refusal, "remote": remote_refusal, "config": config_refusal, "rebase": rebase_refusal,
    "submodule": submodule_refusal, "bisect": bisect_refusal, "fetch": upload_pack_refusal,
    "pull": upload_pack_refusal, "ls-remote": upload_pack_refusal, "clone": clone_refusal, "init": init_refusal,
    "archive": archive_refusal, "difftool": tool_refusal, "mergetool": tool_refusal,
}
# What dev-box mode lets a session do that it refuses today: behind another command
# (``xargs``, ``timeout``, ``find -exec``, ``parallel``) these get arguments the gate
# never sees, so there they stay refused.
PUBLISHING_GIT = frozenset({"push", "remote", "send-pack", "http-push"})


def publishes(words: Sequence[str]) -> bool:
    """A ``git`` command line that pushes or changes a remote."""
    reading = read_globals(list(words[1:]))
    return bool(reading.rest) and reading.rest[0] in PUBLISHING_GIT


__all__ = [
    "KNOWN_GIT", "PUBLISHING_GIT", "SUBCOMMAND_JUDGES", "destination_refusal", "git_refusal", "publishes",
    "push_refusal", "read_globals", "remote_refusal",
]
