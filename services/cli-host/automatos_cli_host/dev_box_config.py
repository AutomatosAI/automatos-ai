"""git's config in a dev-box session: an allowlist of safe keys (dev-box mode).

A config key can change where a push goes (``url.*.insteadOf``, ``remote.*``), which
credential it carries (``credential.*``), or run a program the gate never sees
(``core.pager``, ``diff.external``, ``*.textconv``, ``filter.*.clean``, ``alias.*``,
``gpg.program``, ``core.fsmonitor``…). So a session sets only the keys on
``SAFE_CONFIG_KEYS``, through ``git config``, ``git -c`` or ``--config-env``; a key
ending in a program word (``.cmd``, ``.command``, ``.program``, ``.driver``, ``.clean``,
``.smudge``, ``.textconv``) is refused even under a safe prefix. Reads of any key run.
``git config --edit``, a section renamed or removed, and an option the gate does not
know (git accepts abbreviations, ``--edi`` for ``--edit``) are refused.

Also the word splitter the git judges share (``split_words``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple, Optional

from .dev_box_rules import DevBoxRules, refused

SAFE_CONFIG_KEYS = tuple(re.compile(pattern) for pattern in (
    r"user\.[^.]+",
    r"commit\.(?!template$|gpgsign$)[^.]+",          # not the template, not signing (it runs gpg)
    r"core\.(?:autocrlf|filemode|ignorecase|longpaths)",
    r"pull\.[^.]+", r"fetch\.[^.]+",
    r"rebase\.autosquash", r"init\.defaultbranch", r"merge\.ff", r"push\.autosetupremote",
    r"branch\.autosetuprebase", r"diff\.renames",
    r"color\..+", r"column\..+",
))
PROGRAM_KEY_RE = re.compile(r"\.(?:cmd|command|program|driver|clean|smudge|textconv)$|hook|template")
SAFE_KEYS_SHOWN = ("user.*, commit.*, core.autocrlf/filemode/ignorecase/longpaths, pull.*, fetch.*, "
                   "rebase.autosquash, init.defaultBranch, merge.ff, push.autoSetupRemote, "
                   "branch.autosetuprebase, diff.renames, color.*, column.*")
UNSAFE_KEY_WHY = (f"is not on the dev-box list of safe keys ({SAFE_KEYS_SHOWN}): any other key can change where "
                  "a push goes, which credential it carries, or run a program")

CONFIG_READ, CONFIG_WRITE, CONFIG_SECTION, CONFIG_EDIT = "read", "write", "section", "edit"
CONFIG_SUBCOMMANDS: Dict[str, str] = {
    "get": CONFIG_READ, "list": CONFIG_READ, "set": CONFIG_WRITE, "unset": CONFIG_WRITE,
    "rename-section": CONFIG_SECTION, "remove-section": CONFIG_SECTION, "edit": CONFIG_EDIT,
}
CONFIG_READ_OPTIONS = frozenset({"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--get-color",
                                 "--get-colorbool", "-l", "--list"})
CONFIG_WRITE_OPTIONS = frozenset({"--add", "--replace-all", "--unset", "--unset-all"})
CONFIG_SECTION_OPTIONS = frozenset({"--rename-section", "--remove-section"})
CONFIG_EDIT_OPTIONS = frozenset({"-e", "--edit"})
CONFIG_VALUE_OPTIONS = frozenset({"-f", "--file", "--blob", "--type", "--default", "--comment", "--value", "--url"})
CONFIG_FLAGS = frozenset({
    "--global", "--system", "--local", "--worktree", "--fixed-value", "--bool", "--int", "--bool-or-int", "--path",
    "--expiry-date", "--no-type", "-z", "--null", "--name-only", "--show-origin", "--show-scope", "--includes",
    "--no-includes", "--all", "--regexp", "--append",
})
CONFIG_OPTIONS = (CONFIG_READ_OPTIONS | CONFIG_WRITE_OPTIONS | CONFIG_SECTION_OPTIONS | CONFIG_EDIT_OPTIONS
                  | CONFIG_VALUE_OPTIONS | CONFIG_FLAGS)
# Which option decides what a ``git config`` line does, first match wins.
CONFIG_OPTION_MODES = ((CONFIG_EDIT_OPTIONS, CONFIG_EDIT), (CONFIG_SECTION_OPTIONS, CONFIG_SECTION),
                       (CONFIG_READ_OPTIONS, CONFIG_READ), (CONFIG_WRITE_OPTIONS, CONFIG_WRITE))


@dataclass(frozen=True)
class Words:
    """A subcommand's arguments split into operands and options (with their values dropped)."""
    operands: Tuple[str, ...] = ()
    options: Tuple[str, ...] = ()


def split_words(args: Sequence[str], value_options: FrozenSet[str]) -> Words:
    """Operands and options of a subcommand; an option in ``value_options`` takes the
    next word as its value unless it is glued (``--file=x``)."""
    operands: List[str] = []
    options: List[str] = []
    index = 0
    while index < len(args):
        word = args[index]
        if word == "--":
            operands = [*operands, *args[index + 1:]]
            break
        if not word.startswith("-") or word == "-":
            operands, index = [*operands, word], index + 1
            continue
        options = [*options, word]
        index += 2 if word in value_options else 1
    return Words(tuple(operands), tuple(options))


def safe_key(key: str) -> bool:
    """A config key a dev-box session may set (``SAFE_CONFIG_KEYS``, no program word)."""
    low = key.strip().lower()
    return not PROGRAM_KEY_RE.search(low) and any(pattern.fullmatch(low) for pattern in SAFE_CONFIG_KEYS)


def _config_mode(split: Words) -> Tuple[str, Tuple[str, ...]]:
    """What a ``git config`` line does, and the key it writes: its subcommand says
    (``git config set``), else its options, else a key and a value write and a key alone reads."""
    options = {option.partition("=")[0] for option in split.options}
    operands = split.operands
    if operands and operands[0] in CONFIG_SUBCOMMANDS:
        mode, operands = CONFIG_SUBCOMMANDS[operands[0]], operands[1:]
    else:
        by_operands = CONFIG_WRITE if len(operands) >= 2 else CONFIG_READ
        mode = next((named for flags, named in CONFIG_OPTION_MODES if options & flags), by_operands)
    return mode, tuple(operands[:1])


def _unknown_option(options: Sequence[str]) -> Optional[str]:
    """An option ``git config`` takes that the gate does not know (an abbreviation, say)."""
    return next((option for option in options if option.partition("=")[0] not in CONFIG_OPTIONS), None)


def config_refusal(args: Sequence[str], rules: DevBoxRules, words: Sequence[str]) -> Optional[str]:
    """``git config``: reads run; a write runs only for a safe key."""
    split = split_words(args, CONFIG_VALUE_OPTIONS)
    unknown = _unknown_option(split.options)
    if unknown is not None:
        return refused(words, f"git config {unknown} is an option the gate does not know")
    mode, names = _config_mode(split)
    if mode == CONFIG_EDIT:
        return refused(words, "git config --edit opens the config in an editor the gate cannot follow")
    if mode == CONFIG_SECTION:
        return refused(words, "git config renames or removes a whole section, keys the gate cannot see")
    name = next((n for n in names if not safe_key(n)), None) if mode == CONFIG_WRITE else None
    return refused(words, f"git config {name} {UNSAFE_KEY_WHY}") if name else None


__all__ = ["SAFE_CONFIG_KEYS", "UNSAFE_KEY_WHY", "Words", "config_refusal", "safe_key", "split_words"]
