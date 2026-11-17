"""Which words of a simple command are DATA: a pattern or a program the tool reads.

Pure and stdlib-only; the gate (``policy.py``) asks, this answers. 17 Nov: one
ticket was held and refused for text its commands never ran. ``jq '.. | objects'``
read as a ``..`` traversal and ``grep -v "/tests/"`` as a read of ``/tests/``. A grep
pattern, a sed program and a jq filter are text the tool reads as a pattern or a
program: never a command the shell runs, never a path the tool opens. So the
command-word rules (``ALWAYS_ASK_BASH``), the ``..`` check and the path check leave
those words out. An awk program is read as written (``awk '!/tests\\/|alembic/'`` is
refused for its ``|``): only the path check leaves it out, as before.

Which words, decided per tool from how it takes its arguments:

* ``grep``/``egrep``/``fgrep``, ``rg`` and ``git grep``: every ``-e`` value and, when
  no pattern came by option, the first operand. A ``-f`` file is a path.
* ``git log``/``show``/``diff``/``shortlog``/``rev-list``/``whatchanged``: the value of
  ``--grep``, ``--author``, ``--committer``, ``--grep-reflog``, ``-S`` and ``-G``.
* ``sed``: the program (the ``-e``/``--expression`` values, or the first operand),
  because the gate reads it for the commands that run something or open a file
  (``sed_script``). ``awk``'s program is found the same way (``ScriptVerb``), but
  is data for the path check only. A program read from a file is refused.
* ``jq``: the filter, unless it ``import``s or ``include``s a module, which opens a
  file by path.

Only when the reading is certain. An option a table here does not know, or a pattern
or program option after the first operand (GNU tools permute options, BSD tools do
not, so the same words mean two things there), leaves every word judged as before:
when unsure, a word is a command and a path. Text handed to something that RUNS it
(``bash -c``, ``eval``, ``ssh``, a ``$(…)``) is never data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .sed_script import SedReader

# ── awk and sed: the program is read for what escapes ───────────────────────
# Moved from policy.py. A verb whose ARGUMENT is a program can run commands of its
# own. The verb stays on the allowlist (a ticket needs ``sed -n`` and
# ``awk '{print $1}'``); its program is read for the constructs that escape, and a
# program the gate cannot read at all (``-f progfile``) is refused.
# Every quantifier here is BOUNDED and no alternation is ambiguous: the program
# is a session's own text, so a pattern that can backtrack exponentially on it
# would hang the hook thread the gate answers from (CodeQL py/redos).
# ``@include``/``@load`` (gawk) read code from a file, as ``-f`` does.
AWK_ESCAPE_RE = re.compile(r"system\s{0,8}\(|\bgetline\b|\||print(?:f)?[^;}\n]{0,200}>|@(?:include|load)\b")
# sed's script is read by ``sed_script``, the way sed reads it: ``e``, ``r``/``R``
# and ``w``/``W`` as commands and the ``e``/``w`` flags of ``s`` are refused only where
# sed reads them. Not a regex; the name is the one ``ScriptVerb.escape`` and the
# backtracking test read.
SED_ESCAPE_RE = SedReader()
# A shell reference in a word (``$F``, ``${X}``, ``$(…)``, a bare ``$``): the word's
# text is only known when the line runs, so it is never data.
SHELL_REFERENCE_RE = re.compile(r"\$(?:[{(]|[A-Za-z_0-9@*#?!$])|^\$$")

@dataclass(frozen=True)
class ScriptVerb:
    """How one program-taking verb spells its program (design: one row per verb)."""
    escape: Any                      # the constructs in a program that run a command or read code
    text_options: Tuple[str, ...]    # options whose VALUE is program text
    file_options: Tuple[str, ...]    # options naming a program FILE — refused
    value_options: Tuple[str, ...]   # options with a value that is not a program
    flags: Tuple[str, ...] = ()      # options with no value that this reading knows
    suffix_options: Tuple[str, ...] = ()   # short options that own the rest of their word (``sed -i.bak``)
    program_is_data: bool = True     # False: the program is left out of the path check only

    def runs_a_command(self, text: str) -> bool:
        """The program, as written, can run a command or read code from a file of its own."""
        return bool(self.escape.search(text))


SCRIPT_VERBS: Mapping[str, ScriptVerb] = {
    # gawk's ``-i`` loads a library by name (``-i inplace``): a value, path-checked.
    "awk": ScriptVerb(
        AWK_ESCAPE_RE, ("-e", "--source"), ("-f", "--file", "--include", "-E", "--exec"),
        ("-v", "--assign", "-F", "--field-separator", "-i", "-l", "--load"),
        flags=("-safe", "-version", "--version", "-V", "-h", "--help", "-b", "--characters-as-bytes", "-c",
               "--traditional", "-C", "--copyright", "-g", "--gen-pot", "-k", "--csv", "-M", "--bignum", "-n",
               "--non-decimal-data", "-N", "--use-lc-numeric", "-O", "--optimize", "-P", "--posix", "-r",
               "--re-interval", "-s", "--no-optimize", "-S", "--sandbox", "-t", "--lint-old"),
        suffix_options=("-d", "--dump-variables", "-D", "--debug", "-L", "--lint", "-o", "--pretty-print",
                        "-p", "--profile"),
        program_is_data=False),     # read as written: a ``|`` in a regex or a string is refused
    "sed": ScriptVerb(
        SED_ESCAPE_RE, ("-e", "--expression"), ("-f", "--file"), (),
        flags=("-n", "--quiet", "--silent", "--debug", "--posix", "-E", "-r", "--regexp-extended", "-s",
               "--separate", "--sandbox", "-u", "--unbuffered", "-z", "--null-data", "--zero-terminated",
               "--follow-symlinks", "-b", "--binary", "-a", "--help", "--version"),
        suffix_options=("-i", "--in-place")),
}


@dataclass(frozen=True)
class Program:
    """What a script verb was given to run: each program text, with the index of the
    word that holds it."""
    texts: Tuple[Tuple[int, str], ...]
    from_file: bool = False          # a program file the gate cannot read
    certain: bool = True             # every option known, no program option after the first operand


def _kind(spec: ScriptVerb, option: str) -> str:
    for kind, names in (("text", spec.text_options), ("file", spec.file_options), ("value", spec.value_options),
                        ("suffix", spec.suffix_options), ("flag", spec.flags)):
        if option in names:
            return kind
    return "unknown"


def _option(spec: ScriptVerb, word: str) -> Tuple[str, Optional[str]]:
    """What one option word is (text | file | value | flag | unknown) and the value
    glued to it, None when the next word holds it. Short options combine (``-ne``)
    and take a glued value (``-F,``, ``-es/a/b/``), the way getopt reads them."""
    if word in spec.flags:
        return "flag", None
    if word.startswith("--"):
        name, eq, value = word.partition("=")
        kind = _kind(spec, name)
        return ("flag" if kind == "suffix" else kind), (value if eq else None)
    for position, letter in enumerate(word[1:], start=1):
        kind = _kind(spec, "-" + letter)
        if kind in ("text", "file", "value"):
            return kind, (word[position + 1:] or None)
        if kind != "flag":
            return ("flag" if kind == "suffix" else "unknown"), None
    return "flag", None


def read_program(spec: ScriptVerb, words: Sequence[str]) -> Program:
    """The program a script verb runs, from its words (the verb first): every ``-e``
    text wherever it sits (GNU permutes options) and, when none came before it, the
    first operand. All of them are read by the escape check. A program option after
    the first operand, or an option this reading does not know, makes it uncertain."""
    texts: List[Tuple[int, str]] = []
    first: Optional[int] = None
    early, certain, i = False, True, 1
    while i < len(words):
        word = words[i]
        if word == "--":
            first = i + 1 if first is None and i + 1 < len(words) else first
            break
        if word == "-" or not word.startswith("-"):
            first, i = (i if first is None else first), i + 1
            continue
        kind, glued = _option(spec, word)
        if kind == "file":
            return Program(tuple(texts), from_file=True, certain=False)
        if kind in ("text", "value"):
            at = i if glued is not None else i + 1
            if kind == "text" and at < len(words):
                texts = [*texts, (at, words[at] if glued is None else glued)]
                early, certain = early or first is None, certain and first is None
            i = at + 1
            continue
        certain, i = certain and kind == "flag", i + 1
    if first is not None and not early:
        texts = [(first, words[first]), *texts]
    return Program(tuple(texts), certain=certain)


# ── search tools: the pattern ────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchVerb:
    """How a search tool spells its options (design: one row per tool)."""
    short_flags: str                 # letters with no value
    short_values: str                # letters whose value is the rest of the word, or the next word
    long_flags: FrozenSet[str]       # long options with no value, or with one glued by '='
    long_values: FrozenSet[str]      # long options whose value may be the next word
    short_suffix: str = ""           # letters that own the rest of their word (``git grep -Ovim``)
    patterns: Tuple[str, ...] = ("-e", "--regexp")         # the value IS a pattern
    pattern_files: Tuple[str, ...] = ("-f", "--file")      # the patterns come from a FILE: a path
    no_pattern: FrozenSet[str] = frozenset()               # a mode with no pattern at all
    program_options: FrozenSet[str] = frozenset()          # options that RUN a program: refused
    program_letters: str = ""                              # short options that run one (``git grep -O``)


# GNU and BSD grep. ``--color`` and ``--context`` take a value only when glued with '='.
GREP = SearchVerb(
    short_flags="abcEFGHhIiJLlMnOoPpqRrSsTUuVvwxyZz0123456789", short_values="ABCDdefm",
    long_flags=frozenset({
        "--extended-regexp", "--fixed-strings", "--basic-regexp", "--perl-regexp", "--ignore-case",
        "--no-ignore-case", "--word-regexp", "--line-regexp", "--null-data", "--no-messages", "--invert-match",
        "--byte-offset", "--line-number", "--line-buffered", "--with-filename", "--no-filename",
        "--only-matching", "--quiet", "--silent", "--recursive", "--dereference-recursive",
        "--files-without-match", "--files-with-matches", "--count", "--initial-tab", "--null", "--text",
        "--binary", "--unix-byte-offsets", "--no-group-separator", "--color", "--colour", "--context", "--mmap",
    }),
    long_values=frozenset({
        "--regexp", "--file", "--max-count", "--after-context", "--before-context", "--directories", "--devices",
        "--include", "--exclude", "--exclude-dir", "--include-dir", "--exclude-from", "--label", "--binary-files",
        "--group-separator",
    }),
    no_pattern=frozenset({"--help", "--version"}),
)
RG = SearchVerb(
    short_flags="abcFHhIiLlNnoPpqSsUuVvwxz0.", short_values="ABCdEefgjMmrTt",
    long_flags=frozenset({
        "--hidden", "--no-hidden", "--no-ignore", "--ignore", "--no-ignore-vcs", "--no-ignore-parent",
        "--no-ignore-dot", "--no-ignore-global", "--no-ignore-exclude", "--no-ignore-files", "--no-require-git",
        "--ignore-case", "--case-sensitive", "--smart-case", "--fixed-strings", "--word-regexp", "--line-regexp",
        "--invert-match", "--count", "--count-matches", "--files-with-matches", "--files-without-match",
        "--only-matching", "--line-number", "--no-line-number", "--column", "--no-column", "--heading",
        "--no-heading", "--with-filename", "--no-filename", "--json", "--vimgrep", "--pretty", "--null",
        "--null-data", "--quiet", "--text", "--follow", "--multiline", "--multiline-dotall", "--pcre2",
        "--search-zip", "--unrestricted", "--stats", "--trim", "--byte-offset", "--no-messages", "--no-config",
        "--binary", "--crlf", "--one-file-system", "--passthru", "--sort-files", "--max-columns-preview",
        "--no-unicode", "--line-buffered", "--block-buffered", "--glob-case-insensitive", "--include-zero",
    }),
    long_values=frozenset({
        "--regexp", "--file", "--glob", "--iglob", "--type", "--type-not", "--max-count", "--after-context",
        "--before-context", "--context", "--max-columns", "--threads", "--max-depth", "--encoding", "--type-add",
        "--type-clear", "--ignore-file", "--pre", "--pre-glob", "--replace", "--sort", "--sortr", "--color",
        "--colors", "--path-separator", "--max-filesize", "--engine", "--context-separator",
        "--field-context-separator", "--field-match-separator", "--dfa-size-limit", "--regex-size-limit",
        "--hostname-bin", "--hyperlink-format",
    }),
    no_pattern=frozenset({"--files", "--type-list", "--help", "--version", "--pcre2-version", "--generate"}),
    program_options=frozenset({"--pre", "--pre-glob", "--hostname-bin"}),   # a preprocessor, a hostname program
)
GIT_GREP = SearchVerb(
    short_flags="acEFGhHiIlLnopPqvwWz0123456789", short_values="ABCefm", short_suffix="O",
    long_flags=frozenset({
        "--cached", "--no-index", "--untracked", "--no-exclude-standard", "--exclude-standard",
        "--recurse-submodules", "--no-recurse-submodules", "--text", "--textconv", "--no-textconv",
        "--ignore-case", "--word-regexp", "--invert-match", "--full-name", "--extended-regexp", "--basic-regexp",
        "--perl-regexp", "--fixed-strings", "--line-number", "--column", "--files-with-matches", "--name-only",
        "--files-without-match", "--null", "--only-matching", "--count", "--color", "--no-color", "--break",
        "--heading", "--show-function", "--function-context", "--and", "--or", "--not", "--all-match", "--quiet",
        "--recursive", "--no-recursive", "--open-files-in-pager",
    }),
    long_values=frozenset({"--max-depth", "--threads", "--max-count", "--context", "--after-context",
                           "--before-context"}),
    patterns=("-e",), pattern_files=("-f",),
    program_options=frozenset({"--open-files-in-pager"}), program_letters="O",   # a pager it runs
)
SEARCH_VERBS: Mapping[str, SearchVerb] = {"grep": GREP, "egrep": GREP, "fgrep": GREP, "rg": RG}


def _search_option(spec: SearchVerb, words: Sequence[str], i: int) -> Optional[Tuple[str, Optional[int]]]:
    """One option word: its name and the index of the word holding its value (None for
    a flag). None when this table does not know it."""
    word = words[i]
    if word.startswith("--"):
        name, eq, _ = word.partition("=")
        if name in spec.long_values:
            return name, (i if eq else i + 1)
        return (name, None) if name in spec.long_flags or name in spec.no_pattern else None
    for position, letter in enumerate(word[1:], start=1):
        if letter in spec.short_values:
            return "-" + letter, (i if position + 1 < len(word) else i + 1)
        if letter in spec.short_suffix:
            return "-" + letter, None
        if letter not in spec.short_flags:
            return None
    return word, None


def _search_patterns(spec: SearchVerb, words: Sequence[str], start: int) -> Tuple[int, ...]:
    """Where a search tool's pattern sits: each ``-e`` value and, when no pattern came by
    option, the first operand. Empty when the reading is not certain."""
    given: List[int] = []
    first: Optional[int] = None
    by_option, i = False, start
    while i < len(words):
        word = words[i]
        if word == "--":
            first = i + 1 if first is None and i + 1 < len(words) else first
            break
        if word == "-" or not word.startswith("-"):
            first, i = (i if first is None else first), i + 1
            continue
        option = _search_option(spec, words, i)
        if option is None or option[0] in spec.no_pattern:
            return ()
        name, at = option
        if name in spec.patterns or name in spec.pattern_files:
            if first is not None:
                return ()                    # after an operand: GNU reads an option here, BSD a file
            by_option = True
            given = [*given, at] if name in spec.patterns and at is not None and at < len(words) else given
        i = i + 1 if at is None else at + 1
    if by_option:
        return tuple(given)
    return (first,) if first is not None else ()


# ── git log's patterns and jq's filter ───────────────────────────────────────
GIT_LOG_VERBS = frozenset({"log", "show", "diff", "shortlog", "rev-list", "whatchanged"})
GIT_LOG_PATTERNS = frozenset({"--grep", "--author", "--committer", "--grep-reflog"})
GIT_PICKAXE = ("-S", "-G")

JQ_SHORT_FLAGS = frozenset("abcCehjMnrRsSV")
JQ_LONG_FLAGS = frozenset({
    "--seq", "--stream", "--stream-errors", "--slurp", "--raw-input", "--null-input", "--compact-output", "--tab",
    "--color-output", "--monochrome-output", "--ascii-output", "--unbuffered", "--sort-keys", "--raw-output",
    "--raw-output0", "--join-output", "--exit-status", "--args", "--jsonargs", "--binary", "--version", "--help",
    "--build-configuration",
})
JQ_VALUE_WORDS: Mapping[str, int] = {"--arg": 2, "--argjson": 2, "--slurpfile": 2, "--rawfile": 2, "--indent": 1}
JQ_MODULE_RE = re.compile(r"\b(?:import|include)\b")      # a module directive opens a file by path


def _git_patterns(words: Sequence[str]) -> Tuple[int, ...]:
    """``git grep``'s pattern; ``git log --grep=…``/``-S…``/``-G…`` and their kin."""
    if words[1] == "grep":
        return _search_patterns(GIT_GREP, words, 2)
    if words[1] not in GIT_LOG_VERBS:
        return ()
    found: List[int] = []
    i = 2
    while i < len(words) and words[i] != "--":
        name, eq, _ = words[i].partition("=")
        if (name in GIT_LOG_PATTERNS and not eq) or words[i] in GIT_PICKAXE:
            found, i = [*found, i + 1], i + 2
            continue
        if name in GIT_LOG_PATTERNS or (words[i][:2] in GIT_PICKAXE and len(words[i]) > 2):
            found = [*found, i]
        i += 1
    return tuple(index for index in found if index < len(words))


def _jq_reads_program_file(words: Sequence[str]) -> bool:
    """``-f``/``--from-file`` anywhere: the filter is a file, and every operand an input."""
    for word in words[1:]:
        if word == "--":
            return False
        if word.startswith("--from-file") or (word.startswith("-") and not word.startswith("--") and "f" in word[1:]):
            return True
    return False


def _jq_filter(words: Sequence[str]) -> Tuple[int, ...]:
    """jq's filter, its first operand once the options are read. Never when it reads the
    filter from a file, meets an option this table does not know, or the filter imports
    or includes a module."""
    if _jq_reads_program_file(words):
        return ()
    i = 1
    while i < len(words) and words[i].startswith("-") and words[i] != "-":
        word = words[i]
        if word == "--":
            i += 1
            break
        if word in JQ_VALUE_WORDS:
            i += 1 + JQ_VALUE_WORDS[word]
        elif word in JQ_LONG_FLAGS or (not word.startswith("--") and set(word[1:]) <= JQ_SHORT_FLAGS):
            i += 1
        else:
            return ()
    if i >= len(words) or JQ_MODULE_RE.search(words[i]):
        return ()
    return (i,)


def _positions(words: Sequence[str]) -> Tuple[int, ...]:
    """Where one tool's pattern or program sits, when the reading of its options is certain."""
    head = Path(words[0]).name
    if head in SCRIPT_VERBS:
        program = read_program(SCRIPT_VERBS[head], words)
        return tuple(index for index, _ in program.texts) if program.certain else ()
    if head in SEARCH_VERBS:
        return _search_patterns(SEARCH_VERBS[head], words, 1)
    if head == "jq":
        return _jq_filter(words)
    if head == "git" and len(words) > 1:
        return _git_patterns(words)
    return ()


def data_positions(words: Sequence[str]) -> Tuple[int, ...]:
    """The indexes of the words of one simple command (the verb first; wrappers and
    git's global options already stepped over) that are a pattern or a program its
    tool reads: the command-word rules and the ``..`` check leave them out. Empty for
    every other tool, for an awk program, whenever the reading is not certain, and
    when one of them holds a shell reference (its text is only known when the line runs)."""
    spec = SCRIPT_VERBS.get(Path(words[0]).name) if words else None
    if not words or (spec is not None and not spec.program_is_data):
        return ()
    positions = _positions(words)
    if any(SHELL_REFERENCE_RE.search(words[index]) for index in positions):
        return ()
    return positions


def non_path_positions(words: Sequence[str]) -> Tuple[int, ...]:
    """The words a tool reads as a pattern or a program and never opens as a file: the
    path check leaves them out. An awk or sed program stays out of it, as before, even
    when it holds a ``$`` (``'/x/ {print $1}'``): what it does is the escape check's
    business, which reads it whole."""
    if words and Path(words[0]).name in SCRIPT_VERBS:
        return _positions(words)
    return data_positions(words)


def runs_a_program(words: Sequence[str]) -> Optional[str]:
    """The option that makes a search tool run a program of its own (``rg --pre``,
    ``git grep -O``), or None. Read wherever it sits, glued or not. git takes a unique
    prefix of a long option, so for ``git grep`` any prefix of one counts (``--op=sh``)."""
    head = Path(words[0]).name if words else ""
    spec = GIT_GREP if head == "git" and len(words) > 1 and words[1] == "grep" else SEARCH_VERBS.get(head)
    if spec is None:
        return None
    for word in words[1:]:
        if word == "--":
            return None
        name = word.partition("=")[0]
        short = word.startswith("-") and not word.startswith("--") and set(word[1:]) & set(spec.program_letters)
        prefix = spec is GIT_GREP and len(name) > 2 and any(option.startswith(name) for option in spec.program_options)
        if name in spec.program_options or short or (prefix and name.startswith("--")):
            return name
    return None

__all__ = [
    "AWK_ESCAPE_RE", "GIT_GREP", "GREP", "Program", "RG", "SCRIPT_VERBS", "SEARCH_VERBS", "SED_ESCAPE_RE",
    "SHELL_REFERENCE_RE", "ScriptVerb", "SearchVerb", "data_positions", "non_path_positions",
    "read_program", "runs_a_program",
]
