"""F042, hardened: how a command reaches the platform's secret files without
spelling their names (security review of 28ec7cca9, 2026-09-22).

policy.py refuses a secret file NAMED on a line. The review got past a name in
four ways, each proved on this machine:

* case: APFS is case-insensitive, so ``.ENV`` opens ``.env``. Every name and
  containment test here folds case.
* globs and braces: ``cat .en*`` and ``cat .{e,x}nv`` never spell ``.env``. A
  word that can EXPAND to a secret file is judged by the files it can reach.
* recursion: ``grep -r KEY .`` reads ``.env`` without naming it (BSD grep walks
  dot-files), and ``rg --hidden`` or an ``rg -g`` include glob (which overrides
  .gitignore) does the same. A recursive search whose tree holds a secret file
  it does not filter out is refused.
* ``find … -exec cat {} +``: the same, through find.

The inventory is the set of secret files that EXIST under each protected
checkout. It is walked at most once a minute and skips dependency folders, so a
search over a folder that holds none (``orchestrator/modules``) is untouched.

What no static gate can see is a program that opens the file itself once it
runs (a project script calling ``load_dotenv()``). The secrets are only fully
out of a session's reach when the session does not work in a folder that holds
them: the F042 working-folder decision (options A/B).
"""
from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCAN_DEPTH = 5                    # folder levels walked below a protected checkout
SCAN_MAX_ENTRIES = 50_000         # a bound on one walk; the checkout's own .env family sits at the top
SCAN_TTL_SECONDS = 60.0
# Dependency and cache folders: large, and never where a checkout keeps its own secrets.
SCAN_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".next", ".turbo", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "graphify-out", "coverage",
})
# A pattern is also tested against these, so a secret created since the last walk still counts.
CANONICAL_SECRET_NAMES = (".env", ".env.local", ".env.production", ".env.development", ".credential_key")
MAX_BRACE_WORDS = 64
GLOB_CHARS = frozenset("*?[")

# Verbs that never print a file's contents (a name, a size, a count, a digest):
# a path they are handed is not a leak.
NO_CONTENT_VERBS = frozenset({
    "ls", "stat", "du", "test", "[", "[[", "echo", "printf", "basename", "dirname", "realpath",
    "readlink", "true", "pwd", "which", "date", "seq", "tree", "wc", "file", "md5sum", "sha1sum",
    "sha256sum", "shasum", "cksum",
})
# Verbs that read whole folders into an archive, a copy or a diff: a secret under
# a folder they are handed is read without its name on the line
# (``tar cf - . | tar xOf -`` prints every file). The copiers only when recursive.
TREE_READERS = frozenset({"tar", "bsdtar", "gtar", "zip", "7z", "7za", "cpio", "rsync", "scp", "jar", "ditto"})
RECURSIVE_COPIERS = {"cp": ("-r", "-R", "-a"), "diff": ("-r", "--recursive")}
GREP_VERBS = frozenset({"grep", "egrep", "fgrep", "ugrep", "ug"})

_scans: Dict[str, Tuple[float, Tuple[Path, ...]]] = {}


def clear_inventory_cache() -> None:
    _scans.clear()


def within_folded(path: Path, root: Path) -> bool:
    """``path`` is ``root`` or sits under it, ignoring case (APFS opens ``.ENV`` as ``.env``)."""
    p = [part.casefold() for part in path.parts]
    r = [part.casefold() for part in root.parts]
    return len(p) >= len(r) and p[:len(r)] == r


def secret_files(root: Path, is_secret_name: Callable[[str], bool]) -> Tuple[Path, ...]:
    """The secret-named files under ``root``, walked at most once a minute."""
    key = str(root)
    now = time.monotonic()
    cached = _scans.get(key)
    if cached is not None and now - cached[0] < SCAN_TTL_SECONDS:
        return cached[1]
    found: List[Path] = []
    pending: List[Tuple[Path, int]] = [(root, 0)]
    seen = 0
    while pending and seen < SCAN_MAX_ENTRIES:
        folder, depth = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    seen += 1
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth + 1 < SCAN_DEPTH and entry.name not in SCAN_SKIP_DIRS:
                                pending.append((Path(entry.path), depth + 1))
                        elif is_secret_name(entry.name):
                            found.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    result = tuple(found)
    _scans[key] = (now, result)
    return result


@dataclass(frozen=True)
class Inventory:
    roots: Tuple[Path, ...]        # protected checkouts, resolved
    files: Tuple[Path, ...]        # the secret files that exist under them
    off_limits: Tuple[Path, ...]   # folders no session touches (the host's state)
    granted: Tuple[Path, ...]      # the session's own folders inside those

    @property
    def active(self) -> bool:
        return bool(self.roots or self.off_limits)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted({f.name for f in self.files} | set(CANONICAL_SECRET_NAMES)))

    def reachable_under(self, tree: Path) -> Tuple[Path, ...]:
        """What a walk of ``tree`` meets: the secret files under it, and the
        host's state folder itself when the walk would enter it."""
        hits = [f for f in self.files if within_folded(f, tree)]
        hits += [o for o in self.off_limits
                 if within_folded(o, tree) and not any(within_folded(o, g) for g in self.granted)]
        return tuple(hits)

    def holds_secrets(self, folder: Path) -> bool:
        """A secret sits under the folder a command runs in. A folder elsewhere in
        a protected checkout does not count: a ``$(…)`` whose OUTPUT climbs out
        with ``..`` is a program deciding at run time, the same blind spot as a
        script that opens the file itself (security review 2026-09-22)."""
        return bool(self.reachable_under(folder))


def inventory(secret_roots: Iterable[Path], off_limits: Iterable[Path], granted: Iterable[Path],
              is_secret_name: Callable[[str], bool]) -> Inventory:
    roots = tuple(Path(r).expanduser().resolve() for r in secret_roots)
    return Inventory(
        roots=roots,
        files=tuple(f for r in roots for f in secret_files(r, is_secret_name)),
        off_limits=tuple(Path(o).expanduser().resolve() for o in off_limits),
        granted=tuple(Path(g).expanduser().resolve() for g in granted),
    )


def same_file_as_secret(path: Path, inv: Inventory) -> Optional[Path]:
    """The secret file ``path`` is a hard link to, if any: a hard link has no
    original name to resolve back to, so it is matched by inode."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_nlink < 2:
        return None
    for secret in inv.files:
        try:
            other = os.stat(secret)
        except OSError:
            continue
        if (other.st_dev, other.st_ino) == (st.st_dev, st.st_ino):
            return secret
    return None


# ── words the shell expands ─────────────────────────────────────────────────

def _brace_group(word: str) -> Optional[Tuple[int, int, List[str]]]:
    """The first ``{a,b}`` group with a top-level comma: its span and options.
    ``${…}`` is a parameter, and a group without a comma stays literal (bash)."""
    i = 0
    while i < len(word):
        if word[i] != "{" or (i > 0 and word[i - 1] == "$"):
            i += 1
            continue
        depth, commas = 0, []
        for j in range(i, len(word)):
            if word[j] == "{":
                depth += 1
            elif word[j] == "}":
                depth -= 1
                if depth == 0:
                    if commas:
                        cuts = [i, *commas, j]
                        return i, j, [word[a + 1:b] for a, b in zip(cuts, cuts[1:])]
                    break
            elif word[j] == "," and depth == 1:
                commas.append(j)
        i += 1
    return None


def expand_braces(word: str) -> Tuple[List[str], bool]:
    """Bash brace expansion, bounded: the words, and whether the expansion finished."""
    out: List[str] = []
    pending = [word]
    while pending:
        if len(out) + len(pending) > MAX_BRACE_WORDS:
            return out + pending, False
        current = pending.pop()
        group = _brace_group(current)
        if group is None:
            out.append(current)
            continue
        start, end, options = group
        pending.extend(current[:start] + option + current[end + 1:] for option in options)
    return out, True


def has_glob(text: str) -> bool:
    return any(c in GLOB_CHARS for c in text)


def pattern_may_name(pattern: str, names: Iterable[str], *, leading_dot_literal: bool) -> bool:
    """Whether a file-name pattern can match one of ``names`` (case folded).
    ``leading_dot_literal`` is the shell's rule: ``*``, ``?`` and ``[…]`` never
    match a leading dot, so only a pattern that starts with ``.`` reaches a
    dot-file. A tool's own globs (grep --include, rg -g, find -name) have no
    such rule."""
    folded = pattern.casefold()
    if leading_dot_literal and not folded.startswith("."):
        return False
    return any(fnmatch.fnmatchcase(name.casefold(), folded) for name in names)


def glob_reaches(value: str, base: Path, inv: Inventory, *, leading_dot_literal: bool = True) -> Optional[Path]:
    """The secret, or the host's state folder, a glob can match when read from
    ``base``. ``*`` may cross a ``/`` here where bash's does not, and case is
    folded where bash's globbing is exact: both only ever widen the match."""
    path = Path(value).expanduser()
    full = str(path if path.is_absolute() else base / path)
    first = min(full.find(c) for c in GLOB_CHARS if c in full)
    tree = Path(full[:first].rpartition("/")[0] or "/")
    name = full.rpartition("/")[2]
    for folder in inv.off_limits:
        if any(within_folded(folder, g) for g in inv.granted):
            continue
        if within_folded(tree, folder) or within_folded(folder, tree):
            return folder
    if leading_dot_literal and not name.startswith("."):
        return None                                   # never a dot-file: every secret name is one
    for secret in inv.files:
        if within_folded(secret, tree) and fnmatch.fnmatchcase(str(secret).casefold(), full.casefold()):
            return secret
    return None


# ── recursive readers ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchSpec:
    verb: str
    reads_everything: bool          # dot-files and ignored files are read
    paths: Tuple[str, ...]
    includes: Tuple[str, ...]       # name globs that restrict what is read
    excludes: Tuple[str, ...]
    exclude_dirs: Tuple[str, ...]
    types: Tuple[str, ...]          # rg -t: only these file types are read
    pattern_index: Optional[int]    # which word is the pattern (never a path)


_GREP_VALUE_SHORT = frozenset("efmABCdD")
_GREP_VALUE_LONG = frozenset({
    "--regexp", "--file", "--max-count", "--after-context", "--before-context", "--context",
    "--directories", "--devices", "--include", "--exclude", "--exclude-dir", "--exclude-from",
    "--label", "--color", "--colour", "--binary-files", "--group-separator",
})
# rg's ``-r`` is --replace (a value), not recursion: rg always recurses.
_RG_VALUE_SHORT = frozenset("efgtTmABCMjdEr")
_RG_VALUE_LONG = frozenset({
    "--regexp", "--file", "--glob", "--iglob", "--type", "--type-not", "--max-count",
    "--after-context", "--before-context", "--context", "--max-columns", "--threads", "--max-depth",
    "--encoding", "--type-add", "--type-clear", "--ignore-file", "--pre", "--pre-glob", "--replace",
    "--sort", "--sortr", "--color", "--colors", "--path-separator", "--max-filesize", "--engine",
    "--context-separator", "--field-context-separator", "--field-match-separator",
})


def search_spec(words: Sequence[str]) -> Optional[SearchSpec]:
    """How a grep-family or rg command walks: None when it does not recurse."""
    verb = Path(words[0]).name if words else ""
    if verb in GREP_VERBS:
        return _grep_spec(verb, words)
    if verb == "rg":
        return _rg_spec(words)
    return None


def _option_values(words: Sequence[str], value_short: frozenset, value_long: frozenset):
    """(name, value) for each option, then the operands with their word indexes."""
    options: List[Tuple[str, str]] = []
    operands: List[Tuple[int, str]] = []
    i = 1
    while i < len(words):
        word = words[i]
        if word == "--":
            operands += [(k, w) for k, w in enumerate(words[i + 1:], start=i + 1)]
            break
        if word.startswith("--"):
            name, eq, value = word.partition("=")
            if name in value_long and not eq and i + 1 < len(words):
                value, i = words[i + 1], i + 1
            options.append((name, value))
        elif word.startswith("-") and len(word) > 1:
            for j, flag in enumerate(word[1:], start=1):
                if flag in value_short:
                    value = word[j + 1:]
                    if not value and i + 1 < len(words):
                        value, i = words[i + 1], i + 1
                    options.append(("-" + flag, value))
                    break
                options.append(("-" + flag, ""))
        else:
            operands.append((i, word))
        i += 1
    return options, operands


def search_operands(words: Sequence[str]) -> List[str]:
    """The file operands of a grep-family or rg command: its pattern is not one."""
    verb = Path(words[0]).name if words else ""
    short, long_ = (_RG_VALUE_SHORT, _RG_VALUE_LONG) if verb == "rg" else (_GREP_VALUE_SHORT, _GREP_VALUE_LONG)
    options, operands = _option_values(words, short, long_)
    pattern_given = any(name in ("-e", "-f", "--regexp", "--file") for name, _ in options)
    return [word for i, (_, word) in enumerate(operands) if pattern_given or i > 0]


def _grep_spec(verb: str, words: Sequence[str]) -> Optional[SearchSpec]:
    options, operands = _option_values(words, _GREP_VALUE_SHORT, _GREP_VALUE_LONG)
    names = {name for name, _ in options}
    recursive = bool(names & {"-r", "-R", "--recursive", "--dereference-recursive"}) or any(
        name in ("-d", "--directories") and value == "recurse" for name, value in options) or verb == "ug"
    if not recursive:
        return None
    pattern_given = bool(names & {"-e", "-f", "--regexp", "--file"})
    pattern_index = None if pattern_given or not operands else operands[0][0]
    paths = [w for k, w in operands if k != pattern_index] or ["."]
    return SearchSpec(
        verb=verb, reads_everything=True, paths=tuple(paths),
        includes=tuple(v for n, v in options if n == "--include"),
        excludes=tuple(v for n, v in options if n == "--exclude"),
        exclude_dirs=tuple(v for n, v in options if n == "--exclude-dir"),
        types=(), pattern_index=pattern_index,
    )


def _rg_spec(words: Sequence[str]) -> SearchSpec:
    options, operands = _option_values(words, _RG_VALUE_SHORT, _RG_VALUE_LONG)
    names = [name for name, _ in options]
    unrestricted = names.count("-u") + names.count("--unrestricted")
    hidden = "--hidden" in names or "-." in names or unrestricted >= 2
    globs = [v for n, v in options if n in ("-g", "--glob", "--iglob")]
    pattern_given = any(n in ("-e", "-f", "--regexp", "--file") for n in names)
    pattern_index = None if pattern_given or not operands else operands[0][0]
    paths = [w for k, w in operands if k != pattern_index] or ["."]
    return SearchSpec(
        verb="rg", reads_everything=hidden, paths=tuple(paths),
        includes=tuple(g for g in globs if not g.startswith("!")),
        excludes=tuple(g[1:] for g in globs if g.startswith("!")),
        exclude_dirs=(), types=tuple(v for n, v in options if n in ("-t", "--type")),
        pattern_index=pattern_index,
    )


def reads_whole_tree(words: Sequence[str]) -> bool:
    """An archiver, a syncer, or a recursive copy or diff."""
    head = Path(words[0]).name if words else ""
    if head in TREE_READERS:
        return True
    flags = RECURSIVE_COPIERS.get(head)
    if not flags:
        return False
    return any(w in flags or (w.startswith("-") and not w.startswith("--")
                              and any(f[1] in w[1:] for f in flags if len(f) == 2)) for w in words[1:])


def tool_glob_matches(pattern: str, secret: Path, tree: Path) -> bool:
    """A tool's own glob (grep --include, rg -g, the Grep tool's glob) against
    one file, the way those tools match: by name, or by its path under the tree."""
    folded = pattern.casefold().lstrip("/")
    if fnmatch.fnmatchcase(secret.name.casefold(), folded):
        return True
    try:
        relative = str(secret.relative_to(tree)).casefold()
    except ValueError:
        return False
    return fnmatch.fnmatchcase(relative, folded.replace("**/", "*").replace("/**", "/*"))


# Names no pattern aimed at secrets would match: a pattern that matches one of
# these (``*``, ``**/*``) is a general search, not a search for secrets.
ORDINARY_NAMES = ("main.py", "README.md", "index.ts", "package.json", "notes.txt", "Makefile")


def aimed_at_secrets(pattern: str, names: Iterable[str]) -> bool:
    """A search pattern picked for secrets files: it can match a secret name and
    no ordinary one (``.env``, ``**/.e*``, ``*.ENV`` — never ``*`` or ``*.py``)."""
    name = pattern.rstrip("/").rpartition("/")[2] or pattern
    return pattern_may_name(name, names, leading_dot_literal=False) and not pattern_may_name(
        name, ORDINARY_NAMES, leading_dot_literal=False)


def search_reads(spec: SearchSpec, base: Path, inv: Inventory) -> Optional[Tuple[str, Path]]:
    """The first (search folder, secret) the recursive search would read."""
    for word in spec.paths:
        tree = Path(word).expanduser()
        tree = tree if tree.is_absolute() else base / tree
        for secret in inv.reachable_under(tree):
            if secret not in inv.files:
                return word, secret                       # the host's state folder
            if any(tool_glob_matches(g, secret, tree) for g in spec.excludes):
                continue
            if spec.exclude_dirs and _under_excluded_dir(secret, tree, spec.exclude_dirs):
                continue
            if spec.includes:
                if any(tool_glob_matches(g, secret, tree) for g in spec.includes):
                    return word, secret
                continue
            if spec.types and not any("env" in t.casefold() for t in spec.types):
                continue
            if spec.reads_everything:
                return word, secret
    return None


def _under_excluded_dir(secret: Path, tree: Path, patterns: Sequence[str]) -> bool:
    try:
        folders = secret.relative_to(tree).parts[:-1]
    except ValueError:
        return False
    return any(fnmatch.fnmatchcase(part.casefold(), p.casefold()) for part in folders for p in patterns)


# ── find … -exec ────────────────────────────────────────────────────────────

FIND_EXEC_OPTIONS = ("-exec", "-execdir", "-ok", "-okdir")
_FIND_LEADING = frozenset({"-H", "-L", "-P", "-E", "-X", "-d", "-s", "-x", "-f"})
_FIND_NAME_TESTS = {"-name": "name", "-iname": "name", "-path": "path", "-ipath": "path",
                    "-wholename": "path", "-iwholename": "path"}


@dataclass(frozen=True)
class FindSpec:
    starts: Tuple[str, ...]
    positives: Tuple[Tuple[str, str], ...]     # (name|path, pattern) that must all match
    negatives: Tuple[Tuple[str, str], ...]     # (name|path, pattern) that exclude a hit
    has_or: bool


def find_spec(words: Sequence[str]) -> Optional[Tuple[FindSpec, List[str]]]:
    """A ``find`` that runs a command per hit: its walk, and the command's words."""
    i = 1
    while i < len(words) and (words[i] in _FIND_LEADING or words[i].startswith(("-O", "-D"))):
        i += 1
    starts: List[str] = []
    while i < len(words) and not words[i].startswith("-") and words[i] not in ("(", "!", ")"):
        starts.append(words[i])
        i += 1
    expression = list(words[i:])
    at = next((k for k, w in enumerate(expression) if w in FIND_EXEC_OPTIONS), None)
    if at is None:
        return None
    tests = expression[:at]
    positives: List[Tuple[str, str]] = []
    negatives: List[Tuple[str, str]] = []
    k = 0
    while k < len(tests):
        negated = tests[k] in ("!", "-not")
        k += 1 if negated else 0
        if k + 1 < len(tests) and tests[k] in _FIND_NAME_TESTS:
            entry = (_FIND_NAME_TESTS[tests[k]], tests[k + 1])
            (negatives if negated else positives).append(entry)
            k += 2
            continue
        k += 1
    spec = FindSpec(starts=tuple(starts) or (".",), positives=tuple(positives),
                    negatives=tuple(negatives), has_or=any(w in ("-o", "-or", ",") for w in tests))
    inner = [w for w in expression[at + 1:] if w not in ("{}", "+", ";")]
    return spec, inner


def _find_test_matches(kind: str, pattern: str, printed: str, name: str) -> bool:
    subject = name if kind == "name" else printed
    return fnmatch.fnmatchcase(subject.casefold(), pattern.casefold())


def find_reads(spec: FindSpec, base: Path, inv: Inventory) -> Optional[Tuple[str, Path]]:
    """The first (start, secret) whose per-hit command would open a secret."""
    for start in spec.starts:
        tree = Path(start).expanduser()
        tree = tree if tree.is_absolute() else base / tree
        for secret in inv.reachable_under(tree):
            if secret not in inv.files or spec.has_or:
                return start, secret
            printed = start.rstrip("/") + "/" + str(secret.relative_to(tree))
            if any(_find_test_matches(kind, p, printed, secret.name) for kind, p in spec.negatives):
                continue
            if all(_find_test_matches(kind, p, printed, secret.name) for kind, p in spec.positives):
                return start, secret
    return None
