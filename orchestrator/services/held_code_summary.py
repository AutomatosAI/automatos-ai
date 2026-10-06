"""F370 (ii) (night 10c): a held inline-Python command says in plain words what it does.

The session gate holds ``python3 -c "…"`` and ``python3 - <<'EOF' … EOF`` because it
cannot read code given inline (the host's ``shell_words.opaque_reason``: "python3
runs code given inline"). The owner's card then said only "The agent wants to **run
a Python command**" over the code itself: the Brand Designer's PIL version check,
its colour counts and a contrast function reached a shop owner who could judge none
of them ("As a shop owner I don't know what that is").

The backend can read the code: it is Python, parsed here (``ast``, never run) into
plain words on what it was seen to do, e.g. "It opens an image with PIL and reads its
colours, and prints the result."

The card never says a command is safe. A list of known writes cannot prove there are
none (``io.FileIO(p, 'w')``, ``numpy.memmap(mode='w+')``, pickle, tempfile and many
more write without matching it), so the card DESCRIBES what was seen, WARNS when it
sees a write, the network, another program or a name looked up as it runs, and
otherwise ends on a line that promises nothing: "Only the Python part is described
here; read the command to be sure what it changes."

Only a command that is exactly one Python invocation whose code the shell cannot
alter is described: ``python3 -c '…'`` (or double quotes with no ``$``, backtick or
backslash), or ``python3 - <<'EOF'`` with a QUOTED delimiter and nothing after the
closing one, with at most one ``cd <plain path> && `` in front (the card then says
where it runs). Anything else (a second ``cd``, a ``cd ~`` or ``cd $X``, a ``; rm``, a ``> out.txt``, an
unquoted heredoc whose ``$(…)`` the shell expands first) adds nothing, and the card
stays as it was. Text taken from the code (paths, library names) is shown only inside
code spans, so it cannot write card text of its own.
"""
from __future__ import annotations

import ast
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

# The interpreters whose inline code this reads (``python``, ``python3``, ``python3.11``).
_PYTHON = re.compile(r"^python(\d+(\.\d+)?)?$")
# The whole command: the interpreter, its own flags, then the code. Nothing before or after.
_INTERPRETER = r"^[ \t]*(?P<interpreter>[^\s'\"`$\\;&|<>()*?\[\]{}~=]+)(?P<flags>(?:[ \t]+-[A-Za-z]*)*)"
# ``python3 -c 'CODE'`` or ``python3 -c "CODE"``.
_INLINE = re.compile(_INTERPRETER + r"[ \t]+-c[ \t]+(?:'(?P<single>[^']*)'|\"(?P<double>[^\"]*)\")[ \t]*\Z", re.DOTALL)
# ``python3 - <<'EOF'`` … ``EOF``: a QUOTED delimiter (the shell expands nothing in the body),
# nothing else on the first line, and nothing after the closing line.
_HEREDOC = re.compile(_INTERPRETER + r"[ \t]*<<[ \t]*(?P<quote>['\"])(?P<end>\w+)(?P=quote)[ \t]*\n"
                      r"(?P<body>.*)\n(?P=end)\n?\Z", re.DOTALL)
# What the shell still rewrites inside double quotes.
_SHELL_REWRITES_IN_DOUBLE_QUOTES = re.compile(r"[$`\\]")
_PUNCTUATION = frozenset("();<>|&")
INLINE_FLAG = "-c"
# One leading ``cd <path> && `` is allowed: a literal path of plain characters, never an
# option (``cd -``), the home folder, a variable, a quote, a space or a glob.
_CD_PREFIX = re.compile(r"^[ \t]*cd[ \t]+(?P<folder>[A-Za-z0-9._/][A-Za-z0-9._/-]*)[ \t]+&&[ \t]+")

# Libraries this reads the calls of (an import of any other library is said to be unreadable).
KNOWN_LIBRARIES = frozenset({
    "PIL", "math", "cmath", "colorsys", "json", "re", "statistics", "collections", "itertools", "functools",
    "fractions", "decimal", "string", "textwrap", "sys", "os", "pathlib", "glob", "fnmatch", "hashlib",
    "struct", "datetime", "time", "calendar", "csv", "numpy", "typing", "dataclasses", "random", "base64",
    "binascii", "unicodedata", "zlib", "io", "pprint", "operator", "enum", "heapq", "bisect", "array", "copy",
    "shutil", "platform", "locale", "html", "xml", "difflib",
})
NETWORK_LIBRARIES = frozenset({"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
                               "imaplib", "poplib", "telnetlib", "paramiko", "boto3", "websocket", "websockets"})
PROGRAM_LIBRARIES = frozenset({"subprocess", "pty", "multiprocessing", "importlib", "ctypes", "runpy", "code"})
# Calls that write, delete or move files whatever object they are called on.
WRITE_CALLS = frozenset({
    "save", "savefig", "imsave", "write", "write_text", "write_bytes", "writelines", "unlink", "rmdir",
    "removedirs", "mkdir", "makedirs", "touch", "chmod", "chown", "lchown", "rmtree", "truncate", "to_csv",
    "to_excel", "to_parquet", "to_json", "symlink_to", "hardlink_to", "dump", "mkfifo", "mknod", "utime",
    "savetxt", "savez", "savez_compressed", "tofile", "setxattr", "removexattr",
})
# File operations of ``os`` / ``shutil`` / ``pathlib`` whose names also mean harmless things
# (``str.replace``, ``list.copy``): a write when one of those modules is imported.
FILE_MODULES = frozenset({"os", "shutil", "pathlib"})
FILE_MODULE_WRITES = frozenset({"remove", "rename", "renames", "replace", "copy", "copy2", "copyfile",
                                "copytree", "copymode", "copystat", "move", "link", "symlink"})
PROGRAM_CALLS = frozenset({"system", "popen", "fork", "forkpty", "kill", "killpg", "startfile", "execv",
                           "execve", "execl", "execle", "execlp", "execvp", "execvpe", "spawnv", "spawnl",
                           "posix_spawn", "posix_spawnp"})
# Names looked up while the code runs: the card cannot tell what they reach.
DYNAMIC_CALLS = frozenset({"exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals",
                           "vars", "locals", "input", "breakpoint", "methodcaller", "attrgetter", "modules",
                           "__dict__", "__builtins__", "__getattribute__", "__subclasses__", "__globals__",
                           "__class__", "__loader__", "__spec__"})
READ_CALLS = frozenset({"read", "read_text", "read_bytes", "readlines", "readline", "listdir", "scandir",
                        "iterdir", "walk", "glob", "rglob", "stat", "exists", "is_file", "is_dir", "load",
                        "reader", "DictReader", "getsize"})
COLOUR_CALLS = frozenset({"getcolors", "getpixel", "histogram", "getdata", "getextrema", "quantize",
                          "getpalette", "convert", "getbbox", "split"})
_WRITE_MODE = re.compile(r"[wax+]")
_MODE_LIKE = re.compile(r"^[rwxabt+]{1,3}$")
VERSION_ATTRIBUTE = "__version__"
PRINT, OPEN, PIL_IMAGE = "print", "open", "Image"
TERMINAL_STREAMS = frozenset({"stdout", "stderr"})

# A place named in full (absolute or in the home folder) or a step up: the card names it,
# since the session's own folder is one of many places such a path can be.
_OUTSIDE_PATH = re.compile(r"^(?:/|~|\.\./|[A-Za-z]:[\\/])|/\.\./")
OUTSIDE_NAMES = frozenset({"expanduser", "home", "environ", "getenv", "expandvars"})
HOME_AND_SETTINGS = "your home folder or settings"
MAX_PATHS_SHOWN, MAX_PATH_SHOWN = 3, 120
OUTSIDE = "It names {places}."
# Warnings: each is a positive detection. Nothing here ever says a command is safe.
WRITES = "It writes, moves or deletes files."
NETWORK = "It can reach the internet."
RUNS_PROGRAMS = "It runs other programs or code."
CANNOT_TELL = "It {why}, so this card cannot tell what it does: read the exact command before you allow it."
# The end of every summary that raised no warning: a description, never a promise.
RUNS_IN = "It runs in {folder}."
NOTHING_PROMISED = "Only the Python part is described here; read the command to be sure what it changes."
# Built-ins that only compute: code calling nothing else (and its own functions) "does its own calculations".
CALCULATION_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "chr", "divmod", "enumerate", "filter", "float", "format", "hex", "int",
    "isinstance", "len", "list", "map", "max", "min", "ord", "pow", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "dict", "zip", PRINT,
})


@dataclass
class _Seen:
    """What the code imports, calls and names, gathered in one walk of its syntax tree."""

    imports: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    # Every name and attribute the code mentions, called or not: ``f = os.remove; f(p)``
    # never calls ``remove`` by name, but names it.
    names: Set[str] = field(default_factory=set)
    writes_by_open: bool = False
    open_aliased: bool = False
    outside: List[str] = field(default_factory=list)       # paths it names in full, or a level up
    own_functions: Set[str] = field(default_factory=set)   # names the code defines and calls itself


def inline_python(command: str) -> Optional[str]:
    """The Python a command runs, when the command is exactly one Python invocation whose
    code the shell cannot alter; otherwise ``None``."""
    _folder, command = _after_cd(command or "")
    heredoc = _HEREDOC.match(command)
    if heredoc:
        return _heredoc_code(heredoc)
    inline = _INLINE.match(command)
    if inline is None or not _is_python(inline) or _shell_sees_more(command):
        return None
    if inline.group("single") is not None:
        return inline.group("single")
    code = inline.group("double")
    return None if _SHELL_REWRITES_IN_DOUBLE_QUOTES.search(code) else code


def _after_cd(command: str) -> Tuple[Optional[str], str]:
    """``(folder, the rest)`` for a command that starts ``cd <folder> && ``, else ``(None, command)``.
    Only one: a second ``cd`` is no Python invocation, so the rest is refused."""
    found = _CD_PREFIX.match(command)
    return (found.group("folder"), command[found.end():]) if found else (None, command)


def _is_python(found: re.Match) -> bool:
    """The command's program is Python and its flags are its own (never a second ``-c``)."""
    flags = found.group("flags").split()
    return bool(_PYTHON.match(posixpath.basename(found.group("interpreter")))) and INLINE_FLAG not in flags


def _heredoc_code(found: re.Match) -> Optional[str]:
    """The body of a quoted heredoc, unless a line inside it ends the heredoc early (the
    shell would then run the lines after it as commands of their own)."""
    body = found.group("body")
    if not _is_python(found) or found.group("end") in body.split("\n"):
        return None
    return body


def _shell_sees_more(command: str) -> bool:
    """The shell sees more than ``interpreter [flags] -c CODE``: an operator, a second
    command or another word."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        return True
    flags = [word for word in words[1:-2] if word.startswith("-")]
    return (any(set(word) <= _PUNCTUATION for word in words) or len(words) != len(flags) + 3
            or words[-2] != INLINE_FLAG)


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _to_the_terminal(node: ast.Call) -> bool:
    """``sys.stdout.write(…)`` / ``sys.stderr.write(…)``: printing, not a file write."""
    func = node.func
    stream = func.value if isinstance(func, ast.Attribute) else None
    return (isinstance(stream, ast.Attribute) and stream.attr in TERMINAL_STREAMS
            and isinstance(stream.value, ast.Name) and stream.value.id == "sys")


def _owner(func: ast.expr) -> str:
    """The name a method is called on: ``Image`` for ``Image.open`` and ``PIL.Image.open``."""
    value = func.value if isinstance(func, ast.Attribute) else None
    return value.id if isinstance(value, ast.Name) else value.attr if isinstance(value, ast.Attribute) else ""


def _may_write_mode(node: ast.expr) -> bool:
    """A mode argument that may write: one with w, a, x or +, or one the code computes."""
    if not isinstance(node, ast.Constant):
        return True
    value = node.value
    return isinstance(value, str) and bool(_MODE_LIKE.match(value)) and bool(_WRITE_MODE.search(value))


def _open_writes(node: ast.Call) -> bool:
    """An ``open`` call that may write (or truncate): a write mode, a computed one, or ``os.open``."""
    owner = _owner(node.func)
    if owner == PIL_IMAGE:
        return False                                   # PIL's Image.open only reads
    if owner == "os":
        return True
    modes = [kw.value for kw in node.keywords if kw.arg == "mode"] + list(node.args[1:])
    if isinstance(node.func, ast.Attribute):
        modes += node.args[:1]                         # Path(...).open("w")
    return any(_may_write_mode(mode) for mode in modes)


def _note_import(seen: _Seen, node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        seen.imports |= {alias.name.split(".")[0] for alias in node.names}
    else:
        seen.imports.add((node.module or ".").split(".")[0])
        seen.names |= {alias.name for alias in node.names}       # from os import remove


def _note_call(seen: _Seen, node: ast.Call) -> None:
    name = PRINT if _to_the_terminal(node) else _call_name(node)
    seen.calls.add(name)
    seen.writes_by_open = seen.writes_by_open or (name == OPEN and _open_writes(node))


def code_span(text: str) -> str:
    """``text`` from the code as a markdown code span, where ``*``, ``_``, ``[``, ``<`` and
    links are shown as they are: no backtick (it would close the span), no line break, no
    control character, and at most ``MAX_PATH_SHOWN`` characters."""
    kept = "".join(" " if not ch.isprintable() else "'" if ch == "`" else ch for ch in str(text))
    return f"`{kept[:MAX_PATH_SHOWN].strip() or '?'}`"


def _names_a_path_outside(value: object) -> bool:
    """A string that names a place by full path, in the home folder, or a level up."""
    return isinstance(value, str) and bool(_OUTSIDE_PATH.match(value.strip()))


def _outside_note(seen: _Seen) -> str:
    """The places the code names by full path (and the home folder or settings), for the owner
    to check; empty when it names none."""
    places = [code_span(path) for path in list(dict.fromkeys(seen.outside))[:MAX_PATHS_SHOWN]]
    places += [HOME_AND_SETTINGS] if seen.names & OUTSIDE_NAMES else []
    return OUTSIDE.format(places=", ".join(places)) if places else ""


def _seen(tree: ast.AST) -> _Seen:
    seen = _Seen()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called = {id(node.func) for node in calls}
    printing = {id(node.func) for node in calls if _to_the_terminal(node)}     # sys.stdout.write
    for node in ast.walk(tree):
        _note_node(seen, node, called, printing)
    return seen


def _note_node(seen: _Seen, node: ast.AST, called: Set[int], printing: Set[int]) -> None:
    """Add what one node of the tree imports, calls, names or points at to ``seen``."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        _note_import(seen, node)
    if isinstance(node, ast.FunctionDef) or _assigns_a_lambda(node):
        seen.own_functions |= _defined_names(node)
    if isinstance(node, ast.Call):
        _note_call(seen, node)
    if isinstance(node, ast.Constant) and _names_a_path_outside(node.value):
        seen.outside.append(node.value)
    if isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in printing:
        name = node.id if isinstance(node, ast.Name) else node.attr
        seen.names.add(name)
        seen.open_aliased = seen.open_aliased or (name == OPEN and id(node) not in called)


def _assigns_a_lambda(node: ast.AST) -> bool:
    return isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda)


def _defined_names(node: ast.AST) -> Set[str]:
    """The names a ``def`` or a ``name = lambda …`` gives the code's own functions."""
    if isinstance(node, ast.FunctionDef):
        return {node.name}
    targets = node.targets if isinstance(node, ast.Assign) else []
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _warning(seen: _Seen) -> str:
    """A warning when the code was seen to write, reach the network, run a program, look
    names up as it runs or use a library this cannot read; empty otherwise. Empty is not a
    verdict: much that writes matches nothing here."""
    unknown = sorted(seen.imports - KNOWN_LIBRARIES - NETWORK_LIBRARIES - PROGRAM_LIBRARIES)
    named = seen.names | seen.calls
    file_writes = FILE_MODULE_WRITES if seen.imports & FILE_MODULES else frozenset()
    if named & DYNAMIC_CALLS or seen.open_aliased:
        return CANNOT_TELL.format(why="looks up what to run while it runs")
    if seen.imports & PROGRAM_LIBRARIES or named & PROGRAM_CALLS:
        return RUNS_PROGRAMS
    if seen.imports & NETWORK_LIBRARIES:
        return NETWORK
    if seen.writes_by_open or named & (WRITE_CALLS | file_writes):
        return WRITES
    if unknown:
        return CANNOT_TELL.format(why=f"uses {', '.join(code_span(name) for name in unknown[:3])}")
    return ""


def _description(seen: _Seen) -> str:
    """What the code was seen to do, in plain words; empty when there is nothing plain to say."""
    reads_files = (OPEN in seen.calls and not seen.writes_by_open) or bool(seen.calls & READ_CALLS)
    if seen.calls <= {PRINT} and VERSION_ATTRIBUTE in seen.names and seen.imports:
        libraries = ", ".join(code_span(name) for name in sorted(seen.imports))
        return f"It checks which version of {libraries} is installed, and prints it"
    if "PIL" in seen.imports and OPEN in seen.calls:
        colours = " and reads its colours" if seen.calls & COLOUR_CALLS else ""
        what = f"It opens an image with PIL{colours}"
    elif reads_files:
        what = "It reads files"
    elif not seen.imports and seen.calls <= CALCULATION_BUILTINS | seen.own_functions:
        what = "It does its own calculations with Python's built-in maths"
    else:
        return ""
    return what + (", and prints the result" if PRINT in seen.calls else "")


def plain_summary(command: str) -> Optional[str]:
    """What a single inline-Python command was seen to do, in plain words, ending on a warning
    or on a line that promises nothing; ``None`` when the command is not exactly one Python
    invocation the shell cannot alter, or its code does not parse."""
    code = inline_python(command)
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    seen = _seen(tree)
    description = _description(seen)
    folder, _rest = _after_cd(command)
    runs_in = RUNS_IN.format(folder=code_span(folder)) if folder else ""
    parts = (f"{description}." if description else "", runs_in, _outside_note(seen), _warning(seen) or NOTHING_PROMISED)
    return " ".join(part for part in parts if part)


def with_code_summary(intent: str, command: str) -> str:
    """The card's intent line, with the plain-words summary of any inline Python after it."""
    summary = plain_summary(command)
    return f"{intent} {summary}" if summary else intent


__all__ = ["NOTHING_PROMISED", "code_span", "inline_python", "plain_summary", "with_code_summary"]
