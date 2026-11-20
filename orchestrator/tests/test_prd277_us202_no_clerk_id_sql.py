"""PRD-277 US-202: no SQL outside the Clerk modules names ``users.clerk_user_id``.

Membership and role reads are keyed on the person's own ``users.id``. A Clerk user ID becomes
a ``users.id`` in one place: ``core/auth/clerk_person.py`` (with the other Clerk modules
named in ``EXEMPT_FILES``) and the sign-in link, ``core/auth/user_link.py``. This test scans every
Python file under ``orchestrator/`` and fails, naming the file, the function and the line, when
code anywhere else:

* holds a string that names ``clerk_user_id`` next to a SQL keyword (``SELECT``, ``UPDATE``,
  ``INSERT``, ``DELETE``, ``WHERE``, ``JOIN``, in any case): a ``text()`` statement, also one
  split over adjacent literals, which Python joins into one string;
* reads the column off a class or a table (``User.clerk_user_id``, ``UserModel.clerk_user_id``,
  ``models.User.clerk_user_id``, ``users.c.clerk_user_id``), in a filter, a join or a select;
* filters by it as a keyword (``filter_by(clerk_user_id=...)``), or names it to
  ``getattr`` on a class, ``column()`` or ``literal_column()``.

Exempt: the files named in ``EXEMPT_FILES`` (the Clerk modules in ``core/auth/`` today, and
``core/auth/user_link.py``), ``alembic/versions/`` (history is not rewritten) and the tests. A new
Clerk module is scanned like any other file until it is added to ``EXEMPT_FILES`` on purpose.
``ALLOWED`` names each other statement still allowed, by its file, its function and its own line,
and why; it allows that one statement once, and a second use in the same function fails.

Not caught: anything inside an exempt file, SQL whose column name is put together at run time
(``f"... WHERE {column} = :v"`` with ``column = "clerk_user_id"``), a model reached through a lower-case alias
(``u = aliased(User)``; ``u.clerk_user_id``), SQL with none of the keywords above, and SQL
outside Python files. Reading the attribute off a row or a request context
(``ctx.user.clerk_user_id``, ``user.clerk_user_id``) is not SQL and is left alone, as are
docstrings and comments.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

ORCH = Path(__file__).resolve().parents[1]
COLUMN = "clerk_user_id"
MENTION = re.compile(r"\bclerk_user_id\b")
SQL_KEYWORD = re.compile(r"\b(SELECT|UPDATE|INSERT|DELETE|WHERE|JOIN)\b", re.IGNORECASE)
COLUMN_CALLS = frozenset({"column", "literal_column"})
# Exact files, never a pattern: a new file is exempt only when it is added here.
EXEMPT_FILES = frozenset({
    "core/auth/clerk.py",
    "core/auth/clerk_azp.py",
    "core/auth/clerk_cutover.py",
    "core/auth/clerk_mobile_template.py",
    "core/auth/clerk_person.py",
    "core/auth/user_link.py",
})
EXEMPT_PREFIXES = ("tests/", "alembic/versions/")
SKIPPED_DIRS = frozenset({"__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"})
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
# Enough files that a scan rooted in the wrong place cannot pass by finding nothing.
MIN_SCANNED_FILES = 500

# (file, function, the statement's own line, stripped) -> why it may still name the column. Each
# key allows that one statement once. The one left reads the driving user's Clerk user ID FROM
# their users.id (the way back, not a lookup by Clerk user ID). Moving it to
# core/auth/clerk_person.py (clerk_id_row) means editing a function far over the 50-line limit
# the code-shape check holds a touched function to, so it stays until that function is split.
_DRIVING_CLERK_ID = (
    "US-202: reads the driving user's Clerk user ID from their users.id, inside a function far over "
    "the 50-line limit (scripts/ci/check_changed_code_shape.py); moving it to clerk_id_row means "
    "splitting that function first."
)
AllowedStatement = Tuple[str, str, str]
ALLOWED: Dict[AllowedStatement, str] = {
    ("consumers/chatbot/service.py", "_stream_tool_loop",
     "_row = self.db.query(User.clerk_user_id).filter(User.id == _driving_user).first()"): _DRIVING_CLERK_ID,
}


def _docstring_ids(tree: ast.AST) -> Set[int]:
    """String constants that stand alone as statements: docstrings and string comments."""
    return {id(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)}


def _is_class_or_table(node: ast.AST) -> bool:
    """A model class (``User``, ``models.User``) or a table's column set (``users.c``)."""
    if isinstance(node, ast.Name):
        return node.id[:1].isupper()
    if isinstance(node, ast.Attribute):
        return node.attr == "c" or node.attr[:1].isupper()
    return False


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def _names_the_column(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == COLUMN


def _call_finding(node: ast.Call) -> str:
    name = _call_name(node)
    if name == "filter_by" and any(keyword.arg == COLUMN for keyword in node.keywords):
        return "filters by clerk_user_id"
    if name == "getattr" and len(node.args) >= 2 and _is_class_or_table(node.args[0]) \
            and _names_the_column(node.args[1]):
        return "reads the clerk_user_id column by getattr"
    if name in COLUMN_CALLS and node.args and _names_the_column(node.args[0]):
        return f"names clerk_user_id to {name}()"
    return ""


def _finding(node: ast.AST, docstrings: Set[int]) -> str:
    """What ``node`` does with the column in SQL, or an empty string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
        if MENTION.search(node.value) and SQL_KEYWORD.search(node.value):
            return "names clerk_user_id in SQL text"
    if isinstance(node, ast.Attribute) and node.attr == COLUMN and _is_class_or_table(node.value):
        return "reads the clerk_user_id column"
    if isinstance(node, ast.Call):
        return _call_finding(node)
    return ""


def _functions(tree: ast.AST) -> List[Tuple[int, int, str]]:
    return [(node.lineno, node.end_lineno or node.lineno, node.name)
            for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]


def _enclosing(functions: List[Tuple[int, int, str]], line: int) -> str:
    """The innermost function holding ``line``, or ``<module>``."""
    holding = [(end - start, name) for start, end, name in functions if start <= line <= end]
    return min(holding)[1] if holding else "<module>"


def clerk_id_sql(source: str) -> List[Tuple[int, str, str]]:
    """Every ``(line, function, what)`` in ``source`` where SQL names ``clerk_user_id``."""
    tree = ast.parse(source)
    docstrings, functions = _docstring_ids(tree), _functions(tree)
    found = ((getattr(node, "lineno", 0), _finding(node, docstrings)) for node in ast.walk(tree))
    return sorted((line, _enclosing(functions, line), what) for line, what in found if what)


def _is_exempt(relative: str) -> bool:
    return relative in EXEMPT_FILES or relative.startswith(EXEMPT_PREFIXES)


def scanned_files() -> List[str]:
    """Every Python file under ``orchestrator/`` the rule covers, relative to it."""
    paths = (path for path in ORCH.rglob("*.py") if not SKIPPED_DIRS & set(path.relative_to(ORCH).parts))
    return sorted(rel for rel in (path.relative_to(ORCH).as_posix() for path in paths) if not _is_exempt(rel))


def _sql_in(relative: str) -> List[Tuple[int, str, str]]:
    return clerk_id_sql((ORCH / relative).read_text(encoding="utf-8"))


def _keyed(relative: str, source: str) -> List[Tuple[AllowedStatement, int, str]]:
    """Each finding in ``source`` as ``((relative, function, its line stripped), line, what)``."""
    lines = source.splitlines()
    return [((relative, function, lines[line - 1].strip()), line, what)
            for line, function, what in clerk_id_sql(source)]


def _keyed_findings(relative: str) -> List[Tuple[AllowedStatement, int, str]]:
    return _keyed(relative, (ORCH / relative).read_text(encoding="utf-8"))


def offenders_in(relative: str, source: str) -> List[str]:
    """Findings no allowance covers; an allowed statement found more than once is not covered."""
    found = _keyed(relative, source)
    seen = Counter(key for key, _, _ in found)
    return [f"orchestrator/{relative}:{line} in {key[1]}(): {what}"
            for key, line, what in found if key not in ALLOWED or seen[key] > 1]


def _offenders(relative: str) -> List[str]:
    return offenders_in(relative, (ORCH / relative).read_text(encoding="utf-8"))


def test_no_sql_outside_the_clerk_modules_names_the_clerk_id():
    offenders = [offender for rel in scanned_files() for offender in _offenders(rel)]
    assert not offenders, (
        "read the person by users.id: a Clerk user ID becomes one through core/auth/clerk_person.py "
        "(clerk_person_row, clerk_person_query, clerk_person_pk_expression) or core/auth/person.py "
        "(by_clerk_id), never SQL of your own:\n" + "\n".join(offenders)
    )


def test_every_allowance_is_still_needed():
    for key, reason in ALLOWED.items():
        found = Counter(found_key for found_key, _, _ in _keyed_findings(key[0]))
        assert reason and found[key] == 1, (
            f"{key[0]} {key[1]}() no longer holds exactly this statement once: {key[2]!r}. "
            "Drop or correct its allowance."
        )


def test_every_exempt_file_exists():
    missing = sorted(rel for rel in EXEMPT_FILES if not (ORCH / rel).is_file())
    assert not missing, f"exempt files that no longer exist, drop them: {missing}"


def test_the_scan_covers_the_backend_and_skips_only_the_clerk_modules():
    files = scanned_files()
    assert len(files) >= MIN_SCANNED_FILES
    assert {"core/auth/hybrid.py", "core/auth/workspace_permission.py", "core/auth/workspace_admin.py",
            "core/auth/person.py", "core/auth/principal.py", "api/tools.py"} <= set(files)
    assert not {"core/auth/clerk.py", "core/auth/clerk_person.py", "core/auth/user_link.py"} & set(files)


def test_the_translation_lives_in_the_clerk_module():
    assert _sql_in("core/auth/clerk_person.py"), "the scan no longer sees the one place it allows"


@pytest.mark.parametrize("source", [
    "from sqlalchemy import text\nQ = text('SELECT id FROM users WHERE clerk_user_id = :cid')\n",
    "Q = ('SELECT wm.role FROM workspace_members wm '\n     'JOIN users u ON u.id = wm.user_id '\n"
    "     'WHERE u.clerk_user_id = :cid')\n",
    "Q = 'update users set last_sign_in = now() where clerk_user_id = :cid'\n",
    "def f(db, User, cid):\n    return db.query(User).filter(User.clerk_user_id == cid).first()\n",
    "def f(db, models, pk):\n    return db.query(models.User.clerk_user_id).filter(models.User.id == pk)\n",
    "def f(db, users, cid):\n    return users.c.clerk_user_id == cid\n",
    "def f(db, User, cid):\n    return db.query(User).filter_by(clerk_user_id=cid).first()\n",
    "def f(User):\n    return getattr(User, 'clerk_user_id')\n",
    "from sqlalchemy import column\nC = column('clerk_user_id')\n",
])
def test_the_scan_catches(source):
    assert clerk_id_sql(source)


@pytest.mark.parametrize("source", [
    '"""Rows are found by clerk_user_id: SELECT id FROM users WHERE clerk_user_id = :cid."""\n',
    "def f(ctx):\n    return ctx.user.clerk_user_id  # SELECT ... WHERE clerk_user_id\n",
    "def f(info):\n    return info.get('clerk_user_id')\n",
    "import logging\nlogging.getLogger(__name__).info('Resolved clerk_user_id=%s', 1)\n",
    "def f(post, person):\n    return post(clerk_user_id=person)\n",
    "def f(row):\n    return row.old_clerk_user_id\n",
    "Q = 'SELECT id FROM users WHERE old_clerk_user_id = :old'\n",
])
def test_the_scan_leaves_alone(source):
    assert clerk_id_sql(source) == []


def test_the_scan_names_the_function_it_found_in():
    source = "def outer(db, User, cid):\n    def inner():\n        return User.clerk_user_id == cid\n    return inner\n"
    assert clerk_id_sql(source) == [(3, "inner", "reads the clerk_user_id column")]


_ALLOWED_FILE, _ALLOWED_FUNCTION, _ALLOWED_LINE = next(iter(ALLOWED))


def _in_the_allowed_function(*body: str) -> str:
    return f"def {_ALLOWED_FUNCTION}(self, User, _driving_user):\n" + "".join(f"    {line}\n" for line in body)


def test_the_allowance_covers_its_one_statement():
    assert offenders_in(_ALLOWED_FILE, _in_the_allowed_function(_ALLOWED_LINE)) == []


@pytest.mark.parametrize("body", [
    (_ALLOWED_LINE, _ALLOWED_LINE),
    (_ALLOWED_LINE, "return self.db.query(User).filter(User.clerk_user_id == _driving_user).first()"),
])
def test_a_second_use_in_the_allowed_function_fails(body):
    assert offenders_in(_ALLOWED_FILE, _in_the_allowed_function(*body))
