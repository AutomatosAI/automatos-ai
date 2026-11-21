"""PRD-277 (US-302): the calls the safe-method guard counts as changing state, read from source.

``tests/authz_sweep_probe.py`` hands each route's endpoint and each of its dependencies to
``route_writes``; ``test_prd277_safe_methods_change_nothing.py`` fails a GET, HEAD or OPTIONS
route reachable with the session cookie that makes any of these calls, unless it names the route
with its reason. Counted, anywhere in a function's own body (nested functions included):

* ``WRITER_ATTRIBUTES`` on anything: a commit, a flush, a ``begin`` (a block that commits as it
  leaves), ``add_all``, ``merge`` or a bulk write;
* ``SESSION_WRITER_ATTRIBUTES`` (``add``, ``delete``) on a database session, a name in
  ``SESSION_NAMES``;
* ``QUERY_WRITER_ATTRIBUTES`` (``update``, ``delete``) on a query: a call chain through
  ``query``, ``filter`` or ``filter_by``;
* ``WRITER_FUNCTIONS``: the shared helpers that commit.

Only the function's own source is read, never what it calls. A function whose source cannot be
read (a callable object, a lambda inside an expression) counts no calls.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Iterable, Optional, Tuple

WRITER_ATTRIBUTES = frozenset({
    "commit", "flush", "begin", "add_all", "merge",
    "bulk_save_objects", "bulk_insert_mappings", "bulk_update_mappings",
})
SESSION_WRITER_ATTRIBUTES = frozenset({"add", "delete"})
SESSION_NAMES = frozenset({"db", "session", "db_session", "_db"})
QUERY_WRITER_ATTRIBUTES = frozenset({"update", "delete"})
QUERY_BUILDERS = frozenset({"query", "filter", "filter_by"})
# core.database.database: end_open_transaction commits what is pending, get_db_session commits as
# its block leaves; core.auth.session_audit.commit_removal commits a removal with its audit row.
WRITER_FUNCTIONS = frozenset({"end_open_transaction", "get_db_session", "commit_removal"})


def _parsed(func: Any) -> Optional[ast.AST]:
    """``func``'s own source as a syntax tree, or None when it cannot be read."""
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None


def _final_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _is_query(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _final_name(node.func) in QUERY_BUILDERS


def _writer(call: ast.Call) -> Optional[str]:
    """The name of the write ``call`` makes, or None."""
    name = _final_name(call.func)
    if name in WRITER_FUNCTIONS:
        return name
    if not isinstance(call.func, ast.Attribute):
        return None
    receiver = call.func.value
    if name in WRITER_ATTRIBUTES:
        return name
    if name in SESSION_WRITER_ATTRIBUTES and _final_name(receiver) in SESSION_NAMES:
        return name
    if name in QUERY_WRITER_ATTRIBUTES and _is_query(receiver):
        return name
    return None


def writer_calls(func: Any) -> Tuple[str, ...]:
    """The writes ``func``'s own source makes, by name, sorted and each once."""
    tree = _parsed(func)
    if tree is None:
        return ()
    found = (_writer(node) for node in ast.walk(tree) if isinstance(node, ast.Call))
    return tuple(sorted({name for name in found if name is not None}))


def route_writes(endpoint: Any, dependencies: Iterable[Any]) -> Tuple[str, ...]:
    """The writes a route's endpoint and its dependencies make in their own source."""
    names = set(writer_calls(endpoint))
    for dependency in dependencies:
        names |= set(writer_calls(dependency))
    return tuple(sorted(names))


__all__ = [
    "QUERY_BUILDERS",
    "QUERY_WRITER_ATTRIBUTES",
    "SESSION_NAMES",
    "SESSION_WRITER_ATTRIBUTES",
    "WRITER_ATTRIBUTES",
    "WRITER_FUNCTIONS",
    "route_writes",
    "writer_calls",
]
