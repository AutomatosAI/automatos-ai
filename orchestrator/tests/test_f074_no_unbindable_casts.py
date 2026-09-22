"""F074 — a bind parameter is never followed by a ``::`` cast in raw SQL.

SQLAlchemy's ``text()`` does not bind ``:name::type``: it registers a bogus
bind (``:team::text`` → ``tea``) and sends the literal text to Postgres, which
rejects it with a syntax error at ``":"``. On a request session shared by every
tool in a chat turn, that failed statement leaves the transaction aborted, and
every later tool in the turn fails with "current transaction is aborted".

Night 1, 19:30:09–19:30:26: ``search_multimodal`` and ``search_tables`` failed
on ``:query_embedding::vector``, then query_database (all seven of its night-1
calls), platform_shopify_sync_status and COMPOSIO_SEARCH_TAVILY failed on the
aborted transaction. The multimodal casts were fixed since; seven more sites
were still live. Write ``CAST(:name AS type)``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[1]
UNBINDABLE = re.compile(r"(?<![:\w]):[A-Za-z_]\w*::[A-Za-z_]")
SKIP_DIRS = {"tests", "__pycache__", "node_modules", ".venv", "venv"}


def _string_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _offenders():
    found = []
    for path in ORCH.rglob("*.py"):
        if SKIP_DIRS & set(path.relative_to(ORCH).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for lineno, value in _string_literals(tree):
            for match in UNBINDABLE.finditer(value):
                found.append(f"{path.relative_to(ORCH)}:{lineno}: {match.group(0)}…")
    return found


def test_no_bind_parameter_is_followed_by_a_cast():
    offenders = _offenders()
    assert not offenders, "write CAST(:name AS type) — SQLAlchemy never binds :name::type:\n" + "\n".join(offenders)


@pytest.mark.parametrize("sql,binds", [
    ("select to_jsonb(:team::text)", False),
    ("select to_jsonb(CAST(:team AS text))", True),
    ("select '[]'::jsonb, created_at::date", None),     # casts on literals and columns are fine
])
def test_the_pattern_is_what_sqlalchemy_cannot_bind(sql, binds):
    from sqlalchemy import text

    flagged = bool(UNBINDABLE.search(sql))
    if binds is None:
        assert not flagged
        return
    assert flagged is (not binds)
    assert ("team" in text(sql)._bindparams) is binds
