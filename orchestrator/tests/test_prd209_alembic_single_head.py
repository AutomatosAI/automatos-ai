"""PRD-209 S2 — one clean Alembic head; fresh clones boot; stamped DBs keep upgrading.

The 12-head forest that the PRD-209 spec described has already been collapsed to a
single head by prior merge revisions (PRD-176/203/204/230 …); ``alembic heads``
returns exactly ``prd_workspace_models_backfill`` on this branch. This wave's job is
not to squash again (deleting revision files would STRAND Railway prod — constraint
B) but to (a) lock the single-head invariant against a future divergent head, (b)
prove no stamped database is stranded, and (c) stamp ``init_complete_schema.sql`` so
a fresh compose/init volume boots AT the head instead of replaying (and crashing on)
the forest.

Four properties, all proven here purely (AST over the versions dir + a text read of
the initdb SQL — no database, no alembic import, no Docker):

1. ``len(heads) == 1`` and the head is the pinned ``EXPECTED_HEAD`` (AC1).
2. Every other revision is an ancestor of that single head — i.e. whatever revision
   a stamped DB (Railway prod, at any historical frontier) sits at, ``alembic upgrade
   heads`` walks forward to the head with no manual step (AC2 / constraint B). This is
   the strongest form of "every pre-change frontier head is an ancestor of the new
   single head": it asserts it for *all* revisions, not an enumerated dozen.
3. ``init_complete_schema.sql`` carries an ``alembic_version`` stamp equal to the head
   (AC3) — the fresh-clone boot fix.
4. No revision file was deleted (guarded structurally by properties 1–2: a deleted
   ancestor would break the down_revision chain and orphan the head).
"""
from __future__ import annotations

import ast
import pathlib

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_VERSIONS = _ORCH / "alembic" / "versions"
_INIT_SQL = _ORCH / "core" / "database" / "init_complete_schema.sql"

# The single head on this branch. Changing the head (a new terminal revision) is a
# deliberate act that must update this pin — that is the point of the guard.
EXPECTED_HEAD = "prd_workspace_models_backfill"


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return "<expr>"


def _parse_revisions() -> dict[str, object]:
    """revision id -> down_revision (str | tuple[str,...] | None), by AST."""
    revs: dict[str, object] = {}
    for p in sorted(_VERSIONS.glob("*.py")):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        rev = None
        down = None
        have_rev = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == "revision":
                        rev = _literal(n.value)
                        have_rev = True
                    elif isinstance(t, ast.Name) and t.id == "down_revision":
                        down = _literal(n.value)
        if have_rev and isinstance(rev, str):
            revs[rev] = down
    return revs


def _parents(down: object) -> tuple[str, ...]:
    if down is None:
        return ()
    if isinstance(down, (list, tuple)):
        return tuple(d for d in down if isinstance(d, str))
    if isinstance(down, str):
        return (down,)
    return ()


def _heads(revs: dict[str, object]) -> list[str]:
    referenced: set[str] = set()
    for down in revs.values():
        referenced.update(_parents(down))
    return [r for r in revs if r not in referenced]


def _ancestors(head: str, revs: dict[str, object]) -> set[str]:
    """All revisions reachable by walking down_revision links from ``head``."""
    seen: set[str] = set()
    stack = [head]
    while stack:
        cur = stack.pop()
        for parent in _parents(revs.get(cur)):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def test_prd209_exactly_one_head():
    revs = _parse_revisions()
    heads = _heads(revs)
    assert len(heads) == 1, (
        f"expected exactly one Alembic head, found {len(heads)}: {sorted(heads)}. "
        "A new divergent head shipped — add a merge revision to collapse it (never "
        "delete revision files: that strands stamped databases)."
    )
    assert heads[0] == EXPECTED_HEAD, (
        f"single head is {heads[0]!r}, pin expects {EXPECTED_HEAD!r}. If the head moved "
        "deliberately, update EXPECTED_HEAD and the init_complete_schema.sql stamp together."
    )


def test_prd209_no_stranded_database_every_revision_is_ancestor_of_head():
    # Constraint B: a database stamped at ANY revision (Railway prod's frontier
    # included) must reach the head via plain `alembic upgrade heads`. Proven by
    # showing every non-head revision is an ancestor of the single head.
    revs = _parse_revisions()
    heads = _heads(revs)
    assert len(heads) == 1
    head = heads[0]
    reachable = _ancestors(head, revs)
    stranded = sorted(r for r in revs if r != head and r not in reachable)
    assert not stranded, (
        f"{len(stranded)} revision(s) are NOT ancestors of the head {head!r} — a DB "
        f"stamped there would be stranded by `alembic upgrade heads`: {stranded[:10]}"
    )
    # Non-vacuity: the base initial migration must be reachable (the chain is whole).
    assert "6203026dbac0" in reachable, "initial migration unreachable — broken lineage"


def test_prd209_init_sql_stamps_the_head():
    sql = _INIT_SQL.read_text(encoding="utf-8")
    assert "alembic_version" in sql, "init_complete_schema.sql must create/stamp alembic_version"
    assert EXPECTED_HEAD in sql, (
        f"init_complete_schema.sql must stamp alembic_version at the head {EXPECTED_HEAD!r} "
        "so a fresh compose volume boots at head (no forest replay)."
    )
    # The stamp value must equal the actual computed head — parity, not a stale literal.
    revs = _parse_revisions()
    assert _heads(revs)[0] == EXPECTED_HEAD
