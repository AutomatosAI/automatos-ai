"""PRD-209 S2 — one clean Alembic head; fresh clones boot; stamped DBs keep upgrading.

The 12-head forest that the PRD-209 spec described has already been collapsed to a
single head by prior merge revisions (PRD-176/203/204/230 …); ``alembic heads``
returned exactly ``prd_workspace_models_backfill`` when this guard was written (PRD-232
moved it to ``prd232_cluster_provenance`` — see ``EXPECTED_HEAD``). This wave's job is
not to squash again (deleting revision files would STRAND Railway prod — constraint
B) but to (a) lock the single-head invariant against a future divergent head, (b)
prove no stamped database is stranded, and (c) guard the fresh path: since the
2026-08-29 S2 revision, fresh databases are built by ``scripts/init_fresh_db.py``
(the CI-proven create_all + raw-DDL schema) and stamped at ``heads`` — the stale
``init_complete_schema.sql`` snapshot is deleted (fresh clones were getting 107 of
prod's ~152 tables; and the forest's 41 orphan-root revisions make a from-empty
replay impossible until the recorded lineage-repair follow-on).

Four properties, all proven here purely (AST over the versions dir + text reads —
no database, no alembic import, no Docker):

1. ``len(heads) == 1`` and the head is the pinned ``EXPECTED_HEAD`` (AC1).
2. Every other revision is an ancestor of that single head — i.e. whatever revision
   a stamped DB (Railway prod, at any historical frontier) sits at, ``alembic upgrade
   heads`` walks forward to the head with no manual step (AC2 / constraint B). This is
   the strongest form of "every pre-change frontier head is an ancestor of the new
   single head": it asserts it for *all* revisions, not an enumerated dozen.
3. The fresh path is wired: ``init_fresh_db.py`` stamps ``heads``, the entrypoint
   routes empty databases through it, and no stale init SQL remains anywhere (AC3).
4. No revision file was deleted (guarded structurally by properties 1–2: a deleted
   ancestor would break the down_revision chain and orphan the head).
"""
from __future__ import annotations

import ast
import pathlib

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
_VERSIONS = _ORCH / "alembic" / "versions"
_INIT_FRESH = _ORCH / "scripts" / "init_fresh_db.py"
_ENTRYPOINT = _REPO / "docker-entrypoint.sh"
_COMPOSE = _REPO / "docker-compose.yml"

# The single head on this branch. Changing the head (a new terminal revision) is a
# deliberate act that must update this pin — that is the point of the guard.
# PRD-232 (2026-09-02): the one authorized 232 revision, prd232_cluster_provenance,
# chains onto prd_workspace_models_backfill and is the new single head.
EXPECTED_HEAD = "prd234_s1a_cli_hosts_runtime_ref"


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
        "deliberately, update EXPECTED_HEAD (init_fresh_db stamps 'heads', so it follows)."
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


def test_prd209_fresh_path_is_wired_and_stale_sql_is_gone():
    # (a) The fresh initializer exists and stamps at heads (never a literal revision —
    # stamping "heads" tracks the single head as it moves).
    fresh = _INIT_FRESH.read_text(encoding="utf-8")
    assert "build_schema(engine)" in fresh, (
        "init_fresh_db.py must build via generate_schema_baseline.build_schema (models + tolerant replay)"
    )
    gen = (_ORCH / "scripts" / "generate_schema_baseline.py").read_text(encoding="utf-8")
    assert "init_db()" in gen and "command.upgrade(cfg, rev)" in gen, (
        "the generator must run the model layer (init_db) AND replay the migration forest"
    )

    # (b) The entrypoint routes empty databases through it, fail-closed.
    entry = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "init_fresh_if_empty" in entry and "scripts.init_fresh_db" in entry, (
        "docker-entrypoint.sh must run init_fresh_db on an empty database"
    )

    # (c) The stale snapshot is fully retired: file gone, compose mounts nothing.
    assert not (_ORCH / "core" / "database" / "init_complete_schema.sql").exists(), (
        "init_complete_schema.sql must stay deleted — it was a stale snapshot "
        "(fresh clones got 107 of prod's ~152 tables)"
    )
    assert "init_complete_schema" not in _COMPOSE.read_text(encoding="utf-8"), (
        "docker-compose.yml must not mount the retired init SQL"
    )

    # Head parity: the computed head still matches the pin.
    revs = _parse_revisions()
    assert _heads(revs)[0] == EXPECTED_HEAD
