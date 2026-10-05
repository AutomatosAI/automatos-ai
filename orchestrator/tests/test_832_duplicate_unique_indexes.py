"""#832 — a fresh build carries duplicate unique indexes (``users_email_key``,
``_key1``, ``users_username_key``, ``_key1``…).

``build_schema`` builds ``users`` from the models (``create_all``, with the
model's own ``unique=True`` index) and the creator migrations (e.g.
``128a785a7681``, ``208275450a15``) ALSO declare UNNAMED unique constraints on
the same columns. Each replay of one of those — the tolerant forest pass, then
a residual pass's ``_rerun_upgrade`` — adds another copy, and Postgres
auto-names it ``_key``, ``_key1``, ``_key2``… ``_dedupe_unique_indexes`` is the
repair stage that removes them: keep the lowest-numbered/canonical index per
(table, definition), drop the rest, DROP CONSTRAINT when the duplicate is
backed by a unique constraint (DROP INDEX on those orphans the constraint),
DROP INDEX for a bare duplicate index, and never touch an index a foreign key
still depends on. This mirrors ``test_fresh_init_resume.py``'s style: the real
decision logic is tested directly against fixture rows / a fake engine, no
live Postgres needed.
"""
from __future__ import annotations

from scripts import generate_schema_baseline as gsb

_EMAIL_DEF = "CREATE UNIQUE INDEX ON public.users USING btree (email)"
_USERNAME_DEF = "CREATE UNIQUE INDEX ON public.users USING btree (username)"


def _row(table, index, normdef, constraint_type=None, constraint_name=None, fk_dependent=False):
    return {
        "table_name": table,
        "index_name": index,
        "normdef": normdef,
        "constraint_type": constraint_type,
        "constraint_name": constraint_name,
        "fk_dependent": fk_dependent,
    }


# --------------------------------------------------------------- _duplicate_groups
def test_duplicate_groups_keeps_only_the_lowest_numbered_canonical_first():
    rows = [
        _row("users", "users_email_key1", _EMAIL_DEF),
        _row("users", "users_email_key", _EMAIL_DEF),
        _row("users", "users_email_key2", _EMAIL_DEF),
    ]
    groups = gsb._duplicate_groups(rows)
    assert len(groups) == 1
    assert [r["index_name"] for r in groups[0]] == [
        "users_email_key", "users_email_key1", "users_email_key2",
    ]


def test_duplicate_groups_ignores_a_table_with_no_duplicate():
    assert gsb._duplicate_groups([_row("users", "users_email_key", _EMAIL_DEF)]) == []


def test_duplicate_groups_keeps_different_columns_and_tables_apart():
    rows = [
        _row("users", "users_email_key", _EMAIL_DEF),
        _row("users", "users_username_key", _USERNAME_DEF),
        _row("accounts", "accounts_email_key", _EMAIL_DEF),
    ]
    assert gsb._duplicate_groups(rows) == []


# --------------------------------------------------------------- _dedupe_unique_indexes
class _FakeConn:
    def __init__(self, log: list[str]):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt):
        self._log.append(str(stmt))
        return self


class _FakeEngine:
    """Records every DDL statement ``_dedupe_unique_indexes`` issues."""

    def __init__(self, log: list[str]):
        self._log = log

    def begin(self):
        return _FakeConn(self._log)


def test_dedupe_drops_the_constraint_backed_duplicate_not_the_canonical(monkeypatch):
    rows = [
        _row("users", "users_email_key", _EMAIL_DEF, "u", "users_email_key"),
        _row("users", "users_email_key1", _EMAIL_DEF, "u", "users_email_key1"),
    ]
    monkeypatch.setattr(gsb, "_unique_index_rows", lambda engine: rows)
    log: list[str] = []
    dropped = gsb._dedupe_unique_indexes(_FakeEngine(log))
    assert dropped == ["users.users_email_key1"]
    assert any('DROP CONSTRAINT "users_email_key1"' in s for s in log)
    assert not any("users_email_key\"" in s for s in log)  # the canonical is never touched


def test_dedupe_drops_a_bare_duplicate_index_with_drop_index(monkeypatch):
    rows = [
        _row("users", "users_email_key", _EMAIL_DEF, "u", "users_email_key"),
        # sorts AFTER the canonical name above, so it's the one picked as the duplicate
        _row("users", "users_email_key_dupe", _EMAIL_DEF),  # no backing constraint
    ]
    monkeypatch.setattr(gsb, "_unique_index_rows", lambda engine: rows)
    log: list[str] = []
    dropped = gsb._dedupe_unique_indexes(_FakeEngine(log))
    assert dropped == ["users.users_email_key_dupe"]
    assert any('DROP INDEX "users_email_key_dupe"' in s for s in log)
    assert not any("DROP CONSTRAINT" in s for s in log)


def test_dedupe_never_drops_an_index_a_foreign_key_depends_on(monkeypatch):
    rows = [
        _row("users", "users_email_key", _EMAIL_DEF, "u", "users_email_key"),
        _row("users", "users_email_key1", _EMAIL_DEF, "u", "users_email_key1", fk_dependent=True),
    ]
    monkeypatch.setattr(gsb, "_unique_index_rows", lambda engine: rows)
    log: list[str] = []
    dropped = gsb._dedupe_unique_indexes(_FakeEngine(log))
    assert dropped == []
    assert log == []


def test_dedupe_is_idempotent_nothing_left_to_drop_on_a_second_pass(monkeypatch):
    monkeypatch.setattr(
        gsb, "_unique_index_rows",
        lambda engine: [_row("users", "users_email_key", _EMAIL_DEF, "u", "users_email_key")],
    )
    assert gsb._dedupe_unique_indexes(_FakeEngine([])) == []


# --------------------------------------------------------------- wired into _repair_passes
def test_repair_passes_runs_the_dedupe_stage_last(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gsb, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(gsb, "_reconcile_model_columns", lambda engine: calls.append("columns") or [])
    monkeypatch.setattr(gsb, "_replay_idempotent_raw_sql", lambda engine, script: calls.append("raw_sql") or 0)
    monkeypatch.setattr(gsb, "_drop_relics", lambda engine, script: calls.append("relics") or [])
    monkeypatch.setattr(gsb, "_dedupe_unique_indexes", lambda engine: calls.append("dedupe") or [])
    gsb._repair_passes(object(), object())
    assert calls == ["init_db", "columns", "raw_sql", "relics", "dedupe"]
