"""PRD-209 S4 — the schema-drift check bites, and the real repo is clean.

``scripts/ci/schema_drift_check.py`` is the structural net for "four writers of schema
truth": it flags a table that a migration ``ALTER``s but that no migration and no
``init_complete_schema.sql`` ever ``CREATE``s — the July failure class that crashed a
from-base migration replay with ``relation "..." does not exist`` and that no lane caught.

These guards are pure (import the check module, feed it fixture DDL / fixture migration
dirs, and read the real versions dir + init SQL). No database, no ``Base.metadata`` import,
no live Alembic — the same posture as the other PRD-209 guards, so they run in the required
orchestrator-tests lane and on a laptop with no services.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CHECK_PATH = _REPO_ROOT / "scripts" / "ci" / "schema_drift_check.py"
_TEST_YML = _REPO_ROOT / ".github" / "workflows" / "test.yml"


def _load_check():
    spec = importlib.util.spec_from_file_location("schema_drift_check", _CHECK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec so dataclasses can resolve the module during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sdc = _load_check()


# ----------------------------------------------------------- the pure diff core (AC1)
def test_schema_drift_detects_divergent_column():
    # The named PRD-209 S4 test: a column present on one side (the model/declared schema)
    # but absent from the other (the migration head) is reported as drift. Pure — the diff
    # logic runs against fixture schemas, no live DB.
    declared = {"widgets": {"id", "name", "created_at"}}
    migration_head = {"widgets": {"id", "name"}}  # `created_at` never migrated

    report = sdc.diff_schemas(expected=declared, actual=migration_head)

    assert report.has_drift
    assert report.missing_columns == {"widgets": {"created_at"}}
    assert not report.missing_tables and not report.extra_tables


def test_diff_schemas_reports_table_and_direction():
    a = {"kept": {"id"}, "only_expected": {"id"}}
    b = {"kept": {"id", "extra_col"}, "only_actual": {"id"}}

    report = sdc.diff_schemas(expected=a, actual=b)

    assert report.missing_tables == {"only_expected"}
    assert report.extra_tables == {"only_actual"}
    assert report.extra_columns == {"kept": {"extra_col"}}


def test_diff_schemas_identical_has_no_drift():
    schema = {"t": {"a", "b"}}
    assert not sdc.diff_schemas(expected=schema, actual=dict(schema)).has_drift


# ---------------------------------------------- the check bites on planted drift (AC3)
def _write(tmp_path: pathlib.Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(body, encoding="utf-8")


def test_orphan_alter_check_bites_on_altered_but_never_created(tmp_path):
    # A migration adds a column to `ghost_table`, which no migration and no init SQL
    # ever creates — the exact ALTER-ed-but-never-CREATE-d class. The check must flag it.
    versions = tmp_path / "versions"
    versions.mkdir()
    _write(
        versions,
        "0001_add_col_to_ghost.py",
        "def upgrade():\n"
        "    op.add_column('ghost_table', sa.Column('flag', sa.Boolean()))\n"
        "def downgrade():\n"
        "    op.drop_column('ghost_table', 'flag')\n",
    )
    _write(
        versions,
        "0002_real_table.py",
        "def upgrade():\n"
        "    op.create_table('real_table', sa.Column('id', sa.Integer()))\n"
        "    op.add_column('real_table', sa.Column('name', sa.String()))\n",
    )
    empty_init = tmp_path / "init.sql"
    empty_init.write_text("-- no tables\n", encoding="utf-8")

    orphans = sdc.orphan_alter_tables(versions, empty_init)

    assert "ghost_table" in orphans, "an ALTER-ed-but-never-CREATE-d table must be flagged"
    assert "real_table" not in orphans, "a table created by a migration is not an orphan"


def test_orphan_check_counts_init_sql_as_a_create_writer(tmp_path):
    # A table CREATE-d only in init_complete_schema.sql (not by any migration) but
    # ALTER-ed by a migration is NOT an orphan — init SQL is a writer the fresh clone sees.
    versions = tmp_path / "versions"
    versions.mkdir()
    _write(
        versions,
        "0001_index_seeded.py",
        "def upgrade():\n"
        "    op.create_index('ix_seeded_name', 'seeded_table', ['name'])\n",
    )
    init = tmp_path / "init.sql"
    init.write_text("CREATE TABLE IF NOT EXISTS seeded_table (id serial PRIMARY KEY);\n", encoding="utf-8")

    assert "seeded_table" not in sdc.orphan_alter_tables(versions, init)


def test_orphan_check_reads_foreign_key_and_index_target_table(tmp_path):
    # create_index / create_foreign_key take the table as the SECOND arg — the first
    # (index / constraint name, possibly an op.f(...) call) must not be mistaken for it.
    versions = tmp_path / "versions"
    versions.mkdir()
    _write(
        versions,
        "0001_fk_to_ghost.py",
        "def upgrade():\n"
        "    op.create_foreign_key(op.f('fk_x'), 'fk_source_ghost', 'other', ['a'], ['id'])\n",
    )
    empty_init = tmp_path / "init.sql"
    empty_init.write_text("\n", encoding="utf-8")

    assert "fk_source_ghost" in sdc.orphan_alter_tables(versions, empty_init)
    assert "fk_x" not in sdc.orphan_alter_tables(versions, empty_init)


# ---------------------------------------------- the real repo is clean on this branch (AC2)
def test_real_repo_has_no_unbaselined_orphans():
    # Green on this branch: every orphan-alter table today is in the documented baseline.
    # This is what the CI schema-drift lane asserts; running it here means the required
    # orchestrator-tests lane (and a local run) also redden on a NEW orphan.
    new = sdc.new_orphans(sdc._VERSIONS_DIR, sdc._INIT_SQL)
    assert not new, (
        f"{len(new)} new ALTER-ed-but-never-CREATE-d table(s): {sorted(new)}. Add each to a "
        "migration's create_table or to init_complete_schema.sql (or, if genuinely "
        "create_all-only and accepted, to ORPHAN_ALTER_BASELINE with a reason)."
    )


def test_baseline_has_no_stale_entries():
    # Every baseline entry must still be a real orphan. If a writer now CREATEs one, the
    # entry is stale and must be pruned so the baseline can never quietly over-accept.
    stale = sdc.stale_baseline_entries(sdc._VERSIONS_DIR, sdc._INIT_SQL)
    assert not stale, (
        f"stale ORPHAN_ALTER_BASELINE entries (a writer now CREATEs them — prune): {sorted(stale)}"
    )


def test_baseline_entries_carry_a_reason():
    for table, reason in sdc.ORPHAN_ALTER_BASELINE.items():
        assert isinstance(reason, str) and reason.strip(), f"baseline entry {table!r} needs a reason"


def test_check_main_returns_zero_on_this_branch():
    assert sdc.main() == 0


def test_check_main_returns_nonzero_on_a_planted_orphan(tmp_path, monkeypatch):
    # End-to-end red path: point the check at a fixture forest whose only migration
    # ALTERs a never-created table, and assert main() exits non-zero (reddens the lane).
    versions = tmp_path / "versions"
    versions.mkdir()
    _write(
        versions,
        "0001_orphan.py",
        "def upgrade():\n    op.add_column('brand_new_ghost', sa.Column('x', sa.Integer()))\n",
    )
    init = tmp_path / "init.sql"
    init.write_text("-- empty\n", encoding="utf-8")
    monkeypatch.setattr(sdc, "_VERSIONS_DIR", versions)
    monkeypatch.setattr(sdc, "_INIT_SQL", init)

    assert sdc.main() == 1


# ---------------------------------------------- the CI lane exists and is hard (AC2)
def test_schema_drift_lane_exists_runs_the_script_and_is_hard():
    data = yaml.safe_load(_TEST_YML.read_text(encoding="utf-8"))
    job = data["jobs"].get("schema-drift")
    assert job is not None, "schema-drift lane disappeared from test.yml"

    run_steps = [s.get("run", "") for s in job.get("steps", [])]
    assert any("schema_drift_check.py" in r for r in run_steps), (
        "schema-drift lane must run scripts/ci/schema_drift_check.py"
    )

    masked = [s.get("name", "<unnamed>") for s in job.get("steps", []) if s.get("continue-on-error") is True]
    assert not masked, f"schema-drift lane must not mask drift with continue-on-error: {masked}"

    # Non-vacuity: we added a non-required lane; the required lane is untouched.
    assert "orchestrator-tests" in data["jobs"], "required orchestrator-tests job must remain"
