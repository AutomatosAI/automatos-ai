"""PRD-222 W2·S5 (US-021) — the seeded-onboarding-agent cleanup migration.

Pure tests pin the migration's chain position and, above all, its SCOPE: the
delete predicate is exactly ``is_system_agent = true AND required_role =
'onboarding'``, so no agent created by any other path can match. An @integration
test proves the same with real rows and a surviving fixture agent (skips cleanly
without Postgres; CI runs it).
"""
from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

_MIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic" / "versions" / "prd222_w2s5_drop_onboarding_agents.py"
)
NEW_REVISION = "prd222_w2s5_drop_onboarding_agents"
PRIOR_HEAD = "prd185_s1b_toollog_user_nullable"


def _load_migration():
    spec = importlib.util.spec_from_file_location("_prd222_w2s5_mig", _MIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script_dir():
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(Path(__file__).resolve().parents[1] / "alembic"))


# --------------------------------------------------------------------------- #
# Chain position + single head
# --------------------------------------------------------------------------- #


def test_migration_chains_onto_prior_head():
    mod = _load_migration()
    assert mod.revision == NEW_REVISION
    assert mod.down_revision == PRIOR_HEAD


def test_exactly_one_head_after_this_migration():
    heads = _script_dir().get_heads()
    assert len(heads) == 1, f"expected exactly one alembic head, got {heads}"
    assert heads[0] == NEW_REVISION


# --------------------------------------------------------------------------- #
# Scope — the predicate can only match the seeded onboarding templates
# --------------------------------------------------------------------------- #


def test_predicate_requires_both_system_flag_and_onboarding_role():
    mod = _load_migration()
    pred = mod._SEEDED.lower()
    assert "is_system_agent = true" in pred
    assert "required_role = 'onboarding'" in pred


def test_no_unscoped_agent_delete():
    # Every DELETE FROM agents must carry the scoped predicate — an unscoped
    # `DELETE FROM agents` (no WHERE) would wipe the table.
    src = _MIG_PATH.read_text()
    for line in src.splitlines():
        stripped = line.strip().lower()
        if "delete from agents" in stripped and "where" not in stripped and "target_ids" not in stripped:
            # allow the module constant that builds the sub-select
            if "_seeded" in stripped or "{_seeded}" in stripped:
                continue
            raise AssertionError(f"unscoped agents delete: {line!r}")
    # The agents delete uses the seeded predicate.
    assert 'DELETE FROM agents WHERE {_SEEDED}' in src


# --------------------------------------------------------------------------- #
# @integration — real rows: seeded gone, fixtures spared (skips without a DB).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"retirement-migration integration test needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.mark.integration
def test_migration_removes_seeded_agents_spares_fixture(engine, new_session):
    from sqlalchemy import text

    mod = _load_migration()
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:i AS uuid), :n)"),
        {"i": ws, "n": "mig-test-ws"},
    )

    def mk_agent(name, *, is_system, role):
        return s.execute(
            text(
                "INSERT INTO agents (name, agent_type, workspace_id, is_system_agent, required_role) "
                "VALUES (:n, 'custom', CAST(:w AS uuid), :sys, :role) RETURNING id"
            ),
            {"n": name, "w": ws, "sys": is_system, "role": role},
        ).fetchone()[0]

    seeded = mk_agent("VOYAGER", is_system=True, role="onboarding")
    # Fixtures that MUST survive: a system agent that is NOT onboarding, and a
    # normal user agent.
    sys_other = mk_agent("Auto", is_system=True, role=None)
    regular = mk_agent("Marketing helper", is_system=False, role=None)
    # A dependent agent_skills row on the seeded template (the FK with no cascade).
    # skills requires a NOT NULL skill_type and has no bare short-name column;
    # insert only real columns so the fixture builds under CI Postgres.
    skill_id = s.execute(
        text("INSERT INTO skills (name, skill_type) VALUES ('t', 'technical') RETURNING id"),
    ).fetchone()[0]
    s.execute(
        text("INSERT INTO agent_skills (agent_id, skill_id) VALUES (:a, :s)"),
        {"a": seeded, "s": skill_id},
    )
    s.commit()

    # Run the migration's EXACT scoped SQL (its own constants).
    s.execute(text(f"DELETE FROM agent_skills WHERE agent_id IN ({mod._TARGET_IDS})"))
    s.execute(text(f"DELETE FROM workflow_agents WHERE agent_id IN ({mod._TARGET_IDS})"))
    s.execute(text(f"DELETE FROM workflow_executions WHERE agent_id IN ({mod._TARGET_IDS})"))
    s.execute(text(f"DELETE FROM agents WHERE {mod._SEEDED}"))
    s.commit()

    def exists(aid):
        return s.execute(text("SELECT 1 FROM agents WHERE id=:i"), {"i": aid}).fetchone() is not None

    assert not exists(seeded), "seeded onboarding template should be removed"
    assert exists(sys_other), "a non-onboarding system agent must survive"
    assert exists(regular), "a normal user agent must survive"
    # its dependent skill row went too (FK-safe removal)
    assert s.execute(
        text("SELECT 1 FROM agent_skills WHERE agent_id=:a"), {"a": seeded}
    ).fetchone() is None

    # Idempotent: a second run is a clean no-op.
    s.execute(text(f"DELETE FROM agents WHERE {mod._SEEDED}"))
    s.commit()
