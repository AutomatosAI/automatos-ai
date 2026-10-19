"""PRD-262 US-030: main.py's per-boot ALTERs are now the revision ``prd262_boot_columns``.

Every boot of every worker ran ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` for
``system_prompts.futureagi_eval_enabled`` and ``skills.content_hash``, each taking an
ACCESS EXCLUSIVE lock on its table. Pins:

* **One revision.** It chains onto the head it found (``prd258_voice_timings``) and is
  the only revision chained there; the head pins (test_prd209_alembic_single_head,
  test_prd236_w1_routes) follow it.
* **main.py runs no ALTER.** The block is gone from the lifespan, and ``text``, which
  the block used to import, still resolves for the SQL left in that function.
* **The models declare both columns**, so ``create_all`` and ``init_fresh_db`` build
  them, as the hand-written tables below do.
* **On real Postgres**, in a throwaway schema: the upgrade adds both columns to tables
  that lack them (the flag ``false`` on an existing row), leaves tables that already
  have them, and their values, exactly as they were, and runs twice. The downgrade
  leaves both columns.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import uuid
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from sqlalchemy import create_engine, text  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"
REVISION = VERSIONS / "prd262_boot_columns.py"
MAIN = _ORCH / "main.py"
PARENT = "prd258_voice_timings"
FLAG = ("system_prompts", "futureagi_eval_enabled")
HASH = ("skills", "content_hash")
A_HASH = "ab" * 32

# Where a database stood before the boot ALTERs ever ran: neither column.
LACKING = (
    "CREATE TABLE system_prompts (id SERIAL PRIMARY KEY, slug VARCHAR(100) NOT NULL)",
    "INSERT INTO system_prompts (slug) VALUES ('chatbot-friendly')",
    "CREATE TABLE skills (id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL)",
    "INSERT INTO skills (name) VALUES ('platform-management')",
)
# Where create_all (or init_fresh_db) leaves one: both columns, the flag with no default.
HAVING = (
    "CREATE TABLE system_prompts (id SERIAL PRIMARY KEY, slug VARCHAR(100) NOT NULL, "
    "futureagi_eval_enabled BOOLEAN NOT NULL)",
    "INSERT INTO system_prompts (slug, futureagi_eval_enabled) VALUES ('chatbot-friendly', TRUE)",
    "CREATE TABLE skills (id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL, content_hash VARCHAR(64))",
    f"INSERT INTO skills (name, content_hash) VALUES ('platform-management', '{A_HASH}')",
)
COLUMNS_SQL = (
    "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
    "FROM information_schema.columns WHERE table_schema = :schema ORDER BY table_name, column_name"
)


def _revision():
    spec = importlib.util.spec_from_file_location("prd262_boot_columns_test", REVISION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boot_phase_1_core() -> ast.AsyncFunctionDef:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    return next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_boot_phase_1_core")


# --------------------------------------------------------------------------- static


def test_the_one_revision_chains_onto_the_head_it_found():
    mod = _revision()
    assert (mod.revision, mod.down_revision) == ("prd262_boot_columns", PARENT)
    chained = sorted(
        path.name for path in VERSIONS.glob("*.py")
        if re.search(rf"^down_revision\s*=\s*['\"]{PARENT}['\"]", path.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["prd262_boot_columns.py"]


def test_the_revision_adds_the_columns_main_py_added_and_only_if_missing():
    mod = _revision()
    assert mod.BOOT_COLUMNS == (
        ("system_prompts", "futureagi_eval_enabled", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("skills", "content_hash", "VARCHAR(64)"),
    )
    assert "ADD COLUMN IF NOT EXISTS" in REVISION.read_text(encoding="utf-8")


def test_main_py_runs_no_alter_on_boot():
    source = MAIN.read_text(encoding="utf-8")
    assert "ADD COLUMN" not in source and "futureagi_eval_enabled" not in source
    lifespan = ast.get_source_segment(source, _boot_phase_1_core())
    assert "ALTER TABLE" not in lifespan


def test_the_lifespan_still_resolves_text_for_the_sql_left_in_it():
    """The deleted block imported ``text`` locally, and the LinkedIn update below it used it."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    function = _boot_phase_1_core()
    used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    local = {alias.asname or alias.name for node in ast.walk(function)
             if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    module = {alias.asname or alias.name for node in tree.body
              if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert "text" not in used or "text" in local | module


def test_the_models_declare_both_columns():
    from core.models.core import Skill
    from core.models.system_prompts import SystemPrompt

    flag = SystemPrompt.__table__.c.futureagi_eval_enabled
    assert flag.type.python_type is bool and flag.nullable is False
    content_hash = Skill.__table__.c.content_hash
    assert content_hash.type.length == 64 and content_hash.nullable is True


# --------------------------------------------------------------------------- real Postgres


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the boot-columns migration tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def schema(engine):
    """A schema of its own, so the shared test database's tables are never touched."""
    name = f"prd262_us030_{uuid.uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{name}"'))
    yield name
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


def _in_schema(engine, schema: str, statements=(), steps=()) -> None:
    """Run SQL, then the revision's steps through a real alembic context, inside ``schema``."""
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    mod = _revision()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        for statement in statements:
            conn.execute(text(statement))
        with Operations.context(MigrationContext.configure(conn)):
            for step in steps:
                getattr(mod, step)()


def _columns(engine, schema: str) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(text(COLUMNS_SQL), {"schema": schema}).all()
    return {(row.table_name, row.column_name): tuple(row)[2:] for row in rows}


def _values(engine, schema: str):
    with engine.connect() as conn:
        flag = conn.execute(text(f'SELECT futureagi_eval_enabled FROM "{schema}".system_prompts')).scalar_one()
        content_hash = conn.execute(text(f'SELECT content_hash FROM "{schema}".skills')).scalar_one()
    return flag, content_hash


def test_the_upgrade_adds_both_columns_where_they_are_missing(engine, schema):
    _in_schema(engine, schema, LACKING)
    before = _columns(engine, schema)
    assert FLAG not in before and HASH not in before
    _in_schema(engine, schema, steps=("upgrade", "upgrade"))
    after = _columns(engine, schema)
    assert after[FLAG][0] == "boolean" and after[FLAG][2] == "NO"
    assert after[HASH][:3] == ("character varying", 64, "YES")
    assert _values(engine, schema) == (False, None), "the existing row reads 'not evaluated', no hash"


def test_the_upgrade_leaves_columns_create_all_built_as_they_were(engine, schema):
    _in_schema(engine, schema, HAVING)
    before = _columns(engine, schema)
    _in_schema(engine, schema, steps=("upgrade", "upgrade"))
    assert _columns(engine, schema) == before
    assert before[FLAG][3] is None, "create_all's flag has no server default, and the upgrade adds none"
    assert _values(engine, schema) == (True, A_HASH)


def test_the_downgrade_leaves_both_columns(engine, schema):
    _in_schema(engine, schema, LACKING, steps=("upgrade",))
    upgraded = _columns(engine, schema)
    _in_schema(engine, schema, steps=("downgrade",))
    assert _columns(engine, schema) == upgraded
