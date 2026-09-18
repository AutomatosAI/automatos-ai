"""The multimodal knowledge tables exist on BOTH schema paths, and stay identical.

PRD-209 moved fresh databases to create_all + raw DDL (``scripts/init_test_db``) and
stamps the alembic head without running it; existing databases upgrade through
alembic at boot. A table that lives on one path and not the other is exactly how
``kb_tables`` went missing for three weeks — so the two copies are held together
here, and the API that reads them is checked against the table set the fresh path
actually builds.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "alembic" / "versions" / "kb_multimodal_tables.py"
FRESH = ROOT / "scripts" / "init_test_db.py"
API = ROOT / "api" / "knowledge_multimodal.py"

MULTIMODAL_TABLES = ("kb_tables", "kb_formulas", "kb_images")
KB_TYPE_NAMES = ("document", "codegraph", "table", "image", "formula", "diagram", "knowledge_graph", "memory", "entity")


def _load_migration():
    spec = importlib.util.spec_from_file_location("kb_multimodal_tables", MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _create_statements(src: str) -> dict[str, str]:
    """table name → the body of its CREATE TABLE IF NOT EXISTS (whitespace-normalised)."""
    out = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (kb_\w+) \((.*?)\n\s*\)", src, re.S):
        out[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).strip()
    return out


def test_migration_and_fresh_path_declare_the_same_tables():
    mig, fresh = _create_statements(MIGRATION.read_text()), _create_statements(FRESH.read_text())
    for t in MULTIMODAL_TABLES:
        assert t in mig, f"migration lacks {t}"
        assert t in fresh, f"fresh path lacks {t}"
        assert mig[t] == fresh[t], f"{t} drifted between the migration and the fresh path"


def test_every_statement_is_idempotent_so_production_is_a_noop():
    mod = _load_migration()
    for stmt in (mod.KB_TABLES, mod.KB_FORMULAS, mod.KB_IMAGES):
        assert stmt.lstrip().startswith("CREATE TABLE IF NOT EXISTS")
    for stmt in mod.INDEXES + [mod.VISUAL_EMBEDDING_INDEX]:
        assert stmt.startswith("CREATE INDEX IF NOT EXISTS")
    assert "ON CONFLICT (type_name) DO NOTHING" in mod.KB_TYPES_SEED
    assert mod.down_revision == "tool_execution_logs_workspace_user_idx"


def test_kb_types_seed_carries_the_nine_reference_rows_on_both_paths():
    for src in (MIGRATION.read_text(), FRESH.read_text()):
        for name in KB_TYPE_NAMES:
            assert f"('{name}'," in src, f"kb_types seed lacks {name}"


def test_the_multimodal_api_only_reads_tables_the_fresh_path_builds():
    built = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", FRESH.read_text()))
    referenced = set(re.findall(r"\b(kb_(?:types|tables|formulas|images)|knowledge_items)\b", API.read_text()))
    assert referenced, "the API stopped referencing the multimodal tables"
    assert referenced <= built, f"the Multimodal API reads tables a fresh install never builds: {sorted(referenced - built)}"
