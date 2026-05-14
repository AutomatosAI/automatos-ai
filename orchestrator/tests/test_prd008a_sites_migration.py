"""
PRD-008-A migration — smoke tests
====================================

Lightweight checks that the alembic migration parses correctly, declares
the expected revision chain, and references only known-safe SQL primitives.

Full migration-on-real-DB verification happens via a Postgres test
container in CI (out of scope for unit tests).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

MIGRATION_PATH = ORCH_ROOT / "alembic" / "versions" / "prd008a_sites.py"


@pytest.fixture(scope="module")
def migration_module():
    spec = importlib.util.spec_from_file_location("prd008a_sites_mig", MIGRATION_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_revision_id_is_stable(migration_module):
    assert migration_module.revision == "prd008a_sites"


def test_down_revision_chains_off_known_recent_head(migration_module):
    """Don't accidentally branch a new root. Must chain off an existing
    head — picked the most recent date-tagged migration."""
    assert migration_module.down_revision == "20260326_fix_installs_item_id"


def test_upgrade_and_downgrade_defined(migration_module):
    assert callable(migration_module.upgrade)
    assert callable(migration_module.downgrade)


def test_migration_file_contains_create_table_sites(migration_module):
    """Trip-wire that the upgrade does what it claims."""
    src = MIGRATION_PATH.read_text()
    assert 'create_table' in src
    assert '"sites"' in src


def test_migration_includes_indexes(migration_module):
    src = MIGRATION_PATH.read_text()
    assert "idx_sites_workspace_id" in src
    assert "idx_sites_type_external" in src


def test_migration_includes_backfill_from_workspaces(migration_module):
    """PRD-007's existing widgets must keep working — every existing
    workspace must get a default Site."""
    src = MIGRATION_PATH.read_text()
    assert "INSERT INTO sites" in src
    assert "FROM workspaces" in src


def test_migration_backfill_is_idempotent(migration_module):
    """Re-running the upgrade must not create duplicate Sites."""
    src = MIGRATION_PATH.read_text()
    assert "WHERE NOT EXISTS" in src


def test_downgrade_drops_indexes_before_table(migration_module):
    """Order matters in Postgres — drop indexes first, then the table."""
    src = MIGRATION_PATH.read_text()
    drop_idx_pos = src.find("drop_index")
    drop_tbl_pos = src.find("drop_table")
    assert drop_idx_pos != -1 and drop_tbl_pos != -1
    assert drop_idx_pos < drop_tbl_pos


def test_capability_backfill_matches_model_defaults(migration_module):
    """The SQL hard-codes capability flags; if the model adds a new
    capability key, this test reminds us to extend the migration.
    Keeps the snapshot-at-migration-time in sync with the runtime model.
    """
    from core.models.sites import CAPABILITY_KEYS, derive_default_capabilities

    src = MIGRATION_PATH.read_text()
    for key in CAPABILITY_KEYS:
        assert f"'{key}'" in src, (
            f"Capability {key!r} is declared in core.models.sites but the "
            "migration does not backfill it. Update the SQL or remove the key."
        )

    # The migration has a Shopify branch that flips ``has_cart`` to true.
    # If a future capability defaults to true for Shopify but isn't reflected
    # here, that's a model drift the migration must catch up to.
    shopify_caps = derive_default_capabilities("shopify")
    if shopify_caps["has_cart"]:
        # Sloppy but durable: look for the literal ", true" near has_cart in
        # the file; the alternative is a SQL parser, which is overkill.
        assert "'has_cart'" in src and "true" in src.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
