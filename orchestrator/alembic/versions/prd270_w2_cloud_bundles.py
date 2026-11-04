"""PRD-270 W2: which local Marketplace rows a Sync with Automatos Cloud wrote (31 Oct 2026).

The ONE migration of wave W2 (Cloud bundles); every later story in the wave adds none.

* ``marketplace_cloud_items``: one row per Automatos Cloud item a local Sync wrote, pointing
  at exactly one local row of its type (a marketplace agent, a Playbook, a package or a
  plugin), with the item's Cloud identity (``external_id``, a hash of its lineage), Automatos
  Cloud's version, the bundle's revision (the sha256 of its canonical bytes) and the refs that
  did not resolve locally (``unresolved``). A local row is a Cloud row exactly when a mapping
  points at it, and nothing but the Sync writes one. Two CHECKs keep ``item_type`` to the four
  types and require exactly one local key, the one of its type; one unique constraint keeps one
  mapping per (type, external id), and one partial unique index per local key keeps one
  mapping per local row.
* A table, not W1's ``source`` column on each row: ``agents`` and ``workflow_recipes`` are the
  busiest tables (a CHECK added there scans them), packages and plugins have no ``owner_type``
  for W1's CHECK, and the facts a Cloud row needs are the same for all four types.
* Every foreign key is ON DELETE CASCADE: an operator who deletes a Cloud row locally deletes
  its mapping, and the next Sync lists the item again as new.

``core/models/marketplace_cloud.py`` (``MarketplaceCloudItem``) declares the same shape;
``tests/test_prd270w2_migration.py`` holds the two together.

Create_all-first safe: the table is ``CREATE TABLE IF NOT EXISTS`` and each index ``CREATE
UNIQUE INDEX IF NOT EXISTS``, so a database ``create_all`` built is left as it is, and running
it twice changes nothing. It alters no table and changes no row.

The downgrade drops the table only. It deletes no ``agents``, ``workflow_recipes``,
``marketplace_packages`` or ``marketplace_plugins`` row: a row a Sync wrote stays, as a row of
this install (as W1's downgrade keeps its Cloud themes).

Revision ID: prd270_w2_cloud_bundles
Revises: prd270_w1_cloud_sync
Create Date: 2026-10-31
"""

from alembic import op


revision = "prd270_w2_cloud_bundles"
down_revision = "prd270_w1_cloud_sync"
branch_labels = None
depends_on = None

# (table, CREATE TABLE), as in prd261_revocation_feed.
_TABLES = (
    ("marketplace_cloud_items", """
CREATE TABLE IF NOT EXISTS marketplace_cloud_items (
    id SERIAL PRIMARY KEY,
    item_type VARCHAR(16) NOT NULL,
    external_id VARCHAR(32) NOT NULL,
    version VARCHAR(32) NOT NULL,
    revision VARCHAR(64) NOT NULL,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    recipe_id INTEGER REFERENCES workflow_recipes(id) ON DELETE CASCADE,
    package_id UUID REFERENCES marketplace_packages(id) ON DELETE CASCADE,
    plugin_id UUID REFERENCES marketplace_plugins(id) ON DELETE CASCADE,
    unresolved JSONB NOT NULL DEFAULT '[]'::jsonb,
    synced_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_marketplace_cloud_items_type CHECK (item_type IN ('agent', 'playbook', 'package', 'plugin')),
    CONSTRAINT ck_marketplace_cloud_items_one_row CHECK (num_nonnulls(agent_id, recipe_id, package_id, plugin_id) = 1 AND (agent_id IS NULL OR item_type = 'agent') AND (recipe_id IS NULL OR item_type = 'playbook') AND (package_id IS NULL OR item_type = 'package') AND (plugin_id IS NULL OR item_type = 'plugin')),
    CONSTRAINT uq_marketplace_cloud_items_external UNIQUE (item_type, external_id)
);
"""),
)

# One mapping per local row, as core/models/marketplace_cloud.py declares them.
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_cloud_items_agent ON marketplace_cloud_items "
    "(agent_id) WHERE agent_id IS NOT NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_cloud_items_recipe ON marketplace_cloud_items "
    "(recipe_id) WHERE recipe_id IS NOT NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_cloud_items_package ON marketplace_cloud_items "
    "(package_id) WHERE package_id IS NOT NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_cloud_items_plugin ON marketplace_cloud_items "
    "(plugin_id) WHERE plugin_id IS NOT NULL;",
)

# The table only (its indexes go with it). No local row is deleted.
_DOWNGRADE = (
    "DROP TABLE IF EXISTS marketplace_cloud_items;",
)


def upgrade() -> None:
    for _table, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
