"""PRD-232 US-007 — cluster provenance: seeded cold-start clusters survive the nightly

The synthetic-utterance seed (``seed_tool_routing_graph``) writes
``ToolRoutingIntentCluster`` rows so the tool-routing graph routes day-one, before
any telemetry accrues. The nightly ``edge_builder`` recompute deletes-and-reinserts
its clusters (``core/services/edge_builder._compute_and_upsert_clusters``); without a
provenance marker it would wipe the seeds at 03:00 UTC and the cold-start evaporates.

This adds ONE nullable ``provenance`` column so the rebuild can replace ORGANIC rows
only. This is the WAVE'S ONLY schema change for US-007 — exactly one revision, chained
onto the single head so ``alembic heads`` stays 1.

  - ``provenance`` — 'organic' (default, existing rows + nightly-built) | 'seeded'

``server_default 'organic'`` backfills every pre-migration row as organic, so the
nightly's ``WHERE provenance = 'organic'`` delete keeps behaving exactly as before on
already-built graphs. ``ADD COLUMN IF NOT EXISTS`` makes the step a no-op where it
already applied — safe on fresh clones, CI, and any partially-migrated environment.
"""

from alembic import op


revision = "prd232_cluster_provenance"
down_revision = "prd225_s1_asks_on_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE tool_routing_intent_clusters "
        "ADD COLUMN IF NOT EXISTS provenance VARCHAR(20) DEFAULT 'organic';"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE tool_routing_intent_clusters DROP COLUMN IF EXISTS provenance;"
    )
