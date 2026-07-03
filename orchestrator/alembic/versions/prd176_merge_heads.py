"""PRD-176 F010 (Step 1): collapse the four-headed revision forest to one head.

At 45609d780 ``alembic heads`` returned four divergent heads:

    - 20260612_nl2sql_example_embedding
    - prd158_cloud_default_team
    - prd161_sla_breach
    - prd164_doc_source_type

Four heads make ``alembic upgrade head`` (singular) ambiguous and block a
deterministic from-zero replay — the OSS/fresh-clone deployability bar. This is
a **merge revision only**: it has no schema operations, it simply joins the four
lineages so ``alembic heads`` returns exactly one revision and the wait-migrate
entrypoint (F051) can run ``alembic upgrade heads`` to a single, well-defined
head.

This is Step 1 of the review §7 two-step. Step 2 (LATER, this wave) squashes the
full forest into a single from-zero baseline behind the from-zero CI replay job.
Do NOT add ``CREATE``/``ALTER`` here — a merge revision that mutates schema is
exactly what makes the later squash lossy.

Revision ID: prd176_merge_heads
Revises: 20260612_nl2sql_example_embedding, prd158_cloud_default_team, prd161_sla_breach, prd164_doc_source_type
Create Date: 2026-07-02
"""

# A pure merge point — no operations. Reversible before the Step-2 squash lands.
revision = "prd176_merge_heads"
down_revision = (
    "20260612_nl2sql_example_embedding",
    "prd158_cloud_default_team",
    "prd161_sla_breach",
    "prd164_doc_source_type",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into four heads needs no schema change."""
