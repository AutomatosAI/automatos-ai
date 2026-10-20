"""clerk_user_relinks — the audit record of the Clerk production cutover (19 Oct 2026).

The Clerk production cutover moves each returning ``users`` row from its
development-instance Clerk ID to its production ID once, in an operator step
(``scripts/clerk_cutover_relink.py``), never at sign-in. Each move writes one row
here in the same transaction: the old and new Clerk IDs, the verified email that
matched, and when it was applied or rolled back. ``user_id`` is unique (a row is
relinked at most once) and so is ``new_clerk_user_id`` (one production identity per
row). The names match ``core.models.clerk_user_relinks.ClerkUserRelink``.

A new table only: no existing table or row changes. ``IF NOT EXISTS`` keeps the
upgrade a no-op on a database that ``create_all`` built from the models.

Revision ID: clerk_user_relinks_20261019
Revises: prd263_harness_ledger_grants
Create Date: 2026-10-19
"""
from alembic import op

revision = "clerk_user_relinks_20261019"
down_revision = "prd263_harness_ledger_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS clerk_user_relinks (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            old_clerk_user_id VARCHAR(255) NOT NULL,
            new_clerk_user_id VARCHAR(255) NOT NULL,
            verified_email VARCHAR(255) NOT NULL,
            relinked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            reverted_at TIMESTAMPTZ,
            CONSTRAINT uq_clerk_user_relinks_user UNIQUE (user_id),
            CONSTRAINT uq_clerk_user_relinks_new_id UNIQUE (new_clerk_user_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS clerk_user_relinks;")
