"""mobile_push_tickets and mobile_digest_posts — the push wave's two tables (PRD-266 §4.4, §4.5, 24 Oct 2026).

* ``mobile_push_tickets``: Expo answers every message it accepts with a ticket id, and
  only its receipt (read later, by id) says whether the phone took it. A row keeps the
  ticket until the receipts job reads it (a day at most): which device it went to (a
  ``DeviceNotRegistered`` receipt turns that device's push off), its workspace, and its
  push type (``needs_answer`` | ``review`` | ``auto_message``, a CHECK). Deleting the device
  or the workspace deletes its tickets. ``created_at`` is indexed: the job reads tickets
  old enough for their receipts and prunes the stale ones.
* ``mobile_digest_posts``: one row per workspace, person and local date on which the
  daily digest was posted. The primary key is the claim (``INSERT … ON CONFLICT DO
  NOTHING``), so two ticks, workers or replicas never post one person two digests on one
  day. Deleting the workspace or the person deletes their rows.

Why two tables and not an extended one: mobile_devices is one row per device and neither
Expo's tickets (many per device, kept a day) nor a person's daily digest (one per person
per local date, unique across workers and replicas) fits a column there; notifications
has no slot for either.

New tables only: no existing table or row changes. ``IF NOT EXISTS`` keeps the upgrade
a no-op on a database that ``create_all`` built from the models
(``core.models.mobile_push`` declares the same names), which then gains only a missing
index.

Revision ID: prd266_w2_push
Revises: prd261_revocation_feed
Create Date: 2026-10-24
"""
from alembic import op

revision = "prd266_w2_push"
down_revision = "prd261_revocation_feed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS mobile_push_tickets (
            ticket_id VARCHAR(64) PRIMARY KEY,
            device_id UUID NOT NULL REFERENCES mobile_devices(id) ON DELETE CASCADE,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            push_type VARCHAR(16) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_mobile_push_tickets_push_type CHECK (push_type IN ('needs_answer', 'review', 'auto_message'))
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_mobile_push_tickets_created_at "
        "ON mobile_push_tickets (created_at);"
    )
    op.execute("""
        CREATE TABLE IF NOT EXISTS mobile_digest_posts (
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            local_date DATE NOT NULL,
            posted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (workspace_id, user_id, local_date)
        )
    """)


def downgrade() -> None:
    # Each table's index and constraints go with it.
    op.execute("DROP TABLE IF EXISTS mobile_digest_posts;")
    op.execute("DROP TABLE IF EXISTS mobile_push_tickets;")
