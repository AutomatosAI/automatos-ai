"""The job queue's table moves into its own schema (12 Nov 2026).

PRD-276 §10-7 and §12.2: the queue's write churn (claims, renewals, retries, pruning) sits in its
own schema, ``jobs``, apart from the business tables in the same database. One database, so a job
is still queued in the transaction of the write that caused it; a separate jobs database comes
only with the outbox relay (§12.4). Every query names the table ``jobs.jobs``.

* creates the ``jobs`` schema;
* moves ``public.jobs`` into it with ``ALTER TABLE ... SET SCHEMA``, which carries its rows,
  indexes and foreign key and rewrites nothing;
* create_all-first safe: on a database that ``create_all`` built with the model's schema already
  there, ``jobs_20261111`` will have made an empty ``public.jobs`` as well; any rows it holds are
  copied across (by id, never twice) and it is dropped.

The downgrade moves the table back to ``public`` and drops the schema.

Revision ID: jobs_schema_20261112
Revises: jobs_20261111
Create Date: 2026-11-12
"""
from alembic import op

revision = "jobs_schema_20261112"
down_revision = "jobs_20261111"
branch_labels = None
depends_on = None

_COLUMNS = ("id, kind, workspace_id, payload, status, run_at, attempts, max_attempts, lease_until, claimed_by, "
            "claim_id, idempotency_key, last_error, created_at, updated_at, finished_at")

_MOVE = f"""
DO $$
BEGIN
    IF to_regclass('public.jobs') IS NOT NULL AND to_regclass('jobs.jobs') IS NULL THEN
        ALTER TABLE public.jobs SET SCHEMA jobs;
    ELSIF to_regclass('public.jobs') IS NOT NULL THEN
        INSERT INTO jobs.jobs ({_COLUMNS}) SELECT {_COLUMNS} FROM public.jobs ON CONFLICT DO NOTHING;
        DROP TABLE public.jobs;
    END IF;
END $$;
"""

_BACK = """
DO $$
BEGIN
    IF to_regclass('jobs.jobs') IS NOT NULL AND to_regclass('public.jobs') IS NULL THEN
        ALTER TABLE jobs.jobs SET SCHEMA public;
    END IF;
END $$;
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS jobs")
    op.execute(_MOVE)


def downgrade() -> None:
    op.execute(_BACK)
    op.execute("DROP SCHEMA IF EXISTS jobs")
