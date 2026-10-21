"""PRD-265 W1 — the credit ledger, balances, llm_usage.ask_id, and trials as credits.

- ``credit_ledger`` (append-only, one row per movement per bucket, unique per
  source event and bucket) and ``credit_balances`` (one row per workspace). The
  names match ``core.models.credits``. ``IF NOT EXISTS`` keeps both a no-op on a
  database that ``create_all`` built from the models.
- ``llm_usage.ask_id`` (the ask a call was part of) and ``llm_usage.charge_class``
  (``customer`` until the meter charges it, then ``charged``; ``overhead`` is
  absorbed). Nullable columns added without a default (no table rewrite). Their
  partial indexes are built CONCURRENTLY, outside the migration's transaction, as
  ``llm_usage_execution_index`` does, so a large llm_usage is not locked; the
  meter's index holds only the calls still waiting to be charged.
- ``credit_balances.alert_level``: the last low/run-out alert sent.
- Existing trials become ledger grants (US-005): the dollars left on
  ``workspaces.onboarding.trial`` × 100, floored, as a ``trial`` grant keyed on the
  owner (``source_type = 'trial_grant'``), so a person who already held a trial
  never gets a second one. A workspace with no owner is keyed on itself. When one
  owner has trials on two workspaces, the older workspace keeps its grant (ON
  CONFLICT DO NOTHING). ``onboarding.trial`` itself is left as it was; nothing
  reads it any more. Balances are then recomputed from the ledger, which makes a
  re-run harmless.

Revision ID: prd265_credit_ledger
Revises: clerk_user_relinks_20261019
Create Date: 2026-10-20
"""
import sqlalchemy as sa
from alembic import op

revision = "prd265_credit_ledger"
down_revision = "clerk_user_relinks_20261019"
branch_labels = None
depends_on = None

ASK_INDEX = "ix_llm_usage_ask_id"
METER_INDEX = "ix_llm_usage_to_meter"


def _create_tables() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS credit_ledger (
            id BIGSERIAL PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            kind VARCHAR(16) NOT NULL,
            bucket VARCHAR(16) NOT NULL,
            credits INTEGER NOT NULL,
            expires_at TIMESTAMPTZ,
            source_type VARCHAR(32) NOT NULL,
            source_ref VARCHAR(255) NOT NULL,
            actor VARCHAR(255),
            reason TEXT,
            band VARCHAR(16),
            charged_credits INTEGER,
            actual_cost_usd NUMERIC(14, 6),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_credit_ledger_source UNIQUE (source_type, source_ref, bucket),
            CONSTRAINT ck_credit_ledger_kind
                CHECK (kind IN ('grant', 'topup', 'debit', 'refund', 'adjust', 'expire')),
            CONSTRAINT ck_credit_ledger_bucket
                CHECK (bucket IN ('trial', 'monthly', 'topup', 'overhead'))
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_credit_ledger_workspace_created "
        "ON credit_ledger (workspace_id, created_at)"
    )
    op.execute("""
        CREATE TABLE IF NOT EXISTS credit_balances (
            workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
            trial INTEGER NOT NULL DEFAULT 0,
            monthly INTEGER NOT NULL DEFAULT 0,
            topup INTEGER NOT NULL DEFAULT 0,
            period_end TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            alert_level SMALLINT NOT NULL DEFAULT 0
        )
    """)
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS ask_id VARCHAR(64)")
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS charge_class VARCHAR(16)")


# The dollars left on each workspace's trial, as a once-per-person credit grant.
CONVERT_TRIALS_SQL = """
    INSERT INTO credit_ledger
        (workspace_id, kind, bucket, credits, source_type, source_ref, actor, reason)
    SELECT w.id, 'grant', 'trial',
           FLOOR(GREATEST(
               COALESCE((w.onboarding -> 'trial' ->> 'granted_usd')::numeric, 0)
               - COALESCE((w.onboarding -> 'trial' ->> 'spent_usd')::numeric, 0), 0) * 100)::int,
           'trial_grant',
           COALESCE(w.owner_id::text, 'workspace:' || w.id::text),
           'system',
           'PRD-265: the dollar trial left on this workspace, as credits'
    FROM workspaces w
    WHERE w.onboarding -> 'trial' IS NOT NULL
      AND jsonb_typeof(w.onboarding -> 'trial') = 'object'
    ORDER BY w.created_at
    ON CONFLICT ON CONSTRAINT uq_credit_ledger_source DO NOTHING
"""

# Every balance, recomputed from the ledger (so running it twice changes nothing).
REBUILD_BALANCES_SQL = """
    INSERT INTO credit_balances (workspace_id, trial, monthly, topup)
    SELECT workspace_id,
           COALESCE(SUM(credits) FILTER (WHERE bucket = 'trial'), 0),
           COALESCE(SUM(credits) FILTER (WHERE bucket = 'monthly'), 0),
           COALESCE(SUM(credits) FILTER (WHERE bucket = 'topup'), 0)
    FROM credit_ledger
    GROUP BY workspace_id
    ON CONFLICT (workspace_id) DO UPDATE
        SET trial = EXCLUDED.trial, monthly = EXCLUDED.monthly,
            topup = EXCLUDED.topup, updated_at = now()
"""


def _concurrent_index(name: str, definition: str) -> None:
    """Build ``name`` CONCURRENTLY; an INVALID leftover of a failed build is dropped
    first, because IF NOT EXISTS would keep it for good."""
    invalid = op.get_bind().execute(sa.text(
        "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE c.relname = :name AND NOT i.indisvalid"), {"name": name}).first()
    if invalid:
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
    op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} {definition}")


def upgrade() -> None:
    _create_tables()
    op.execute(CONVERT_TRIALS_SQL)
    op.execute(REBUILD_BALANCES_SQL)
    with op.get_context().autocommit_block():
        _concurrent_index(ASK_INDEX, "ON llm_usage (ask_id) WHERE ask_id IS NOT NULL")
        _concurrent_index(METER_INDEX, "ON llm_usage (id) WHERE charge_class = 'customer'")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {METER_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {ASK_INDEX}")
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS charge_class")
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS ask_id")
    op.execute("DROP TABLE IF EXISTS credit_balances")
    op.execute("DROP TABLE IF EXISTS credit_ledger")
