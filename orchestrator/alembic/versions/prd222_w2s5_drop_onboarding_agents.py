"""PRD-222 W2·S5 (US-021) — remove the retired seeded onboarding agents.

The four Mission Zero onboarding template agents (VOYAGER, BLUEPRINT, SCRIBE,
FORGE) were global (``workspace_id IS NULL``) system agents seeded with
``is_system_agent = true`` and ``required_role = 'onboarding'`` — a role no real
user ever holds, so they were hidden from every workspace roster. The seed, the
ephemeral-clone/cleanup machinery, the ``source='mission_zero'`` special-casing,
and the hierarchy-allowlist entries are all deleted in this wave (D1), so these
template rows are now dead. This migration removes them from live databases.

Safety contract:
  * **Scoped** — the WHERE clause is exactly ``is_system_agent = true AND
    required_role = 'onboarding'``. No agent created by any other path matches
    (no real agent carries ``required_role = 'onboarding'``), so a normal or
    fixture agent is provably untouched.
  * **Idempotent / guarded** — every statement is a set-based DELETE with that
    predicate; on a database where the rows are already absent (fresh clones, CI,
    a second run) each affects zero rows and the migration is a no-op.
  * **FK-safe** — the seed's ONLY dependent rows are in ``agent_skills`` (no
    ``ON DELETE`` on its ``agent_id`` FK), so those are cleared first. The two
    other no-cascade agent references (``workflow_agents`` /
    ``workflow_executions``) are cleared defensively though the templates never
    ran a workflow; ``cloned_from_id`` / ``reports_to_id`` are ``ON DELETE SET
    NULL`` and need no handling.

Downgrade is a no-op: the seed is deleted, so the rows cannot be recreated.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "prd222_w2s5_drop_onboarding_agents"
down_revision = "prd185_s1b_toollog_user_nullable"
branch_labels = None
depends_on = None

# The single, precise predicate identifying the seeded onboarding template agents.
_SEEDED = "is_system_agent = true AND required_role = 'onboarding'"
_TARGET_IDS = f"SELECT id FROM agents WHERE {_SEEDED}"


def upgrade():
    # Clear dependent rows first (no ON DELETE on these agent_id FKs), then the
    # agents themselves. All scoped to the seeded onboarding templates.
    op.execute(f"DELETE FROM agent_skills WHERE agent_id IN ({_TARGET_IDS})")
    op.execute(f"DELETE FROM workflow_agents WHERE agent_id IN ({_TARGET_IDS})")
    op.execute(f"DELETE FROM workflow_executions WHERE agent_id IN ({_TARGET_IDS})")
    op.execute(f"DELETE FROM agents WHERE {_SEEDED}")


def downgrade():
    # Irreversible by design: the onboarding-agent seed is deleted in this wave,
    # so the removed template rows cannot be recreated. No-op.
    pass
