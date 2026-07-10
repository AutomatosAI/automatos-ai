"""PRD-191 S1+S4 — agent_skills: dedupe links, UNIQUE(agent_id, skill_id), real priority

The junction table has been a bare two-column Table since inception — no PK,
no unique — and the concurrency-unsafe seeders running on hot request paths
(hybrid/chat/workspaces, multiple workers) duplicated links freely: prod's
workspace Auto agent is linked to platform-management 4×, and SkillsSection
renders every link, so the same ~26.5 KB body taxes every Auto turn twice
(~5k duplicated tokens). The earlier dedupe migration
(dedupe_skills_unique_workspace_name.py:52) claims an "implicit unique
(agent_id, skill_id)" that never existed — its link-dedupe had nothing
backing it.

Order is load-bearing: dedupe FIRST (the live 4× links would fail the
constraint), then constrain. The defensive re-dedupe is safe even where the
earlier best-effort dedupe already ran (Gerard's-call box, option a).

Also adds the per-attachment ``priority`` column (S4, closes F054): the
uncapped-primary slot becomes a decision — mirroring AgentAssignedPlugin's
real priority — instead of relationship-load order sorted by a phantom
attribute.

Revision ID: prd191_agent_skills_unique_and_priority
Revises: prd187_s5_drop_memory_relics  (stacked — merge PRD-187's #525 first)
Create Date: 2026-07-10
"""
import sqlalchemy as sa
from alembic import op

revision = "prd191_agent_skills_unique_and_priority"
down_revision = "prd187_s5_drop_memory_relics"
branch_labels = None
depends_on = None


def survivors(rows):
    """PURE dedupe policy: given (row_id, agent_id, skill_id) tuples, return
    the row_ids to KEEP — exactly one per (agent_id, skill_id) pair, the
    lowest row_id winning. (The SQL below is this function's ctid mirror.)"""
    keep = {}
    for row_id, agent_id, skill_id in sorted(rows):
        keep.setdefault((agent_id, skill_id), row_id)
    return sorted(keep.values())


def upgrade() -> None:
    # 1. Collapse duplicate links — lowest physical row survives (same
    #    conflict-safe shape as dedupe_skills_unique_workspace_name's dedupe).
    op.execute(
        """
        DELETE FROM agent_skills a
        USING agent_skills b
        WHERE a.agent_id = b.agent_id
          AND a.skill_id = b.skill_id
          AND a.ctid > b.ctid
        """
    )
    # 2. The attachment-level priority (S4) — default 0, NOT NULL.
    op.add_column(
        "agent_skills",
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
    )
    # 3. Now the identity guarantee the seeders' ON CONFLICT targets (S1).
    op.create_unique_constraint(
        "uq_agent_skills_agent_id_skill_id", "agent_skills", ["agent_id", "skill_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_agent_skills_agent_id_skill_id", "agent_skills", type_="unique")
    op.drop_column("agent_skills", "priority")
    # The deleted duplicate links are not recreated — they were corruption.
