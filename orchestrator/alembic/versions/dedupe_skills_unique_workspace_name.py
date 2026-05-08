"""Dedupe skills and enforce uniqueness on (workspace_id, name).

After fork-on-edit shipped (commit 7ea2ff274), workspace forks of marketplace
skills started surfacing in the marketplace tab alongside the originals. The
GET /api/v1/skills endpoint was filtering with `workspace_id IS NULL OR
workspace_id == ctx.workspace_id`, so each fork appeared as a duplicate.

This migration:
  1. For each (workspace_id, name) group with multiple rows, picks the oldest
     (lowest id) as the survivor.
  2. Re-points agent_skills.skill_id from dupes to the survivor, handling
     existing-link conflicts by deleting the redundant dupe link.
  3. Re-points workspace_enabled_skills.skill_id similarly.
  4. Sets skill_audit_logs.skill_id to NULL on dupes (its FK is already
     ondelete=SET NULL but doing it explicitly avoids relying on that).
  5. Deletes the dupe rows. CASCADE removes skill_files and skill_versions.
  6. Adds UNIQUE (workspace_id, name) so this can never recur.

Standalone migration — safe to run anytime.
"""

from alembic import op


revision = "dedupe_skills_unique_workspace_name"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()

    # 1. Identify dupes: same (workspace_id, name), keep the lowest id as survivor.
    bind.execute(
        """
        CREATE TEMPORARY TABLE skill_dedupe_map AS
        WITH ranked AS (
            SELECT
                id,
                workspace_id,
                name,
                MIN(id) OVER (PARTITION BY workspace_id, name) AS survivor_id
            FROM skills
        )
        SELECT id AS dupe_id, survivor_id
        FROM ranked
        WHERE id <> survivor_id;
        """
    )

    # 2. Re-point agent_skills. Junction has implicit unique (agent_id, skill_id);
    #    if the agent is already linked to the survivor, drop the dupe link instead.
    bind.execute(
        """
        DELETE FROM agent_skills a
        USING skill_dedupe_map m
        WHERE a.skill_id = m.dupe_id
          AND EXISTS (
              SELECT 1 FROM agent_skills b
              WHERE b.agent_id = a.agent_id
                AND b.skill_id = m.survivor_id
          );
        """
    )
    bind.execute(
        """
        UPDATE agent_skills a
        SET skill_id = m.survivor_id
        FROM skill_dedupe_map m
        WHERE a.skill_id = m.dupe_id;
        """
    )

    # 3. Re-point workspace_enabled_skills. Same conflict handling.
    bind.execute(
        """
        DELETE FROM workspace_enabled_skills w
        USING skill_dedupe_map m
        WHERE w.skill_id = m.dupe_id
          AND EXISTS (
              SELECT 1 FROM workspace_enabled_skills v
              WHERE v.workspace_id = w.workspace_id
                AND v.skill_id = m.survivor_id
          );
        """
    )
    bind.execute(
        """
        UPDATE workspace_enabled_skills w
        SET skill_id = m.survivor_id
        FROM skill_dedupe_map m
        WHERE w.skill_id = m.dupe_id;
        """
    )

    # 4. Detach audit logs from dupes (FK is already SET NULL on delete, but explicit is safer).
    bind.execute(
        """
        UPDATE skill_audit_logs
        SET skill_id = NULL
        WHERE skill_id IN (SELECT dupe_id FROM skill_dedupe_map);
        """
    )

    # 5. Delete dupes. skill_files / skill_versions / workspace_enabled_skills cascade.
    bind.execute(
        """
        DELETE FROM skills
        WHERE id IN (SELECT dupe_id FROM skill_dedupe_map);
        """
    )

    bind.execute("DROP TABLE IF EXISTS skill_dedupe_map;")

    # 6. Enforce uniqueness going forward. Postgres treats NULL workspace_id as
    #    distinct in a plain UNIQUE, which is exactly what we want — but we also
    #    need marketplace skills (workspace_id IS NULL) to be unique by name. Add
    #    two constraints: a regular UNIQUE for workspace-owned, and a partial
    #    unique index for marketplace skills.
    op.create_unique_constraint(
        "uq_skills_workspace_id_name",
        "skills",
        ["workspace_id", "name"],
    )
    bind.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_marketplace_name
        ON skills (name)
        WHERE workspace_id IS NULL;
        """
    )


def downgrade():
    bind = op.get_bind()
    bind.execute("DROP INDEX IF EXISTS uq_skills_marketplace_name;")
    op.drop_constraint("uq_skills_workspace_id_name", "skills", type_="unique")
    # Dedupe is not reversible — dupe rows are gone.
