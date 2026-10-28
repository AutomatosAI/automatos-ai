"""proactive_suggestions, proactive_runs and proactive_person_settings — Proactive Auto's records (PRD-268 §6.2.4, §6.9, 28 Oct 2026).

The ONE migration of wave W1 (Check-ins); every later story in the wave adds none.

* ``proactive_suggestions``: one row per thing a check-in found for one person (the
  addressee, §12-Q2): what Auto says (``title``, ``detail``), the reads it cites
  (``evidence``), what Start does (``proposed_action``), its dedupe key
  (``subject_key``), its streak count (``occurrences``, ``last_seen_at``) and the
  person's decision (``status``, ``decided_via``, ``hide_reason``, ``decided_at``,
  ``started_ref``). At most one ``open`` row per workspace, person and subject: a
  partial unique index, so two check-ins racing on one subject cannot both file it,
  while a hidden row and a new open one may share the key.
* ``proactive_runs``: the audit, one row per check-in run or refusal: its window, its
  outcome, the signals it read, how many suggestions it made, and its model, tokens
  and cost. Indexed by (workspace, agent, created_at): the streak rules and the caps
  read an agent's latest runs.
* ``proactive_person_settings``: one row per workspace and person: the pause and the
  quiet hours. The primary key (workspace_id, user_id) is the unique.

``user_id`` and the agent ids are plain integers, not foreign keys: the run audit and
a decided suggestion outlive the agent or person they name, as the audit log does.
``status`` and ``outcome`` are CHECKed against the values in
``core.models.proactive``. Every foreign key leads an index, so deleting a workspace
never scans these tables.

New tables only: no existing table or row changes. ``IF NOT EXISTS`` keeps the upgrade
a no-op on a database that ``create_all`` built from the models
(``core.models.proactive`` declares the same names), and running it twice changes
nothing. ``tests/test_prd268_proactive_models.py`` holds the two together.

Revision ID: prd268_w1_proactive
Revises: prd266_w2_push
Create Date: 2026-10-28
"""
from alembic import op

revision = "prd268_w1_proactive"
down_revision = "prd266_w2_push"
branch_labels = None
depends_on = None

# (table, CREATE TABLE), in creation order.
_TABLES = (
    ("proactive_suggestions", """
CREATE TABLE IF NOT EXISTS proactive_suggestions (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    source_type VARCHAR(16) NOT NULL,
    source_agent_id INTEGER,
    run_id UUID,
    title VARCHAR(120) NOT NULL,
    detail VARCHAR(600) NOT NULL DEFAULT '',
    evidence JSONB NOT NULL DEFAULT '[]',
    proposed_action JSONB NOT NULL,
    subject_key VARCHAR(200) NOT NULL,
    severity VARCHAR(16) NOT NULL DEFAULT 'can_wait',
    status VARCHAR(16) NOT NULL DEFAULT 'open',
    occurrences INTEGER NOT NULL DEFAULT 1,
    last_seen_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    started_ref JSONB,
    decided_via VARCHAR(16),
    hide_reason VARCHAR(32),
    decided_at TIMESTAMP WITH TIME ZONE,
    est_credits INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    delivered_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT ck_proactive_suggestions_status CHECK (status IN ('open', 'started', 'hidden', 'expired'))
);
"""),
    ("proactive_runs", """
CREATE TABLE IF NOT EXISTS proactive_runs (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_id INTEGER,
    kind VARCHAR(32) NOT NULL,
    window_since TIMESTAMP WITH TIME ZONE,
    window_until TIMESTAMP WITH TIME ZONE,
    outcome VARCHAR(32) NOT NULL,
    signals JSONB NOT NULL DEFAULT '[]',
    suggestions INTEGER NOT NULL DEFAULT 0,
    model VARCHAR(128),
    tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(14, 6) NOT NULL DEFAULT 0,
    error_class VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_proactive_runs_outcome CHECK (outcome IN ('silent', 'suggested', 'error', 'skipped_quiet', 'skipped_cap', 'skipped_credits', 'skipped_paused', 'skipped_off'))
);
"""),
    ("proactive_person_settings", """
CREATE TABLE IF NOT EXISTS proactive_person_settings (
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    paused BOOLEAN NOT NULL DEFAULT false,
    quiet_start TIME WITHOUT TIME ZONE,
    quiet_end TIME WITHOUT TIME ZONE,
    timezone VARCHAR(64),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, user_id)
);
"""),
)

_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_proactive_suggestions_open_subject "
    "ON proactive_suggestions (workspace_id, user_id, subject_key) WHERE status = 'open';",
    "CREATE INDEX IF NOT EXISTS ix_proactive_suggestions_workspace_status_created "
    "ON proactive_suggestions (workspace_id, status, created_at);",
    "CREATE INDEX IF NOT EXISTS ix_proactive_runs_workspace_agent_created "
    "ON proactive_runs (workspace_id, agent_id, created_at);",
)


def upgrade() -> None:
    for _table, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # Each table's indexes and constraints go with it.
    for table, _statement in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table};")
