"""Deliverables: a CLI agent's heartbeat report is a heartbeat, so the feed hides it (4 Oct 2026).

Pinned:

* **The writer:** a heartbeat ticket's task report is filed as ``report_type='heartbeat'``;
  any other ticket's stays ``task``.
* **The migration** ``outputs_heartbeat_reports`` chains onto ``prd251c_wave4`` and is the only
  revision chained there. On Postgres, in a throwaway schema with the view's source tables:
  * the view classes a ``heartbeat`` report as source ``heartbeat``, and the rest as before;
  * a ``task`` report linked to a heartbeat ticket is re-typed, while one linked to a user
    ticket, or unlinked, is not;
  * running it twice changes nothing more;
  * the downgrade restores the old classification and types.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from api.board_tasks import report_type_for  # noqa: E402
from tests.test_prd251w2_campaigns import VERSIONS, _load, _run  # noqa: E402

MIGRATION = VERSIONS / "outputs_heartbeat_reports.py"
WS = uuid.uuid4()

# The columns v_workspace_outputs reads from its three sources (prd133b), and the ticket table.
SOURCE_TABLES = """
CREATE TABLE board_tasks (id serial PRIMARY KEY, workspace_id uuid NOT NULL, source_type varchar(30) NOT NULL);
CREATE TABLE blog_posts (
    id uuid PRIMARY KEY, workspace_id uuid, author_agent_id integer, author_name varchar, title varchar,
    excerpt text, file_path text, slug varchar, category varchar, tags text[], published_at timestamptz,
    reading_time_minutes integer, status varchar, deleted_at timestamptz, created_at timestamptz,
    updated_at timestamptz);
CREATE TABLE agent_reports (
    id uuid PRIMARY KEY, workspace_id uuid, heartbeat_result_id uuid, orchestration_task_id uuid,
    agent_id integer, agent_name varchar, title varchar, summary text, file_path text, file_type varchar,
    file_size_bytes integer, metrics jsonb, report_type varchar(50), grade varchar, grade_notes text,
    attachments jsonb, linked_task_ids jsonb, status varchar, deleted_at timestamptz,
    created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now());
CREATE TABLE deliverables (
    id uuid PRIMARY KEY, workspace_id uuid, source_type varchar, source_id text, agent_id integer,
    agent_name varchar, artifact_type varchar, title varchar, summary text, storage_type varchar,
    file_path text, file_name varchar, file_type varchar, file_size_bytes bigint, preview_url text,
    preview_type varchar, extra jsonb, status varchar, deleted_at timestamptz, created_at timestamptz,
    updated_at timestamptz);
"""


def test_a_heartbeat_tickets_report_is_a_heartbeat_report():
    assert report_type_for(SimpleNamespace(source_type="heartbeat")) == "heartbeat"
    assert report_type_for(SimpleNamespace(source_type="user")) == "task"
    assert report_type_for(SimpleNamespace()) == "task"


def test_the_revision_chains_onto_prd251c_wave4_alone():
    mod = _load(MIGRATION)
    assert (mod.revision, mod.down_revision) == ("outputs_heartbeat_reports", "prd251c_wave4")
    chained = sorted(
        p.name for p in VERSIONS.glob("*.py")
        if re.search(r"^down_revision\s*=\s*['\"]prd251c_wave4['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["outputs_heartbeat_reports.py"]


def test_the_view_is_prd133bs_with_one_rule_added():
    mod = _load(MIGRATION)
    assert mod.VIEW_SQL.replace("                WHEN ar.report_type = 'heartbeat'         THEN 'heartbeat'\n", "") == mod.PRIOR_VIEW_SQL
    assert mod.PRIOR_VIEW_SQL.strip() in (VERSIONS / "prd133b_outputs_view.py").read_text()


@pytest.fixture
def pg_schema():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the migration test needs the Postgres test database: {exc}")
    schema = f"heartbeat_reports_{uuid.uuid4().hex[:8]}"
    with engine.connect() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
        conn.execute(sa.text(f'SET search_path TO "{schema}"'))
        conn.execute(sa.text(SOURCE_TABLES))
        conn.commit()
        try:
            yield conn
        finally:
            conn.rollback()
            conn.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
            conn.commit()
    engine.dispose()


def _ticket(conn, source_type: str) -> int:
    return conn.execute(sa.text("INSERT INTO board_tasks (workspace_id, source_type) VALUES (:ws, :st) RETURNING id"),
                        {"ws": WS, "st": source_type}).scalar_one()


def _report(conn, title: str, linked=None) -> uuid.UUID:
    ident = uuid.uuid4()
    conn.execute(sa.text(
        "INSERT INTO agent_reports (id, workspace_id, title, report_type, status, linked_task_ids) "
        "VALUES (:id, :ws, :title, 'task', 'ok', CAST(:linked AS jsonb))"),
        {"id": ident, "ws": WS, "title": title, "linked": None if linked is None else str(linked)})
    return ident


def _state(conn) -> dict:
    rows = conn.execute(sa.text(
        "SELECT o.title, o.source_type, ar.report_type FROM v_workspace_outputs o "
        "JOIN agent_reports ar ON ar.id = o.id")).all()
    return {title: (source, kind) for title, source, kind in rows}


def test_the_migration_classes_heartbeat_reports_and_re_types_the_old_ones(pg_schema):
    conn = pg_schema
    beat, user = _ticket(conn, "heartbeat"), _ticket(conn, "user")
    _report(conn, "Task: Heartbeat: ATLAS", linked=[beat])
    _report(conn, "Task: Draft the email", linked=[user])
    _report(conn, "Task: Heartbeat: lookalike", linked=None)  # the title alone never counts
    _run(conn, VERSIONS / "prd133b_outputs_view.py", "upgrade")
    assert _state(conn)["Task: Heartbeat: ATLAS"] == ("chat", "task")  # the bug: a chat report

    _run(conn, MIGRATION, "upgrade")
    after = {
        "Task: Heartbeat: ATLAS": ("heartbeat", "heartbeat"),
        "Task: Draft the email": ("chat", "task"),
        "Task: Heartbeat: lookalike": ("chat", "task"),
    }
    assert _state(conn) == after
    _run(conn, MIGRATION, "upgrade")
    assert _state(conn) == after  # twice changes nothing more

    _run(conn, MIGRATION, "downgrade")
    assert _state(conn)["Task: Heartbeat: ATLAS"] == ("chat", "task")
