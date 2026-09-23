"""F049 (night 1) — a deleted agent's spend keeps the agent's name.

``llm_usage.agent_id`` has no foreign key, so deleting an agent left its usage
rows with an id that names nothing: Analytics showed "Agent #273" (CELLAR) and
the KPI card's top spenders dropped it entirely (an inner join). The delete path
now stamps the name on the rows first; both readers use it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"
WS = uuid4()


def test_the_column_is_declared_and_migrated_idempotently():
    from core.models.core import LLMUsage

    assert "agent_name" in {c.name for c in LLMUsage.__table__.columns}
    migration = (VERSIONS / "llm_usage_agent_name.py").read_text()
    assert 'down_revision = "kb_multimodal_tables"' in migration
    assert "ADD COLUMN IF NOT EXISTS agent_name" in migration


# ── the delete path ─────────────────────────────────────────────────────────

class _Result:
    def fetchall(self):
        return []


class _Savepoint:
    def commit(self):
        pass

    def rollback(self):
        pass


class _DeleteDb:
    def __init__(self, agent):
        self.agent = agent
        self.log = []

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return self.agent

    def execute(self, statement, params=None):
        self.log.append(("sql", " ".join(str(statement).split()), params))
        return _Result()

    def begin_nested(self):
        return _Savepoint()

    def delete(self, obj):
        self.log.append(("delete", obj.id, None))

    def commit(self):
        self.log.append(("commit", None, None))

    def rollback(self):
        pass


def test_deleting_an_agent_stamps_its_name_on_its_spend_first():
    from api.agents import delete_agent

    cellar = NS(id=273, name="CELLAR", workspace_id=WS)
    db = _DeleteDb(cellar)
    asyncio.run(delete_agent(273, NS(workspace_id=WS), db))
    stamp = next(i for i, (kind, sql, _) in enumerate(db.log) if kind == "sql" and sql.startswith("UPDATE llm_usage"))
    gone = next(i for i, (kind, _, _) in enumerate(db.log) if kind == "delete")
    assert stamp < gone
    assert db.log[stamp][2] == {"name": "CELLAR", "agent_id": 273, "workspace_id": WS}
    assert "agent_name IS NULL" in db.log[stamp][1]      # a rename-then-delete keeps the first stamp


# ── the readers, on Postgres ────────────────────────────────────────────────

_DROP = "DROP TABLE IF EXISTS pg_temp.llm_usage, pg_temp.agents"


@pytest.fixture
def db(test_engine):
    # Temp tables shadow the real ones on this connection (pg_temp is searched
    # first); dropped schema-qualified on the way in and out.
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text(_DROP))
        session.execute(text("CREATE TEMP TABLE llm_usage (LIKE public.llm_usage INCLUDING DEFAULTS)"))
        session.execute(text("CREATE TEMP TABLE agents (id int, name varchar)"))
        session.execute(text("INSERT INTO agents VALUES (58, 'WRITER')"))
        now = datetime.now() - timedelta(hours=1)
        for agent_id, name, cost in ((273, "CELLAR", 3.0), (273, "CELLAR", 1.0), (58, None, 0.5), (15, None, 0.2)):
            session.execute(text(
                "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, agent_id, agent_name, request_type, "
                "input_tokens, output_tokens, total_tokens, input_cost, output_cost, total_cost, created_at) "
                "VALUES (:ws, 'm', 'openrouter', 'direct', :aid, :name, 'agent', 1, 1, 2, 0, :cost, :cost, :at)"),
                {"ws": WS, "aid": agent_id, "name": name, "cost": cost, "at": now})
        yield session
        session.rollback()
        session.execute(text(_DROP))
        session.commit()
        session.close()


def test_analytics_names_a_deleted_agent_by_its_stamped_name(db):
    from api.llm_analytics import _agent_facts

    facts = _agent_facts(db, WS, datetime.now() - timedelta(days=1), [273, 58, 15])
    assert facts["273"]["label"] == "CELLAR"
    assert facts["58"]["label"] == "WRITER"            # the live name wins
    assert facts["15"]["label"] == "Agent #15"         # deleted before names were kept


def test_the_kpi_card_keeps_a_deleted_agents_spend_in_its_top_spenders(db):
    from api.kpi_api import get_cost_tracker

    out = asyncio.run(get_cost_tracker(period="7d", ctx=NS(workspace_id=WS), db=db))
    assert [a["name"] for a in out["top_agents"]] == ["CELLAR", "WRITER", "Agent #15"]
    assert out["top_agents"][0]["cost"] == 4.0
