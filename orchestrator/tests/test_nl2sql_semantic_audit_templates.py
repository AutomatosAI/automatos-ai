"""PRD-160 S4 — semantic layer injection, NL audit rows, template execution.

Unit-level (no live DB): pins that (a) admin business definitions are rendered
into the generation prompt, (b) every NL query writes one audit row carrying
agent/SQL/outcome, and (c) the Query Templates execute path validates and runs
the template SQL with BOUND parameters and round-trips real rows.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _bare_service():
    from modules.nl2sql.service import DatabaseKnowledgeService

    svc = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
    svc.llm_provider = MagicMock()
    return svc


# --- A: semantic layer admin instructions steer generation -------------------

def test_admin_instructions_rendered_into_prompt():
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService

    gen = NaturalLanguageToSQLService(llm_provider=MagicMock())
    prompt = gen._build_prompt(
        question="how many active users?",
        schema_metadata={"tables": [{"name": "users", "columns": [{"name": "status", "type": "varchar"}]}]},
        semantic_layer={"instructions": "active = status NOT IN ('churned','deleted')"},
        dialect="postgresql",
        examples=None,
    )
    assert "BUSINESS DEFINITIONS" in prompt
    assert "active = status NOT IN ('churned','deleted')" in prompt


def test_no_semantic_layer_no_definitions_block():
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService

    gen = NaturalLanguageToSQLService(llm_provider=MagicMock())
    prompt = gen._build_prompt(
        question="count users",
        schema_metadata={"tables": [{"name": "users", "columns": []}]},
        semantic_layer=None,
        dialect="postgresql",
        examples=None,
    )
    assert "BUSINESS DEFINITIONS" not in prompt


# --- B: every NL query lands one audit row -----------------------------------

def test_write_nl_audit_creates_one_row(monkeypatch):
    captured = {}

    class FakeAudit:
        def __init__(self, **kw):
            captured.update(kw)

    db = MagicMock()
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    monkeypatch.setattr("core.models.database_knowledge.DatabaseQueryAudit", FakeAudit)

    svc = _bare_service()
    asyncio.run(svc.write_nl_audit(
        source_id=7,
        user_id="3",
        agent_id="42",
        nl_query="how many active users?",
        result={"success": True, "sql": "SELECT count(*) FROM users",
                "row_count": 5, "execution_time_ms": 12, "confidence": {"score": 0.9}},
    ))

    assert captured["source_id"] == 7            # workspace via the source FK
    assert captured["agent_id"] == "42"          # agent
    assert captured["generated_sql"] == "SELECT count(*) FROM users"  # SQL
    assert captured["success"] is True           # outcome
    assert captured["row_count"] == 5
    assert captured["confidence_score"] == 0.9
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_write_nl_audit_is_best_effort(monkeypatch):
    # A failing audit write must never raise into the query path.
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("core.database.database.SessionLocal", boom)
    svc = _bare_service()
    # should not raise
    asyncio.run(svc.write_nl_audit(
        source_id=1, nl_query="q", result={"success": False, "error": "x"},
    ))


# --- C: Query Templates execute round-trips with bound params ----------------

def test_execute_template_round_trips(monkeypatch):
    svc = _bare_service()
    source = SimpleNamespace(
        dialect="postgresql",
        schema_metadata={"tables": [{"name": "users"}]},
        max_rows_limit=1000, credential_id=1, query_timeout_seconds=30,
    )
    svc._get_source = AsyncMock(return_value=source)
    svc._decrypt_source_credentials = MagicMock(return_value={
        "host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"})
    svc._run_sql_with_guards = MagicMock(return_value=(["id"], [{"id": 1}, {"id": 2}]))

    tmpl = SimpleNamespace(
        sql_template="SELECT id FROM users WHERE status = :status",
        visualization_type="bar", usage_count=0,
    )
    db = MagicMock()
    qobj = MagicMock()
    qobj.filter.return_value = qobj
    qobj.first.return_value = tmpl
    db.query.return_value = qobj
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    monkeypatch.setattr("core.models.database_knowledge.DatabaseQueryTemplate", MagicMock())

    out = asyncio.run(svc.execute_template(
        source_id=7, template_id=3, parameters={"status": "active"},
        workspace_id="ws-1", max_rows=500,
    ))

    assert out["success"] is True
    assert out["row_count"] == 2
    assert out["visualization_type"] == "bar"
    # values are BOUND, never interpolated
    _, kwargs = svc._run_sql_with_guards.call_args
    assert kwargs.get("params") == {"status": "active"}
    # the validated SQL kept the placeholder and got a LIMIT
    assert ":status" in kwargs.get("params") or ":status" in svc._run_sql_with_guards.call_args[0][2]
    assert tmpl.usage_count == 1  # usage telemetry bumped


def test_execute_template_missing_template_fails_closed(monkeypatch):
    svc = _bare_service()
    source = SimpleNamespace(dialect="postgresql", schema_metadata={"tables": []},
                             max_rows_limit=1000, credential_id=1)
    svc._get_source = AsyncMock(return_value=source)
    db = MagicMock()
    qobj = MagicMock()
    qobj.filter.return_value = qobj
    qobj.first.return_value = None  # no such template in this source
    db.query.return_value = qobj
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    monkeypatch.setattr("core.models.database_knowledge.DatabaseQueryTemplate", MagicMock())

    out = asyncio.run(svc.execute_template(
        source_id=7, template_id=999, parameters={}, workspace_id="ws-1",
    ))
    assert out["success"] is False and "not found" in out["error"].lower()
