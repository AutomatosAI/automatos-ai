"""PRD-160 S2 — accuracy stack: execution guards, value sampling, self-correction.

Unit-level (no live DB): the DB connection is faked so we can pin the order and
content of what hits the wire — a per-statement timeout, an EXPLAIN dry-run
before the real query, and the bounded self-correction loop recovering from a
bad-column attempt within the 2-retry budget.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SERVICE = "modules.nl2sql.service"


def _bare_service():
    from modules.nl2sql.service import DatabaseKnowledgeService

    svc = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
    svc.llm_provider = MagicMock()
    return svc


class _FakeConn:
    """A connection that records every executed statement and can fail on a
    chosen one. Acts as its own context manager (``with engine.connect()``)."""

    def __init__(self, fail_on=None, distinct=None):
        self.executed: list[str] = []
        self._fail_on = fail_on or (lambda s: False)
        self._distinct = distinct or {}

    def execute(self, clause, params=None):
        sql = str(clause)
        self.executed.append(sql)
        if self._fail_on(sql):
            raise RuntimeError(f"db error on: {sql[:40]}")
        res = MagicMock()
        res.keys.return_value = ["id", "name"]
        # value-sampling DISTINCT probe → return per-column canned values
        for col, vals in self._distinct.items():
            if f'"{col}"' in sql or f"`{col}`" in sql:
                res.fetchall.return_value = [(v,) for v in vals]
                return res
        res.fetchall.return_value = [(1, "a")]
        return res

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeEngine:
    def __init__(self, conn):
        self._conn = conn
        self.disposed = False

    def connect(self):
        return self._conn

    def dispose(self):
        self.disposed = True


_CREDS = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"}


# --- statement timeout -------------------------------------------------------

def test_statement_timeout_sql_by_dialect():
    from modules.nl2sql.service import DatabaseKnowledgeService as S

    assert S._statement_timeout_sql("postgresql", 7) == "SET statement_timeout = 7000"
    assert S._statement_timeout_sql("mysql", 7) == "SET max_execution_time = 7000"
    assert S._statement_timeout_sql("sqlite", 7) is None


# --- EXPLAIN dry-run + execution guards --------------------------------------

def test_guards_set_timeout_and_explain_before_query():
    svc = _bare_service()
    conn = _FakeConn()
    source = SimpleNamespace(dialect="postgresql", query_timeout_seconds=9)
    with patch(f"{SERVICE}.create_engine", return_value=_FakeEngine(conn)):
        cols, rows = svc._run_sql_with_guards(source, _CREDS, "SELECT * FROM users")

    joined = conn.executed
    assert "SET statement_timeout = 9000" in joined
    explain_idx = next(i for i, s in enumerate(joined) if s.startswith("EXPLAIN"))
    select_idx = next(i for i, s in enumerate(joined) if s == "SELECT * FROM users")
    assert explain_idx < select_idx, "EXPLAIN dry-run must precede the real query"
    assert cols == ["id", "name"]
    assert rows == [{"id": 1, "name": "a"}]


def test_explain_failure_propagates_without_executing():
    svc = _bare_service()
    conn = _FakeConn(fail_on=lambda s: s.startswith("EXPLAIN"))
    source = SimpleNamespace(dialect="postgresql", query_timeout_seconds=30)
    with patch(f"{SERVICE}.create_engine", return_value=_FakeEngine(conn)):
        with pytest.raises(RuntimeError):
            svc._run_sql_with_guards(source, _CREDS, "SELECT bad_col FROM users")
    # the real query must NOT run once EXPLAIN rejects it
    assert not any(s == "SELECT bad_col FROM users" for s in conn.executed)


# --- low-cardinality value sampling ------------------------------------------

def test_value_sampling_populates_low_cardinality_only():
    svc = _bare_service()
    schema = {"tables": [{"name": "users", "columns": [
        {"name": "id", "type": "integer"},
        {"name": "status", "type": "varchar"},
        {"name": "bio", "type": "text"},
    ]}]}
    conn = _FakeConn(distinct={"status": ["active", "churned", "trial"]})
    source = SimpleNamespace(dialect="postgresql")
    with patch(f"{SERVICE}.create_engine", return_value=_FakeEngine(conn)):
        svc._augment_schema_with_samples(source, _CREDS, schema)

    cols = {c["name"]: c for c in schema["tables"][0]["columns"]}
    assert cols["status"].get("samples") == ["active", "churned", "trial"]
    assert "samples" not in cols["id"]   # numeric → never probed
    assert "samples" not in cols["bio"]  # free-form TEXT → never probed
    # only the categorical column was probed
    probes = [s for s in conn.executed if s.startswith("SELECT DISTINCT")]
    assert len(probes) == 1 and "status" in probes[0]


def test_value_sampling_skips_high_cardinality():
    svc = _bare_service()
    schema = {"tables": [{"name": "users", "columns": [
        {"name": "country", "type": "varchar"},
    ]}]}
    # 13 distinct (> max_distinct=12) → must be dropped, not injected
    conn = _FakeConn(distinct={"country": [f"c{i}" for i in range(13)]})
    source = SimpleNamespace(dialect="postgresql")
    with patch(f"{SERVICE}.create_engine", return_value=_FakeEngine(conn)):
        svc._augment_schema_with_samples(source, _CREDS, schema)
    assert "samples" not in schema["tables"][0]["columns"][0]


# --- bounded self-correction (retries=2) -------------------------------------

def test_self_correction_recovers_within_two_retries():
    """A bad-column first attempt fails the EXPLAIN/exec; the loop feeds the
    error back, regenerates, and the second attempt succeeds — all inside the
    2-retry budget."""
    svc = _bare_service()
    source = SimpleNamespace(
        dialect="postgresql", workspace_id="ws-1", credential_id=1,
        schema_metadata={"tables": [{"name": "users"}]},
    )
    svc._get_source = AsyncMock(return_value=source)
    svc._augment_schema_with_samples = MagicMock()
    svc._get_example_store = MagicMock(return_value=None)
    svc._calculate_confidence = MagicMock(return_value={"score": 1.0})
    # execution: fail once (bad column), then succeed
    svc._run_sql_with_guards = MagicMock(side_effect=[
        RuntimeError('column "bad_col" does not exist'),
        (["id"], [{"id": 1}]),
    ])

    gen = MagicMock()
    gen.generate_sql.side_effect = [
        ("SELECT bad_col FROM users", "first try", {"success": True}),
        ("SELECT id FROM users", "corrected", {"success": True}),
    ]

    cred_store = MagicMock()
    cred_store.get_credential.return_value = SimpleNamespace(encrypted_data=b"x")
    enc = MagicMock()
    enc.decrypt_dict.return_value = _CREDS

    with patch(f"{SERVICE}.NaturalLanguageToSQLService", return_value=gen), \
         patch(f"{SERVICE}._emit_nl2sql_primitive", MagicMock()), \
         patch("core.database.database.SessionLocal", MagicMock()), \
         patch("core.credentials.service.CredentialStore", return_value=cred_store), \
         patch("core.credentials.encryption.EncryptionService", return_value=enc), \
         patch("modules.context.ContextService", MagicMock()):
        result = asyncio.run(svc.query_database(
            source_id="1",
            natural_language_query="how many users are active?",
            user_id="u-1",
            agent_id="a-1",
            workspace_id="ws-1",
        ))

    assert result["success"] is True
    assert result["attempts"] == 2, "should recover on the 2nd attempt"
    # the corrected generation saw the prior error as context
    assert gen.generate_sql.call_count == 2
    second_kwargs = gen.generate_sql.call_args_list[1].kwargs
    assert second_kwargs.get("error_context"), "2nd generation must receive error_context"
    assert second_kwargs.get("previous_attempts"), "2nd generation must receive previous_attempts"
