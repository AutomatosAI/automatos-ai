"""PRD-154 S12 — NL2SQL structured-error + dialect honoring, entity-KG auth gate,
clone-error token strip.

Verified ground truth (main 2026-06-10):
  * modules/nl2sql/query/nl2sql_service.py generate_sql, on LLM failure, returned
    an EXECUTABLE ``SELECT 'Error generating SQL: …' as error`` — which validates
    then EXECUTES, surfacing the error text as a fake result row.
  * api/database_knowledge.py create endpoint received ``source.dialect`` but
    ``add_database_source`` hardcoded ``dialect='postgresql'`` (# TODO) — a MySQL
    source was stored as postgres and generated postgres SQL.
  * api/knowledge_graph.py entity endpoints (84-559) had NO auth dependency —
    unauthenticated cross-tenant reads.
  * modules/codegraph/codegraph_service.py clone-failure ValueError embedded git's
    error text, which echoes the authed clone URL (https://<token>@github.com/…).

No DB, no network: the LLM provider is faked, ``git.Repo.clone_from`` is patched to
raise, and the entity-KG 401 sweep runs against a fresh FastAPI app with REQUIRE_AUTH
forced on and ``get_db`` overridden (the hybrid dependency 401s before any query).
"""
from __future__ import annotations

import os

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern); the port
# points at nothing so any stray connect fails fast. CI exports real values.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio
import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# CI collection-order safety net: this module imports the real modules.*/api.*
# chain at collection time; restore real app modules before that (no-op once
# conftest has run, which it always has under pytest).
import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import core.auth.hybrid as hybrid_mod  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from api.knowledge_graph import router as kg_router  # noqa: E402
from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService  # noqa: E402
from modules.nl2sql.service import DatabaseKnowledgeService  # noqa: E402
from modules.codegraph.codegraph_service import CodeGraphService  # noqa: E402

_ORCH = Path(__file__).resolve().parents[1]
_NL2SQL_SRC = (_ORCH / "modules" / "nl2sql" / "query" / "nl2sql_service.py").read_text(encoding="utf-8")
_SERVICE_SRC = (_ORCH / "modules" / "nl2sql" / "service.py").read_text(encoding="utf-8")
_DBK_API_SRC = (_ORCH / "api" / "database_knowledge.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# S12.3 — generation failure returns a structured error, no executable SQL
# ---------------------------------------------------------------------------
class _RaisingLLM:
    def generate_response_sync(self, messages):
        raise RuntimeError("LLM upstream 503")


def test_generate_sql_failure_returns_no_executable_sql():
    svc = NaturalLanguageToSQLService(llm_provider=_RaisingLLM())
    sql, explanation, metadata = svc.generate_sql(
        question="how many orders today",
        schema_metadata={"tables": [{"name": "orders", "columns": [{"name": "id", "type": "int"}]}]},
        dialect="postgresql",
    )
    assert metadata.get("success") is False
    assert metadata.get("error")
    # No executable SQL on failure: empty, never a runnable SELECT.
    assert sql == ""
    assert re.match(r"^\s*SELECT\b", sql or "", flags=re.IGNORECASE) is None


def test_nl2sql_source_drops_error_string_select():
    assert "SELECT 'Error generating SQL" not in _NL2SQL_SRC


def test_service_short_circuits_generation_failure_with_no_sql():
    # The orchestrator returns a structured error (sql=None) on a failed
    # generation instead of validating/executing the empty string.
    assert 'metadata.get("success") is False' in _SERVICE_SRC
    assert '"sql": None' in _SERVICE_SRC


# ---------------------------------------------------------------------------
# S12.2 — the dialect field is honored end-to-end (no postgres hardcode)
# ---------------------------------------------------------------------------
def test_add_database_source_accepts_dialect_param():
    sig = inspect.signature(DatabaseKnowledgeService.add_database_source)
    assert "dialect" in sig.parameters


def test_service_does_not_hardcode_postgres_dialect_on_create():
    assert "dialect='postgresql'" not in _SERVICE_SRC
    assert "detect from credentials" not in _SERVICE_SRC
    assert "dialect=dialect" in _SERVICE_SRC


def test_create_endpoint_forwards_dialect():
    assert "dialect=source.dialect" in _DBK_API_SRC


# ---------------------------------------------------------------------------
# S12.4 — entity-KG endpoints removed in PRD-165 (graph consolidation)
# ---------------------------------------------------------------------------
# The PRD-21 entity explorer (/entities/*, /stats/entities) was deleted in
# PRD-165 — the workspace graph is the single canonical surface and agents query
# it via the platform_graph_* tools, not REST. With REQUIRE_AUTH on and no
# credentials a *deleted* route 404s (routing misses before auth runs), so this
# now guards against the endpoints being reintroduced.
_ENTITY_REQUESTS = [
    ("GET", "/api/knowledge/entities"),
    ("GET", "/api/knowledge/entities/search?query=foo&limit=10"),
    ("GET", "/api/knowledge/entities/1"),
    ("GET", "/api/knowledge/entities/1/graph"),
    ("GET", "/api/knowledge/entities/1/documents"),
    ("POST", "/api/knowledge/entities/find-connection?entity_a=a&entity_b=b"),
    ("GET", "/api/knowledge/stats/entities"),
]


def _kg_client(monkeypatch):
    monkeypatch.setattr(hybrid_mod.config, "REQUIRE_AUTH", True)
    app = FastAPI()
    app.include_router(kg_router)

    def _override_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("method,path", _ENTITY_REQUESTS)
def test_entity_kg_endpoint_removed(monkeypatch, method, path):
    client = _kg_client(monkeypatch)
    resp = client.request(method, path)
    assert resp.status_code == 404, f"{method} {path} should be deleted -> {resp.status_code} {resp.text[:200]}"


def _dependency_calls(dependant):
    out = []
    for sub in dependant.dependencies:
        if sub.call is not None:
            out.append(sub.call)
        out.extend(_dependency_calls(sub))
    return out


def test_every_knowledge_graph_route_carries_hybrid_gate():
    # Non-vacuity + regression: EVERY remaining route on the router (the
    # workspace-graph import/build/delete endpoints) carries the hybrid auth
    # dependency, so a future graph route is gated at birth.
    routes = [r for r in kg_router.routes if isinstance(r, APIRoute)]
    assert len(routes) >= 3
    for route in routes:
        assert get_request_context_hybrid in _dependency_calls(route.dependant), (
            f"{route.path} is not auth-gated"
        )


# ---------------------------------------------------------------------------
# S12.5 — clone error strips the GitHub auth token
# ---------------------------------------------------------------------------
def test_clone_error_strips_github_token():
    svc = CodeGraphService.__new__(CodeGraphService)
    token = "ghp_SUPERSECRETTOKEN1234567890"
    # git echoes the authed clone URL in its failure text — simulate it.
    leaky = (
        "Cmd('git') failed: git clone "
        f"https://{token}@github.com/fake/repo.git /tmp/x "
        "stderr: 'fatal: Authentication failed'"
    )
    with patch("git.Repo.clone_from", side_effect=Exception(leaky)):
        with pytest.raises(ValueError) as exc:
            asyncio.run(svc._clone_github_repo("https://github.com/fake/repo.git", None, token))
    msg = str(exc.value)
    assert token not in msg
    # The redacted, token-free URL is still present so the error stays useful.
    assert "github.com/fake/repo.git" in msg
