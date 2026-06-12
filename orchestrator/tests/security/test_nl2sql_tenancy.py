"""PRD-156 S3 — NL2SQL: unsafe path OFF, off the chat surface, endpoints scoped.

The NL2SQL tools (``query_database`` / ``smart_query_database``) executed raw
LLM-generated SQL against the ENTIRE main DB with no workspace filter, and made
unauthenticated HTTP self-calls. S3:
  * both executors return a disabled response; ``query_main_database`` is deleted
    (OFF, not shimmed);
  * the tools are removed from the chat surface (intent_classifier / service.py),
    and ``query_main_database`` is unreachable from any chat-routing module;
  * the analytics + audit endpoints are workspace-scoped.

Pure unit / structural tests — no live DB. The cross-workspace DB matrix runs in
CI. ``exec_research`` is now import-light; a camelot leaf-stub keeps the
``modules.tools`` package import from breaking collection in a lean venv.
"""
from __future__ import annotations

import asyncio
import importlib.util as _ilu
import pathlib
import re
import sys as _sys

import pytest

ORCH = pathlib.Path(__file__).resolve().parents[2]  # .../orchestrator


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))


# --- A: re-enabled path fails closed without scope; query_main_database deleted -
#
# PRD-160 S1 re-enables both executors as workspace-scoped, in-process tools.
# The security contract is no longer "always disabled" but "refuses to run
# without a workspace context" — the defense-in-depth backstop if the path is
# reached (e.g. a Playbook step) without scope. The unscoped raw-SQL helper
# stays deleted.

def test_database_tool_refuses_without_workspace():
    from modules.tools.execution import exec_research

    r = asyncio.run(exec_research.execute_database_tool(None, "query_database", {"query": "x"}, 1))
    assert r["success"] is False
    assert "disabled" not in r  # no longer a hard-disabled stub
    assert "workspace" in r["error"].lower()


def test_smart_database_tool_refuses_without_workspace():
    from modules.tools.execution import exec_research

    r = asyncio.run(
        exec_research.execute_smart_database_tool(None, "smart_query_database", {"query": "x"}, 1)
    )
    assert r["success"] is False
    assert "disabled" not in r
    assert "workspace" in r["error"].lower()


def test_query_main_database_stays_deleted():
    """The unscoped raw-SQL fallback must never come back, even re-enabled."""
    from modules.tools.execution import exec_research

    assert not hasattr(exec_research, "query_main_database")


def test_executors_make_no_http_self_call():
    """PRD-160 S1 grep gate: the in-process path imports no HTTP client."""
    code = (ORCH / "modules/tools/execution/exec_research.py").read_text()
    body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
    assert "httpx" not in body and "requests" not in body and "aiohttp" not in body


# --- B: query_main_database unreachable from chat; tools off the surface -------

@pytest.mark.parametrize(
    "rel",
    [
        "consumers/chatbot/service.py",
        "consumers/chatbot/intent_classifier.py",
        "consumers/chatbot/smart_tool_router.py",
        "modules/tools/execution/exec_research.py",
    ],
)
def test_no_query_main_database_call_in_chat_surface(rel):
    code = "\n".join(
        line
        for line in (ORCH / rel).read_text().splitlines()
        if not line.strip().startswith("#")
    )
    assert "query_main_database(" not in code, f"{rel} still calls query_main_database"


def test_nl2sql_tools_present_in_chat_registry():
    """PRD-160 S1: NL2SQL is back on the chat surface (workspace-scoped)."""
    svc = (ORCH / "consumers/chatbot/service.py").read_text()
    block = re.search(r"SEARCH_TOOLS\s*=\s*\{(.*?)\}", svc, re.S).group(1)
    code = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))
    assert "'smart_query_database'" in code  # re-enabled

    ic = (ORCH / "consumers/chatbot/intent_classifier.py").read_text()
    data_branch = ic.split("Check for data/analytics queries")[1][:500]
    assert "smart_query_database" in data_branch  # suggested again, scoped
    assert "suggested = []" not in data_branch


# --- C: analytics + audit endpoints workspace-scoped --------------------------

def test_analytics_endpoints_scope_by_workspace():
    txt = (ORCH / "api/database_analytics.py").read_text()
    joins = txt.count("JOIN database_knowledge_sources dks ON dks.id = dqa.source_id")
    filters = txt.count("dks.workspace_id = :workspace_id::uuid")
    assert joins >= 3 and filters >= 3, f"unscoped analytics query (joins={joins} filters={filters})"


def test_audit_endpoint_verifies_source_ownership():
    txt = (ORCH / "api/database_knowledge.py").read_text()
    audit = txt.split('@router.get("/{source_id}/audit")')[1][:800]
    assert "DatabaseKnowledgeSource.workspace_id == ctx.workspace_id" in audit
    assert "404" in audit  # fails closed when the source is not in the caller's workspace
