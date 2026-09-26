"""F078 (containment) — Auto is not offered a tool that can only error, and a
workspace with one database never has to be asked which one.

Night 3, one database connected: Auto chose platform_query_prometheus four
times; on the local stack every call answered "Prometheus is only accessible
within the Railway internal network". The monitoring actions now carry an
availability check — the handler's own precondition — and the registry leaves
an unavailable action out of every surface it builds (dispatcher enum,
first-class schemas, OpenAI tools, platform_find_tools). Called anyway, the
handler refuses without dialing anything.

Renaming or merging the data tools is Gerard's call and is not done here.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from config import config
from modules.tools.discovery import handlers_monitoring as mon
from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry

MONITORING = ("platform_query_prometheus", "platform_query_loki_logs", "platform_get_logs", "platform_list_services")


def _registry(*actions):
    registry = ActionRegistry()
    registry._initialized = True               # only the actions under test
    for action in actions:
        registry.register(action)
    return registry


def _action(name, available=None, promoted=False):
    return ActionDefinition(name=name, description=name, category="t",
                            parameters={"type": "object", "properties": {}}, promoted=promoted,
                            available=available)


def _enum(registry, **kw):
    schema = registry.to_dispatcher_schema(exclude_promoted=False, include_super_admin=True, **kw)
    return schema["function"]["parameters"]["properties"]["action"].get("enum", [])


# ── the registry leaves an unavailable action out of every surface ──────────

def test_an_unavailable_action_is_in_no_surface():
    def broken():
        raise RuntimeError("check blew up")

    registry = _registry(_action("works"), _action("down", available=lambda: False),
                         _action("up", available=lambda: True), _action("broken", available=broken),
                         _action("promoted_down", available=lambda: False, promoted=True))
    assert _enum(registry) == ["up", "works"]
    assert _enum(registry, allowed_names=["down", "up"]) in (["up"], ["up", "works"])  # never "down"
    assert [t["function"]["name"] for t in registry.to_first_class_schemas(include_super_admin=True)] == []
    assert sorted(t["function"]["name"] for t in registry.to_openai_tools(include_super_admin=True)) == ["up", "works"]
    assert registry.get("down") is not None     # still callable by name: its handler refuses


# ── the monitoring checks are the handlers' own preconditions ───────────────

@pytest.fixture
def unconfigured(monkeypatch):
    for key in ("PROMETHEUS_CONFIGURED", "LOKI_CONFIGURED"):
        monkeypatch.setattr(config, key, False, raising=False)
    for key in ("GRAFANA_URL", "GRAFANA_SERVICE_ACCOUNT_TOKEN", "RAILWAY_API_TOKEN", "RAILWAY_PROJECT_ID"):
        monkeypatch.setattr(config, key, "", raising=False)
    return monkeypatch


def test_nothing_configured_means_no_monitoring_action_is_available(unconfigured):
    assert not mon.prometheus_available() and not mon.loki_available() and not mon.railway_api_available()


def test_each_check_passes_on_its_own_configuration(unconfigured):
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    unconfigured.setattr(config, "GRAFANA_URL", "https://grafana.example")
    unconfigured.setattr(config, "GRAFANA_SERVICE_ACCOUNT_TOKEN", "glsa_x")
    unconfigured.setattr(config, "RAILWAY_API_TOKEN", "t")
    unconfigured.setattr(config, "RAILWAY_PROJECT_ID", "p")
    assert mon.prometheus_available() and mon.loki_available() and mon.railway_api_available()


def test_an_explicit_loki_url_is_enough(unconfigured):
    unconfigured.setattr(config, "LOKI_CONFIGURED", True)
    assert mon.loki_available()


def test_the_real_registry_offers_monitoring_only_where_it_can_run(unconfigured):
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    offered = set(_enum(registry))
    assert not offered & set(MONITORING)
    assert "platform_get_alerts" in offered                       # reads our own DB: always runs
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    assert "platform_query_prometheus" in set(_enum(registry))


def test_prometheus_called_anyway_refuses_without_dialing(unconfigured):
    import httpx

    def no_network(*_a, **_k):
        raise AssertionError("an unconfigured Prometheus must not be dialed")

    unconfigured.setattr(httpx, "AsyncClient", no_network)
    result = asyncio.run(mon.query_prometheus(None, uuid4(), {"query": "health"}))
    assert result["success"] is False and "PROMETHEUS_URL" in result["error"]
    loki = asyncio.run(mon.query_loki_logs(None, uuid4(), {}))
    assert loki["success"] is False and "LOKI_URL" in loki["error"]
    # F121 (night 3): the local edition's refusal names no host — five calls
    # answered "Cannot reach Prometheus at http://prometheus.railway.internal:9090".
    for refusal in (result["error"], loki["error"]):
        assert "railway.internal" not in refusal and "http" not in refusal


# ── one database: the data tools use it without asking ──────────────────────

def test_smart_query_database_with_one_source_and_no_name_queries_it(monkeypatch):
    from modules.nl2sql.service import DatabaseKnowledgeService
    from modules.tools.execution import exec_research

    class _Service(DatabaseKnowledgeService):
        def __init__(self):
            self.queried = None

        async def active_sources(self, workspace_id, db_session=None):
            return [(36, "harbourline_shop")]

        async def smart_query(self, source_id, text, user_id, agent_id=None, workspace_id=None, owner_question=None):
            self.queried = source_id
            return {"success": True, "data": [], "columns": [], "row_count": 0}

        async def write_nl_audit(self, **_kw):
            pass

    service = _Service()
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: service)
    result = asyncio.run(exec_research.execute_smart_database_tool(
        NS(db=None), "smart_query_database", {"query": "how many orders?"}, 1,
        workspace_id="ws-1", caller_context={"user_id": "u"}))
    assert result["success"] is True and service.queried == "36"


def test_the_data_tools_never_require_a_database_to_be_named():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.registry.tool_registry import get_tool_registry

    tools = get_tool_registry()
    for name in ("query_database", "smart_query_database"):
        [param] = [p for p in tools.get_tool(name).parameters if p.name == "database_name"]
        assert param.required is False and "used automatically" in param.description
    query_data = get_action_registry().get("platform_query_data").parameters
    assert "database_id" not in query_data.get("required", [])
