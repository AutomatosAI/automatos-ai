"""PRD-277 D-12 (#1605): an agent's ``http_request`` tool names and calls only the install's own hosts in enterprise.

In ``enterprise`` the platform's API is the install's own, so the tool's allowed hosts are its
internal API and web hosts and the loopback names, without Automatos's public API and web hosts,
and the description the model reads names the same (``agent_http_own_hosts_only``). In ``local``
and ``saas`` the list and the text are exactly what they were. Nothing else in the tool changes:
a call to a host off the list gets the tool's existing refusal, and no client is made.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
import pytest

from config import config
from modules.tools.execution import exec_shell
from modules.tools.execution.http_hosts import fitted_to_edition
from modules.tools.registry.tool_registry import ToolRegistry

PUBLIC = ("api.automatos.app", "ui.automatos.app")


def _before() -> set:
    """The list as ``_get_allowed_http_domains`` wrote it before D-12, for every edition."""
    return {config.INTERNAL_API_HOSTNAME, config.INTERNAL_FRONTEND_HOSTNAME, *PUBLIC, "localhost", "127.0.0.1"}


def _description(public: str) -> str:
    """The registry's description, naming the public hosts as ``public`` (``""`` for none)."""
    return (
        "Make HTTP requests to INTERNAL platform URLs only. "
        "Use for testing API endpoints, checking health status, and verifying responses. "
        f"RESTRICTED to whitelisted domains: {config.INTERNAL_API_HOSTNAME}, {config.INTERNAL_FRONTEND_HOSTNAME}, "
        f"{public}localhost. "
        "Will FAIL on external URLs. For external web requests, use web search tools or Composio actions."
    )


@pytest.fixture
def edition(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(config, "AUTH_EDITION", name)
    return use


class _Response:
    status_code = 200
    headers: Dict[str, str] = {"content-type": "application/json"}
    text = "{}"

    def json(self) -> Dict[str, Any]:
        return {}


@pytest.fixture
def requests(monkeypatch) -> List[str]:
    """Every client made and every address requested, through a stand-in for httpx's client."""
    made: List[str] = []

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            made.append("client")

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any) -> _Response:
            made.append(url)
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return made


def _call(url: str) -> Dict[str, Any]:
    return asyncio.run(exec_shell.execute_http_request(None, "http_request", {"url": url}, agent_id=1))


@pytest.mark.parametrize("name", ["local", "saas"])
def test_local_and_saas_keep_the_list_and_the_text(edition, name):
    edition(name)
    assert exec_shell._get_allowed_http_domains() == _before()
    spec = ToolRegistry().get_tool("http_request")
    assert spec.description == _description("api.automatos.app, ui.automatos.app, ")
    assert spec.metadata["allowed_domains"] == [
        config.INTERNAL_API_HOSTNAME, config.INTERNAL_FRONTEND_HOSTNAME, *PUBLIC, "localhost", "127.0.0.1",
    ]


def test_enterprise_lists_and_names_only_the_installs_hosts(edition):
    edition("enterprise")
    assert exec_shell._get_allowed_http_domains() == _before() - set(PUBLIC)
    spec = ToolRegistry().get_tool("http_request")
    assert spec.description == _description("")
    assert spec.metadata["allowed_domains"] == [
        config.INTERNAL_API_HOSTNAME, config.INTERNAL_FRONTEND_HOSTNAME, "localhost", "127.0.0.1",
    ]
    assert "automatos.app" not in spec.description
    assert spec.metadata["max_response_bytes"] == 1_000_000


def test_the_tool_keeps_its_place_and_everything_else_in_enterprise(edition):
    edition("local")
    local = ToolRegistry()
    edition("enterprise")
    enterprise = ToolRegistry()
    assert list(enterprise.tools) == list(local.tools)
    before, after = local.get_tool("http_request"), enterprise.get_tool("http_request")
    assert after.parameters == before.parameters and after.examples == before.examples
    assert after.security_level == before.security_level and after.executor_method == before.executor_method


@pytest.mark.parametrize("host", PUBLIC)
def test_enterprise_refuses_a_call_to_automatos_public_hosts_and_requests_nothing(edition, requests, host):
    edition("enterprise")
    answer = _call(f"https://{host}/api/agents")
    assert answer["success"] is False
    assert answer["error"].startswith(f"Domain '{host}' is not whitelisted. Allowed: ")
    assert "automatos.app" not in answer["error"].split("Allowed: ")[1]
    assert requests == []


@pytest.mark.parametrize("name", ["local", "saas"])
def test_local_and_saas_still_call_automatos_public_hosts(edition, requests, name):
    edition(name)
    answer = _call("https://api.automatos.app/health")
    assert answer["success"] is True
    assert requests == ["client", "https://api.automatos.app/health"]


def test_enterprise_still_calls_the_installs_own_api(edition, requests):
    edition("enterprise")
    url = f"http://{config.INTERNAL_API_HOSTNAME}/health"
    assert _call(url)["success"] is True
    assert requests == ["client", url]


def test_outside_enterprise_the_registry_spec_is_returned_as_it_is(edition):
    edition("local")
    spec = object()
    assert fitted_to_edition(spec) is spec
