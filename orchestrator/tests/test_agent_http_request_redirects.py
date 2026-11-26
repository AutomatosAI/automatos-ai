"""#1656: an agent's ``http_request`` tool checks every redirect against its host list, and TLS outside ``local``.

The tool checks the first address's host against ``allowed_http_hosts`` before it sends anything.
It follows redirects, so each hop is checked against the same list too: a redirect from an allowed
host to a host off the list is refused before it is sent, with a clear tool error, and a redirect
between allowed hosts is still followed. The server's certificate is checked in every edition but
``local`` (``agent_http_verifies_tls``).

The requests go through a real httpx client on a ``MockTransport``, so httpx's own redirect
handling runs; nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
import pytest

from config import config
from modules.tools.execution import exec_shell

_REAL_CLIENT = httpx.AsyncClient
START = "http://localhost/api/v1/auth/start"


@pytest.fixture
def edition(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(config, "AUTH_EDITION", name)
    use("local")
    return use


@pytest.fixture
def server(monkeypatch) -> Dict[str, Any]:
    """A stand-in for the network: ``redirects`` maps an address to where it redirects; any other answers 200."""
    state: Dict[str, Any] = {"redirects": {}, "sent": [], "clients": []}

    def answer(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        state["sent"].append(url)
        if url in state["redirects"]:
            return httpx.Response(302, headers={"location": state["redirects"][url]})
        return httpx.Response(200, json={"at": url})

    def client(**kwargs: Any) -> httpx.AsyncClient:
        state["clients"].append(kwargs)
        return _REAL_CLIENT(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return state


def _call(url: str) -> Dict[str, Any]:
    return asyncio.run(exec_shell.execute_http_request(None, "http_request", {"url": url}, agent_id=1))


@pytest.mark.parametrize("target", ["https://idp.example.com/authorize", "http://169.254.169.254/latest/"])
def test_a_redirect_off_the_list_is_refused_before_it_is_sent(edition, server, target):
    server["redirects"][START] = target
    answer = _call(START)
    assert answer["success"] is False
    host = httpx.URL(target).host
    assert answer["error"].startswith(f"Redirect to '{host}' refused: that host is not whitelisted. Allowed: ")
    assert server["sent"] == [START]


def test_a_redirect_between_allowed_hosts_is_still_followed(edition, server):
    server["redirects"][START] = "http://127.0.0.1/api/v1/auth/done"
    answer = _call(START)
    assert answer["success"] is True
    assert answer["status_code"] == 200
    assert answer["body"] == {"at": "http://127.0.0.1/api/v1/auth/done"}
    assert server["sent"] == [START, "http://127.0.0.1/api/v1/auth/done"]


def test_every_hop_of_a_chain_is_checked(edition, server):
    server["redirects"][START] = "http://127.0.0.1/next"
    server["redirects"]["http://127.0.0.1/next"] = "https://elsewhere.example.net/"
    answer = _call(START)
    assert answer["success"] is False
    assert "'elsewhere.example.net'" in answer["error"]
    assert server["sent"] == [START, "http://127.0.0.1/next"]


@pytest.mark.parametrize("name, followed", [("saas", True), ("enterprise", False)])
def test_a_redirect_follows_the_editions_own_list(edition, server, name, followed):
    """Automatos's public API host is on saas's list, and off enterprise's (D-12)."""
    edition(name)
    server["redirects"][START] = "https://api.automatos.app/health"
    assert _call(START)["success"] is followed
    sent: List[str] = [START, "https://api.automatos.app/health"] if followed else [START]
    assert server["sent"] == sent


@pytest.mark.parametrize("name, verify", [("local", False), ("saas", True), ("enterprise", True)])
def test_tls_is_verified_outside_local(edition, server, name, verify):
    edition(name)
    assert _call(f"http://{config.INTERNAL_API_HOSTNAME}/health")["success"] is True
    assert [kwargs["verify"] for kwargs in server["clients"]] == [verify]
