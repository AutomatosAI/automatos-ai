"""PRD-240 — agents read and search the web as a platform capability.

Pure tests: stubbed HTTP transports, a stubbed private-range check, no DB, no
network, no provider. Locked here:

* the switch (local on / saas off / explicit wins) and the denylist (suffix
  match, adds to — never replaces — the private-range refusal);
* the two actions are registered with schema truth and reach the executor;
* ``web_fetch`` refuses bad schemes, denied and private hosts, redirects that
  land somewhere refused, non-text content; caps bytes; reduces HTML to text;
  answers ``{available:false}`` when the switch is off — never an exception;
* ``web_search`` resolves openrouter → composio → searxng, honours the pin and
  ``off``, names the three options when it has nothing, and turns a backend
  failure into a result;
* the OpenRouter route attaches its server-side search tool only when
  allowed and never twice; citations become ``LLMResponse.citations`` and a
  Sources footer in both the streaming and non-streaming paths.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434): an earlier test can leave a stale,
# spec-less `modules`/`services` entry in sys.modules that breaks these imports.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Dict, List  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
from uuid import uuid4  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from config import config  # noqa: E402
import core.security.web_access as wa  # noqa: E402
from core.llm.web_citations import citations_from_annotations, sources_markdown  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_web import register_web_actions  # noqa: E402
import modules.tools.discovery.handlers_web as hw  # noqa: E402
import services.web_search as ws  # noqa: E402

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
WS = uuid4()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", True)
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ())
    # No real DNS in a unit test: everything public resolves fine unless a test
    # marks a host private.
    monkeypatch.setattr(wa, "validate_webhook_url", lambda url: (True, "OK"))
    yield


@pytest.fixture
def web_off(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", False)
    yield


def _mark_private(monkeypatch, *hosts: str):
    def _check(url: str):
        host = httpx.URL(url).host
        if host in hosts:
            return False, "Resolved to blocked range (10.0.0.0/8)"
        return True, "OK"
    monkeypatch.setattr(wa, "validate_webhook_url", _check)


def _stub_http(monkeypatch, handler):
    """Route web_fetch's client through an httpx.MockTransport."""
    transport = httpx.MockTransport(handler)

    def _factory(**kwargs):
        kwargs.pop("verify", None)
        return httpx.AsyncClient(transport=transport, **kwargs)

    monkeypatch.setattr(hw, "_async_client", _factory)


# ---------------------------------------------------------------------------
# S1 — the switch and the denylist
# ---------------------------------------------------------------------------


def test_switch_reads_the_config_flag(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", True)
    assert wa.web_access_enabled() is True
    monkeypatch.setattr(config, "WEB_ACCESS", False)
    assert wa.web_access_enabled() is False


def test_config_defaults_follow_the_edition():
    """The parsing rule in config.py: unset → on only for the local edition."""
    src = (_ORCH / "config.py").read_text(encoding="utf-8")
    assert 'else AUTH_EDITION == "local"' in src
    assert '_WEB_ACCESS_RAW in ("on", "true", "1", "yes")' in src


def test_denylist_is_suffix_matched(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("example.com", "corp.internal"))
    assert wa.host_denied("example.com")
    assert wa.host_denied("www.example.com")
    assert wa.host_denied("API.Corp.Internal")
    assert not wa.host_denied("notexample.com")
    assert not wa.host_denied("example.com.evil.net")
    assert not wa.host_denied("")


def test_validate_outbound_url_order(monkeypatch, web_on):
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("blocked.test",))
    _mark_private(monkeypatch, "postgres", "host.docker.internal")
    assert wa.validate_outbound_url("https://docs.python.org/3/")[0] is True
    ok, reason, _ = wa.validate_outbound_url("ftp://docs.python.org/x")
    assert not ok and "http" in reason
    ok, reason, _ = wa.validate_outbound_url("https://www.blocked.test/page")
    assert not ok and "WEB_ACCESS_DENY" in reason
    ok, reason, _ = wa.validate_outbound_url("http://postgres:5432/")
    assert not ok and "blocked range" in reason
    ok, reason, _ = wa.validate_outbound_url("http://host.docker.internal/")
    assert not ok


def test_switch_off_refuses_before_resolving(monkeypatch, web_off):
    calls: List[str] = []
    monkeypatch.setattr(wa, "validate_webhook_url", lambda url: calls.append(url) or (True, "OK"))
    ok, reason, _ = wa.validate_outbound_url("https://example.org/")
    assert not ok and "WEB_ACCESS=off" in reason
    assert calls == []


def test_private_ranges_are_refused_even_with_an_empty_denylist(monkeypatch, web_on):
    _mark_private(monkeypatch, "169.254.169.254")
    ok, reason, _ = wa.validate_outbound_url("http://169.254.169.254/latest/meta-data/")
    assert not ok and "not reachable" in reason


# ---------------------------------------------------------------------------
# S2/S3 — registration + schema truth, and the executor knows them
# ---------------------------------------------------------------------------


def _defs():
    reg = ActionRegistry()
    register_web_actions(reg)
    return reg._actions


def test_both_actions_registered_with_schema_truth():
    d = _defs()
    assert d["web_fetch"].parameters["required"] == ["url"]
    assert d["web_search"].parameters["required"] == ["query"]
    for name in ("web_fetch", "web_search"):
        assert d[name].category == "web"
        assert d[name].permission_level == "read"
        assert d[name].requires_confirmation is False
        assert d[name].promoted is False  # discoverable, not shipped in every turn


def test_executor_maps_both_handlers():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ex = PlatformActionExecutor(MagicMock(), WS)
    assert ex._handlers["web_fetch"] is hw.web_fetch
    assert ex._handlers["web_search"] is hw.web_search


def test_rag_hint_points_at_the_native_actions():
    src = (_ORCH / "modules/tools/registry/tool_registry.py").read_text(encoding="utf-8")
    assert "use web_search / web_fetch" in src
    assert "TAVILY_SEARCH" not in src


# ---------------------------------------------------------------------------
# S2 — web_fetch
# ---------------------------------------------------------------------------


_PAGE = """<html><head><title>Hello Page</title><script>alert(1)</script></head>
<body><nav>Menu Menu</nav><h1>Welcome</h1><p>First paragraph.</p>
<ul><li>one</li><li>two</li></ul><footer>foot</footer></body></html>"""


@pytest.mark.asyncio
async def test_fetch_reduces_html_to_text(monkeypatch, web_on):
    _stub_http(monkeypatch, lambda req: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=_PAGE))
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/hello"})
    assert out["success"] is True
    data = out["data"]
    assert data["title"] == "Hello Page"
    assert "# Welcome" in data["content"] and "First paragraph." in data["content"]
    assert "- one" in data["content"]
    assert "alert(1)" not in data["content"] and "Menu Menu" not in data["content"] and "foot" not in data["content"]
    assert data["truncated"] is False and data["status_code"] == 200


@pytest.mark.asyncio
async def test_fetch_requires_url_and_refuses_bad_schemes(monkeypatch, web_on):
    assert (await hw.web_fetch(MagicMock(), WS, {}))["success"] is False
    out = await hw.web_fetch(MagicMock(), WS, {"url": "file:///etc/passwd"})
    assert out["success"] is False and "http" in out["error"]


@pytest.mark.asyncio
async def test_fetch_refuses_denied_and_private_hosts(monkeypatch, web_on):
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("blocked.test",))
    _mark_private(monkeypatch, "minio")
    called = []
    _stub_http(monkeypatch, lambda req: called.append(str(req.url)) or httpx.Response(200, text="x"))
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://blocked.test/"})
    assert out["success"] is False and "WEB_ACCESS_DENY" in out["error"]
    out = await hw.web_fetch(MagicMock(), WS, {"url": "http://minio:9000/bucket"})
    assert out["success"] is False and "not reachable" in out["error"]
    assert called == []  # refused before any request was made


@pytest.mark.asyncio
async def test_fetch_rechecks_the_url_after_a_redirect(monkeypatch, web_on):
    _mark_private(monkeypatch, "internal.host")

    def handler(req: httpx.Request):
        if req.url.host == "public.test":
            return httpx.Response(302, headers={"location": "http://internal.host/secret"})
        return httpx.Response(200, text="SECRET")

    _stub_http(monkeypatch, handler)
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://public.test/go"})
    assert out["success"] is False and "Redirected" in out["error"]
    assert "SECRET" not in json.dumps(out)


@pytest.mark.asyncio
async def test_fetch_refuses_binary_content_types(monkeypatch, web_on):
    _stub_http(monkeypatch, lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/a.pdf"})
    assert out["success"] is False and "application/pdf" in out["error"]


@pytest.mark.asyncio
async def test_fetch_caps_bytes_and_chars(monkeypatch, web_on):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_BYTES", 100)
    _stub_http(monkeypatch, lambda req: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"a" * 1000))
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/big.txt"})
    assert out["success"] is True
    assert out["data"]["truncated"] is True
    assert len(out["data"]["content"]) <= 100

    monkeypatch.setattr(config, "WEB_FETCH_MAX_BYTES", 10_000)
    _stub_http(monkeypatch, lambda req: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"b" * 5000))
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/big.txt", "max_chars": 600})
    assert out["data"]["truncated"] is True and len(out["data"]["content"]) == 600


@pytest.mark.asyncio
async def test_fetch_is_unavailable_not_an_error_when_switched_off(monkeypatch, web_off):
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/"})
    assert out["success"] is True
    assert out["data"]["available"] is False and "WEB_ACCESS=off" in out["data"]["reason"]


@pytest.mark.asyncio
async def test_fetch_turns_transport_errors_into_results(monkeypatch, web_on):
    def handler(req):
        raise httpx.ConnectError("boom")
    _stub_http(monkeypatch, handler)
    out = await hw.web_fetch(MagicMock(), WS, {"url": "https://example.org/"})
    assert out["success"] is False and "Fetch failed" in out["error"]


# ---------------------------------------------------------------------------
# S3 — web_search: resolver, handler, backends
# ---------------------------------------------------------------------------


def _keys(monkeypatch, *, openrouter=False, composio=False, searxng=False, pin="auto"):
    monkeypatch.setattr(ws, "openrouter_key_available", lambda db, ws_id: openrouter)
    monkeypatch.setattr(ws, "composio_key_available", lambda: composio)
    monkeypatch.setattr(ws, "searxng_available", lambda: searxng)
    monkeypatch.setattr(config, "WEB_SEARCH_PROVIDER", pin)


def test_resolver_order_and_pin(monkeypatch):
    _keys(monkeypatch)
    assert ws.resolve_backend() is None
    _keys(monkeypatch, searxng=True)
    assert ws.resolve_backend() == "searxng"
    _keys(monkeypatch, composio=True, searxng=True)
    assert ws.resolve_backend() == "composio"
    _keys(monkeypatch, openrouter=True, composio=True, searxng=True)
    assert ws.resolve_backend() == "openrouter"
    _keys(monkeypatch, openrouter=True, composio=True, pin="composio")
    assert ws.resolve_backend() == "composio"
    _keys(monkeypatch, openrouter=True, pin="searxng")  # pinned but not configured
    assert ws.resolve_backend() is None
    _keys(monkeypatch, openrouter=True, pin="off")
    assert ws.resolve_backend() is None


@pytest.mark.asyncio
async def test_search_names_the_options_when_nothing_is_configured(monkeypatch, web_on):
    _keys(monkeypatch)
    out = await hw.web_search(MagicMock(), WS, {"query": "fastapi release notes"})
    assert out["success"] is True
    assert out["data"]["available"] is False
    options = " ".join(out["data"]["options"])
    assert "OpenRouter" in options and "COMPOSIO_KEY" in options and "profile search" in options


@pytest.mark.asyncio
async def test_search_is_unavailable_when_switched_off(monkeypatch, web_off):
    out = await hw.web_search(MagicMock(), WS, {"query": "x"})
    assert out["data"]["available"] is False


@pytest.mark.asyncio
async def test_search_delegates_and_filters_denied_hosts(monkeypatch, web_on):
    _keys(monkeypatch, searxng=True)
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("blocked.test",))

    async def fake(query, max_results):
        return [
            {"title": "A", "url": "https://ok.test/a", "snippet": ""},
            {"title": "B", "url": "https://www.blocked.test/b", "snippet": ""},
        ]
    monkeypatch.setattr(ws, "_search_searxng", fake)
    out = await hw.web_search(MagicMock(), WS, {"query": "q", "max_results": 50})
    assert out["success"] is True
    assert out["data"]["backend"] == "searxng"
    assert [r["url"] for r in out["data"]["results"]] == ["https://ok.test/a"]


@pytest.mark.asyncio
async def test_search_backend_failure_is_a_result(monkeypatch, web_on):
    _keys(monkeypatch, searxng=True)

    async def boom(query, max_results):
        raise RuntimeError("SearXNG refused the JSON format")
    monkeypatch.setattr(ws, "_search_searxng", boom)
    out = await hw.web_search(MagicMock(), WS, {"query": "q"})
    assert out["success"] is False and "SearXNG refused" in out["error"] and out["backend"] == "searxng"


@pytest.mark.asyncio
async def test_openrouter_backend_reads_citations_and_never_double_attaches(monkeypatch):
    captured: Dict[str, Any] = {}

    class _Manager:
        async def generate_response(self, messages, tools=None):
            captured["tools"] = tools
            captured["messages"] = messages
            return SimpleNamespace(
                citations=[{"title": "Doc", "url": "https://docs.test/x", "snippet": "s"}],
                content="ignored",
            )

    import core.llm.manager as mgr
    monkeypatch.setattr(mgr, "create_llm_manager", lambda **kw: _Manager())
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("blocked.test",))
    results = await ws._search_openrouter("fastapi", 3, WS)
    assert results == [{"title": "Doc", "url": "https://docs.test/x", "snippet": "s"}]
    (tool,) = captured["tools"]
    assert tool["type"] == "openrouter:web_search" and tool["max_uses"] == 1 and tool["max_results"] == 3
    assert tool["excluded_domains"] == ["blocked.test"]
    assert "fastapi" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_openrouter_backend_falls_back_to_urls_in_text(monkeypatch):
    class _Manager:
        async def generate_response(self, messages, tools=None):
            return SimpleNamespace(citations=None, content="1. FastAPI docs — https://fastapi.tiangolo.com/ — the site\n2. no link here")

    import core.llm.manager as mgr
    monkeypatch.setattr(mgr, "create_llm_manager", lambda **kw: _Manager())
    results = await ws._search_openrouter("fastapi", 5, WS)
    assert results == [{"title": "FastAPI docs", "url": "https://fastapi.tiangolo.com/", "snippet": ""}]


def test_composio_payload_shapes_are_parsed_defensively():
    assert ws._parse_composio({"results": [{"title": "T", "link": "https://a.test", "snippet": "S"}]}) == [
        {"title": "T", "url": "https://a.test", "snippet": "S"}
    ]
    assert ws._parse_composio({"data": {"organic": [{"title": "T", "url": "https://b.test"}]}})[0]["url"] == "https://b.test"
    assert ws._parse_composio([{"url": "javascript:alert(1)"}]) == []
    assert ws._parse_composio("garbage") == []
    assert ws._parse_composio({"unexpected": 1}) == []


@pytest.mark.asyncio
async def test_searxng_backend_explains_a_disabled_json_format(monkeypatch):
    monkeypatch.setattr(config, "SEARXNG_URL", "http://searxng:8080")

    class _Resp:
        status_code = 403
    class _Client:
        def __init__(self, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None): return _Resp()
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError) as excinfo:
        await ws._search_searxng("q", 5)
    assert "settings.yml" in str(excinfo.value)


# ---------------------------------------------------------------------------
# S4 — the OpenRouter route's server tool + citations
# ---------------------------------------------------------------------------


def _client_stub(tool_type="openrouter:web_search"):
    from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider

    stub = SimpleNamespace(spec=SimpleNamespace(web_search_tool=tool_type))
    return lambda tools: OpenAICompatibleProvider._web_search_server_tool(stub, tools)


def test_server_tool_attached_only_when_allowed(monkeypatch, web_on):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_USES_PER_TURN", 3)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULTS", 5)
    attach = _client_stub()
    tool = attach(None)
    assert tool == {"type": "openrouter:web_search", "max_uses": 3, "max_results": 5}
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("blocked.test",))
    assert attach([])["excluded_domains"] == ["blocked.test"]
    # never twice: a caller that already carries one keeps its own settings
    assert attach([{"type": "openrouter:web_search", "max_uses": 1}]) is None
    # a provider without a server tool never gets one
    assert _client_stub(None)(None) is None


def test_server_tool_never_attached_when_switched_off(monkeypatch, web_off):
    assert _client_stub()(None) is None


def test_server_tool_survives_the_tool_sanitiser():
    from core.llm.clients.base import BaseLLMProvider

    out = BaseLLMProvider._sanitize_tools([{"type": "openrouter:web_search", "max_uses": 3}, {"name": "f", "parameters": {}}])
    assert out[0] == {"type": "openrouter:web_search", "max_uses": 3}
    assert out[1]["type"] == "function"


def test_citations_normalise_and_render():
    anns = [
        {"type": "url_citation", "url_citation": {"url": "https://a.test/1", "title": "One", "content": "c1"}},
        {"type": "url_citation", "url_citation": {"url": "https://a.test/1", "title": "dup"}},
        {"url": "https://b.test/2"},
        "junk",
    ]
    cites = citations_from_annotations(anns)
    assert cites == [
        {"title": "One", "url": "https://a.test/1", "snippet": "c1"},
        {"title": "https://b.test/2", "url": "https://b.test/2", "snippet": ""},
    ]
    md = sources_markdown(cites)
    assert md.startswith("\n\n**Sources**\n") and "- [One](https://a.test/1)" in md
    assert sources_markdown([]) == ""


def _chunk(delta: Dict[str, Any], finish=None):
    return SimpleNamespace(
        model="m",
        usage=None,
        choices=[SimpleNamespace(finish_reason=finish, delta=SimpleNamespace(model_dump=lambda: delta))],
    )


def test_stream_assembler_collects_annotations_into_citations_and_sources():
    from core.llm.clients.openai_compatible_client import _StreamAssembler

    a = _StreamAssembler()
    assert a.feed(_chunk({"content": "Paris is the capital."})) == [("text", "Paris is the capital.")]
    a.feed(_chunk({"content": "", "annotations": [{"type": "url_citation", "url_citation": {"url": "https://w.test/paris", "title": "Paris"}}]}, finish="stop"))
    assert a.sources().strip().startswith("**Sources**")
    resp = a.response("openrouter", streamed=True)
    assert resp.citations == [{"title": "Paris", "url": "https://w.test/paris", "snippet": ""}]
    assert resp.content.startswith("Paris is the capital.") and "- [Paris](https://w.test/paris)" in resp.content


def test_stream_assembler_without_annotations_is_unchanged():
    from core.llm.clients.openai_compatible_client import _StreamAssembler

    a = _StreamAssembler()
    a.feed(_chunk({"content": "hi"}, finish="stop"))
    resp = a.response("openrouter", streamed=True)
    assert resp.citations is None and resp.content == "hi" and a.sources() == ""


# ---------------------------------------------------------------------------
# D8 — every dial is declared, forwarded and documented
# ---------------------------------------------------------------------------

_DIALS = (
    "WEB_ACCESS", "WEB_ACCESS_DENY", "WEB_SEARCH_PROVIDER", "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_MAX_USES_PER_TURN", "WEB_SEARCH_OPENROUTER_MODEL", "SEARXNG_URL",
    "WEB_FETCH_MAX_BYTES", "WEB_FETCH_TIMEOUT_SECONDS",
)


def test_every_dial_is_in_the_config_surface_and_on_the_config():
    names = set(json.loads((_ORCH / "reports/config-surface.json").read_text())["settings"])
    for dial in _DIALS:
        assert dial in names, f"{dial} missing from reports/config-surface.json"
        assert hasattr(config, dial), f"config.{dial} does not exist"


def test_compose_forwards_every_dial():
    compose = (_REPO / "docker-compose.yml").read_text(encoding="utf-8")
    for dial in _DIALS:
        assert re.search(rf"^\s*{dial}:\s*\$\{{{dial}", compose, re.MULTILINE), (
            f"docker-compose.yml does not forward {dial} — a .env line for it would be inert"
        )


def test_search_profile_is_opt_in_with_json_enabled():
    compose = (_REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'profiles: ["search"]' in compose
    settings = (_REPO / "envs/searxng/settings.yml").read_text(encoding="utf-8")
    assert "- json" in settings


def test_docs_name_the_switch_the_recommendation_and_the_three_engines():
    quick = (_REPO / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "Recommended: OpenRouter" in quick
    assert "WEB_ACCESS=off" in quick and "WEB_ACCESS_DENY" in quick
    for engine in ("OpenRouter", "Composio", "SearXNG"):
        assert engine in quick
    env = (_REPO / ".env.example").read_text(encoding="utf-8")
    assert "WEB_ACCESS" in env and "SEARXNG_URL" in env
    guide = (_REPO / "docs/getting-started/self-hosting.md").read_text(encoding="utf-8")
    assert "Web access for agents" in guide and "openrouter:web_search" in guide
