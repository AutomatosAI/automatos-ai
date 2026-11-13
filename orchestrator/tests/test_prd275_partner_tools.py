"""PRD-275: partner tools from a pack file. Pure: stubbed HTTP, a stubbed resolver, no database.

A pack declares read-only tools on a partner's own API; the server builds each GET and adds the
key. Locked here, on the path a model's call takes (``runner.run`` and the executor's seam):

* the pack format, and the six guardrails: a call that is not a GET, a call to another host, a
  redirect, invalid parameters, an oversized response, and the key in the tool's input, its
  output or a log line;
* one audit row for each call that read the key, and no answer for a call that cannot be recorded;
* only an agent that lists a pack is offered its tools, on a full turn and on a lean one, and
  only that agent's call runs, behind the executor's own gates;
* nothing is offered or run while ``PARTNER_TOOLS_ENABLED`` is off.
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

import asyncio  # noqa: E402
import copy  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Dict, List  # noqa: E402
from uuid import uuid4  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from config import config  # noqa: E402
import core.security.web_access as wa  # noqa: E402
from modules.tools.partner import answer, client, offer, runner, shape  # noqa: E402
from modules.tools.partner import request as req  # noqa: E402
from modules.tools.partner.pack import PackError, pack_schemas, parse_pack, split_tool_name  # noqa: E402
from modules.tools.partner.request import PartnerRefusal  # noqa: E402

WS = uuid4()
BASE = "https://hub.partner.example"
KEY = "partnerkey" * 3  # not a real key: the value a call must send and must never show
PUBLIC_IP = "93.184.216.34"
PRIVATE_HOSTS = {"internal.partner.example": "10.0.0.9"}
TOOL = "acme_find_orders"
AGENT, OTHER_AGENT, CREDENTIAL = 3, 4, 7


def _pack() -> Dict[str, Any]:
    return {
        "schema": "automatos-partner-pack/v1",
        "name": "acme",
        "title": "Acme Hub",
        "version": "0.1.0",
        "description": "Orders at Acme.",
        "tools": [{
            "name": "find_orders",
            "description": "Search and look up orders.",
            "scope": "orders:read",
            "operations": [
                {"name": "search", "description": "List orders, newest first.", "path": "/api/v1/orders",
                 "parameters": {"type": "object", "properties": {
                     "status": {"type": "string", "enum": ["open", "shipped"], "description": "Orders in this state."},
                     "limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
                 "query": {"sort": "-createdAt"},
                 "response": {"root": "data", "keep": ["id", "status", "customer.name"], "max_items": 3}},
                {"name": "get", "description": "One order by its id.", "path": "/api/v1/orders/{order_id}",
                 "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"]},
                 "response": {"root": "data", "keep": ["id", "status", "total"]}},
            ],
        }],
    }


def _changed(change: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """The pack with one thing changed; ``change`` edits a copy."""
    pack = copy.deepcopy(_pack())
    change(pack)
    return pack


def _search(pack: Dict[str, Any]) -> Dict[str, Any]:
    return pack["tools"][0]["operations"][0]


def _fake_getaddrinfo(host, port, proto=None):
    """No real DNS: a partner's name resolves to one public address, an internal one privately."""
    return [(2, 1, 6, "", (PRIVATE_HOSTS.get(host, PUBLIC_IP), port))]


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ())
    monkeypatch.setattr(wa, "_getaddrinfo", _fake_getaddrinfo)


@pytest.fixture
def stored(monkeypatch):
    """The store without a database: the credential a call reads and the audit rows it writes."""
    state = SimpleNamespace(rows=[], reads=0, credential={"base_url": BASE, "api_key": KEY, "auth_method": "bearer"})

    def credential_fields(db, workspace_id, credential_id):
        state.reads += 1
        return dict(state.credential)

    monkeypatch.setattr(runner, "on_own_session", lambda work, *args, **kwargs: work(None, *args, **kwargs))
    monkeypatch.setattr(runner, "credential_fields", credential_fields)
    monkeypatch.setattr(runner, "record_call",
                        lambda db, credential_id, **row: state.rows.append({"credential_id": credential_id, **row}))
    return state


def _partner(monkeypatch, handler) -> List[httpx.Request]:
    """Route the client into an httpx.MockTransport; returns every request actually sent."""
    sent: List[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    def factory(**kwargs):
        kwargs.pop("verify", None)
        return httpx.AsyncClient(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr(client, "_async_client", factory)
    return sent


def _orders(count: int = 1) -> httpx.Response:
    rows = [{"id": f"SO-{n}", "status": "open", "customer": {"name": "Ada", "email": "ada@acme.example"},
             "internal_notes": "margin 41%"} for n in range(1, count + 1)]
    return httpx.Response(200, json={"data": rows, "page": 1})


def _resolved() -> runner.Resolved:
    return runner.Resolved("acme", parse_pack(_pack()).tools[0], TOOL, CREDENTIAL)


def _run(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(runner.run(_resolved(), arguments, workspace_id=WS, agent_id=AGENT))


# --- the format ------------------------------------------------------------------------------

def test_a_pack_names_its_tools_and_one_argument_picks_the_operation():
    schema = pack_schemas(parse_pack(_pack()))[0]["function"]
    assert schema["name"] == TOOL and split_tool_name(TOOL) == ("acme", "find_orders")
    parameters = schema["parameters"]
    assert parameters["required"] == ["operation"]
    assert parameters["properties"]["operation"]["enum"] == ["search", "get"]
    said = parameters["properties"]["operation"]["description"]
    assert "search: List orders, newest first. (may take status, limit)" in said
    assert "get: One order by its id. (needs order_id)" in said
    assert set(parameters["properties"]) == {"operation", "status", "limit", "order_id"}


def test_a_tool_with_one_operation_takes_that_operations_parameters():
    one = _changed(lambda pack: pack["tools"][0]["operations"].pop(0))
    parameters = pack_schemas(parse_pack(one))[0]["function"]["parameters"]
    assert parameters == {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}


def test_a_stored_pack_reads_back_as_it_was_written():
    pack = parse_pack(_pack())
    assert pack.as_stored()["schema"] == "automatos-partner-pack/v1"
    assert parse_pack(pack.as_stored()) == pack


@pytest.mark.parametrize("change, said", [
    (lambda pack: _search(pack).update(path="https://evil.example/orders"), "path must start with '/'"),
    (lambda pack: _search(pack).update(path="//evil.example/orders"), "path must start with '/'"),
    (lambda pack: _search(pack).update(path="/api/../admin"), "path must start with '/'"),
    (lambda pack: _search(pack).update(path="/api/v1/orders?all=1"), "path must start with '/'"),
    (lambda pack: _search(pack).update(path="/api/v1/orders/{id}"), "path parameter 'id'"),
    (lambda pack: _search(pack).update(query={"status": "open"}), "fixed query names"),
    (lambda pack: _search(pack)["parameters"]["properties"].update(operation={"type": "string"}), "reserved"),
    (lambda pack: _search(pack)["parameters"]["properties"].update(tags={"type": "array"}), "tags"),
    (lambda pack: _search(pack)["response"].update(keep=[]), "keep"),
    (lambda pack: pack.update(name="platform"), "platform's own tools"),
    (lambda pack: pack.update(name="Acme"), "name"),
    (lambda pack: pack.update(name="a" * 21), "name"),      # "<pack>_<tool>" must fit a provider's 64 characters
    (lambda pack: pack.update(description="x" * 250_000), "at most 200000 characters"),
    (lambda pack: pack.update(schema="automatos-partner-pack/v9"), "schema"),
    (lambda pack: pack.update(base_url="https://hub.partner.example"), "base_url"),
    (lambda pack: pack["tools"].append(copy.deepcopy(pack["tools"][0])), "tool names must be unique"),
])
def test_a_pack_that_breaks_the_format_is_refused(change, said):
    with pytest.raises(PackError) as refused:
        parse_pack(_changed(change))
    assert said in str(refused.value)


def test_two_operations_cannot_declare_one_parameter_differently():
    def clash(pack):
        pack["tools"][0]["operations"][1]["parameters"]["properties"]["status"] = {"type": "integer"}

    with pytest.raises(PackError, match="declared differently"):
        parse_pack(_changed(clash))


# --- guardrail 1: GET only -------------------------------------------------------------------

def test_a_pack_cannot_ask_for_another_method():
    with pytest.raises(PackError, match="method"):
        parse_pack(_changed(lambda pack: _search(pack).update(method="POST")))


def test_every_call_is_a_get_and_the_model_cannot_ask_for_another(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    refused = _run({"operation": "search", "method": "POST"})
    assert refused["success"] is False and refused["error_code"] == req.BAD_ARGUMENTS
    assert sent == [] and stored.reads == 0  # nothing was sent and the key was not read

    assert _run({"operation": "search", "status": "open", "limit": 2})["success"] is True
    assert [request.method for request in sent] == ["GET"]


# --- guardrail 2: only the partner's own host, on a path the pack declares -------------------

def test_the_call_goes_to_the_credentials_host_at_the_address_that_was_checked(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    assert _run({"operation": "search", "status": "open", "limit": 2})["success"] is True
    request = sent[0]
    assert request.url.host == PUBLIC_IP and request.headers["host"] == "hub.partner.example"
    assert request.url.scheme == "https" and request.url.path == "/api/v1/orders"
    assert dict(request.url.params) == {"sort": "-createdAt", "status": "open", "limit": "2"}


@pytest.mark.parametrize("order_id", [
    "../admin", "a/b", "//evil.example", "https://evil.example/x", "x?all=1", "x#y", "..", ".", "a b", "%2e%2e", "",
])
def test_a_path_value_cannot_leave_the_declared_path(on, stored, monkeypatch, order_id):
    sent = _partner(monkeypatch, lambda request: _orders())
    refused = _run({"operation": "get", "order_id": order_id})
    assert refused["success"] is False and refused["error_code"] == req.BAD_ARGUMENTS
    assert sent == [] and stored.reads == 0


def test_a_plain_id_is_one_path_segment(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: httpx.Response(200, json={"data": {"id": "SO-1001", "total": 12.5}}))
    result = _run({"operation": "get", "order_id": "SO-1001"})
    assert result["data"] == {"id": "SO-1001", "total": 12.5}
    assert sent[0].url.path == "/api/v1/orders/SO-1001"


@pytest.mark.parametrize("base_url, code", [
    ("http://hub.partner.example", req.BAD_BASE_URL),                 # the key would travel in clear
    ("https://reader:pw@hub.partner.example", req.BAD_BASE_URL),      # an address only
    ("https://hub.partner.example?next=1", req.BAD_BASE_URL),
    ("hub.partner.example", req.BAD_BASE_URL),
    ("https://internal.partner.example", client.HOST_REFUSED),        # resolves into a private range
])
def test_a_base_url_the_platform_will_not_call_sends_nothing(on, stored, monkeypatch, base_url, code):
    sent = _partner(monkeypatch, lambda request: _orders())
    stored.credential["base_url"] = base_url
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == code
    assert sent == [] and "reader:pw" not in json.dumps(refused)


def test_a_base_url_on_another_port_is_called_there_and_named_in_the_host_header(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    stored.credential["base_url"] = "https://hub.partner.example:8443/hub/"
    assert _run({"operation": "search"})["success"] is True
    assert sent[0].url.port == 8443 and sent[0].headers["host"] == "hub.partner.example:8443"
    assert sent[0].url.path == "/hub/api/v1/orders"


def test_a_url_off_the_base_urls_origin_is_not_the_same_origin():
    assert req.same_origin(BASE, f"{BASE}/api/v1/orders?x=1")
    assert not req.same_origin(BASE, "https://evil.example/api/v1/orders")
    assert not req.same_origin(BASE, "http://hub.partner.example/api/v1/orders")
    assert not req.same_origin(BASE, "https://hub.partner.example:8443/api/v1/orders")


# --- guardrail 3: a redirect is never followed -----------------------------------------------

def test_a_redirect_is_refused_and_its_location_is_never_fetched(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: httpx.Response(302, headers={"location": "https://evil.example/take"}))
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == client.REDIRECT_REFUSED
    assert len(sent) == 1 and sent[0].headers["host"] == "hub.partner.example"
    assert stored.rows[0]["status_code"] == 302 and stored.rows[0]["success"] is False


# --- guardrail 4: invalid parameters ---------------------------------------------------------

@pytest.mark.parametrize("arguments", [
    {"operation": "search", "status": "lost"},        # not in the enum
    {"operation": "search", "limit": "ten"},          # not an integer
    {"operation": "search", "limit": 500},            # over the maximum
    {"operation": "search", "colour": "red"},         # not a parameter of the operation
    {"operation": "search", "order_id": "SO-1"},      # another operation's parameter
    {"operation": "get"},                             # a required parameter is missing
    {"operation": "delete"},                          # not an operation
    {"status": "open"},                               # no operation named
])
def test_arguments_the_operation_does_not_take_are_refused_before_the_key_is_read(on, stored, monkeypatch, arguments):
    sent = _partner(monkeypatch, lambda request: _orders())
    refused = _run(arguments)
    assert refused["success"] is False and refused["error_code"] == req.BAD_ARGUMENTS
    assert sent == [] and stored.reads == 0 and stored.rows == []


def test_a_value_longer_than_a_phrase_is_refused():
    free_text = _changed(lambda pack: _search(pack)["parameters"]["properties"].update(search={"type": "string"}))
    operation = parse_pack(free_text).tools[0].operations[0]
    assert req.checked_arguments(operation, {"search": "late orders"}) == {"search": "late orders"}
    with pytest.raises(PartnerRefusal) as refused:
        req.checked_arguments(operation, {"search": "x" * (req.MAX_ARGUMENT_CHARS + 1)})
    assert refused.value.code == req.BAD_ARGUMENTS


# --- guardrail 5: an oversized answer is capped ----------------------------------------------

def test_a_long_list_is_cut_to_the_packs_cap_and_only_the_listed_fields_are_kept(on, stored, monkeypatch):
    _partner(monkeypatch, lambda request: _orders(40))
    result = _run({"operation": "search"})
    assert result["data"] == [{"id": f"SO-{n}", "status": "open", "customer": {"name": "Ada"}} for n in (1, 2, 3)]
    assert result["metadata"] == {"source": "GET /api/v1/orders", "status_code": 200, "truncated": True,
                                  "items": 3, "total_items": 40}
    assert "internal_notes" not in json.dumps(result) and "ada@acme.example" not in json.dumps(result)


def test_a_body_over_the_byte_cap_is_refused_unread(on, stored, monkeypatch):
    monkeypatch.setattr(client, "MAX_RESPONSE_BYTES", 2_000)
    _partner(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "x" * 50_000}]}))
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == client.RESPONSE_TOO_LARGE
    assert len(json.dumps(refused)) < 1_000


def test_a_partner_that_answers_too_slowly_is_cut_off_by_one_deadline(on, stored, monkeypatch):
    async def dripping(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return _orders()

    monkeypatch.setattr(client, "TOTAL_SECONDS", 0.05)
    _partner(monkeypatch, dripping)
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == client.TIMED_OUT
    assert stored.rows[0]["error_code"] == client.TIMED_OUT and stored.rows[0]["duration_ms"] < 2_000


def test_records_are_dropped_from_the_end_to_fit_the_character_cap():
    response = parse_pack(_changed(lambda pack: _search(pack)["response"].update(max_items=200, max_chars=200)))
    rows = [{"id": f"SO-{n:04d}", "status": "open", "customer": {"name": "Ada Lovelace"}} for n in range(50)]
    kept = shape.shaped({"data": rows}, response.tools[0].operations[0].response)
    assert 0 < kept.items < 50 and kept.total_items == 50 and kept.truncated is True
    assert kept.data == [{key: row[key] for key in ("id", "status", "customer")} for row in rows[:kept.items]]
    assert len(json.dumps(kept.data)) <= 200


def test_one_record_too_large_to_return_is_refused_whole():
    response = parse_pack(_pack()).tools[0].operations[1].response
    with pytest.raises(PartnerRefusal) as refused:
        shape.shaped({"data": {"id": "SO-1", "status": "x" * 30_000}}, response)
    assert refused.value.code == shape.ANSWER_TOO_LARGE


def test_a_field_listed_whole_is_kept_whole_and_a_list_is_pruned_record_by_record():
    tree = shape.keep_tree(["id", "lines.sku", "customer", "customer.name"])
    assert tree == {"id": None, "customer": None, "lines": {"sku": None}}
    order = {"id": 1, "secret": "x", "customer": {"name": "Ada", "tier": "gold"},
             "lines": [{"sku": "A", "cost": 3}, {"sku": "B", "cost": 4}]}
    assert shape.pruned(order, tree) == {"id": 1, "customer": {"name": "Ada", "tier": "gold"},
                                         "lines": [{"sku": "A"}, {"sku": "B"}]}


# --- guardrail 6: the key is sent in a header and appears nowhere else -----------------------

def test_the_key_is_in_the_header_and_never_in_the_input_the_output_or_a_log_line(on, stored, monkeypatch, caplog):
    def echoing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "SO-1", "status": "open",
                                                   "debug": request.headers.get("authorization")}]})

    sent = _partner(monkeypatch, echoing)
    caplog.set_level(logging.DEBUG)
    result = _run({"operation": "search"})
    assert result["success"] is True
    assert sent[0].headers["authorization"] == f"Bearer {KEY}"         # it was used
    assert KEY not in str(sent[0].url)                                 # never in the address
    assert KEY not in json.dumps(pack_schemas(parse_pack(_pack())))    # the input the model is shown
    assert KEY not in json.dumps(result)                               # the output, an echo included
    assert KEY not in answer.answer_text(result, TOOL)                 # what the model reads
    assert KEY not in json.dumps(stored.rows)                          # the audit row
    assert KEY not in caplog.text                                      # every log line of the call


def test_a_failure_that_quotes_the_key_is_logged_without_it(on, stored, monkeypatch, caplog):
    def choking(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot write header authorization: Bearer {KEY}")

    _partner(monkeypatch, choking)
    caplog.set_level(logging.DEBUG)
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == client.UNREACHABLE
    assert KEY not in json.dumps(refused) and KEY not in caplog.text and KEY not in json.dumps(stored.rows)


def test_the_key_goes_in_the_header_the_credential_names(on, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    stored.credential.update(auth_method="header", header_name="X-Hub-Key")
    assert _run({"operation": "search"})["success"] is True
    assert sent[0].headers["x-hub-key"] == KEY and "authorization" not in sent[0].headers


@pytest.mark.parametrize("fields", [
    {"auth_method": "query"},                               # the key would be in the address
    {"auth_method": "basic"},
    {"auth_method": "header", "header_name": "Host"},       # a header the client owns
    {"auth_method": "header", "header_name": "X-HTTP-Method-Override", "api_key": "DELETE"},  # not a plain GET
    {"auth_method": "header", "header_name": "X-Forwarded-Token"},                            # a forwarded address
    {"auth_method": "header", "header_name": "Authorization"},                                # that is "bearer"
    {"api_key": "two\r\nX-Injected: 1"},                    # would rewrite the request
    {"api_key": ""},
])
def test_a_credential_that_cannot_carry_the_key_safely_sends_nothing(on, stored, monkeypatch, fields):
    sent = _partner(monkeypatch, lambda request: _orders())
    stored.credential.update(fields)
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == req.BAD_CREDENTIAL and sent == []


# --- what the partner answers ----------------------------------------------------------------

@pytest.mark.parametrize("respond, code", [
    (lambda: httpx.Response(404, json={"error": "not found"}), runner.NOT_FOUND),   # another organisation's order
    (lambda: httpx.Response(403, json={"error": "forbidden"}), runner.KEY_REFUSED),
    (lambda: httpx.Response(429), runner.RATE_LIMITED),
    (lambda: httpx.Response(500, text="stack trace"), runner.HTTP_ERROR),
    (lambda: httpx.Response(200, text="<html>login</html>"), shape.UNEXPECTED_ANSWER),
    (lambda: httpx.Response(200, json={"items": []}), shape.UNEXPECTED_ANSWER),     # no "data" to read
    (lambda: httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(b'{"data": []}')),
     client.UNREADABLE),
])
def test_an_answer_that_is_not_the_records_is_a_refusal_the_model_can_read(on, stored, monkeypatch, respond, code):
    _partner(monkeypatch, lambda request: respond())
    refused = _run({"operation": "search"})
    assert refused["success"] is False and refused["error_code"] == code
    assert refused[answer.PARTNER_KEY] == {"pack": "acme", "tool": TOOL, "operation": "search"}
    assert "stack trace" not in json.dumps(refused) and "login" not in json.dumps(refused)


def test_a_refused_key_names_the_scope_the_tool_needs(on, stored, monkeypatch):
    _partner(monkeypatch, lambda request: httpx.Response(403))
    assert "orders:read" in _run({"operation": "search"})["error"]


# --- the audit row ---------------------------------------------------------------------------

def test_each_call_that_read_the_key_writes_one_audit_row(on, stored, monkeypatch):
    _partner(monkeypatch, lambda request: _orders())
    assert _run({"operation": "search"})["success"] is True
    row = stored.rows[0]
    assert len(stored.rows) == 1 and isinstance(row.pop("duration_ms"), int)
    assert row == {"credential_id": CREDENTIAL, "tool": TOOL, "operation": "search", "status_code": 200,
                   "success": True, "agent_id": AGENT, "error_code": None}


def test_a_call_that_cannot_be_recorded_gives_no_answer(on, stored, monkeypatch):
    def failing(db, credential_id, **row):
        raise RuntimeError("audit table is gone")

    _partner(monkeypatch, lambda request: _orders())
    monkeypatch.setattr(runner, "record_call", failing)
    withheld = _run({"operation": "search"})
    assert withheld["success"] is False and withheld["error_code"] == runner.AUDIT_FAILED and "data" not in withheld


def test_a_credential_that_is_gone_sends_nothing_and_records_nothing(on, stored, monkeypatch):
    def gone(db, workspace_id, credential_id):
        raise PartnerRefusal("partner_no_credential", "The pack has no credential in this workspace.")

    sent = _partner(monkeypatch, lambda request: _orders())
    monkeypatch.setattr(runner, "credential_fields", gone)
    refused = _run({"operation": "search"})
    assert refused["error_code"] == "partner_no_credential" and sent == [] and stored.rows == []


# --- who is offered a pack's tools -----------------------------------------------------------

@pytest.fixture
def installed(monkeypatch):
    """A workspace with the pack installed for one agent (``AGENT``), which lists it."""
    packs = [SimpleNamespace(pack=parse_pack(_pack()), credential_id=CREDENTIAL, agents=(AGENT,))]
    monkeypatch.setattr(offer, "granted_packs", lambda db, agent_id, workspace_id: ("acme",) if agent_id == AGENT else ())
    monkeypatch.setattr(offer, "installed_packs", lambda db, workspace_id, names=None: packs)
    return packs


def _named(schemas: List[Dict[str, Any]]) -> List[Any]:
    return [offer.schema_name(schema) for schema in schemas]


BUILT_IN = [{"type": "function", "function": {"name": "search_knowledge", "parameters": {}}}]


def test_only_an_agent_that_lists_the_pack_is_offered_its_tools(on, installed):
    assert _named(offer.with_partner_tools(BUILT_IN, object(), AGENT, WS)) == ["search_knowledge", TOOL]
    assert offer.with_partner_tools(BUILT_IN, object(), OTHER_AGENT, WS) == BUILT_IN


def test_an_agent_that_lists_the_pack_without_the_installs_leave_is_offered_nothing(on, installed, monkeypatch):
    # An editor may edit an agent (agents:update) but not install a pack (workspace:manage): listing the
    # pack on an agent the install does not name gives it nothing, on a full turn or a lean one.
    from consumers.chatbot import lean_turn

    monkeypatch.setattr(offer, "granted_packs", lambda db, agent_id, workspace_id: ("acme",))
    assert offer.with_partner_tools(BUILT_IN, object(), OTHER_AGENT, WS) == BUILT_IN
    edited = SimpleNamespace(**{**vars(_lean_agent()), "id": OTHER_AGENT})
    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)
    assert lean_turn.lean_grant_for(_db_with(edited), OTHER_AGENT) == lean_turn.granted_names(edited)
    assert offer.granted_partner_schemas(_db_with(edited), OTHER_AGENT, [TOOL]) == []


def test_a_name_already_offered_is_never_offered_twice(on, installed):
    taken = [*BUILT_IN, {"type": "function", "function": {"name": TOOL}}]
    assert _named(offer.with_partner_tools(taken, object(), AGENT, WS)) == ["search_knowledge", TOOL]


def test_nothing_is_offered_while_partner_tools_are_off(installed, monkeypatch):
    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", False, raising=False)
    assert offer.with_partner_tools(BUILT_IN, object(), AGENT, WS) is BUILT_IN


def test_a_pack_that_cannot_be_read_costs_the_turn_nothing(on, installed, monkeypatch):
    def broken(db, workspace_id, names=None):
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr(offer, "installed_packs", broken)
    assert offer.with_partner_tools(BUILT_IN, None, AGENT, WS) == BUILT_IN
    assert offer.with_partner_tools(BUILT_IN, SimpleNamespace(), AGENT, WS) == BUILT_IN


def test_the_tool_sources_seam_adds_the_agents_partner_tools(on, installed):
    @offer.offers_partner_tools
    def core(**kwargs):
        return list(BUILT_IN)

    offered = core(agent_id=AGENT, session_used=object(), workspace_id=WS, query=None)
    assert _named(offered) == ["search_knowledge", TOOL]
    assert _named(core(agent_id=OTHER_AGENT, session_used=object(), workspace_id=WS)) == ["search_knowledge"]


def test_the_router_and_the_executor_and_the_formatter_carry_the_seams():
    from modules.tools import tool_router
    from modules.tools.execution.unified_executor import UnifiedToolExecutor
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    def layers(fn):
        while fn is not None:
            yield fn.__code__
            fn = getattr(fn, "__wrapped__", None)

    def wrapper_of(decorator):
        return decorator(lambda *args, **kwargs: None).__code__

    assert wrapper_of(offer.offers_partner_tools) in layers(tool_router._get_tools_for_agent_core)
    assert wrapper_of(runner.runs_partner_tools) in layers(UnifiedToolExecutor.execute_tool)
    assert wrapper_of(answer.reads_a_partner_answer_whole) in layers(ToolResultFormatter.format_for_llm)


def test_a_pack_cannot_take_the_name_of_one_of_the_platforms_own_tools():
    def named(pack_name: str, tool_name: str):
        def rename(pack):
            pack["name"], pack["tools"][0]["name"] = pack_name, tool_name

        return parse_pack(_changed(rename))

    assert offer.taken_names(None, named("search", "documents")) == ["search_documents"]   # an alias the executor reads
    assert offer.taken_names(None, named("query", "database")) == ["query_database"]       # a tool in the registry
    assert offer.taken_names(None, parse_pack(_pack())) == []


def test_an_agents_partner_packs_must_be_a_list_of_pack_names(on, monkeypatch):
    assert offer.partner_pack_errors({}) == [] and offer.partner_pack_errors({"partner_packs": ["acme"]}) == []
    for wrong in ("acme", ["Acme"], [7], ["acme\n"], ["acme"] * 11, {"acme": True}):
        assert offer.partner_pack_errors({"partner_packs": wrong}), wrong
    assert offer.packs_named({"partner_packs": ["acme", "acme", "Bad Name", "acme\n", 7]}) == ("acme",)
    assert offer.packs_named({"partner_packs": "acme"}) == () and offer.packs_named(None) == ()
    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", False, raising=False)
    assert offer.partner_pack_errors({"partner_packs": "acme"}) == []  # off: the key is not read, so not judged


# --- a lean turn ------------------------------------------------------------------------------

def _lean_agent() -> SimpleNamespace:
    return SimpleNamespace(id=AGENT, workspace_id=WS, is_system_agent=False, configuration={
        "runtime_profile": "lean", "session_tool_groups": [], "partner_packs": ["acme"]})


def _db_with(agent: Any) -> SimpleNamespace:
    """A session whose one agent query answers ``agent``."""
    return SimpleNamespace(query=lambda model: SimpleNamespace(
        filter=lambda *conditions: SimpleNamespace(first=lambda: agent)))


def test_a_lean_agents_grants_name_its_partner_tools(on, installed, monkeypatch):
    from consumers.chatbot import lean_turn

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)
    agent = _lean_agent()
    grants = lean_turn.lean_grant_for(_db_with(agent), AGENT)
    assert grants[-1] == TOOL and grants[:-1] == lean_turn.granted_names(agent)


def test_a_lean_turn_is_offered_the_partner_tools_it_is_granted(on, installed, monkeypatch):
    from consumers.chatbot import lean_turn
    from modules.tools.partner import store

    agent = _lean_agent()
    monkeypatch.setattr(store, "on_own_session", lambda work, *args, **kwargs: work(_db_with(agent), *args, **kwargs))
    monkeypatch.setattr(lean_turn, "lean_tool_surface", lambda granted, **kwargs: [])

    @lean_turn.offers_a_lean_turn_its_grants
    async def get_tools(chat, agent_id, **kwargs):
        return ["the full surface"]

    def chat(granted):
        return SimpleNamespace(_lean_actions=granted, db=None)

    assert _named(asyncio.run(get_tools(chat([TOOL]), AGENT, is_super_admin=False))) == [TOOL]
    assert asyncio.run(get_tools(chat(["platform_list_playbooks"]), AGENT, is_super_admin=False)) == []
    assert asyncio.run(get_tools(chat(None), AGENT)) == ["the full surface"]  # a full turn as it was


# --- the executor's seam ----------------------------------------------------------------------

@pytest.fixture
def executor(monkeypatch, stored):
    """An executor with no tools of its own, its telemetry captured, and a lookup that knows
    one agent (``AGENT``) lists the pack."""
    from modules.tools.execution import telemetry, unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    fired: List[Dict[str, Any]] = []
    monkeypatch.setattr(telemetry, "fire_telemetry", lambda **row: fired.append(row))
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **row: fired.append(row))
    monkeypatch.setattr(unified_executor, "capture_tool_outcome", lambda **outcome: None)
    monkeypatch.setattr(unified_executor, "fire_tool_trace", lambda **trace: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(runner, "resolve", lambda db, workspace_id, agent_id, name:
                        _resolved() if (agent_id, name) == (AGENT, TOOL) else None)
    ex = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    ex.db, ex.composio_actions, ex.tool_routes = None, {}, {}
    ex._tool_registry = SimpleNamespace(get_tool=lambda name: None)
    ex.fired = fired
    return ex


def _call(ex, agent_id: int = AGENT, arguments: Any = None) -> Dict[str, Any]:
    arguments = {"operation": "search"} if arguments is None else arguments
    return asyncio.run(ex.execute_tool(TOOL, arguments, agent_id=agent_id, workspace_id=WS))


def test_a_listed_agents_call_runs_on_the_executors_seam(on, executor, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    result = _call(executor)
    assert result["success"] is True and result["data"][0]["id"] == "SO-1" and len(sent) == 1
    assert [(row["tool_name"], row["result"]["success"]) for row in executor.fired] == [(TOOL, True)]
    assert executor.fired[0]["parameters"] == {"operation": "search"} and executor.fired[0]["agent_id"] == AGENT


def test_an_agent_that_does_not_list_the_pack_has_no_such_tool(on, executor, stored, monkeypatch):
    sent = _partner(monkeypatch, lambda request: _orders())
    result = _call(executor, agent_id=OTHER_AGENT)
    assert result["success"] is False and "Unknown tool" in result["error"]
    assert sent == [] and stored.reads == 0 and stored.rows == []


def test_while_partner_tools_are_off_the_name_is_unknown_and_nothing_is_looked_up(executor, stored, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("looked up a partner tool while partner tools are off")

    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", False, raising=False)
    monkeypatch.setattr(runner, "resolve", never)
    sent = _partner(monkeypatch, lambda request: _orders())
    result = _call(executor)
    assert result["success"] is False and "Unknown tool" in result["error"] and sent == []


def test_a_lean_turn_runs_a_partner_tool_only_when_it_is_granted(on, executor, stored, monkeypatch):
    from core.security.lean_grants import LEAN_REFUSAL, lean_grants

    sent = _partner(monkeypatch, lambda request: _orders())
    with lean_grants(["search_knowledge"]):
        refused = _call(executor)
    assert refused["permission_denied"] is True and refused["error"] == LEAN_REFUSAL
    assert sent == [] and stored.reads == 0
    with lean_grants(["search_knowledge", TOOL]):
        assert _call(executor)["success"] is True
    assert [row["result"].get("success") for row in executor.fired] == [False, True]  # both calls are counted


def test_a_name_the_executor_routes_itself_is_never_a_partner_tool(on, executor):
    executor.tool_routes = {"acme_find_orders": object()}
    assert runner.might_be_partner_tool(executor, TOOL) is False
    executor.tool_routes = {}
    assert runner.might_be_partner_tool(executor, TOOL) is True
    for name in ("platform_list_agents", "workspace_read_file", "search_documents", "GMAIL_SEND_EMAIL",
                 "composio_execute", "scratchpad", None):
        assert runner.might_be_partner_tool(executor, name) is False, name


# --- how the model reads an answer ------------------------------------------------------------

def test_the_model_reads_every_record_a_partner_tool_returns():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    operation = parse_pack(_pack()).tools[0].operations[0]
    rows = [{"id": f"SO-{n}"} for n in range(1, 7)]
    result = answer.succeeded("acme", TOOL, operation, 200, shape.Shaped(rows, items=6, total_items=6))
    text = ToolResultFormatter.format_for_llm(result, TOOL)
    assert text.startswith(f"Tool: {TOOL}\nStatus: success")
    assert all(f'"SO-{n}"' in text for n in range(1, 7)) and '"source": "GET /api/v1/orders"' in text
    refusal = answer.refused("acme", TOOL, PartnerRefusal(runner.NOT_FOUND, "The partner has no such record (HTTP 404)."))
    assert ToolResultFormatter.format_for_llm(refusal, TOOL) == f"Tool {TOOL} failed: The partner has no such record (HTTP 404)."
    cut = ToolResultFormatter.format_for_llm(result, TOOL, 120)  # a caller with a smaller budget is told of the cut
    assert len(cut) == 120 and cut.endswith(answer.CUT_NOTE) and cut.startswith(text[:60])


# --- installing a pack ------------------------------------------------------------------------

@pytest.fixture
def routes(monkeypatch):
    import api.partner_packs as api

    installs: List[Any] = []

    def install(db, workspace_id, pack, credential_id, agent_ids=()):
        installs.append((workspace_id, pack.name, credential_id, tuple(agent_ids)))
        return SimpleNamespace(pack=pack, credential_id=credential_id, agents=tuple(agent_ids))

    monkeypatch.setattr(api, "install_pack", install)
    monkeypatch.setattr(api, "taken_names", lambda db, pack: [])
    return SimpleNamespace(api=api, installs=installs, ctx=SimpleNamespace(workspace_id=WS))


def _install(routes, name: str = "acme", pack: Any = None, agents: Any = (AGENT,)) -> Dict[str, Any]:
    body = routes.api.InstallRequest(pack=_pack() if pack is None else pack, credential_id=CREDENTIAL,
                                     agents=list(agents))
    return routes.api.install_partner_pack(name, body, ctx=routes.ctx, db=object())


def test_a_manager_installs_a_pack_with_the_credential_its_calls_use(on, routes):
    described = _install(routes)
    assert routes.installs == [(WS, "acme", CREDENTIAL, (AGENT,))] and described["agents"] == [AGENT]
    assert described["tools"] == [{"name": TOOL, "description": "Search and look up orders.", "scope": "orders:read",
                                   "operations": ["search", "get"]}]
    assert described["credential_id"] == CREDENTIAL and KEY not in json.dumps(described)


def test_a_pack_installed_for_no_agent_is_installed_for_no_one(on, routes):
    body = routes.api.InstallRequest(pack=_pack(), credential_id=CREDENTIAL)  # no agents named
    assert routes.api.install_partner_pack("acme", body, ctx=routes.ctx, db=object())["agents"] == []
    with pytest.raises(ValueError):  # more agents than a pack may be allowed to
        routes.api.InstallRequest(pack=_pack(), credential_id=CREDENTIAL, agents=list(range(1, 52)))


def test_a_pack_that_cannot_be_installed_says_why(on, routes, monkeypatch):
    for name, pack, said in (("other", None, "names itself 'acme'"),
                             ("acme", _changed(lambda p: _search(p).update(method="POST")), "method")):
        with pytest.raises(HTTPException) as refused:
            _install(routes, name, pack)
        assert refused.value.status_code == 422 and said in refused.value.detail
    monkeypatch.setattr(routes.api, "taken_names", lambda db, pack: [TOOL])
    with pytest.raises(HTTPException) as clash:
        _install(routes)
    assert clash.value.status_code == 422 and TOOL in clash.value.detail and routes.installs == []


def test_the_routes_answer_404_while_partner_tools_are_off(routes, monkeypatch):
    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", False, raising=False)
    for call in (lambda: _install(routes),
                 lambda: routes.api.list_partner_packs(ctx=routes.ctx, db=object()),
                 lambda: routes.api.remove_partner_pack("acme", ctx=routes.ctx, db=object())):
        with pytest.raises(HTTPException) as off:
            call()
        assert off.value.status_code == 404
    assert routes.installs == []


def test_every_route_is_for_the_workspaces_managers():
    import api.partner_packs as api
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    served = {(method, route.path) for route in api.router.routes for method in route.methods - {"HEAD", "OPTIONS"}}
    assert served == {("GET", "/api/partner-packs"), ("PUT", "/api/partner-packs/{name}"),
                      ("DELETE", "/api/partner-packs/{name}")}
    for route in api.router.routes:
        gates = [getattr(dependency.call, PERMISSION_MARKER_ATTR, None) for dependency in route.dependant.dependencies]
        assert "workspace:manage" in gates, route.path
