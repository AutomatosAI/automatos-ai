"""PRD-262, the 19 Oct breakdown's row #20: core.qdrant speaks Qdrant's REST API as qdrant-client did.

qdrant-client served the memory stores over Qdrant's REST API and brought grpcio and
portalocker into the image for a gRPC transport the code never used. core.qdrant's
QdrantRestClient replaces it on the httpx the image already has. Every method the code
calls is driven here through httpx's MockTransport: the request it sends (method, path,
``wait``, body, the ``api-key`` header) and what it returns. Also checked:

- the errors keep qdrant-client's shape, so core/qdrant_writes.py still tells a timeout
  (a Qdrant 408, the client's own timeout) from anything else, and an F103 write that
  times out is tried once more;
- payload values JSON has no rule for are written as qdrant-client wrote them, a
  payload's None stays null, and a filter's clause left None is not sent;
- an answer that is not Qdrant's shape is refused where it enters;
- every client method the stores, the snapshot job and the orphan sweep call exists;
- qdrant-client is gone from requirements.in and requirements.lock with grpcio and
  portalocker, and stays only in requirements-dev.txt, for the PRD-108 proofs that search
  with its in-memory engine (test.yml's "no test tools in the production image" step
  then checks it is not in the built image).

MockTransport only; no Qdrant, no network.
"""
from __future__ import annotations

import ast
import asyncio
import datetime as dt
import decimal
import enum
import json
import re
import uuid
from pathlib import Path
from typing import Any, List, Optional

import httpx
import pytest

from core import qdrant_writes
from core.qdrant import (
    CountResult,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    QdrantRestClient,
    Record,
    ResponseHandlingException,
    ScoredPoint,
    SnapshotDescription,
    UnexpectedResponse,
    VectorParams,
)
from core.qdrant.client import encode

ORCH = Path(__file__).resolve().parents[1]
URL = "http://qdrant.test:6333"
COLLECTION = "field_memory"
FIELD = Filter(must=[FieldCondition(key="field_id", match=MatchValue(value="f-1"))])
FIELD_JSON = {"must": [{"key": "field_id", "match": {"value": "f-1"}}]}
POINT_ID = "5c56c793-69f3-4fbf-87e6-c4bf54c28c26"


class _Qdrant:
    """A MockTransport handler: records every request and answers from its queue."""

    def __init__(self, *answers: Any) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def sent(self) -> tuple:
        """(method, path, query, body) of the one request sent."""
        assert len(self.requests) == 1, self.requests
        request = self.requests[0]
        body = json.loads(request.content) if request.content else None
        return request.method, request.url.path, dict(request.url.params), body


def _ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})


def _call(qdrant: _Qdrant, method: str, *args: Any, api_key: Optional[str] = None, **kwargs: Any) -> Any:
    async def go():
        client = QdrantRestClient(URL, api_key=api_key, timeout=30, transport=httpx.MockTransport(qdrant))
        try:
            return await getattr(client, method)(*args, **kwargs)
        finally:
            await client.aclose()
    return asyncio.run(go())


# --------------------------------------------------------------------------- collections


@pytest.mark.parametrize("exists", [True, False])
def test_collection_exists(exists):
    qdrant = _Qdrant(_ok({"exists": exists}))
    assert _call(qdrant, "collection_exists", COLLECTION) is exists
    assert qdrant.sent == ("GET", "/collections/field_memory/exists", {}, None)


def test_get_collections():
    qdrant = _Qdrant(_ok({"collections": [{"name": "field_memory"}, {"name": "durable_memory"}]}))
    found = _call(qdrant, "get_collections")
    assert [collection.name for collection in found.collections] == ["field_memory", "durable_memory"]
    assert qdrant.sent == ("GET", "/collections", {}, None)


def test_create_collection():
    qdrant = _Qdrant(_ok(True))
    assert _call(qdrant, "create_collection", collection_name=COLLECTION,
                 vectors_config=VectorParams(size=2048, distance=Distance.COSINE, on_disk=True),
                 hnsw_config=HnswConfigDiff(on_disk=True), on_disk_payload=True) is True
    assert qdrant.sent == ("PUT", "/collections/field_memory", {}, {
        "vectors": {"size": 2048, "distance": "Cosine", "on_disk": True},
        "hnsw_config": {"on_disk": True}, "on_disk_payload": True})


def test_delete_collection():
    qdrant = _Qdrant(_ok(True))
    assert _call(qdrant, "delete_collection", "field_0f2c") is True
    assert qdrant.sent == ("DELETE", "/collections/field_0f2c", {}, None)


@pytest.mark.parametrize("schema,sent", [(PayloadSchemaType.KEYWORD, "keyword"), (PayloadSchemaType.INTEGER, "integer")])
def test_create_payload_index_waits(schema, sent):
    qdrant = _Qdrant(_ok({"operation_id": 1, "status": "completed"}))
    _call(qdrant, "create_payload_index", collection_name=COLLECTION, field_name="agent_id", field_schema=schema,
          wait=True)
    assert qdrant.sent == ("PUT", "/collections/field_memory/index", {"wait": "true"},
                           {"field_name": "agent_id", "field_schema": sent})


# --------------------------------------------------------------------------- points


def test_upsert_waits_and_keeps_a_null_payload_value():
    qdrant = _Qdrant(_ok({"operation_id": 2, "status": "completed"}))
    point = PointStruct(id=POINT_ID, vector=[0.1, 0.2], payload={"field_id": "f-1", "expired_at": None})
    _call(qdrant, "upsert", collection_name=COLLECTION, points=[point])
    assert qdrant.sent == ("PUT", "/collections/field_memory/points", {"wait": "true"},
                           {"points": [{"id": POINT_ID, "vector": [0.1, 0.2],
                                        "payload": {"field_id": "f-1", "expired_at": None}}]})


@pytest.mark.parametrize("selector,body", [
    (FilterSelector(filter=FIELD), {"filter": FIELD_JSON}),
    (["a", "b"], {"points": ["a", "b"]}),
])
def test_delete_by_filter_or_by_ids(selector, body):
    qdrant = _Qdrant(_ok({"operation_id": 3, "status": "completed"}))
    _call(qdrant, "delete", collection_name=COLLECTION, points_selector=selector)
    assert qdrant.sent == ("POST", "/collections/field_memory/points/delete", {"wait": "true"}, body)


@pytest.mark.parametrize("points,selected", [([POINT_ID], {"points": [POINT_ID]}), (FIELD, {"filter": FIELD_JSON})])
def test_set_payload_on_ids_or_a_filter(points, selected):
    qdrant = _Qdrant(_ok({"operation_id": 4, "status": "completed"}))
    _call(qdrant, "set_payload", collection_name=COLLECTION, payload={"access_count": 2}, points=points)
    assert qdrant.sent == ("POST", "/collections/field_memory/points/payload", {"wait": "true"},
                           {"payload": {"access_count": 2}, **selected})


def test_a_selector_of_another_kind_is_refused():
    with pytest.raises(TypeError, match="select points with"):
        _call(_Qdrant(), "delete", collection_name=COLLECTION, points_selector="f-1")


def test_retrieve():
    qdrant = _Qdrant(_ok([{"id": POINT_ID, "payload": {"access_count": 1}, "vector": None}]))
    found = _call(qdrant, "retrieve", COLLECTION, ids=[POINT_ID])
    assert found == [Record(id=POINT_ID, payload={"access_count": 1})]
    assert qdrant.sent == ("POST", "/collections/field_memory/points", {},
                           {"ids": [POINT_ID], "with_payload": True, "with_vector": False})


def test_scroll_a_page_from_an_offset():
    qdrant = _Qdrant(_ok({"points": [{"id": 7, "payload": {"key": "k"}}], "next_page_offset": 8}))
    points, offset = _call(qdrant, "scroll", collection_name=COLLECTION, scroll_filter=FIELD, limit=256, offset=6,
                           with_payload=True, with_vectors=False)
    assert (points, offset) == ([Record(id=7, payload={"key": "k"})], 8)
    assert qdrant.sent == ("POST", "/collections/field_memory/points/scroll", {}, {
        "limit": 256, "offset": 6, "with_payload": True, "with_vector": False, "filter": FIELD_JSON})


def test_scroll_defaults_as_qdrant_client_did_and_selects_payload_keys():
    qdrant = _Qdrant(_ok({"points": [], "next_page_offset": None}))
    assert _call(qdrant, "scroll", collection_name=COLLECTION, with_payload=["field_id"]) == ([], None)
    assert qdrant.sent[3] == {"limit": 10, "with_payload": ["field_id"], "with_vector": False}


def test_count():
    qdrant = _Qdrant(_ok({"count": 3}))
    assert _call(qdrant, "count", collection_name=COLLECTION, count_filter=FIELD, exact=True) == CountResult(3)
    assert qdrant.sent == ("POST", "/collections/field_memory/points/count", {}, {"exact": True, "filter": FIELD_JSON})


def test_query_points():
    hit = {"id": POINT_ID, "version": 4, "score": 0.83, "payload": {"key": "k"}, "vector": None}
    qdrant = _Qdrant(_ok({"points": [hit]}))
    found = _call(qdrant, "query_points", collection_name=COLLECTION, query=[0.1, 0.2], query_filter=FIELD, limit=21)
    assert found.points == [ScoredPoint(id=POINT_ID, score=0.83, payload={"key": "k"}, version=4)]
    assert qdrant.sent == ("POST", "/collections/field_memory/points/query", {}, {
        "query": {"nearest": [0.1, 0.2]}, "limit": 21, "with_payload": True, "filter": FIELD_JSON})


# --------------------------------------------------------------------------- snapshots


def test_create_snapshot_waits():
    made = {"name": "field_memory-1.snapshot", "creation_time": "2026-10-19T05:00:00", "size": 1024, "checksum": "ab"}
    qdrant = _Qdrant(_ok(made))
    assert _call(qdrant, "create_snapshot", collection_name=COLLECTION) == SnapshotDescription(**made)
    assert qdrant.sent == ("POST", "/collections/field_memory/snapshots", {"wait": "true"}, None)


def test_list_snapshots():
    qdrant = _Qdrant(_ok([{"name": "a.snapshot", "creation_time": None, "size": 0}]))
    assert _call(qdrant, "list_snapshots", collection_name=COLLECTION) == [SnapshotDescription(name="a.snapshot")]
    assert qdrant.sent == ("GET", "/collections/field_memory/snapshots", {}, None)


def test_delete_snapshot_names_it_in_the_path():
    qdrant = _Qdrant(_ok(True))
    _call(qdrant, "delete_snapshot", collection_name=COLLECTION, snapshot_name="field_memory-2026-10-12.snapshot")
    assert qdrant.sent == ("DELETE", "/collections/field_memory/snapshots/field_memory-2026-10-12.snapshot",
                           {"wait": "true"}, None)


# --------------------------------------------------------------------------- the key, errors, answers


@pytest.mark.parametrize("api_key", ["k-123", None])
def test_the_api_key_header_is_sent_only_when_a_key_is_set(api_key):
    qdrant = _Qdrant(_ok({"collections": []}))
    _call(qdrant, "get_collections", api_key=api_key)
    assert qdrant.requests[0].headers.get("api-key") == api_key


def _error(answer: Any) -> Exception:
    with pytest.raises((UnexpectedResponse, ResponseHandlingException)) as caught:
        _call(_Qdrant(answer), "upsert", collection_name=COLLECTION, points=[])
    return caught.value


def test_a_408_is_a_timeout_and_a_400_is_not():
    timed_out = _error(httpx.Response(408, text="timeout"))
    assert isinstance(timed_out, UnexpectedResponse) and timed_out.status_code == 408
    assert qdrant_writes.is_timeout(timed_out)
    refused = _error(httpx.Response(400, json={"status": {"error": "wrong vector size"}}))
    assert refused.status_code == 400 and b"wrong vector size" in refused.content
    assert not qdrant_writes.is_timeout(refused)


def test_the_clients_own_timeout_is_a_timeout_and_a_refused_connection_is_not():
    timed_out = _error(httpx.ReadTimeout("timed out"))
    assert isinstance(timed_out, ResponseHandlingException) and isinstance(timed_out.source, httpx.ReadTimeout)
    assert qdrant_writes.is_timeout(timed_out)
    refused = _error(httpx.ConnectError("connection refused"))
    assert isinstance(refused.source, httpx.ConnectError) and not qdrant_writes.is_timeout(refused)


@pytest.mark.parametrize("answer", [
    httpx.Response(200, text="<html>proxy error</html>"),
    httpx.Response(200, json={"status": "ok"}),
])
def test_an_answer_that_is_not_qdrants_json_is_refused(answer):
    assert isinstance(_error(answer), ResponseHandlingException)


@pytest.mark.parametrize("method,args,result", [
    ("scroll", {"collection_name": COLLECTION}, {"next_page_offset": None}),
    ("query_points", {"collection_name": COLLECTION, "query": [0.1]}, {"points": [{"id": 1, "score": "high"}]}),
    ("retrieve", {"collection_name": COLLECTION, "ids": [1]}, [{"id": [1], "payload": {}}]),
    ("count", {"collection_name": COLLECTION}, {}),
])
def test_a_result_of_the_wrong_shape_is_refused_where_it_enters(method, args, result):
    with pytest.raises(ResponseHandlingException):
        _call(_Qdrant(_ok(result)), method, **args)


def test_a_write_that_times_out_is_tried_once_more_through_the_client(monkeypatch):
    """F103 over the REST client: a 408, then the write lands."""
    monkeypatch.setattr(qdrant_writes.config, "MEMORY_WRITE_RETRY_PAUSE_S", 0, raising=False)
    qdrant = _Qdrant(httpx.Response(408, text="timeout"), _ok({"operation_id": 5, "status": "completed"}))

    async def go():
        client = QdrantRestClient(URL, transport=httpx.MockTransport(qdrant))
        await qdrant_writes.upsert_retrying_a_timeout_once(client, COLLECTION, [PointStruct(id=1, vector=[0.5])])
        await client.aclose()

    asyncio.run(go())
    assert [request.url.path for request in qdrant.requests] == ["/collections/field_memory/points"] * 2


# --------------------------------------------------------------------------- what is sent


class _Kind(enum.Enum):
    RULE = "rule"


class _Vector:
    """numpy's shape: an array or scalar that knows its plain-Python form."""

    def tolist(self) -> List[float]:
        return [0.25, 0.5]


def test_payload_values_json_has_no_rule_for_are_written_as_qdrant_client_wrote_them():
    when = dt.datetime(2026, 10, 19, 5, 0, tzinfo=dt.timezone.utc)
    body = {"at": when, "day": dt.date(2026, 10, 19), "id": uuid.UUID(POINT_ID), "kind": _Kind.RULE,
            "price": decimal.Decimal("1.5"), "tags": {"a"}, "vector": _Vector(), "gone": None}
    assert json.loads(encode(body)) == {"at": "2026-10-19T05:00:00+00:00", "day": "2026-10-19", "id": POINT_ID,
                                        "kind": "rule", "price": 1.5, "tags": ["a"], "vector": [0.25, 0.5],
                                        "gone": None}
    with pytest.raises(TypeError, match="not JSON serializable"):
        encode({"bad": object()})


def test_a_filter_sends_the_clauses_it_was_given():
    rules = FieldCondition(key="metadata.kind", match=MatchAny(any=("rule", "preference")))
    assert Filter(must=[], should=[rules]).to_json() == {
        "must": [], "should": [{"key": "metadata.kind", "match": {"any": ["rule", "preference"]}}]}
    assert Filter().to_json() == {}
    assert Filter(must_not=list(FIELD.must)).to_json() == {"must_not": FIELD_JSON["must"]}


# --------------------------------------------------------------------------- the callers and the locks


# Each caller, and the name its Qdrant client goes by there.
CALLERS = {
    "modules/context/adapters/vector_field.py": "_client",
    "modules/memory/durable_store.py": "_client",
    "core/qdrant_writes.py": "client",
    "services/qdrant_snapshots.py": "qdrant",
    "scripts/nuke_orphan_field_collections.py": "client",
}


def _owner(node: ast.expr) -> str:
    """``self._client`` -> ``_client``, ``qdrant`` -> ``qdrant``; anything else -> ''."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _client_calls(path: Path, owner: str) -> set:
    """Every method a module calls on its Qdrant client."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _owner(node.func.value) == owner}


def test_every_method_the_callers_call_is_the_clients():
    called = set().union(*(_client_calls(ORCH / path, owner) for path, owner in CALLERS.items()))
    assert {"scroll", "query_points", "upsert", "create_snapshot", "delete_collection"} <= called, called
    missing = {name for name in called if not callable(getattr(QdrantRestClient, name, None))}
    assert missing == set(), "a caller calls a method the client does not have"


def _imports_qdrant_client(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    return any(name.split(".")[0] == "qdrant_client" for name in names)


def test_no_module_of_the_app_imports_qdrant_client():
    found = [path.relative_to(ORCH).as_posix() for path in sorted(ORCH.rglob("*.py"))
             if "tests" not in path.relative_to(ORCH).parts and _imports_qdrant_client(path)]
    assert found == []


def _pinned(lock: str) -> set:
    return {match.group(1).lower() for match in re.finditer(r"^([A-Za-z0-9][\w.-]*)==", lock, re.M)}


def test_qdrant_client_left_the_lock_with_grpcio_and_portalocker():
    requirements = (ORCH / "requirements.in").read_text(encoding="utf-8")
    assert not re.search(r"^qdrant-client", requirements, re.M)
    assert not {"qdrant-client", "grpcio", "portalocker"} & _pinned((ORCH / "requirements.lock").read_text())


def test_only_the_prd108_proofs_keep_qdrant_client_for_ci():
    assert re.search(r"^qdrant-client==", (ORCH / "requirements-dev.txt").read_text(encoding="utf-8"), re.M)
    assert "qdrant-client" in _pinned((ORCH / "requirements-dev.lock").read_text())
    proofs = [path.name for path in sorted((ORCH / "tests").glob("test_*.py")) if _imports_qdrant_client(path)]
    assert proofs == ["test_prd108_proof.py", "test_prd108_scenarios.py"]
