"""Qdrant's REST API on the httpx the API already ships (PRD-262, the 19 Oct breakdown's row #20).

The memory stores (modules/context/adapters/vector_field.py, modules/memory/
durable_store.py), the snapshot job (services/qdrant_snapshots.py) and the orphan
sweep (scripts/nuke_orphan_field_collections.py) call these fifteen methods and no
others. qdrant-client served them over the same REST API and brought grpcio and
portalocker into the image with it (about 23 MB), for a gRPC transport the code never
used. Each method takes the arguments qdrant-client's took, sends the request it sent
(``wait=true`` where it waited by default, the ``api-key`` header when a key is set)
and returns the same shape, so the callers read as they did:

- a transport error (a refused connection, the client's own timeout) raises
  ``ResponseHandlingException`` with the error as ``source``, an answer that is not a
  success raises ``UnexpectedResponse`` with its status, and an answer that is not
  Qdrant's JSON raises ``ResponseHandlingException`` too, as qdrant-client did;
- payload values the standard JSON encoder can't write (datetimes, UUIDs, enums,
  decimals, sets, numpy values) are encoded as qdrant-client's encoder wrote them.

tests/test_prd262_qdrant_rest_client.py drives every method through httpx's
MockTransport and checks the request and the parsed answer.
"""
from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from urllib.parse import quote

import httpx

from core.qdrant.errors import ResponseHandlingException, UnexpectedResponse
from core.qdrant.models import (
    CollectionDescription,
    CollectionsResponse,
    CountResult,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    PayloadSchemaType,
    PointId,
    PointStruct,
    QueryResponse,
    Record,
    ScoredPoint,
    SnapshotDescription,
    VectorParams,
)

API_KEY_HEADER = "api-key"
JSON_HEADERS = {"Content-Type": "application/json"}
DEFAULT_TIMEOUT_S = 5.0  # qdrant-client's default; the stores pass their own
DEFAULT_LIMIT = 10  # qdrant-client's default page for scroll and query
SUCCESS = frozenset({200, 201, 202})

PointsSelector = Union[FilterSelector, Filter, Sequence[PointId]]
_Parsed = TypeVar("_Parsed")


def _jsonable(value: Any) -> Any:
    """What the JSON encoder writes for a value it has no rule for, as qdrant-client's encoder did."""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def encode(body: Dict[str, Any]) -> bytes:
    """A request body as Qdrant's REST JSON. Pure."""
    return json.dumps(body, default=_jsonable).encode("utf-8")


def _selector(selector: PointsSelector) -> Dict[str, Any]:
    """The points a delete or a payload update applies to: a filter, or a list of ids."""
    if isinstance(selector, FilterSelector):
        return {"filter": selector.filter.to_json()}
    if isinstance(selector, Filter):
        return {"filter": selector.to_json()}
    if isinstance(selector, (list, tuple)):
        return {"points": list(selector)}
    raise TypeError(f"select points with a Filter, a FilterSelector or a list of ids, not {type(selector).__name__}")


def _filtered(body: Dict[str, Any], flt: Optional[Filter]) -> Dict[str, Any]:
    return body if flt is None else {**body, "filter": flt.to_json()}


def _parsed(parse: Callable[[Any], _Parsed], result: Any) -> _Parsed:
    """``parse(result)``; a shape Qdrant never sends raises ResponseHandlingException."""
    try:
        return parse(result)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ResponseHandlingException(error) from error


def _scroll_page(result: Any) -> Tuple[List[Record], Optional[PointId]]:
    return [Record.parse(item) for item in result["points"]], result.get("next_page_offset")


def _collections(result: Any) -> CollectionsResponse:
    return CollectionsResponse([CollectionDescription(str(item["name"])) for item in result["collections"]])


class QdrantRestClient:
    """An async client for the Qdrant REST endpoints the platform calls. One httpx client
    per instance, made with it, as qdrant-client's AsyncQdrantClient made one."""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_S,
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        self._http = httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()

    async def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                       wait: bool = False) -> Any:
        """Send one request; return the answer's ``result``."""
        params = {"wait": "true"} if wait else None
        content, headers = (None, None) if body is None else (encode(body), JSON_HEADERS)
        try:
            response = await self._http.request(method, path, params=params, content=content, headers=headers)
        except httpx.HTTPError as error:
            raise ResponseHandlingException(error) from error
        if response.status_code not in SUCCESS:
            raise UnexpectedResponse.for_response(response)
        return _parsed(lambda answer: answer.json()["result"], response)

    @staticmethod
    def _collection(name: str) -> str:
        return f"/collections/{quote(name, safe='')}"

    # ── collections ──────────────────────────────────────────────

    async def collection_exists(self, collection_name: str) -> bool:
        """True when the collection exists."""
        result = await self._request("GET", f"{self._collection(collection_name)}/exists")
        return _parsed(lambda found: bool(found["exists"]), result)

    async def get_collections(self) -> CollectionsResponse:
        """Every collection on the node; the stores' health check."""
        return _parsed(_collections, await self._request("GET", "/collections"))

    async def create_collection(self, collection_name: str, vectors_config: VectorParams,
                                hnsw_config: Optional[HnswConfigDiff] = None,
                                on_disk_payload: Optional[bool] = None) -> bool:
        """Create a collection of ``vectors_config`` vectors."""
        body: Dict[str, Any] = {"vectors": vectors_config.to_json()}
        if hnsw_config is not None:
            body["hnsw_config"] = hnsw_config.to_json()
        if on_disk_payload is not None:
            body["on_disk_payload"] = on_disk_payload
        return bool(await self._request("PUT", self._collection(collection_name), body))

    async def delete_collection(self, collection_name: str) -> bool:
        """Drop a collection and its points."""
        return bool(await self._request("DELETE", self._collection(collection_name)))

    async def create_payload_index(self, collection_name: str, field_name: str,
                                   field_schema: PayloadSchemaType, wait: bool = True) -> None:
        """Index the payload key ``field_name``; Qdrant answers a second call for the same key as the first."""
        body = {"field_name": field_name, "field_schema": PayloadSchemaType(field_schema).value}
        await self._request("PUT", f"{self._collection(collection_name)}/index", body, wait=wait)

    # ── points ───────────────────────────────────────────────────

    async def upsert(self, collection_name: str, points: Sequence[PointStruct], wait: bool = True) -> None:
        """Insert the points, or replace the ones whose id is taken."""
        body = {"points": [point.to_json() for point in points]}
        await self._request("PUT", f"{self._collection(collection_name)}/points", body, wait=wait)

    async def delete(self, collection_name: str, points_selector: PointsSelector, wait: bool = True) -> None:
        """Delete the points a filter matches, or the listed ids."""
        await self._request("POST", f"{self._collection(collection_name)}/points/delete",
                            _selector(points_selector), wait=wait)

    async def set_payload(self, collection_name: str, payload: Dict[str, Any], points: PointsSelector,
                          wait: bool = True) -> None:
        """Set these payload keys on the selected points, leaving their other keys as they are."""
        body = {"payload": payload, **_selector(points)}
        await self._request("POST", f"{self._collection(collection_name)}/points/payload", body, wait=wait)

    async def retrieve(self, collection_name: str, ids: Sequence[PointId], with_payload: Union[bool, List[str]] = True,
                       with_vectors: bool = False) -> List[Record]:
        """The points with these ids that exist."""
        body = {"ids": list(ids), "with_payload": with_payload, "with_vector": with_vectors}
        result = await self._request("POST", f"{self._collection(collection_name)}/points", body)
        return _parsed(lambda items: [Record.parse(item) for item in items], result)

    async def scroll(self, collection_name: str, scroll_filter: Optional[Filter] = None, limit: int = DEFAULT_LIMIT,
                     offset: Optional[PointId] = None, with_payload: Union[bool, List[str]] = True,
                     with_vectors: bool = False) -> Tuple[List[Record], Optional[PointId]]:
        """One page of the points ``scroll_filter`` matches, from ``offset``, and the next page's offset
        (None after the last page)."""
        body: Dict[str, Any] = {"limit": limit, "with_payload": with_payload, "with_vector": with_vectors}
        if offset is not None:
            body["offset"] = offset
        result = await self._request("POST", f"{self._collection(collection_name)}/points/scroll",
                                     _filtered(body, scroll_filter))
        return _parsed(_scroll_page, result)

    async def count(self, collection_name: str, count_filter: Optional[Filter] = None, exact: bool = True) -> CountResult:
        """How many points ``count_filter`` matches."""
        result = await self._request("POST", f"{self._collection(collection_name)}/points/count",
                                     _filtered({"exact": exact}, count_filter))
        return _parsed(lambda found: CountResult(int(found["count"])), result)

    async def query_points(self, collection_name: str, query: Sequence[float], query_filter: Optional[Filter] = None,
                           limit: int = DEFAULT_LIMIT, with_payload: Union[bool, List[str]] = True) -> QueryResponse:
        """The ``limit`` points nearest ``query`` among those ``query_filter`` matches, best first."""
        body = {"query": {"nearest": list(query)}, "limit": limit, "with_payload": with_payload}
        result = await self._request("POST", f"{self._collection(collection_name)}/points/query",
                                     _filtered(body, query_filter))
        return _parsed(lambda found: QueryResponse([ScoredPoint.parse(item) for item in found["points"]]), result)

    # ── snapshots ────────────────────────────────────────────────

    async def create_snapshot(self, collection_name: str, wait: bool = True) -> SnapshotDescription:
        """Snapshot the collection on the node."""
        result = await self._request("POST", f"{self._collection(collection_name)}/snapshots", wait=wait)
        return _parsed(SnapshotDescription.parse, result)

    async def list_snapshots(self, collection_name: str) -> List[SnapshotDescription]:
        """The collection's snapshots on the node."""
        result = await self._request("GET", f"{self._collection(collection_name)}/snapshots")
        return _parsed(lambda items: [SnapshotDescription.parse(item) for item in items], result)

    async def delete_snapshot(self, collection_name: str, snapshot_name: str, wait: bool = True) -> None:
        """Delete one of the collection's snapshots from the node."""
        path = f"{self._collection(collection_name)}/snapshots/{quote(snapshot_name, safe='')}"
        await self._request("DELETE", path, wait=wait)
