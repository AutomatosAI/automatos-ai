"""PRD-268 US-C02d: spies that see every write a handler makes, for the read-action sweep.

- ``spy_session``: a Session where every read finds nothing (a query's ``first`` is None,
  ``all`` is [], ``count`` is 0, an ``execute`` returns no rows) and every write is recorded
  instead of made: ``add``, ``add_all``, ``delete``, ``merge``, ``flush``, ``commit``, the
  ``bulk_*`` calls, ``Query.update`` and ``Query.delete``, and an ``execute`` (or ``scalar``,
  ``scalars``, or one on the session's connection) of an INSERT, UPDATE or DELETE.
- ``recording_http``: httpx's transports (sync and async) and requests' adapter, patched to
  answer 200 with ``{}`` and record each request that writes; no httpx or requests call is
  sent. A request to the platform's own vector store (``config.QDRANT_URL``), a datastore
  like Postgres, is recorded only when it changes the store (a collection or index created,
  an upsert, a delete, a payload set), not when it reads (a GET, a retrieve, a search, a
  scroll, a count). Every other request leaves the platform, so every one is recorded.
- ``minimal_params``: an action's required parameters, each with its minimal valid value.

What the spies cannot see: a write reached only when rows exist (the database is empty), a
session the code opens itself (``SessionLocal``), Redis, and clients that are neither httpx
nor requests (boto3).

Pure: no database, no network.
"""
from __future__ import annotations

import io
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import httpx
import requests
from sqlalchemy import false, literal, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import UpdateBase
from sqlalchemy.sql.elements import TextClause

MINIMAL_STRING = "1"
MINIMAL_NUMBER = 1

# What the session records instead of doing. ``merge`` hands back what it was given.
SESSION_WRITES = ("add", "add_all", "delete", "merge", "flush", "commit",
                  "bulk_save_objects", "bulk_insert_mappings", "bulk_update_mappings")
# A statement writes when it is an INSERT, UPDATE or DELETE construct, or SQL text that
# starts with a statement that changes data or schema, holds an INSERT, UPDATE or DELETE
# (a CTE), sends a notification or moves a sequence. "SELECT ... FOR UPDATE" reads.
_WRITE_SQL = re.compile(
    r"^\s*(?:INSERT|UPDATE|DELETE|MERGE|TRUNCATE|CREATE|ALTER|DROP|COPY|REFRESH)\b"
    r"|\bINSERT\s+INTO\b|\bDELETE\s+FROM\b|\bUPDATE\s+\S+\s+SET\b|\bpg_notify\b|\bnextval\b|\bsetval\b",
    re.IGNORECASE,
)
# The Query methods that narrow or shape a query: each hands back the same empty query.
_QUERY_CHAINING = frozenset({
    "add_columns", "add_entity", "autoflush", "correlate", "distinct", "enable_assertions",
    "enable_eagerloads", "except_", "except_all", "execution_options", "filter", "filter_by",
    "from_statement", "group_by", "having", "intersect", "intersect_all", "join", "limit", "offset",
    "options", "order_by", "outerjoin", "params", "populate_existing", "prefix_with", "select_from",
    "set_label_style", "slice", "suffix_with", "union", "union_all", "where", "with_entities",
    "with_for_update", "with_hint", "with_parent", "with_session", "yield_per",
})
_RESULT_CHAINING = frozenset({"columns", "mappings", "scalars", "tuples", "unique", "yield_per"})
# The vector store's reads: any GET or HEAD, and a POST to a collection's points (a retrieve),
# to one of their read endpoints, or to its facets. Every other request to it changes it.
_VECTOR_READ_METHODS = frozenset({"GET", "HEAD"})
_VECTOR_READ_POST = re.compile(
    r"/points(?:/(?:scroll|count|query|search|recommend|discover)(?:/batch|/groups|/matrix/(?:pairs|offsets))?)?/?$"
    r"|/facet/?$"
)
# A query turned into SQL (a subquery, a CTE, an EXISTS) is a SELECT that finds nothing.
_NOTHING = select(literal(None).label("nothing")).where(false())


@dataclass
class WriteLog:
    """What one handler call wrote (``writes``) and how the call ended (``outcome``)."""

    writes: List[str] = field(default_factory=list)
    outcome: str = "returned"

    def record(self, what: str) -> None:
        self.writes.append(what)


def writes_data(statement: Any) -> bool:
    """Whether ``statement`` is an INSERT, UPDATE or DELETE (a construct, text or a string)."""
    if isinstance(statement, UpdateBase):
        return True
    text = getattr(getattr(statement, "element", statement), "text", statement)
    if isinstance(statement, TextClause) or isinstance(text, str):
        return bool(_WRITE_SQL.search(str(text)))
    return False


def _statement_name(statement: Any) -> str:
    text = getattr(getattr(statement, "element", statement), "text", None)
    return " ".join(str(text if text is not None else statement).split())[:80]


class EmptyResult:
    """The result of a statement that found no rows."""

    rowcount = 0
    returns_rows = True

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name in _RESULT_CHAINING:
            return lambda *args, **kwargs: self
        raise AttributeError(f"the spy's empty result has no {name}")

    def partitions(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        return iter(())

    def all(self) -> List[Any]:
        return []

    fetchall = all

    def fetchmany(self, *args: Any, **kwargs: Any) -> List[Any]:
        return []

    def first(self) -> None:
        return None

    fetchone = scalar = scalar_one_or_none = one_or_none = first

    def one(self) -> Any:
        raise NoResultFound("No row was found when one was required")

    scalar_one = one

    def keys(self) -> List[str]:
        return []

    def close(self) -> None:
        return None


class EmptyQuery:
    """``session.query(...)`` over an empty database. ``update`` and ``delete`` are recorded."""

    def __init__(self, log: WriteLog) -> None:
        self._log = log

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name in _QUERY_CHAINING:
            return lambda *args, **kwargs: self
        raise AttributeError(f"the spy's empty query has no {name}")

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    @property
    def column_descriptions(self) -> List[Any]:
        return []

    def __getitem__(self, item: Any) -> List[Any]:
        if isinstance(item, slice):
            return []
        raise IndexError(item)

    def all(self) -> List[Any]:
        return []

    def first(self) -> None:
        return None

    scalar = one_or_none = first

    def get(self, *args: Any, **kwargs: Any) -> None:
        return None

    def one(self) -> Any:
        raise NoResultFound("No row was found when one was required")

    def count(self) -> int:
        return 0

    def update(self, *args: Any, **kwargs: Any) -> int:
        self._log.record("Query.update")
        return 0

    def delete(self, *args: Any, **kwargs: Any) -> int:
        self._log.record("Query.delete")
        return 0

    @property
    def statement(self) -> Any:
        return _NOTHING

    def subquery(self, *args: Any, **kwargs: Any) -> Any:
        return _NOTHING.subquery(*args, **kwargs)

    def cte(self, *args: Any, **kwargs: Any) -> Any:
        return _NOTHING.cte(*args, **kwargs)

    def scalar_subquery(self) -> Any:
        return _NOTHING.scalar_subquery()

    as_scalar = scalar_subquery

    def exists(self) -> Any:
        return _NOTHING.exists()

    def label(self, name: str) -> Any:
        return _NOTHING.scalar_subquery().label(name)


class _Nested:
    """``begin_nested()`` / ``begin()``: a savepoint that holds nothing of its own. The writes
    made inside it are recorded where they are made."""

    def __enter__(self) -> "_Nested":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def _executing(log: WriteLog, verb: str, empty: Any) -> Callable[..., Any]:
    """An ``execute``-like call: records the statement when it writes, and finds nothing."""
    def run(statement: Any = None, *args: Any, **kwargs: Any) -> Any:
        if writes_data(statement):
            log.record(f"{verb} {_statement_name(statement)}")
        return empty() if callable(empty) else empty
    return run


def _recording(log: WriteLog, name: str) -> Callable[..., Any]:
    def write(*args: Any, **kwargs: Any) -> Any:
        target = type(args[0]).__name__ if args else ""
        log.record(f"{name}({target})" if target else name)
        return args[0] if name == "merge" and args else None
    return write


def spy_connection(log: WriteLog) -> MagicMock:
    """The session's connection (and its engine's): the same execute spy, and commit recorded."""
    connection = MagicMock(name="spy_connection")
    connection.execute.side_effect = _executing(log, "connection.execute", EmptyResult)
    connection.exec_driver_sql.side_effect = _executing(log, "connection.exec_driver_sql", EmptyResult)
    connection.scalar.side_effect = _executing(log, "connection.scalar", None)
    connection.commit.side_effect = _recording(log, "connection.commit")
    connection.__enter__.return_value = connection
    connection.begin.side_effect = lambda *a, **k: _Nested()
    return connection


def _spy_engine(log: WriteLog) -> MagicMock:
    engine = MagicMock(name="spy_engine")
    engine.dialect.name = "postgresql"
    engine.connect.side_effect = lambda *a, **k: spy_connection(log)
    engine.begin.side_effect = lambda *a, **k: spy_connection(log)
    return engine


def spy_session(log: WriteLog) -> MagicMock:
    """A Session over an empty database that records every write into ``log``."""
    session = MagicMock(spec=Session, name="spy_session")
    session.query.side_effect = lambda *args, **kwargs: EmptyQuery(log)
    session.execute.side_effect = _executing(log, "execute", EmptyResult)
    session.scalar.side_effect = _executing(log, "scalar", None)
    session.scalars.side_effect = _executing(log, "scalars", EmptyResult)
    session.get.return_value = None
    for name in SESSION_WRITES:
        getattr(session, name).side_effect = _recording(log, name)
    session.begin_nested.side_effect = lambda *a, **k: _Nested()
    session.begin.side_effect = lambda *a, **k: _Nested()
    session.connection.side_effect = lambda *a, **k: spy_connection(log)
    session.bind = _spy_engine(log)
    session.get_bind.return_value = session.bind
    session.no_autoflush = nullcontext()
    session.info = {}
    session.new = session.dirty = session.deleted = frozenset()
    session.is_active = True
    session.in_transaction.return_value = False
    return session


def _requests_answer(request: Any) -> requests.Response:
    """200 ``{}``, already read, so ``with``, ``stream=True`` and ``close`` all work."""
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response.raw = io.BytesIO(b"{}")
    response._content = b"{}"
    response._content_consumed = True
    response.headers["Content-Type"] = "application/json"
    response.encoding = "utf-8"
    response.url = request.url
    response.request = request
    return response


def _origin(url: Any) -> tuple:
    parts = urlsplit(str(url))
    return (parts.scheme, parts.hostname, parts.port)


def _vector_store_origin() -> Optional[tuple]:
    """Where the platform's own vector store answers (``config.QDRANT_URL``), or None when off."""
    from config import config

    url = getattr(config, "QDRANT_URL", "") or ""
    return _origin(url) if url else None


def http_write(method: str, url: Any) -> Optional[str]:
    """What a request writes, or None when it only reads the platform's own vector store."""
    path = urlsplit(str(url)).path
    if _vector_store_origin() == _origin(url):
        reads = method in _VECTOR_READ_METHODS or (method == "POST" and _VECTOR_READ_POST.search(path))
        return None if reads else f"vector store {method} {path}"
    return f"HTTP {method} {urlsplit(str(url)).hostname or url}"


@contextmanager
def recording_http(log: WriteLog) -> Iterator[WriteLog]:
    """Every httpx (sync or async) and requests call is answered 200 ``{}`` and none is
    sent; each that writes (``http_write``) is recorded in ``log``."""
    def note(method: str, url: Any) -> None:
        wrote = http_write(str(method).upper(), url)
        if wrote:
            log.record(wrote)

    def sync_send(transport: Any, request: httpx.Request) -> httpx.Response:
        note(request.method, request.url)
        return httpx.Response(200, json={}, request=request)

    async def async_send(transport: Any, request: httpx.Request) -> httpx.Response:
        note(request.method, request.url)
        return httpx.Response(200, json={}, request=request)

    def requests_send(adapter: Any, request: Any, *args: Any, **kwargs: Any) -> requests.Response:
        note(request.method, request.url)
        return _requests_answer(request)

    with patch.object(httpx.HTTPTransport, "handle_request", sync_send), \
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", async_send), \
            patch.object(requests.adapters.HTTPAdapter, "send", requests_send):
        yield log


def minimal_value(schema: Any) -> Any:
    """The minimal valid value for a JSON schema: an enum's first value; "1" for a string
    ("1" repeated to its minLength); its minimum, or 1, for a number; False; [] (or minItems
    copies); an object of its required properties."""
    if not isinstance(schema, dict):
        return MINIMAL_STRING
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if schema.get(key) and "type" not in schema:
            return minimal_value(schema[key][0])
    kind = schema.get("type") or ("object" if "properties" in schema else "array" if "items" in schema else "string")
    if isinstance(kind, list):
        kind = next((one for one in kind if one != "null"), "null")
    return _MINIMAL_BY_TYPE.get(kind, lambda s: MINIMAL_STRING)(schema)


def _minimal_number(schema: Dict[str, Any]) -> Any:
    """Its minimum (one past an exclusive one), else 1, and never past its maximum."""
    exclusive = schema.get("exclusiveMinimum")
    if isinstance(exclusive, bool):  # draft 4: a flag on ``minimum``
        exclusive = schema.get("minimum") if exclusive else None
    if exclusive is not None:
        value = exclusive + 1
    else:
        value = schema.get("minimum", MINIMAL_NUMBER)
    return min(value, schema["maximum"]) if "maximum" in schema else value


def minimal_params(schema: Dict[str, Any]) -> Dict[str, Any]:
    """An action's parameters: its required fields only, each with its minimal valid value."""
    properties = (schema or {}).get("properties") or {}
    return {name: minimal_value(properties.get(name)) for name in (schema or {}).get("required") or ()}


_MINIMAL_BY_TYPE: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "string": lambda s: MINIMAL_STRING * max(1, int(s.get("minLength", 1))),
    "integer": _minimal_number,
    "number": _minimal_number,
    "boolean": lambda s: False,
    "null": lambda s: None,
    "array": lambda s: [minimal_value(s.get("items")) for _ in range(int(s.get("minItems", 0)))],
    "object": minimal_params,
}
