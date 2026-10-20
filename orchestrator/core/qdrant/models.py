"""The Qdrant request and response shapes the memory stores use (PRD-262 row #20).

The same names, fields and JSON as qdrant-client's models for the subset the code
calls, so the field memory (modules/context/adapters/vector_field.py), the durable
memory (modules/memory/durable_store.py) and the snapshots read as they did. Requests
serialise to Qdrant's REST JSON with ``to_json``; responses are parsed by ``*.parse``,
which fails with ValueError on a shape Qdrant never sends, so a bad answer is refused
where it enters, not three calls later.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

PointId = Union[str, int]


class Distance(str, Enum):
    """How a collection compares vectors."""

    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"
    MANHATTAN = "Manhattan"


class PayloadSchemaType(str, Enum):
    """The type of a payload index."""

    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    GEO = "geo"
    DATETIME = "datetime"
    TEXT = "text"
    UUID = "uuid"


@dataclass(frozen=True)
class MatchValue:
    """A payload key equal to ``value``."""

    value: Any

    def to_json(self) -> Dict[str, Any]:
        return {"value": self.value}


@dataclass(frozen=True)
class MatchAny:
    """A payload key equal to any of ``any``."""

    any: Sequence[Any]

    def to_json(self) -> Dict[str, Any]:
        return {"any": list(self.any)}


@dataclass(frozen=True)
class FieldCondition:
    """One condition on the payload key ``key``."""

    key: str
    match: Union[MatchValue, MatchAny]

    def to_json(self) -> Dict[str, Any]:
        return {"key": self.key, "match": self.match.to_json()}


@dataclass(frozen=True)
class Filter:
    """Every ``must`` condition, at least one ``should`` one, no ``must_not`` one. A clause
    left None is not sent; an empty list is sent as it is, as qdrant-client did."""

    must: Optional[Sequence[FieldCondition]] = None
    should: Optional[Sequence[FieldCondition]] = None
    must_not: Optional[Sequence[FieldCondition]] = None

    def to_json(self) -> Dict[str, Any]:
        clauses = (("must", self.must), ("should", self.should), ("must_not", self.must_not))
        return {name: [condition.to_json() for condition in conditions]
                for name, conditions in clauses if conditions is not None}


@dataclass(frozen=True)
class FilterSelector:
    """Selects the points ``filter`` matches, for a delete."""

    filter: Filter


@dataclass(frozen=True)
class PointStruct:
    """A point to upsert: its id, vector and payload."""

    id: PointId
    vector: Sequence[float]
    payload: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "vector": list(self.vector), "payload": self.payload}


@dataclass(frozen=True)
class VectorParams:
    """A collection's vectors: their size and distance, and whether they live on disk."""

    size: int
    distance: Distance
    on_disk: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        found = {"size": self.size, "distance": Distance(self.distance).value}
        return found if self.on_disk is None else {**found, "on_disk": self.on_disk}


@dataclass(frozen=True)
class HnswConfigDiff:
    """The HNSW index settings a collection overrides."""

    on_disk: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        return {} if self.on_disk is None else {"on_disk": self.on_disk}


def _point_id(value: Any) -> PointId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"a point id is a string or an integer, not {value!r}")
    return value


def _payload(value: Any) -> Optional[Dict[str, Any]]:
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"a payload is an object, not {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Record:
    """A point as scroll and retrieve return it."""

    id: PointId
    payload: Optional[Dict[str, Any]] = None
    vector: Any = None

    @classmethod
    def parse(cls, item: Any) -> "Record":
        if not isinstance(item, dict):
            raise ValueError(f"a point is an object, not {type(item).__name__}")
        return cls(id=_point_id(item.get("id")), payload=_payload(item.get("payload")), vector=item.get("vector"))


@dataclass(frozen=True)
class ScoredPoint:
    """A point as a query returns it, with its similarity."""

    id: PointId
    score: float
    payload: Optional[Dict[str, Any]] = None
    version: int = 0
    vector: Any = None

    @classmethod
    def parse(cls, item: Any) -> "ScoredPoint":
        record = Record.parse(item)
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"a scored point has a numeric score, not {score!r}")
        return cls(id=record.id, score=float(score), payload=record.payload, version=int(item.get("version") or 0),
                   vector=record.vector)


@dataclass(frozen=True)
class QueryResponse:
    """A query's points, best first."""

    points: List[ScoredPoint]


@dataclass(frozen=True)
class CountResult:
    """How many points a filter matches."""

    count: int


@dataclass(frozen=True)
class CollectionDescription:
    """A collection, by name."""

    name: str


@dataclass(frozen=True)
class CollectionsResponse:
    """Every collection on the node."""

    collections: List[CollectionDescription]


@dataclass(frozen=True)
class SnapshotDescription:
    """A snapshot on the node: its file name, when it was made, its size."""

    name: str
    creation_time: Optional[str] = None
    size: int = 0
    checksum: Optional[str] = None

    @classmethod
    def parse(cls, item: Any) -> "SnapshotDescription":
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"a snapshot description has a name: {item!r}")
        return cls(name=item["name"], creation_time=item.get("creation_time"), size=int(item.get("size") or 0),
                   checksum=item.get("checksum"))
