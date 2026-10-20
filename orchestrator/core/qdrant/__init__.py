"""Qdrant over its REST API, on httpx (PRD-262, the 19 Oct breakdown's row #20).

The memory stores' client (``QdrantRestClient``), the request and response shapes
they use, and what a call raises. It replaced qdrant-client, which brought grpcio and
portalocker for a gRPC transport the code never used.
"""
from core.qdrant.client import QdrantRestClient
from core.qdrant.errors import ResponseHandlingException, UnexpectedResponse
from core.qdrant.models import (
    CollectionDescription,
    CollectionsResponse,
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
    QueryResponse,
    Record,
    ScoredPoint,
    SnapshotDescription,
    VectorParams,
)

__all__ = [
    "CollectionDescription",
    "CollectionsResponse",
    "CountResult",
    "Distance",
    "FieldCondition",
    "Filter",
    "FilterSelector",
    "HnswConfigDiff",
    "MatchAny",
    "MatchValue",
    "PayloadSchemaType",
    "PointStruct",
    "QdrantRestClient",
    "QueryResponse",
    "Record",
    "ResponseHandlingException",
    "ScoredPoint",
    "SnapshotDescription",
    "UnexpectedResponse",
    "VectorParams",
]
