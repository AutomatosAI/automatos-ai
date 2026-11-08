"""Memory-pipeline eval: recall through the real chat-turn path (#1430 Wave 0).

``evals.memory_recall`` (PRD-185 S10) scores a retriever over a labelled corpus with an
offline bag-of-words proxy. It never runs the path a chat turn takes, which is where
#1434's bug lived: facts filed where distillation files them (the workspace namespace),
recalled by the Context Router for an agent's turn (Auto has an agent id too). This eval
plugs that path into the same corpus, gold set and metrics through ``retriever_factory``:

  * each tenant's facts go into a fresh workspace through ``store_long_term`` (the
    distillation write), into the real ``DurableMemoryStore``;
  * each query runs ``ContextRouter.retrieve_context`` for an agent's chat turn, through
    the real namespaces, relevance floor and injection filter; the ranked ids come from
    the bundle's long-term memories (at most the router's five).

Two stores:
  * **offline** (default, CI): an in-memory vector store and a deterministic hashed
    bag-of-words embedder. It proves the path, not the embedding model: a namespace or
    filter regression drops recall to zero here.
  * **live** (``--live``): the configured Qdrant (``QDRANT_URL``) and OpenRouter
    embeddings (``OPENROUTER_API_KEY``), in a throwaway collection created for the run and
    dropped after it. Never the durable collection.

``--lookalikes N`` adds N look-alikes per fact (other names, other numbers, same shape),
so recall has to pick the right fact among near copies; N=8 gives 414 facts.

The number is published, never a gate (exit 0), like ``evals.memory_recall``.

Run (from orchestrator/):
    python -m evals.memory_pipeline                    # offline, the bundled corpus
    python -m evals.memory_pipeline --lookalikes 8     # offline, 414 facts
    python -m evals.memory_pipeline --live --json      # real Qdrant and embeddings
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import uuid
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.qdrant.models import FilterSelector, QueryResponse, Record, ScoredPoint
from evals.memory_recall import (
    MemoryDoc,
    MemoryRecallReport,
    Retriever,
    _tokenize,
    load_corpus,
    load_gold_set,
    run_memory_recall_eval,
)

AGENT_ID = 1  # the chat turn's agent: any agent; the workspace's facts must still come back
OFFLINE_DIMENSION = 256
LIVE_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"  # the platform's seeded default (seed_system_settings)
LIVE_DIMENSION = 2048
FACT_IMPORTANCE = 0.6

# ---------------------------------------------------------------------------
# The offline store: hashed bag-of-words vectors, filters as Qdrant applies them
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic: each token hashed into a fixed-width, L2-normalised vector."""

    def __init__(self, dimension: int = OFFLINE_DIMENSION) -> None:
        self.dimension = dimension

    async def generate_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for token in _tokenize(text):
            vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _payload_value(payload: Dict[str, Any], key: str) -> Any:
    for part in key.split("."):
        payload = payload.get(part) if isinstance(payload, dict) else None
    return payload


def _holds(payload: Dict[str, Any], condition: Any) -> bool:
    value, match = _payload_value(payload, condition.key), condition.match
    return value in match.any if hasattr(match, "any") else value == match.value


def matches(payload: Dict[str, Any], flt: Any) -> bool:
    """Qdrant's filter semantics: every must, at least one should, no must_not."""
    if flt is None:
        return True
    if not all(_holds(payload, c) for c in flt.must or []):
        return False
    if flt.should and not any(_holds(payload, c) for c in flt.should):
        return False
    return not any(_holds(payload, c) for c in flt.must_not or [])


class InMemoryVectors:
    """The QdrantRestClient calls the durable store makes, over a dict, with cosine ranking."""

    def __init__(self) -> None:
        self.points: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    async def upsert(self, collection_name: str, points: Sequence[Any], wait: bool = True) -> None:
        for p in points:
            self.points[str(p.id)] = (list(p.vector), dict(p.payload or {}))

    async def query_points(self, collection_name: str, query: Sequence[float], query_filter: Any = None,
                           limit: int = 10, **_: Any) -> QueryResponse:
        scored = [ScoredPoint(id=pid, score=sum(a * b for a, b in zip(query, vec)), payload=payload)
                  for pid, (vec, payload) in self.points.items() if matches(payload, query_filter)]
        scored.sort(key=lambda p: -p.score)  # stable: a tie keeps the order the facts were stored in
        return QueryResponse(points=scored[:limit])

    async def scroll(self, collection_name: str, scroll_filter: Any = None, limit: int = 10,
                     offset: Optional[int] = None, **_: Any) -> Tuple[List[Record], Optional[int]]:
        hits = [Record(id=pid, payload=payload) for pid, (_, payload) in self.points.items()
                if matches(payload, scroll_filter)]
        start = int(offset or 0)
        return hits[start:start + limit], (start + limit if start + limit < len(hits) else None)

    async def retrieve(self, collection_name: str, ids: Sequence[Any], **_: Any) -> List[Record]:
        return [Record(id=str(i), payload=self.points[str(i)][1]) for i in ids if str(i) in self.points]

    def _selected(self, selector: Any) -> List[str]:
        if isinstance(selector, FilterSelector):
            return [pid for pid, (_, payload) in self.points.items() if matches(payload, selector.filter)]
        return [str(i) for i in selector]

    async def delete(self, collection_name: str, points_selector: Any, wait: bool = True) -> None:
        for pid in self._selected(points_selector):
            self.points.pop(pid, None)

    async def set_payload(self, collection_name: str, payload: Dict[str, Any], points: Any,
                          wait: bool = True) -> None:
        for pid in self._selected(points):
            if pid in self.points:
                vec, old = self.points[pid]
                self.points[pid] = (vec, {**old, **payload})


async def _nothing(*_: Any, **__: Any) -> List[Any]:
    return []


def service_over(client: Any, embedder: Any, collection: str, dimension: int, bootstrapped: bool) -> Any:
    """A UnifiedMemoryService over this store, as the memory tests build one: the real
    write, search and namespaces; no Redis cache; the short-term and daily layers empty
    (they are Postgres's, not this eval's); and no substrate metric, which would write a
    row to the configured database for every search (PRD-197 S4)."""
    from modules.memory.durable_store import DurableMemoryStore
    from modules.memory.unified_memory_service import UnifiedMemoryService

    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done = True, bootstrapped
    store._client, store._embedder = client, embedder
    store._collection, store._dimension = collection, dimension
    store._record_search = lambda *_args, **_kwargs: None
    service = UnifiedMemoryService.__new__(UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    service.search_short_term = service.search_short_term_semantic = service.get_all_daily_logs = _nothing
    return service


def offline_service() -> Any:
    return service_over(InMemoryVectors(), HashEmbedder(), "eval_memory", OFFLINE_DIMENSION, bootstrapped=True)


class OpenRouterEmbedder:
    """Embeddings straight from OpenRouter's endpoint. Not the platform's provider, which
    records every call as LLM usage in the configured database, for its default workspace:
    an eval never writes usage or cost rows."""

    def __init__(self, model: str, api_key: str, base_url: str, dimension: int) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model, self._dimension = model, dimension

    async def generate_embedding(self, text: str) -> List[float]:
        # As the platform's provider does: the model's native vector (Qwen3-Embedding-8B: 4096),
        # cut to the store's dimension (core/llm/clients/openrouter_embedding.py).
        response = await self._client.embeddings.create(model=self._model, input=[text])
        return list(response.data[0].embedding)[: self._dimension]


def live_service(model: str, dimension: int) -> Tuple[Any, Callable[[], Any]]:
    """The configured Qdrant and OpenRouter embeddings, in a collection made for this run.
    Returns the service and the coroutine function that drops the collection."""
    from config import config
    from core.qdrant import QdrantRestClient

    if not (config.QDRANT_URL and config.OPENROUTER_API_KEY):
        raise SystemExit("--live needs QDRANT_URL and OPENROUTER_API_KEY")
    client = QdrantRestClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=30)
    embedder = OpenRouterEmbedder(model, config.OPENROUTER_API_KEY, config.OPENROUTER_BASE_URL, dimension)
    collection = f"eval_memory_{uuid.uuid4().hex[:12]}"
    return (service_over(client, embedder, collection, dimension, bootstrapped=False),
            lambda: client.delete_collection(collection))


# ---------------------------------------------------------------------------
# The chat-turn retriever and the look-alikes
# ---------------------------------------------------------------------------


def chat_turn_retriever_factory(service: Any, run: Callable[[Any], Any]) -> Callable[[List[MemoryDoc]], Retriever]:
    """``retriever_factory`` for ``run_memory_recall_eval``: file a tenant's facts the way
    distillation does, then answer each query from an agent's chat-turn recall."""
    from modules.memory.context_router import ContextRouter

    router = ContextRouter(service=service)

    def factory(docs: List[MemoryDoc]) -> Retriever:
        workspace = str(uuid.uuid4())
        for d in docs:
            stored = run(service.store_long_term(
                workspace_id=workspace, content=d.text,
                metadata={"memory_id": d.memory_id, "type": d.content_type, "importance": FACT_IMPORTANCE},
            ))
            if not stored.get("success"):
                raise RuntimeError(f"the store refused {d.memory_id}: {stored.get('error')}")

        def retrieve(query: str) -> List[str]:
            bundle = run(router.retrieve_context(workspace_id=workspace, agent_id=AGENT_ID, query=query))
            return [str((m.get("metadata") or {}).get("memory_id")) for m in bundle.long_term_memories]

        return retrieve

    return factory


_NAMES = ["Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark", "Wayne", "Soylent", "Tyrell", "Wonka"]
_PEOPLE = ["Priya", "Tomasz", "Aiko", "Bruno", "Leila", "Owen", "Ines", "Kofi", "Marta", "Sven"]
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
_MID_SENTENCE_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_NUMBER = re.compile(r"\d+(?:st|nd|rd|th)?")
_CALENDAR = frozenset(
    "January February March April May June July August September October November December "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
)


def proper_names(corpus: Sequence[MemoryDoc]) -> frozenset:
    """The corpus's names: words written capitalised somewhere other than a sentence's start
    (months and weekdays aside: a look-alike moves the date, it does not rename it)."""
    return frozenset(w for d in corpus for w in _MID_SENTENCE_PROPER.findall(d.text)) - _CALENDAR


def _renumber(m: re.Match, i: int) -> str:
    """A different number, an ordinal kept an ordinal (1st → 3rd for i=1)."""
    digits = re.match(r"\d+", m.group(0)).group(0)
    n = int(digits) * (i + 1) + i
    if m.group(0) == digits:
        return str(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def lookalike(doc: MemoryDoc, i: int, names: frozenset) -> MemoryDoc:
    """Another fact of the same shape about someone else: its names and numbers swapped (i ≥ 1)."""
    pool = iter(_NAMES[(i + k) % len(_NAMES)] if k % 2 == 0 else _PEOPLE[(i + k) % len(_PEOPLE)] for k in range(99))
    swapped: Dict[str, str] = {}

    def name(m: re.Match) -> str:
        word = m.group(0)
        return swapped.setdefault(word, next(pool)) if word in names else word

    text = _PROPER.sub(name, doc.text)
    text = _NUMBER.sub(lambda m: _renumber(m, i), text)
    return replace(doc, memory_id=f"{doc.memory_id}~{i}", text=text)


def with_lookalikes(corpus: List[MemoryDoc], n: int) -> List[MemoryDoc]:
    """The corpus plus ``n`` look-alikes of every fact that has a name or a number to change,
    in the same tenant. A fact with neither has no look-alike: a copy is not another fact
    (and the store would fold it into the original by its content hash)."""
    names = proper_names(corpus)
    out = list(corpus)
    for doc in corpus:
        out.extend(twin for twin in (lookalike(doc, i, names) for i in range(1, n + 1)) if twin.text != doc.text)
    return out


# ---------------------------------------------------------------------------
# Running and reporting
# ---------------------------------------------------------------------------


def run_pipeline_eval(lookalikes: int = 0, live: bool = False, model: str = LIVE_EMBEDDING_MODEL,
                      dimension: int = LIVE_DIMENSION) -> Dict[str, Any]:
    """The bag-of-words proxy and the chat-turn path over the same corpus and gold set."""
    corpus, gold = with_lookalikes(load_corpus(), lookalikes), load_gold_set()
    loop = asyncio.new_event_loop()
    try:
        service, drop = live_service(model, dimension) if live else (offline_service(), None)
        try:
            pipeline = run_memory_recall_eval(corpus, gold, chat_turn_retriever_factory(service, loop.run_until_complete))
        finally:
            if drop is not None:
                loop.run_until_complete(drop())
    finally:
        loop.close()
    return {
        "store": f"live ({model})" if live else "offline (hashed bag-of-words)",
        "facts": len(corpus),
        "lookalikes_per_fact": lookalikes,
        "proxy": _summary(run_memory_recall_eval(corpus, gold)),
        "chat_turn": _summary(pipeline),
    }


def _summary(report: MemoryRecallReport) -> Dict[str, Any]:
    return {
        "recall_at_5": round(report.mean_recall_at_5, 3),
        "mrr": round(report.mean_mrr, 3),
        "slices": {k: {"n": int(v["n"]), "recall_at_5": round(v["recall_at_5"], 3), "mrr": round(v["mrr"], 3)}
                   for k, v in sorted(report.slices.items())},
    }


def render(result: Dict[str, Any]) -> str:
    rows = ["| Recall | recall@5 | MRR | " + " | ".join(f"{s} recall@5" for s in result["proxy"]["slices"]) + " |",
            "|---|---|---|" + "---|" * len(result["proxy"]["slices"])]
    for label, key in (("Bag-of-words proxy (memory_recall)", "proxy"), ("Chat-turn path", "chat_turn")):
        s = result[key]
        slices = " | ".join(f"{v['recall_at_5']:.2f} (n={v['n']})" for v in s["slices"].values())
        rows.append(f"| {label} | {s['recall_at_5']:.2f} | {s['mrr']:.2f} | {slices} |")
    head = (f"# Memory-pipeline eval\n\nStore: {result['store']}. Facts: {result['facts']} "
            f"({result['lookalikes_per_fact']} look-alikes per fact).\n\n")
    return head + "\n".join(rows) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lookalikes", type=int, default=0, help="look-alikes per fact (8 gives 414 facts)")
    parser.add_argument("--live", action="store_true", help="the configured Qdrant and OpenRouter embeddings")
    parser.add_argument("--model", default=LIVE_EMBEDDING_MODEL, help="the embedding model for --live")
    parser.add_argument("--dimension", type=int, default=LIVE_DIMENSION, help="its dimension")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    result = run_pipeline_eval(args.lookalikes, args.live, args.model, args.dimension)
    sys.stdout.write((json.dumps(result, indent=2) if args.json else render(result)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
