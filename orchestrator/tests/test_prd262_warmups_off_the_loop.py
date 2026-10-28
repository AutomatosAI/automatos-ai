"""PRD-262 US-034: the boot's warm-ups do their synchronous work in worker threads.

The warm-ups launched at boot (``core/boot/startup_tasks.py``) now run while the server
already serves. Their synchronous parts used to run on the event loop and held it for
about 5.7 s on a warm boot (APScheduler logged a missed job). Each now goes through
``asyncio.to_thread``, and only network calls are awaited on the loop:

* ``embed_all_agents_on_startup``: the embedding provider's set-up, the workspace list
  and the closing counts; per workspace, ``embed_workspace_agents`` reads the agents,
  and ``embed_agent`` builds each agent's text (persona, skills, apps, plugins) and
  commits its new embedding in a thread;
* ``warm_action_index_on_startup``: building the index (the ActionRegistry, the cache
  client); ``ActionSemanticIndex.warm`` loads the provider and the utterance corpus,
  and the build reads and writes its Redis cache, in threads;
* ``ensure_field_memory_collection``: the shared-context factory.

Each test records the thread a synchronous step ran on and compares it with the loop's.
No database, no Redis, no network.
"""
from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

# startup_tasks imports the error telemetry, which builds the engine; it refuses to
# without POSTGRES_* values. Nothing here connects; a real .env still wins.
for _key, _value in {"POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test", "POSTGRES_HOST": "localhost",
                     "POSTGRES_PORT": "5432", "POSTGRES_DB": "test"}.items():
    os.environ.setdefault(_key, _value)

import core.boot.startup_tasks as startup_tasks  # noqa: E402
import core.routing.semantic_indexer as semantic_indexer  # noqa: E402
from modules.tools.discovery import utterance_corpus  # noqa: E402
from modules.tools.discovery.action_registry import ActionDefinition  # noqa: E402
from modules.tools.discovery.action_semantic_index import ActionSemanticIndex  # noqa: E402

DIM = 4


def _here() -> int:
    return threading.get_ident()


def _run(scenario):
    """Run ``scenario()`` and return (its result, the loop's thread)."""
    async def wrapped():
        return await scenario(), _here()
    return asyncio.run(wrapped())


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *criteria):
        return self

    def all(self):
        return list(self._rows)

    def count(self):
        return len(self._rows)


class _Db:
    """A session stand-in that records the thread of every query and commit."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.threads = {"query": [], "commit": []}
        self.added = []
        self.closed = False

    def query(self, *entities):
        self.threads["query"].append(_here())
        return _Query(self.rows)

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.threads["commit"].append(_here())

    def close(self):
        self.closed = True


class _EmbeddingManager:
    """The embedding manager: the provider set-up is synchronous, the embedding call is not."""

    def __init__(self):
        self.threads = {"ensure_provider": [], "embed": []}
        self.provider = SimpleNamespace(config=SimpleNamespace(model="fake"))

    def _ensure_provider(self):
        self.threads["ensure_provider"].append(_here())

    def get_provider_info(self):
        return {"provider": "fake", "model": "fake", "dimension": DIM}

    def get_dimension(self):
        return DIM

    async def generate_embedding(self, text):
        self.threads["embed"].append(_here())
        return [1.0] * DIM

    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        self.threads["embed"].append(_here())
        return [[1.0] * DIM for _ in texts]


# --------------------------------------------------------------------------- the agents' embeddings


def test_embed_all_agents_reads_the_database_and_sets_up_the_provider_in_threads(monkeypatch):
    import core.database.database as database
    import core.llm.embedding_manager as embedding_manager

    db = _Db(rows=[SimpleNamespace(id="ws-1"), SimpleNamespace(id="ws-2")])
    manager = _EmbeddingManager()
    embedded = []
    record_error = MagicMock()

    async def embed_workspace_agents(ws_id, session):
        embedded.append((ws_id, _here()))
        return 1

    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    monkeypatch.setattr(embedding_manager, "get_embedding_manager", lambda: manager)
    monkeypatch.setattr(semantic_indexer, "embed_workspace_agents", embed_workspace_agents)
    monkeypatch.setattr(startup_tasks, "record_error", record_error)

    _, loop_thread = _run(startup_tasks.embed_all_agents_on_startup)

    assert len(db.threads["query"]) == 2 and loop_thread not in db.threads["query"]  # workspaces, counts
    assert manager.threads["ensure_provider"] and loop_thread not in manager.threads["ensure_provider"]
    assert embedded == [("ws-1", loop_thread), ("ws-2", loop_thread)]  # awaited on the loop
    assert db.closed
    record_error.assert_not_called()


def _agent(agent_id, text_hash, embedding):
    return SimpleNamespace(id=agent_id, name=f"agent-{agent_id}", semantic_text_hash=text_hash,
                           semantic_embedding=embedding)


def test_a_workspaces_agents_are_read_built_and_stored_in_threads_and_embedded_on_the_loop(monkeypatch):
    import core.llm.embedding_manager as embedding_manager

    stale, current = _agent(1, "old", None), _agent(2, "hash:agent-2", [0.5] * DIM)
    db = _Db(rows=[stale, current])
    manager = _EmbeddingManager()
    built = []

    def build(agent, session):
        built.append(_here())
        return agent.name

    monkeypatch.setattr(embedding_manager, "get_embedding_manager", lambda: manager)
    monkeypatch.setattr(semantic_indexer, "build_agent_semantic_text", build)
    monkeypatch.setattr(semantic_indexer, "compute_text_hash", lambda text: f"hash:{text}")

    count, loop_thread = _run(lambda: semantic_indexer.embed_workspace_agents("ws-1", db))

    assert count == 1  # the current one was skipped
    assert (stale.semantic_embedding, stale.semantic_text_hash) == ([1.0] * DIM, "hash:agent-1")
    assert db.added == [stale]
    assert len(built) == 2 and loop_thread not in built
    assert db.threads["query"] and loop_thread not in db.threads["query"]
    assert db.threads["commit"] and loop_thread not in db.threads["commit"]
    assert manager.threads["embed"] == [loop_thread]


# --------------------------------------------------------------------------- the action index


def test_the_action_index_is_built_in_a_thread_and_warmed_on_the_loop(monkeypatch):
    import modules.tools.discovery.action_semantic_index as action_semantic_index

    threads = {}

    class _Index:
        async def warm(self):
            threads["warm"] = _here()

    def build():
        threads["build"] = _here()
        return _Index()

    monkeypatch.setattr(action_semantic_index, "get_action_semantic_index", build)

    _, loop_thread = _run(startup_tasks.warm_action_index_on_startup)

    assert threads["build"] != loop_thread
    assert threads["warm"] == loop_thread


def _index(manager, cache, actions):
    """An ActionSemanticIndex over ``actions``, built without its singletons."""
    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = manager
    idx._cache = cache
    idx._registry = SimpleNamespace(get_all=lambda: list(actions))
    idx._action_embeddings = {}
    idx._indexed = False
    idx._corpus_hash = None
    idx._lock = None
    idx._inflight = {}
    idx._rank_inflight = {}
    return idx


class _Cache:
    """The text-addressed embedding cache, recording the thread of each call."""

    def __init__(self):
        self.store = {}
        self.threads = []

    def get_embeddings_batch(self, texts, model="default"):
        self.threads.append(_here())
        return {text: self.store.get(text) for text in texts}

    def set_embeddings_batch(self, embeddings, model="default"):
        self.threads.append(_here())
        self.store.update(embeddings)


def _corpus(monkeypatch, threads=None):
    def corpus_hash():
        if threads is not None:
            threads.append(_here())
        return "0123456789abcdef"

    monkeypatch.setattr(utterance_corpus, "corpus_hash", corpus_hash)
    monkeypatch.setattr(utterance_corpus, "utterances_for", lambda name: [])


def _action(name):
    return ActionDefinition(name=name, description=f"{name} does a thing", category="agents",
                            parameters={"type": "object", "properties": {}}, permission_level="read")


def test_warm_loads_the_provider_and_the_corpus_in_a_thread_before_the_build(monkeypatch):
    manager, corpus_threads, order = _EmbeddingManager(), [], []
    _corpus(monkeypatch, corpus_threads)
    idx = _index(manager, _Cache(), [])

    async def build():
        order.append(("build", _here()))

    idx._index_build = build

    _, loop_thread = _run(idx.warm)

    assert manager.threads["ensure_provider"] and loop_thread not in manager.threads["ensure_provider"]
    assert corpus_threads and loop_thread not in corpus_threads
    assert order == [("build", loop_thread)]


def test_the_index_build_reads_and_writes_its_cache_in_threads(monkeypatch):
    _corpus(monkeypatch)
    manager, cache = _EmbeddingManager(), _Cache()
    idx = _index(manager, cache, [_action("platform_one"), _action("platform_two")])

    _, loop_thread = _run(lambda: idx.ensure_indexed(exclude_admin=False, exclude_promoted=False))

    assert set(idx._action_embeddings) == {"platform_one", "platform_two"}
    assert len(cache.threads) == 2 and loop_thread not in cache.threads  # one read, one write
    assert manager.threads["embed"] == [loop_thread]


# --------------------------------------------------------------------------- the shared context


def test_the_shared_context_is_built_in_a_thread(monkeypatch):
    import modules.context.factory as factory

    threads = []
    monkeypatch.setattr(factory, "get_shared_context", lambda: threads.append(_here()) or MagicMock())
    monkeypatch.setattr(startup_tasks, "record_error", MagicMock())

    _, loop_thread = _run(startup_tasks.ensure_field_memory_collection)

    assert threads and threads[0] != loop_thread
    startup_tasks.record_error.assert_not_called()
