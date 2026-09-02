"""
PRD-232 US-006 — the corpus reaches the embeddings (+ parameter enums).
=======================================================================

C6: ``_build_embedding_text`` embedded only name+description+tags+examples+
category, so "close the blocked tickets" never reached ``platform_update_task_status``
(its phrasing lived nowhere in the embedded text) and enum values like the
``done`` status were invisible to the ranker. US-006 folds each action's seeded
utterance corpus (US-005) AND its parameter enum values into the embedded text,
and hardens ``ensure_indexed`` to re-embed when the corpus changes.

Both invalidation layers are covered:
  * Redis embedding cache is keyed on the TEXT (embedding:{model}:sha256(text)),
    so changing an action's utterances changes its text and re-embeds only that
    action — proven by the counting-provider test below.
  * The per-process dict is keyed by action NAME, so it needs the explicit
    ``corpus_hash()`` guard — proven by asserting the dict is dropped + rebuilt.

The ranking fixture is a real, deterministic *lexical* embedding (hashed
bag-of-words) — NOT ``DeterministicEmbeddingProvider`` (hash-random, no semantic
signal, banned outside test fixtures by PRD-185 S3). Because the corpus injects
the query's vocabulary into the right action's text, lexical overlap is enough to
prove the wiring end-to-end without any network/LLM call. Async bodies run via
``asyncio.run`` to match the sibling US-003 suite (asyncio_mode = strict).
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import types
from pathlib import Path

from modules.tools.discovery.action_registry import ActionDefinition, get_action_registry
from modules.tools.discovery.action_semantic_index import ActionSemanticIndex
from modules.tools.discovery import utterance_corpus

_ORCH_ROOT = Path(__file__).resolve().parent.parent


# ── lexical embedding fixture (deterministic, hermetic, semantically real) ───
_LEX_DIM = 1024
_STOP = {
    "the", "a", "an", "to", "of", "for", "and", "or", "my", "me", "i", "is",
    "are", "in", "on", "at", "with", "this", "that", "it", "any", "have", "do",
    "from", "now", "please",
}


def _lex_tokens(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _lex_vec(text: str):
    v = [0.0] * _LEX_DIM
    for tok in _lex_tokens(text):
        v[int(hashlib.sha1(tok.encode()).hexdigest(), 16) % _LEX_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


class _LexEM:
    """A real (if simple) lexical embedding provider. Deterministic, no network."""

    def __init__(self):
        self.provider = types.SimpleNamespace(config=types.SimpleNamespace(model="lex"))
        self.embedded_texts = []

    def get_provider_info(self):
        return {"provider": "lex", "model": "lex", "dimension": _LEX_DIM}

    def get_dimension(self):
        return _LEX_DIM

    async def generate_embedding(self, text):
        return _lex_vec(text)

    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        self.embedded_texts.extend(texts)
        return [_lex_vec(t) for t in texts]


class _TextCache:
    """Text-addressed embedding cache, exactly like CacheService: keyed on
    (model, text). Records writes so a test can see which texts re-embedded."""

    def __init__(self):
        self.store = {}
        self.sets = []

    def get_embeddings_batch(self, texts, model="default"):
        bucket = self.store.get(model, {})
        return {t: bucket.get(t) for t in texts}

    def set_embeddings_batch(self, embeddings, model="default"):
        self.store.setdefault(model, {}).update(embeddings)
        self.sets.extend(embeddings.keys())


def _index_with(em, cache, registry):
    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = em
    idx._cache = cache
    idx._registry = registry
    idx._action_embeddings = {}
    idx._indexed = False
    idx._corpus_hash = None
    idx._lock = None
    idx._inflight = {}
    idx._rank_inflight = {}
    return idx


def _stub_action(name, description, category):
    return ActionDefinition(name=name, description=description, category=category,
                            parameters={"type": "object", "properties": {}})


class _StubRegistry:
    def __init__(self, actions):
        self._actions = list(actions)

    def get_all(self):
        return list(self._actions)


def _install_corpus(monkeypatch, mapping):
    """Point the runtime corpus loader at an in-memory mapping (no disk), with a
    content hash derived from it — exactly what _load_from_disk would produce."""
    hasher = hashlib.sha256()
    for name in sorted(mapping):
        hasher.update(name.encode())
        for text in mapping[name]:
            hasher.update(b"\x00")
            hasher.update(text.encode())
    digest = hasher.hexdigest()[:16]
    monkeypatch.setattr(utterance_corpus, "utterances_for", lambda n: list(mapping.get(n, [])))
    monkeypatch.setattr(utterance_corpus, "corpus_hash", lambda: digest)


# ── AC: the embedded text now carries utterances + enum values ───────────────
def test_build_embedding_text_includes_utterances_and_enums(monkeypatch):
    action = ActionDefinition(
        name="platform_update_task_status",
        description="Change a task's status.",
        category="tasks",
        parameters={"type": "object", "properties": {
            "status": {"type": "string", "enum": ["inbox", "assigned", "in_progress", "review", "done"]},
        }},
        examples=["mark task 3 as done"],
        tags=["tasks", "write"],
    )
    monkeypatch.setattr(
        utterance_corpus, "utterances_for",
        lambda name: ["close the blocked tickets", "kill task 9"] if name == action.name else [],
    )
    text = ActionSemanticIndex._build_embedding_text(action)
    assert "Options:" in text
    for value in ("inbox", "assigned", "in_progress", "review", "done"):
        assert value in text, f"enum value {value!r} missing from embedded text"
    assert "Utterances:" in text
    assert "close the blocked tickets" in text
    assert "kill task 9" in text


def test_collect_enum_values_dedupes_and_flattens():
    params = {"properties": {
        "status": {"enum": ["inbox", "done", "done"]},
        "tags": {"type": "array", "items": {"enum": ["urgent", "inbox"]}},
        "title": {"type": "string"},
    }}
    vals = ActionSemanticIndex._collect_enum_values(params)
    assert vals == ["inbox", "done", "urgent"]  # order-preserving, deduped, incl. items.enum
    assert ActionSemanticIndex._collect_enum_values(None) == []


# ── AC: ranking — "close the blocked tickets" ranks the board-write top-5 ────
def test_close_blocked_tickets_ranks_update_task_status_top5():
    async def _run():
        idx = _index_with(_LexEM(), _TextCache(), get_action_registry())
        ranked = await idx.rank_actions("close the blocked tickets", top_k=5,
                                        exclude_admin=True, exclude_promoted=True)
        return [n for n, _ in ranked]

    names = asyncio.run(_run())
    assert "platform_update_task_status" in names, (
        f"board-write action absent from top-5 for the VECTOR query: {names}"
    )


def test_register_variant_pairs_rank_top5():
    """Two more register-variant pairs from the corpus (no mail/inbox family
    exists): the delete register and the cancel/kill register land their target
    action in the top-5 semantic floor."""
    async def _run(query):
        idx = _index_with(_LexEM(), _TextCache(), get_action_registry())
        ranked = await idx.rank_actions(query, top_k=5, exclude_admin=True, exclude_promoted=True)
        return [n for n, _ in ranked]

    for query, target in [
        ("trash that document from the knowledge base", "platform_delete_document"),
        ("kill the mission", "platform_cancel_mission"),
    ]:
        names = asyncio.run(_run(query))
        assert target in names, f"{query!r} did not rank {target} in top-5: {names}"


def test_enum_value_reaches_ranker():
    """The task-status enum value 'done' (embedded via Options:) lets 'mark it as
    done' reach the board-write action."""
    async def _run():
        idx = _index_with(_LexEM(), _TextCache(), get_action_registry())
        ranked = await idx.rank_actions("mark it as done", top_k=5,
                                        exclude_admin=True, exclude_promoted=True)
        return [n for n, _ in ranked]

    assert "platform_update_task_status" in asyncio.run(_run())


# ── AC: corpus-hash change forces re-embed — BOTH layers ─────────────────────
def test_corpus_hash_change_forces_reembed_both_layers(monkeypatch):
    """A corpus change re-embeds exactly the changed action: the per-process dict
    is dropped by the hash guard (per-process layer) and the Redis text-address
    misses only the changed text (Redis layer)."""
    reg = _StubRegistry([
        _stub_action("platform_update_task_status", "Change a task's status.", "tasks"),
        _stub_action("platform_list_agents", "List the agents.", "agents"),
    ])
    em, cache = _LexEM(), _TextCache()
    idx = _index_with(em, cache, reg)

    _install_corpus(monkeypatch, {
        "platform_update_task_status": ["close the ticket"],
        "platform_list_agents": ["show agents"],
    })
    asyncio.run(idx.ensure_indexed(exclude_admin=False, exclude_promoted=False))
    assert idx._indexed and len(idx._action_embeddings) == 2
    hash_a = idx._corpus_hash
    assert len(em.embedded_texts) == 2  # both texts embedded from cold

    # change ONE action's utterances -> new global hash + new text for that action
    _install_corpus(monkeypatch, {
        "platform_update_task_status": ["close the blocked tickets now"],
        "platform_list_agents": ["show agents"],
    })
    em.embedded_texts.clear()
    asyncio.run(idx.ensure_indexed(exclude_admin=False, exclude_promoted=False))

    # per-process layer: hash moved, dict was dropped and rebuilt to 2
    assert idx._corpus_hash != hash_a
    assert len(idx._action_embeddings) == 2
    # Redis layer: text-addressed, so ONLY the changed action re-embedded upstream
    assert len(em.embedded_texts) == 1, em.embedded_texts
    assert "platform_update_task_status" in em.embedded_texts[0]
    assert "blocked tickets now" in em.embedded_texts[0]


def test_unchanged_corpus_does_not_reembed(monkeypatch):
    reg = _StubRegistry([_stub_action("platform_list_agents", "List the agents.", "agents")])
    em = _LexEM()
    idx = _index_with(em, _TextCache(), reg)
    _install_corpus(monkeypatch, {"platform_list_agents": ["show agents"]})
    asyncio.run(idx.ensure_indexed(exclude_admin=False, exclude_promoted=False))
    em.embedded_texts.clear()
    asyncio.run(idx.ensure_indexed(exclude_admin=False, exclude_promoted=False))  # same hash
    assert em.embedded_texts == []  # nothing re-embedded


# ── AC: DeterministicEmbeddingProvider never used outside test fixtures ───────
def test_no_deterministic_embedding_provider_in_index_or_corpus():
    for rel in (
        "modules/tools/discovery/action_semantic_index.py",
        "modules/tools/discovery/utterance_corpus.py",
    ):
        src = (_ORCH_ROOT / rel).read_text()
        assert "DeterministicEmbeddingProvider" not in src, f"{rel} references the banned provider"


def test_index_source_references_utterances():
    """Acceptance parity (grep -qi 'utterance' on the index): the embedding-text
    builder names the corpus."""
    src = (_ORCH_ROOT / "modules/tools/discovery/action_semantic_index.py").read_text()
    assert "utterance" in src.lower()
