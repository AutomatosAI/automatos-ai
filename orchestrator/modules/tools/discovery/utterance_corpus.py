"""
PRD-232 US-006 — runtime loader for the synthetic utterance corpus.
===================================================================

Loads the hand-authored seed utterances (``core/seeds/utterances/<category>.yaml``,
US-005) into ``{action_name -> [utterance texts]}`` and exposes a content hash.

``ActionSemanticIndex._build_embedding_text`` folds each action's utterances (and
its parameter enum values) into the text it embeds, so a query phrased like any
seeded utterance lands near the right action. Because the Redis embedding cache
is keyed on the *text* (``embedding:{model}:sha256(text)``), changing an action's
utterances changes its text and re-embeds THAT action automatically. The
per-process embedding dict is keyed by action *name*, so it needs the explicit
``corpus_hash()`` guard in ``ensure_indexed`` — the two together are the
"invalidate both layers on corpus change" contract.

Cached at module level (load once per process, like the skill loader). Tests
reset via ``reset_cache()`` or monkeypatch the module functions. No LLM, no
network, no DB — pure YAML read.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# orchestrator/modules/tools/discovery/utterance_corpus.py -> parents[3] == orchestrator/
CORPUS_DIR = Path(__file__).resolve().parents[3] / "core" / "seeds" / "utterances"

_lock = threading.Lock()
# (corpus mapping, content hash) — None until first load.
_cache: Optional[Tuple[Dict[str, List[str]], str]] = None


def _load_from_disk() -> Tuple[Dict[str, List[str]], str]:
    """Parse every ``<category>.yaml`` into ``{action_name: [texts]}`` + a stable
    content hash. Tolerant: a missing dir or unreadable file yields an empty
    contribution rather than breaking indexing (embedding text just omits the
    utterance section)."""
    import yaml

    corpus: Dict[str, List[str]] = {}
    if CORPUS_DIR.is_dir():
        for path in sorted(CORPUS_DIR.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except Exception:
                logger.warning("utterance_corpus: could not parse %s — skipping", path.name)
                continue
            actions = raw.get("actions") if isinstance(raw, dict) else None
            if not isinstance(actions, dict):
                continue
            for name, utts in actions.items():
                if not isinstance(utts, list):
                    continue
                texts = [
                    u["text"].strip()
                    for u in utts
                    if isinstance(u, dict) and isinstance(u.get("text"), str) and u["text"].strip()
                ]
                if texts:
                    corpus.setdefault(name, []).extend(texts)

    hasher = hashlib.sha256()
    for name in sorted(corpus):
        hasher.update(name.encode("utf-8"))
        for text in corpus[name]:
            hasher.update(b"\x00")
            hasher.update(text.encode("utf-8"))
    return corpus, hasher.hexdigest()[:16]


def _ensure_loaded() -> Tuple[Dict[str, List[str]], str]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load_from_disk()
    return _cache


def load_utterance_corpus() -> Dict[str, List[str]]:
    """``{action_name -> [utterance texts]}`` for every action that has seeds."""
    return _ensure_loaded()[0]


def corpus_hash() -> str:
    """Stable 16-hex content hash of the whole corpus (invalidation key)."""
    return _ensure_loaded()[1]


def utterances_for(action_name: str) -> List[str]:
    """Seed utterances for one action ( [] if none seeded )."""
    return load_utterance_corpus().get(action_name, [])


def reset_cache() -> None:
    """Drop the module-level cache so the next call re-reads disk (test hook)."""
    global _cache
    with _lock:
        _cache = None
