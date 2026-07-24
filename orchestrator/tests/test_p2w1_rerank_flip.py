"""PRD-188 S1: rerank flip — the pipeline's highest-precision stage, dark → on.

The Cohere integration and its call site were always correct and already
degrade gracefully without a key; reranking never ran because of two config
values and one hardcode. These tests pin the flip:

* a fresh RAGConfig defaults reranking ON (via the canonical config accessor);
* an explicit caller value still wins (the S5 eval lever grid depends on it);
* an admin system-setting of "false" still turns it off (default flipped,
  control kept);
* rerank reorders by rerank_score when a key is present, and returns the input
  order unchanged — without error — when it is not (the graceful path stays
  graceful);
* the documents.py search endpoint no longer forces reranking off;
* the model default lives in config.py, not in rerank_manager's old hardcode;
* the dead pre-PRD-136 'rag_rerank_enabled' flat-key read is gone from
  service.py (the migration renamed the key, so that lookup always missed and
  silently pinned reranking off).

All pure — the Cohere boundary is mocked; no network, no DB.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

import core.llm.manager as llm_manager_mod  # noqa: E402
import core.llm.rerank_manager as rerank_manager_mod  # noqa: E402
import modules.rag.service as rag_service_mod  # noqa: E402
from core.llm.rerank_manager import RerankResult  # noqa: E402
from modules.rag.service import RAGConfig, RAGService  # noqa: E402


@pytest.fixture
def hermetic_settings(monkeypatch):
    """No DB, no env: the flat rag-settings blob is empty and the canonical
    get_system_setting returns its passed default — so tests read the shipped
    defaults, not this machine's state."""
    monkeypatch.setattr(rag_service_mod, "_load_rag_settings", lambda force=False: {})
    monkeypatch.setattr(
        llm_manager_mod, "get_system_setting", lambda group, key, default=None: default
    )
    monkeypatch.delenv("RAG_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RAG_RERANK_MODEL", raising=False)


# ---------------------------------------------------------------------------
# The default flip
# ---------------------------------------------------------------------------

def test_rerank_enabled_by_default(hermetic_settings):
    cfg = RAGConfig()
    assert cfg.enable_reranking is True


def test_explicit_caller_value_wins(hermetic_settings):
    assert RAGConfig(enable_reranking=False).enable_reranking is False
    assert RAGConfig(enable_reranking=True).enable_reranking is True


def test_admin_setting_off_is_respected(hermetic_settings, monkeypatch):
    def _setting(group, key, default=None):
        if (group, key) == ("rag", "rerank_enabled"):
            return "false"
        return default

    monkeypatch.setattr(llm_manager_mod, "get_system_setting", _setting)
    assert RAGConfig().enable_reranking is False


def test_dead_prd136_flat_key_read_is_deleted():
    """PRD-136 renamed the settings key; the old flat-map lookup could only
    ever return its default. The read must be gone, not defaulted around."""
    src = Path(rag_service_mod.__file__).read_text()
    assert '_get_rag_setting_str("rag_rerank_enabled"' not in src


# ---------------------------------------------------------------------------
# The rerank stage itself (Cohere boundary mocked)
# ---------------------------------------------------------------------------

def _service():
    return RAGService(RAGConfig(enable_reranking=True))


def test_rerank_runs_when_key_present(hermetic_settings, monkeypatch):
    fake = MagicMock()
    fake.is_available.return_value = True
    fake.rerank = AsyncMock(
        return_value=[
            RerankResult(index=1, relevance_score=0.92),
            RerankResult(index=0, relevance_score=0.31),
        ]
    )
    monkeypatch.setattr(rerank_manager_mod, "get_rerank_manager", lambda: fake)

    candidates = [{"content": "first"}, {"content": "second"}]
    out = asyncio.run(_service()._rerank_candidates("query", candidates))

    assert [c["content"] for c in out] == ["second", "first"]
    assert out[0]["rerank_score"] == 0.92 and out[1]["rerank_score"] == 0.31
    # Immutability at the boundary: the input candidates were not mutated.
    assert "rerank_score" not in candidates[0] and "rerank_score" not in candidates[1]


def test_rerank_degrades_without_key(hermetic_settings, monkeypatch):
    fake = MagicMock()
    fake.is_available.return_value = False
    monkeypatch.setattr(rerank_manager_mod, "get_rerank_manager", lambda: fake)

    candidates = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    out = asyncio.run(_service()._rerank_candidates("query", candidates))

    # Identity order, same objects, no error — the graceful path stays graceful.
    assert out == candidates
    fake.rerank.assert_not_called()


def test_rerank_error_falls_back_to_input_order(hermetic_settings, monkeypatch):
    fake = MagicMock()
    fake.is_available.return_value = True
    fake.rerank = AsyncMock(side_effect=RuntimeError("cohere down"))
    monkeypatch.setattr(rerank_manager_mod, "get_rerank_manager", lambda: fake)

    candidates = [{"content": "a"}, {"content": "b"}]
    out = asyncio.run(_service()._rerank_candidates("query", candidates))
    assert out == candidates


# ---------------------------------------------------------------------------
# The hardcodes are gone
# ---------------------------------------------------------------------------

def test_search_endpoint_does_not_force_rerank_off():
    documents_src = (Path(_orchestrator_root) / "api" / "documents.py").read_text()
    assert "enable_reranking=False" not in documents_src


def test_rerank_model_default_lives_in_config(hermetic_settings):
    from config import config as app_config

    assert app_config.RAG_RERANK_MODEL == "rerank-v3.5"
    # The rerank_manager module-level hardcode is deleted, not shadowed.
    assert not hasattr(rerank_manager_mod, "DEFAULT_RERANK_MODEL")
    assert rerank_manager_mod.RerankManager()._model == "rerank-v3.5"
