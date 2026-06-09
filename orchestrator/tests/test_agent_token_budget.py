"""
Agent output-token budget is resolved from the agent's own settings.
=====================================================================

The per-agent "Max Output Tokens" setting is the source of truth for an
agent's output budget. When the agent has not set one, the budget falls back
to the selected model's registry ceiling (``LLMModel.max_output_tokens``), and
only then to the single named constant ``DEFAULT_MAX_OUTPUT_TOKENS``.

Power mode plays NO role in the token budget — it governs only the LLM tier
and tool-iteration count. There are no hardcoded token literals (the old
silent ``2000`` defaults and the ``min(2000, ceiling)`` clamp are gone).

These tests are DB-free: ``_model_max_output_tokens`` is exercised against a
mock session, and the bound method is reached via ``__new__`` so no real
AgentFactory construction (and no DB) is required.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure orchestrator package is importable
_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from core.llm.defaults import DEFAULT_MAX_OUTPUT_TOKENS, get_default_model_config
from modules.agents.factory.agent_factory import AgentFactory, AgentMetadata, ModelConfiguration


# --- The single named default -------------------------------------------------

def test_default_model_config_uses_named_constant():
    """The canonical default config carries DEFAULT_MAX_OUTPUT_TOKENS — not a literal."""
    assert get_default_model_config()["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_named_default_is_not_the_old_2000_literal():
    """Guard against a silent regression back to the old 2000 cap."""
    assert DEFAULT_MAX_OUTPUT_TOKENS != 2000
    assert DEFAULT_MAX_OUTPUT_TOKENS >= 8000


# --- ModelConfiguration defaults ----------------------------------------------

def test_model_configuration_dataclass_default():
    """A ModelConfiguration with no explicit max_tokens defaults to the constant."""
    mc = ModelConfiguration(provider="openrouter", model_id="x")
    assert mc.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_from_dict_defaults_to_constant_when_absent():
    """from_dict with no max_tokens key falls back to the constant, not 2000."""
    mc = ModelConfiguration.from_dict({"provider": "openrouter", "model_id": "x"})
    assert mc.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_from_dict_honours_explicit_agent_setting():
    """An explicit agent setting (the slider value) wins over every fallback."""
    mc = ModelConfiguration.from_dict({"provider": "openrouter", "model_id": "x", "max_tokens": 16000})
    assert mc.max_tokens == 16000


def test_get_model_config_preferred_model_defaults_to_constant():
    """The legacy preferred_model path defaults max_tokens to the constant."""
    meta = AgentMetadata(name="A", agent_type="t", preferred_model="some/model")
    assert meta.get_model_config().max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_get_model_config_preferred_model_honours_explicit():
    """An explicit max_tokens on the legacy path is preserved."""
    meta = AgentMetadata(name="A", agent_type="t", preferred_model="some/model", max_tokens=12000)
    assert meta.get_model_config().max_tokens == 12000


# --- _model_max_output_tokens: the model-ceiling fallback ---------------------

def _factory_with_session(session):
    """Build an AgentFactory shell without running __init__ (no DB needed)."""
    factory = AgentFactory.__new__(AgentFactory)
    factory.db_session = session
    factory.logger = logging.getLogger("test_agent_token_budget")
    return factory


def test_model_ceiling_used_when_model_in_registry():
    """When the model exists in the registry, its own ceiling is returned."""
    model_row = MagicMock()
    model_row.max_output_tokens = 16384
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = model_row

    factory = _factory_with_session(db)
    assert factory._model_max_output_tokens("gpt-4o") == 16384


def test_constant_fallback_when_model_not_in_registry():
    """An unknown model falls back to the single named default."""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None

    factory = _factory_with_session(db)
    assert factory._model_max_output_tokens("ghost/model") == DEFAULT_MAX_OUTPUT_TOKENS


def test_constant_fallback_when_no_db_session():
    """With no DB session there is no registry to consult — use the constant."""
    factory = _factory_with_session(None)
    assert factory._model_max_output_tokens("gpt-4o") == DEFAULT_MAX_OUTPUT_TOKENS


def test_constant_fallback_when_no_model_id():
    """A missing model_id resolves to the constant, never a literal."""
    factory = _factory_with_session(MagicMock())
    assert factory._model_max_output_tokens(None) == DEFAULT_MAX_OUTPUT_TOKENS


def test_db_error_falls_back_to_constant():
    """A registry lookup failure must not raise — the constant is returned."""
    db = MagicMock()
    db.query.side_effect = RuntimeError("db down")

    factory = _factory_with_session(db)
    assert factory._model_max_output_tokens("gpt-4o") == DEFAULT_MAX_OUTPUT_TOKENS
