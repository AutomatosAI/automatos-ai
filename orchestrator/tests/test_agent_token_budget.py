"""
Agent output-token budget is resolved from the agent's own settings.
=====================================================================

The per-agent "Max Output Tokens" setting is the source of truth for an
agent's output budget. When the agent has not set one, the budget is an agent
run's (8,000). F196 (night 6): it used to be the selected model's registry
ceiling, and 65,535 reserved per call was refused at a low provider balance. The
ceiling (``LLMModel.max_output_tokens``) now only caps a budget.

Power mode plays NO role in the token budget — it governs only the LLM tier
and tool-iteration count. There are no hardcoded token literals (the old
silent ``2000`` defaults and the ``min(2000, ceiling)`` clamp are gone).

These tests are DB-free: ``_model_ceiling`` is exercised against a mock
session, and the bound method is reached via ``__new__`` so no real
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


# --- _model_ceiling caps; _output_budget decides (F196) --------------------------

def _factory_with_session(session):
    """Build an AgentFactory shell without running __init__ (no DB needed)."""
    factory = AgentFactory.__new__(AgentFactory)
    factory.db_session = session
    factory.logger = logging.getLogger("test_agent_token_budget")
    return factory


def test_model_ceiling_read_from_the_registry():
    """When the model exists in the registry, its own ceiling is returned."""
    model_row = MagicMock()
    model_row.max_output_tokens = 16384
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = model_row

    factory = _factory_with_session(db)
    assert factory._model_ceiling("gpt-4o") == 16384


def test_no_ceiling_when_model_not_in_registry():
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None

    factory = _factory_with_session(db)
    assert factory._model_ceiling("ghost/model") is None


def test_no_ceiling_when_no_db_session():
    assert _factory_with_session(None)._model_ceiling("gpt-4o") is None


def test_no_ceiling_when_no_model_id():
    assert _factory_with_session(MagicMock())._model_ceiling(None) is None


def test_db_error_means_no_ceiling():
    """A registry lookup failure must not raise."""
    db = MagicMock()
    db.query.side_effect = RuntimeError("db down")

    assert _factory_with_session(db)._model_ceiling("gpt-4o") is None


def test_the_budget_is_the_agents_setting_else_an_agent_runs_capped_by_the_ceiling(monkeypatch):
    from core.llm import output_budget

    monkeypatch.setattr(output_budget, "_stored", lambda purpose: None)
    assert AgentFactory._output_budget(None, 16384) == 8000       # never the ceiling
    assert AgentFactory._output_budget(None, 4096) == 4096        # the ceiling caps it
    assert AgentFactory._output_budget(12000, 16384) == 12000     # the agent's own setting wins
    assert AgentFactory._output_budget(20000, 16384) == 16384
    assert AgentFactory._output_budget(None, None) == 8000
