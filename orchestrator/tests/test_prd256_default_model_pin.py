"""PRD-256 Wave 2, US-009: Auto's default model is one setting, with one fallback.

The owner's four-arm run picked Auto's default model. Since 19 Oct 2026 (Gerard) that
decision is data: system_settings ``orchestrator_llm`` provider and model. The code
keeps one last-resort pair, in core/llm/defaults.py. These tests pin that:

  * a newly seeded Auto row with no setting carries DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL;
  * with the setting, the seeded row carries the setting, with no second copy;
  * an existing Auto row keeps the model the owner chose in Settings;
  * the seed module names no model id of its own.

PURE (no database): the seed runs against a MagicMock session; settings are faked.
"""
from __future__ import annotations

import inspect
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.llm.defaults as llm_defaults  # noqa: E402
import core.seeds.seed_auto_agent as auto_seed  # noqa: E402
from core.llm import manager as llm_manager  # noqa: E402
from core.models.core import Agent  # noqa: E402

OWNER_CHOSEN = {"provider": "anthropic", "model_id": "claude-sonnet-5", "max_tokens": 4000}


def _settings(monkeypatch, rows):
    """These system-settings rows, and no stored key for any provider."""
    stored = dict(rows)
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: stored.get((c, k), d))
    monkeypatch.setattr(llm_manager, "get_credential_data",
                        lambda provider, environment=None, service_name="orchestrator": {})


def _seed_session(monkeypatch, existing=None):
    """A session whose Auto lookup returns ``existing``; side seeds are stubbed."""
    monkeypatch.setattr(auto_seed, "ensure_builtin_skill", lambda db, name: None)
    monkeypatch.setattr(auto_seed, "seed_brand_designer", lambda db, ws: None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = existing
    return db


def test_a_new_auto_with_no_setting_is_seeded_with_the_default_constants(monkeypatch):
    _settings(monkeypatch, {})
    db = _seed_session(monkeypatch)
    agent = auto_seed.seed_auto_agent(db, uuid.uuid4())
    assert agent.model_config["provider"] == llm_defaults.DEFAULT_LLM_PROVIDER
    assert agent.model_config["model_id"] == llm_defaults.DEFAULT_LLM_MODEL


def test_the_owners_decision_is_the_orchestrator_setting(monkeypatch):
    _settings(monkeypatch, {("orchestrator_llm", "provider"): "anthropic",
                            ("orchestrator_llm", "model"): "claude-sonnet-5"})
    db = _seed_session(monkeypatch)
    agent = auto_seed.seed_auto_agent(db, uuid.uuid4())
    assert agent.model_config["provider"] == "anthropic"
    assert agent.model_config["model_id"] == "claude-sonnet-5"


def test_the_seed_never_rewrites_the_model_on_an_existing_auto(monkeypatch):
    monkeypatch.setattr(auto_seed, "_backfill_auto_persona", lambda agent: "current")
    existing = Agent(name="Auto", model_config=dict(OWNER_CHOSEN))
    db = _seed_session(monkeypatch, existing=existing)
    agent = auto_seed.seed_auto_agent(db, uuid.uuid4())
    assert agent is existing
    assert agent.model_config == OWNER_CHOSEN
    db.add.assert_not_called()


def test_the_seed_names_no_model_of_its_own():
    source = inspect.getsource(auto_seed)
    assert llm_defaults.DEFAULT_LLM_MODEL not in source
    assert f'"{llm_defaults.DEFAULT_LLM_PROVIDER}"' not in source
    assert "DEFAULT_LLM_MODEL =" not in source
