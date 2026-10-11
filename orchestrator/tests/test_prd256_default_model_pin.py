"""PRD-256 Wave 2, US-009: Auto's default model is one constant.

The owner's four-arm run picks Auto's default model; the decision must stay a
one-line change in core/llm/defaults.py (plus the Settings row for agent 1 in
an existing workspace, which the seed never rewrites). These tests pin that:

  * a newly seeded Auto row carries DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL;
  * changing the constant changes the seeded row, with no second copy;
  * an existing Auto row keeps the model the owner chose in Settings;
  * the seed module names no model id of its own.

PURE (no database): the seed runs against a MagicMock session.
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
from core.models.core import Agent  # noqa: E402

OWNER_CHOSEN = {"provider": "anthropic", "model_id": "claude-sonnet-5", "max_tokens": 4000}


def _seed_session(monkeypatch, existing=None):
    """A session whose Auto lookup returns ``existing``; side seeds are stubbed."""
    monkeypatch.setattr(auto_seed, "ensure_builtin_skill", lambda db, name: None)
    monkeypatch.setattr(auto_seed, "seed_brand_designer", lambda db, ws: None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = existing
    return db


def test_a_new_auto_is_seeded_with_the_default_constants(monkeypatch):
    db = _seed_session(monkeypatch)
    agent = auto_seed.seed_auto_agent(db, uuid.uuid4())
    assert agent.model_config["provider"] == llm_defaults.DEFAULT_LLM_PROVIDER
    assert agent.model_config["model_id"] == llm_defaults.DEFAULT_LLM_MODEL


def test_the_owners_decision_is_a_one_line_change_in_defaults(monkeypatch):
    monkeypatch.setattr(llm_defaults, "DEFAULT_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm_defaults, "DEFAULT_LLM_MODEL", "claude-sonnet-5")
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
