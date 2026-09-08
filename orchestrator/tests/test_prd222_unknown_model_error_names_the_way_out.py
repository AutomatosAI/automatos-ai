"""PRD-223 — an unknown model id on agent creation uses the governed default and says so.

Prod 2026-09-02 (post-#672): told "omit model_id to use the default", the model
retried the SAME unknown id twice and the onboarding build stalled. The registry
decides the model; an unknown id now means the workspace default, stated in the
result and the log — the agent is still created. A known id still passes the
PRD-223 policy gate.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import api.llm_marketplace as marketplace
import core.llm.model_policy as policy
from modules.tools.discovery.handlers_agents import create_agent


def test_unknown_model_creates_the_agent_on_the_default_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(marketplace, "_get_or_create_from_cache", lambda db, model_id, provider=None: None)
    with caplog.at_level(logging.WARNING, logger="modules.tools.discovery.handlers_agents"):
        res = asyncio.run(create_agent(MagicMock(), uuid4(), {"name": "Booking Assistant", "model_id": "anthropic/claude-sonnet-4-20250514"}))
    assert res["success"] is True
    assert "created successfully" in res["message"]
    assert "not in the catalog" in res["model_note"] and "workspace default (" in res["model_note"]
    assert res["model_note"] in res["message"]
    assert any("unknown model" in r.getMessage() for r in caplog.records)


def test_known_model_still_passes_the_policy_gate(monkeypatch):
    monkeypatch.setattr(
        marketplace, "_get_or_create_from_cache",
        lambda db, model_id, provider=None: SimpleNamespace(provider="openrouter", serving_provider="openrouter"),
    )
    seen = {}

    def gate(db, workspace_id, model_id, orchestrator_seat=False, provider=None):
        seen["model_id"] = model_id
        return False, "not allowed on this plan"

    monkeypatch.setattr(policy, "check_model_for_agent", gate)
    res = asyncio.run(create_agent(MagicMock(), uuid4(), {"name": "X", "model_id": "google/gemini-2.5-pro"}))
    assert res["success"] is False and "Model rejected" in res["error"]
    assert seen["model_id"] == "google/gemini-2.5-pro"
