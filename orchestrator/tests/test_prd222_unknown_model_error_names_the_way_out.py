"""PRD-222 — an unknown model id on agent creation names the way out (live-test 2026-09-02)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import api.llm_marketplace as marketplace
from modules.tools.discovery.handlers_agents import create_agent


def test_unknown_model_error_points_at_the_default(monkeypatch):
    monkeypatch.setattr(marketplace, "_get_or_create_from_cache", lambda db, model_id: None)
    res = asyncio.run(create_agent(MagicMock(), uuid4(), {"name": "Booking Assistant", "model_id": "anthropic/claude-sonnet-4-20250514"}))
    assert res["success"] is False
    assert "not found in the model catalog" in res["error"]
    assert "Omit model_id" in res["error"]
    assert "workspace default (" in res["error"]
