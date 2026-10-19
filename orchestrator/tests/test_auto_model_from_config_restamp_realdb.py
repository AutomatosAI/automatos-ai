"""The Auto model restamp against the real schema (Gerard, 19 Oct 2026).

``restamp_inheriting_auto_agents`` selects every workspace's Auto row (a system
agent, slug ``auto-%``, a workspace set) and moves only the ones still on the
platform default. Here, on two workspaces: their Auto rows on shipped seed pairs
move and persist (the JSON column is flagged); a third workspace's Auto chosen in
Settings stays; a workspace's own agent on the same seed pair, and a global system
agent named ``auto-…``, are never touched. Runs in ``db_session``'s transaction,
rolled back after; rows other tests left are not asserted on.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from core.llm import auto_model_sync as sync
from core.llm import new_agent_model
from core.models.core import Agent

HAIKU = ("anthropic", "claude-haiku-5-5")
GEMINI_SEED = {"provider": "openrouter", "model_id": "google/gemini-2.5-flash", "temperature": 0.7, "max_tokens": 4000}
GPT4O_SEED = {"provider": "openrouter", "model_id": "openai/gpt-4o", "temperature": 0.7, "max_tokens": 4000}
CHOSEN = {"provider": "openrouter", "model_id": "openai/gpt-5.5", "temperature": 0.2, "max_tokens": 4000}


def _agent(db, name, model_config, *, ws=None, slug=None, system=False):
    agent = Agent(name=name, agent_type="system" if system else "custom", status="active",
                  workspace_id=UUID(ws) if ws else None, slug=slug, is_system_agent=system,
                  model_config=dict(model_config))
    db.add(agent)
    db.flush()
    return agent


@pytest.fixture
def world(db_session, seed_workspace, monkeypatch):
    monkeypatch.setattr(new_agent_model, "default_route", lambda: HAIKU)
    a, b, c = seed_workspace(name="restamp-a"), seed_workspace(name="restamp-b"), seed_workspace(name="restamp-c")
    rows = {
        "auto_a": _agent(db_session, "Auto", GEMINI_SEED, ws=a, slug=f"auto-{a}", system=True),
        "auto_b": _agent(db_session, "Auto", GPT4O_SEED, ws=b, slug=f"auto-{b}", system=True),
        "auto_c": _agent(db_session, "Auto", CHOSEN, ws=c, slug=f"auto-{c}", system=True),
        "worker_a": _agent(db_session, "Barista", GEMINI_SEED, ws=a, slug=f"barista-{a}"),
        "global": _agent(db_session, "Auto CTO", GEMINI_SEED, slug=f"auto-cto-{a}", system=True),
    }
    return db_session, rows


def _reloaded(db, agent):
    db.expire(agent)
    return db.get(Agent, agent.id).model_config


def test_the_restamp_moves_only_inheriting_auto_rows_across_workspaces(world):
    db, rows = world
    selected = {agent.id for agent in sync.auto_agent_rows(db)}
    assert {rows["auto_a"].id, rows["auto_b"].id, rows["auto_c"].id} <= selected
    assert rows["worker_a"].id not in selected and rows["global"].id not in selected

    sync.restamp_inheriting_auto_agents(db)

    for key in ("auto_a", "auto_b"):
        stored = _reloaded(db, rows[key])
        assert (stored["provider"], stored["model_id"]) == HAIKU
        assert stored["source"] == sync.PLATFORM_DEFAULT_SOURCE and stored["max_tokens"] == 4000
    assert _reloaded(db, rows["auto_c"]) == CHOSEN
    assert _reloaded(db, rows["worker_a"]) == GEMINI_SEED
    assert _reloaded(db, rows["global"]) == GEMINI_SEED
