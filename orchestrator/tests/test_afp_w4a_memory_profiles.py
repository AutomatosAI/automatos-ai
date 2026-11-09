"""Wave 4a: memory profiles say what distillation keeps as a fact, by domain (§5.6).

An agent's ``configuration.memory_profile`` picks one, else its workspace's
``settings.memory_profile``, else ``general``, which is today's prompt unchanged. A profile adds
its focus, what to keep and skip, and examples before the prompt's writing rules. Profiles are
seeded rows (``memory_profiles``); a workspace's own replaces a seeded one with its slug. Off
unless ``MEMORY_PROFILES_ENABLED``. The real database for the table, the seeds and the route.
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from modules.memory import memory_profiles as mp

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "memory_profiles_20261109.py"
SAID = "My March invoice was charged twice. Can you refund one?"
REPLIED = "I've raised a refund for the duplicate March invoice; you'll have it by 20 November."


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.memory_profiles import MemoryProfile

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    MemoryProfile.__table__.create(eng, checkfirst=True)  # a test database built before the table
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    from core.seeds.seed_memory_profiles import seed_memory_profiles

    s = new_session()
    seed_memory_profiles(s)
    s.close()


@pytest.fixture
def profiles_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "MEMORY_PROFILES_ENABLED", True)


def _workspace(new_session, settings=None) -> str:
    from sqlalchemy import text

    ws = str(uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, settings) VALUES (CAST(:id AS uuid), 'Wave 4a', "
                   "CAST(:settings AS jsonb))"), {"id": ws, "settings": __import__("json").dumps(settings or {})})
    s.commit()
    s.close()
    return ws


def _drop(new_session, ws):
    from sqlalchemy import text

    s = new_session.sweep()
    s.execute(text("DELETE FROM memory_profiles WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


# --- the table -----------------------------------------------------------------------------------

def test_the_migration_declares_the_models_shape():
    from core.models.memory_profiles import SEEDED_WHERE, WORKSPACE_WHERE, MemoryProfile

    source = MIGRATION.read_text()
    tree = ast.parse(source)
    table = next(node.value.value for node in tree.body
                 if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_TABLE")
    declared = set(re.findall(r"^\s+([a-z_]+) [A-Z]", table, re.M)) - {"CREATE"}
    assert declared == {c.name for c in MemoryProfile.__table__.columns}
    assert f"WHERE {SEEDED_WHERE};" in source and f"WHERE {WORKSPACE_WHERE};" in source
    for index in MemoryProfile.__table__.indexes:
        assert index.name in source


def test_the_seeds_are_idempotent_and_general_is_todays_prompt(new_session):
    from core.seeds.seed_memory_profiles import SEEDED_PROFILES, seed_memory_profiles
    from modules.memory.write_contract import MEMORY_FACT_TYPES

    s = new_session()
    seed_memory_profiles(s)
    assert seed_memory_profiles(s) == (0, 0)
    s.close()
    by_slug = {p["slug"]: p for p in SEEDED_PROFILES}
    assert set(by_slug) == {"general", "customer", "incident", "case"}
    assert not by_slug["general"]["focus"] and not by_slug["general"]["examples"]
    for profile in SEEDED_PROFILES:
        assert all(e["type"] in MEMORY_FACT_TYPES for e in profile["examples"]), profile["slug"]


# --- the choice ----------------------------------------------------------------------------------

def test_the_agents_profile_wins_then_the_workspaces_then_general():
    assert mp.chosen_slug({"memory_profile": "customer"}, {"memory_profile": "incident"}) == "customer"
    assert mp.chosen_slug({}, {"memory_profile": "incident"}) == "incident"
    assert mp.chosen_slug(None, None) == mp.GENERAL


def test_an_agents_profile_is_a_slug():
    from api.agents import _reject_invalid_runtime

    assert mp.slug_errors({"memory_profile": "customer"}) == [] and mp.slug_errors({}) == []
    with pytest.raises(HTTPException) as refused:
        _reject_invalid_runtime({"memory_profile": ["customer"]})
    assert refused.value.status_code == 400


def test_with_the_dial_off_every_distillation_is_general(new_session, seeded, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "MEMORY_PROFILES_ENABLED", False)
    ws = _workspace(new_session, {"memory_profile": "customer"})
    try:
        s = new_session()
        assert mp.profile_for(s, ws, None) is None
        s.close()
    finally:
        _drop(new_session, ws)


def test_a_workspace_picks_a_seeded_profile_and_its_own_replaces_it(new_session, seeded, profiles_on):
    from core.models.memory_profiles import MemoryProfile

    ws = _workspace(new_session, {"memory_profile": "customer"})
    try:
        s = new_session()
        assert mp.profile_for(s, ws, None).name == "Customer"
        s.add(MemoryProfile(workspace_id=ws, slug="customer", name="Acme customers", focus="Acme's own focus."))
        s.commit()
        assert mp.profile_for(s, ws, None).focus == "Acme's own focus."
        s.close()
    finally:
        _drop(new_session, ws)


def test_general_and_an_unknown_slug_distil_as_today(new_session, seeded, profiles_on, caplog):
    ws = _workspace(new_session, {"memory_profile": "general"})
    other = _workspace(new_session, {"memory_profile": "astrology"})
    try:
        s = new_session()
        assert mp.profile_for(s, ws, None) is None
        assert mp.profile_for(s, other, None) is None
        s.close()
        assert "astrology" in caplog.text
    finally:
        _drop(new_session, ws)
        _drop(new_session, other)


# --- the prompt ----------------------------------------------------------------------------------

def _customer() -> mp.Profile:
    from core.seeds.seed_memory_profiles import SEEDED_PROFILES

    p = next(p for p in SEEDED_PROFILES if p["slug"] == "customer")
    return mp.Profile(slug=p["slug"], name=p["name"], focus=p["focus"], keep=tuple(p["keep"]),
                      skip=tuple(p["skip"]), examples=tuple(p["examples"]))


def test_general_leaves_the_distillation_prompt_as_it_is():
    from consumers.chatbot.smart_memory import SmartMemoryManager

    prompt = SmartMemoryManager._build_distill_prompt(SAID, REPLIED)
    assert mp.with_profile(prompt, None) is prompt


def test_a_profile_goes_before_the_writing_rules():
    from consumers.chatbot.smart_memory import SmartMemoryManager

    prompt = SmartMemoryManager._build_distill_prompt(SAID, REPLIED)
    shaped = mp.with_profile(prompt, _customer())
    block_at, rules_at = shaped.index(mp.PROFILE_HEADING), shaped.index(mp.WRITING_RULES_ANCHOR)
    assert block_at < rules_at < shaped.index("Return ONLY a JSON array")
    assert '"type": "open_loop"' in shaped and "Northwind" in shaped
    assert shaped.replace(mp.profile_block(_customer()) + "\n\n", "") == prompt  # nothing else changes


def test_the_distillation_asks_with_the_profile(monkeypatch):
    from consumers.chatbot import smart_memory

    asked = []

    class _LLM:
        async def generate_response(self, messages):
            asked.append(messages[0]["content"])
            return SimpleNamespace(content="[]")

    async def _profile(workspace_id, agent_id):
        return _customer()

    monkeypatch.setattr("core.llm.create_llm_manager", lambda **kwargs: _LLM())
    monkeypatch.setattr("modules.memory.memory_profiles.distillation_profile", _profile)
    manager = object.__new__(smart_memory.SmartMemoryManager)
    facts = asyncio.run(manager._distill_durable_facts(SAID, REPLIED, workspace_id="ws", agent_id=7))
    assert facts == [] and mp.PROFILE_HEADING in asked[0]


# --- the route -----------------------------------------------------------------------------------

def test_the_route_lists_the_profiles_a_workspace_can_pick(new_session, seeded, profiles_on):
    from api.memory_profiles import list_memory_profiles
    from core.models.memory_profiles import MemoryProfile

    ws = _workspace(new_session)
    try:
        s = new_session()
        s.add(MemoryProfile(workspace_id=ws, slug="incident", name="Acme on-call", focus="Acme's runbook."))
        s.commit()
        listed = list_memory_profiles(ctx=SimpleNamespace(workspace_id=ws), db=s)
        s.close()
        by_slug = {p["slug"]: p for p in listed["profiles"]}
        assert {"general", "customer", "incident", "case"} <= set(by_slug) and listed["enabled"] is True
        assert by_slug["incident"]["name"] == "Acme on-call" and by_slug["incident"]["seeded"] is False
        assert by_slug["customer"]["seeded"] is True
    finally:
        _drop(new_session, ws)
