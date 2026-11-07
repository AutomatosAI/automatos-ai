"""Auto's soul has one source: its agent row (7 Nov 2026).

On Automatos Cloud a workspace's chat spoke with ``settings.orchestrator.custom_soul`` (8,553
characters) while Settings → Orchestrator showed the Auto row's ``custom_persona_prompt`` (2,187),
and when the two were equal the same soul went into the prompt twice: once as the personality
block, again as "Persona & Communication Style". Now:

* the chat's personality block for Auto is the row's soul, exactly once, whatever the settings hold;
* a non-Auto agent keeps its own persona section, as before;
* Settings GET reports the soul the chat uses, and a save writes the row and never the setting;
* a soul Auto writes on its own row (platform_update_agent) is marked ``custom``, so Settings
  shows it rather than a preset's name over it.

The identity section and the base prompt run for real; the routes run with a fake workspace and
Auto row (no DB), as tests/test_prd263_workspace_harness_mode_api.py does.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.workspaces as workspaces_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from consumers.chatbot import personality  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.context.sections.base import SectionContext  # noqa: E402
from modules.context.sections.identity import IdentitySection  # noqa: E402
from services.orchestrator_auto_agent import is_workspace_auto, mark_soul_custom  # noqa: E402

WS_ID = uuid.uuid4()
ROUTE = "/api/workspaces/current/orchestrator"
ROW_SOUL = "I'm Auto, born in Galway. I answer straight, I keep my word, and I never waffle."
STALE_SOUL = "I'm the settings copy nobody can see in Settings any more."
PERSONA_HEADING = "## Persona & Communication Style"
CUSTOM = {"personality_mode": "custom", "communication_style": "balanced"}


def _auto_row(soul=ROW_SOUL, mode="custom", **overrides):
    row = SimpleNamespace(
        id=7, name="Auto", slug=f"auto-{WS_ID}", is_system_agent=True, workspace_id=WS_ID,
        use_custom_persona=True, custom_persona_prompt=soul, persona=None, description=None,
        configuration={"personality_mode": mode}, model_config={},
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _chat_prompt(agent, orchestrator_settings) -> str:
    """The chatbot identity block, as a chat turn builds it."""
    ctx = SectionContext(agent=agent, workspace_id=str(WS_ID), workspace_name="Galway Roasters",
                         kwargs={"personality": True, "agent_name": agent.name,
                                 "orchestrator_settings": orchestrator_settings})
    return asyncio.run(IdentitySection().render(ctx))


# ── the chat's prompt ───────────────────────────────────────────────────────

def test_the_chat_speaks_with_the_rows_soul_once_not_the_settings_copy():
    """The drift on Automatos Cloud: the setting held one soul, the row another."""
    prompt = _chat_prompt(_auto_row(), {**CUSTOM, "custom_soul": STALE_SOUL})

    assert prompt.count(ROW_SOUL) == 1
    assert STALE_SOUL not in prompt
    assert PERSONA_HEADING not in prompt


def test_the_same_soul_in_both_places_is_still_said_once():
    prompt = _chat_prompt(_auto_row(), {**CUSTOM, "custom_soul": ROW_SOUL})

    assert prompt.count(ROW_SOUL) == 1


def test_a_preset_mode_speaks_with_the_rows_preset_and_doctrine_once():
    """A preset save writes the voice and the Manager's Doctrine to the row: that is the soul."""
    from core.seeds.seed_auto_agent import MANAGER_DOCTRINE_BLOCK, _default_persona

    prompt = _chat_prompt(_auto_row(soul=_default_persona(), mode="friendly"),
                          {"personality_mode": "friendly", "communication_style": "balanced"})

    assert prompt.count(MANAGER_DOCTRINE_BLOCK) == 1
    assert personality._FRIENDLY_PERSONALITY not in prompt  # the preset is not stacked on the soul


def test_an_auto_row_with_no_soul_falls_back_to_its_mode_preset():
    prompt = _chat_prompt(_auto_row(soul=None, use_custom_persona=False), {"personality_mode": "technical"})

    assert personality._TECHNICAL_PERSONALITY in prompt


def test_a_non_auto_agent_keeps_its_own_persona_section():
    shop = SimpleNamespace(id=9, name="Shop helper", slug=f"shop-helper-{WS_ID}", is_system_agent=False,
                           workspace_id=WS_ID, use_custom_persona=True, custom_persona_prompt="Talk about beans.",
                           persona=None, description=None)

    prompt = _chat_prompt(shop, {"personality_mode": "friendly"})

    assert f"{PERSONA_HEADING}\nTalk about beans." in prompt
    assert personality._FRIENDLY_PERSONALITY in prompt


def test_no_settings_key_carries_a_soul_any_more():
    """load_orchestrator_settings reads only the keys its defaults name."""
    assert "custom_soul" not in personality._ORCHESTRATOR_DEFAULTS
    assert "custom_soul" not in workspaces_api._ORCHESTRATOR_FIELDS


@pytest.mark.parametrize("agent, auto", [
    (_auto_row(), True),
    (_auto_row(slug="auto-cto"), False),                       # the global CTO agent
    (_auto_row(is_system_agent=False), False),                  # a user agent named like Auto
    (_auto_row(workspace_id=uuid.uuid4()), False),              # another workspace's slug
    (_auto_row(workspace_id=None), False),
])
def test_only_the_workspaces_own_auto_row_is_auto(agent, auto):
    assert is_workspace_auto(agent) is auto


# ── Settings shows what the chat uses ──────────────────────────────────────

class _Query:
    def __init__(self, db):
        self._db = db

    def get(self, ident):
        return self._db.workspace if ident == WS_ID else None


class _FakeDB:
    def __init__(self, settings):
        self.workspace = SimpleNamespace(id=WS_ID, settings=settings)

    def query(self, _model):
        return _Query(self)

    def commit(self):
        pass


@pytest.fixture
def routes(monkeypatch):
    """A client on the workspaces router for an owner, the Auto row faked; returns (client, row)."""
    row = _auto_row()
    monkeypatch.setattr(workspaces_api, "_get_or_seed_auto_agent", lambda db, workspace_id: row)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda instance, key: None)

    def client(db):
        app = FastAPI()
        app.include_router(workspaces_api.router)
        app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
            workspace_id=WS_ID, user=UserContext(id="owner-1", clerk_user_id="clerk_owner", system_role="user"),
            auth_type="clerk")
        app.dependency_overrides[get_db] = lambda: db
        return TestClient(app)

    return client, row


def test_settings_shows_the_soul_the_chat_speaks_with(routes):
    client, row = routes
    settings = {"orchestrator": {**CUSTOM, "custom_soul": STALE_SOUL}}

    shown = client(_FakeDB(settings)).get(ROUTE).json()
    prompt = _chat_prompt(row, settings["orchestrator"])

    assert (shown["personality_mode"], shown["custom_soul"]) == ("custom", ROW_SOUL)
    assert prompt.count(shown["custom_soul"]) == 1


def test_a_save_writes_the_soul_to_the_row_and_never_to_the_settings(routes):
    client, row = routes
    db = _FakeDB({"orchestrator": {"thinking_level": "low"}})
    new_soul = "I'm Auto. Short answers, warm voice."

    resp = client(db).put(ROUTE, json={"personality_mode": "custom", "custom_soul": new_soul})

    assert resp.status_code == 200
    assert row.custom_persona_prompt == new_soul and row.use_custom_persona is True
    assert db.workspace.settings["orchestrator"] == {"thinking_level": "low", "personality_mode": "custom"}
    assert resp.json()["orchestrator"]["custom_soul"] == new_soul  # the response reads the row too


# ── a soul written outside Settings ─────────────────────────────────────────

def test_a_soul_auto_writes_on_its_own_row_is_shown_as_custom(monkeypatch):
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda instance, key: None)
    row = _auto_row(mode="friendly", configuration={"personality_mode": "friendly", "thinking_level": "high"})

    mark_soul_custom(row)

    assert row.configuration == {"personality_mode": "custom", "thinking_level": "high"}
    assert workspaces_api._resolve_persona_view(row.configuration["personality_mode"], row.custom_persona_prompt) == {
        "personality_mode": "custom", "custom_soul": ROW_SOUL}


def test_another_agents_prompt_changes_no_mode():
    other = _auto_row(is_system_agent=False, configuration={"thinking_level": "high"})

    mark_soul_custom(other)

    assert other.configuration == {"thinking_level": "high"}


def test_platform_update_agent_marks_autos_soul_custom(monkeypatch):
    from modules.tools.discovery import handlers_agents

    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda instance, key: None)
    row = _auto_row(mode="friendly", configuration={"personality_mode": "friendly"})

    changes = handlers_agents._apply_persona_org_and_tag_fields(row, {"system_prompt": ROW_SOUL}, False, None)

    assert changes == ["system prompt updated"]
    assert row.custom_persona_prompt == ROW_SOUL and row.configuration["personality_mode"] == "custom"
