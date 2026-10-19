"""PRD-263 US-062: each Auto turn records which skill body served it.

A shipped edit to Auto's platform-management body could not be compared before and
after in production: the turn's ``messages.context_trace`` said how many tokens the
skills section took, never which body it was. The skills section now records the
``content_hash`` of each always-on body it renders (the sha256 of the body as
rendered, which for a seeded body is the row's own hash), ``ContextService`` carries
it into ``ContextResult.skill_bodies``, and ``to_assembly_trace()`` (what the turn
writer saves on the message) includes it.

Mocked like ``test_context/test_service.py``: no DB, no model.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.security.surface import WIDGET, turn_surface
from modules.context.modes import ContextMode
from modules.context.result import ContextResult
from modules.context.sections.base import SectionContext
from modules.context.sections.skills import SKILL_BODIES_KEY, SkillsSection
from modules.context.service import ContextService

CORE = "platform-management"
BODY = "## Platform management\nYou run this workspace for its owner."


def _sha(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


SHIPPED_HASH = _sha(BODY)                         # the seeder's content_hash (read_seed: sha256 of the body)


def _skill(name, body, content_hash=None, sid=None, implementation=None):
    return NS(id=sid, name=name, prompt_template=body, description=f"{name} skill",
              content_hash=content_hash, is_active=True, tools_schema=None,
              implementation=implementation, skill_source=None, filesystem_path=None)


def _auto(*skills):
    return NS(id=322, name="Auto", agent_type="assistant", description=None, use_custom_persona=False,
              custom_persona_prompt=None, persona=None, skills=list(skills))


@pytest.fixture(autouse=True)
def core_set(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "SKILL_CORE_ALWAYS_ON", [CORE])


def _render(agent, db_session=None):
    ctx = SectionContext(agent=agent, workspace_id="ws-1", db_session=db_session)
    return asyncio.run(SkillsSection().render(ctx)), ctx


# --- the skills section records the body it rendered ---


def test_the_core_body_that_served_is_recorded_by_its_content_hash():
    out, ctx = _render(_auto(_skill(CORE, BODY, SHIPPED_HASH, sid=7), _skill("growth", "GROWTH BODY", "g1", sid=8)))
    assert BODY in out
    assert ctx.kwargs[SKILL_BODIES_KEY] == [{"name": CORE, "content_hash": SHIPPED_HASH}]


def test_a_fork_with_no_content_hash_is_recorded_by_the_hash_of_its_body():
    # Forking on edit (api/workspace_skills.py) writes the row with content_hash NULL.
    fork = "## Platform management\nThe owner's own rules."
    out, ctx = _render(_auto(_skill(CORE, fork, None, sid=70)))
    assert fork in out
    assert ctx.kwargs[SKILL_BODIES_KEY] == [{"name": CORE, "content_hash": _sha(fork)}]


def test_the_fallback_body_comes_from_the_linked_row_not_another_row_of_that_name(monkeypatch, tmp_path):
    # Auto's linked row has an empty prompt_template, so the body comes from the loader.
    # A lookup by name resolves another workspace's fork; the loader must read the linked row.
    from modules.agents.services import skill_loader

    monkeypatch.setattr(skill_loader.SkillLoaderConfig, "SKILLS_BASE_DIR", str(tmp_path))
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = _skill(
        CORE, "Another workspace's rules.", None, sid=71)
    loader = skill_loader.SkillLoader(db)
    monkeypatch.setattr(skill_loader, "get_skill_loader", lambda db_session=None: loader)

    out, ctx = _render(_auto(_skill(CORE, "", None, sid=7, implementation="The linked row's rules.")), db)
    assert "The linked row's rules." in out and "Another workspace" not in out
    assert ctx.kwargs[SKILL_BODIES_KEY] == [{"name": CORE, "content_hash": _sha("The linked row's rules.")}]


def test_a_skill_offered_by_its_description_only_is_not_a_body_that_served():
    out, ctx = _render(_auto(_skill("growth", "GROWTH BODY", "g1", sid=8)))
    assert "growth" in out and "GROWTH BODY" not in out
    assert ctx.kwargs[SKILL_BODIES_KEY] == []


def test_a_widget_turn_renders_no_body_and_records_none():
    with turn_surface(WIDGET, ("chat",), None):
        out, ctx = _render(_auto(_skill(CORE, BODY, SHIPPED_HASH, sid=7)))
    assert out == ""
    assert SKILL_BODIES_KEY not in ctx.kwargs


# --- the result carries it into the persisted trace ---


def test_the_assembly_trace_carries_the_skill_bodies():
    result = ContextResult(system_prompt="", skill_bodies=[{"name": CORE, "content_hash": SHIPPED_HASH}])
    trace = json.loads(json.dumps(result.to_assembly_trace()))   # it is saved as JSONB
    assert trace["skill_bodies"] == [{"name": CORE, "content_hash": SHIPPED_HASH}]
    assert ContextResult(system_prompt="").to_assembly_trace()["skill_bodies"] == []


# --- the whole assembly: Auto's chat turn ---


def _db():
    db = MagicMock()                       # the chainable session test_context/test_service.py uses
    query = db.query.return_value
    query.join.return_value = query.filter.return_value = query.order_by.return_value = query
    query.all.return_value, query.first.return_value = [], None
    db.execute.return_value.scalars.return_value.all.return_value = []
    return db


def _build(agent, *extra_patches):
    from modules.context.sections.memory import MemorySection
    from modules.context.sections.platform_actions import PlatformActionsSection
    from modules.context.sections.tools import ToolsSection

    async def _no_memory(self, ctx):
        return ""

    with patch.object(PlatformActionsSection, "_build", return_value=""), \
            patch.object(MemorySection, "render", _no_memory), \
            patch.object(ToolsSection, "load_tools", AsyncMock(return_value=([], "auto"))):
        for extra in extra_patches:
            extra.start()
        try:
            return asyncio.run(ContextService(_db()).build_context(
                mode=ContextMode.CHATBOT, agent=agent, workspace_id="ws-1",
                messages=[{"role": "user", "content": "Put the launch on the board"}], skip_memory=True,
            ))
        finally:
            for extra in extra_patches:
                extra.stop()


def test_an_auto_turn_trace_names_the_platform_management_body_by_hash():
    result = _build(_auto(_skill(CORE, BODY, SHIPPED_HASH, sid=7)))
    assert BODY in result.system_prompt
    assert result.skill_bodies == [{"name": CORE, "content_hash": SHIPPED_HASH}]
    assert result.to_assembly_trace()["skill_bodies"] == [{"name": CORE, "content_hash": SHIPPED_HASH}]


def test_an_edited_body_is_told_apart_by_its_new_hash():
    # An in-place edit (api/workspace_skills.py) changes prompt_template and leaves content_hash stale.
    edited = BODY + "\nNew rule."
    result = _build(_auto(_skill(CORE, edited, SHIPPED_HASH, sid=7)))
    assert result.to_assembly_trace()["skill_bodies"] == [{"name": CORE, "content_hash": _sha(edited)}]
    assert _sha(edited) != SHIPPED_HASH


def test_an_agent_without_the_core_skill_records_no_body():
    result = _build(_auto(_skill("sentinel", "You are SENTINEL.", "s1", sid=9)))
    assert "skills" in result.sections_included
    assert result.to_assembly_trace()["skill_bodies"] == []


def test_a_skills_section_the_budget_dropped_records_no_body():
    import modules.context.service as service

    def _drop_skills(rendered, budget):
        return [s for s in rendered if s.name != "skills"], ["skills"]

    result = _build(_auto(_skill(CORE, BODY, SHIPPED_HASH, sid=7)),
                    patch.object(service._budget_manager, "allocate", side_effect=_drop_skills))
    assert BODY not in result.system_prompt
    assert result.to_assembly_trace()["skill_bodies"] == []
