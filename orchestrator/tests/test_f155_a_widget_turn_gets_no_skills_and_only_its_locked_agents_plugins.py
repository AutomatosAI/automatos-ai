"""F155: a widget turn's prompt carries no skills, and plugins only from the
agent its key is locked to.

The skills section rendered each skill attached to the answering agent. For
the core always-on set (Auto's platform manual) it rendered the whole body.
On a widget turn load_skill is refused, so the skill lines pointed at bodies
the turn could not load. The plugins section rendered the answering agent's
plugins. Gerard, 25 Sep:
- skills: nothing on a widget turn;
- plugins: kept for a named agent the key is locked to, none when any agent
  may answer the key (Auto included).
The widget mark now carries the key's agent lock. A widget-born mission's or
playbook's origin carries it too (only when the key has one), so their tasks
follow the same rule.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from core.security.surface import WIDGET, turn_surface
from modules.context.sections.base import SectionContext

LOCKED_AGENT = 42


def _ctx(agent_id):
    return SectionContext(agent=NS(id=agent_id, skills=[]), workspace_id="ws-1", db_session=object())


def _render_plugins(monkeypatch, agent_id, **mark):
    from modules.context.sections.plugins import PluginsSection

    monkeypatch.setattr(PluginsSection, "_build", lambda self, ctx: f"## Plugins of agent {ctx.agent.id}")
    section = PluginsSection()
    if mark:
        with turn_surface(WIDGET, ("chat",), None, **mark):
            return asyncio.run(section.render(_ctx(agent_id)))
    return asyncio.run(section.render(_ctx(agent_id)))


def test_only_the_locked_agent_brings_its_plugins_to_a_widget_turn(monkeypatch):
    assert _render_plugins(monkeypatch, LOCKED_AGENT, agent_lock=LOCKED_AGENT) == f"## Plugins of agent {LOCKED_AGENT}"
    assert _render_plugins(monkeypatch, 7, agent_lock=LOCKED_AGENT) == ""   # another agent runs the work
    assert _render_plugins(monkeypatch, 7, agent_lock=None) == ""           # any agent may answer the key
    assert _render_plugins(monkeypatch, 7) == "## Plugins of agent 7"       # a dashboard turn


def test_a_widget_turn_gets_no_skills(monkeypatch):
    from modules.context.sections.skills import SkillsSection

    monkeypatch.setattr(SkillsSection, "_build", lambda self, ctx: "## Skills\n- platform-management")
    with turn_surface(WIDGET, ("chat",), None, agent_lock=LOCKED_AGENT):
        assert asyncio.run(SkillsSection().render(_ctx(LOCKED_AGENT))) == ""
    assert asyncio.run(SkillsSection().render(_ctx(LOCKED_AGENT))) == "## Skills\n- platform-management"


def test_widget_born_work_carries_the_keys_agent_lock():
    from core.security.surface import origin_surface, stamp_origin, widget_agent_lock

    with turn_surface(WIDGET, ("chat",), None, agent_lock=LOCKED_AGENT):
        locked = stamp_origin({})
    with turn_surface(WIDGET, ("chat",), None):
        unlocked = stamp_origin({})
    assert locked["origin_agent_lock"] == LOCKED_AGENT and "origin_agent_lock" not in unlocked
    assert "origin_agent_lock" not in stamp_origin({"origin_agent_lock": LOCKED_AGENT})  # never a caller's
    with origin_surface(locked):
        assert widget_agent_lock() == LOCKED_AGENT
