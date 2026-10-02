"""F223 (night 6, 2 Oct, iteration 13): an installed playbook listed, as active, a
helper the plan doesn't allow.

The owner installed the marketplace's "Weekly social posts" (POST
/api/workflow-recipes/install/107, playbook #111). Its helper, the Social Media
Director, was refused by the plan's agent limit ("Failed to install agent 'Social
Media Director': Your basic plan includes 5 agents..."), but the playbook's steps
kept the marketplace agent. So the installed playbook showed a helper the
workspace does not have, as if it were there. Now such a step is left with no
helper, naming the one it needs, and the install warns which step needs one.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS

from core.models.core import WorkflowTemplate
from modules.tools.discovery import cascade_installer as ci

DIRECTOR = "Social Media Director"
REFUSAL = "Your basic plan includes 5 agents and this workspace has 8."


class _Rows:
    """db.query(...).filter(...).first() answering the given rows in call order."""

    def __init__(self, *rows):
        self._rows = list(rows)

    def query(self, *entities):
        return self

    def filter(self, *clauses):
        return self

    def first(self):
        return self._rows.pop(0)

    def flush(self):
        pass


def _install(monkeypatch, *, refused, steps, remap=True):
    director = NS(id=61, name=DIRECTOR, install_count=0)
    clone = NS(id=501, name=DIRECTOR)

    def _clone(*args, **kwargs):
        if refused:
            raise ValueError(REFUSAL)
        return clone, DIRECTOR

    async def _no_deps(db, ws, marketplace_agent, cloned_agent):
        return ci.CascadeResult()

    monkeypatch.setattr(ci, "workspace_clone_of", lambda db, ws, agent: None)
    monkeypatch.setattr(ci, "clone_agent_to_workspace", _clone)
    monkeypatch.setattr(ci, "cascade_agent_dependencies", _no_deps)
    recipe = NS(recommended_agents=[DIRECTOR], required_tools=[], template_definition={}, name="Weekly social posts")
    installed = WorkflowTemplate(name="Weekly social posts", steps=steps)
    result = asyncio.run(ci.cascade_recipe_dependencies(_Rows(director), uuid.uuid4(), recipe, installed,
                                                        remap_steps=remap))
    return installed, result


def test_a_step_whose_helper_the_plan_refused_names_no_helper(monkeypatch):
    steps = [{"agent_id": 61, "agent_name": DIRECTOR, "prompt_template": "Plan the week"}]
    installed, result = _install(monkeypatch, refused=True, steps=steps)

    assert installed.steps[0]["agent_id"] is None                 # night 6: the marketplace agent, "active"
    assert installed.steps[0]["needs_helper"] == DIRECTOR
    assert f"Failed to install agent '{DIRECTOR}': {REFUSAL}" in result.warnings
    assert ci.HELPER_NOT_INSTALLED.format(number=1, helper=DIRECTOR) in result.warnings
    assert steps[0]["agent_id"] == 61                              # the marketplace's own steps untouched


def test_a_helper_that_was_installed_is_the_steps_helper(monkeypatch):
    installed, result = _install(monkeypatch, refused=False, steps=[{"agent_id": 61, "agent_name": DIRECTOR}])
    assert installed.steps[0]["agent_id"] == 501 and "needs_helper" not in installed.steps[0]
    assert not [w for w in result.warnings if "has no helper" in w]


def test_a_step_naming_any_agent_the_workspace_lacks_names_no_helper(monkeypatch):
    """A step pinned to an agent no install gave this workspace (another workspace's)."""
    steps = [{"agent_id": 61, "agent_name": DIRECTOR}, {"agent_id": 9, "agent_name": "Brand Designer"}]
    installed, result = _install(monkeypatch, refused=False, steps=steps)
    assert [s["agent_id"] for s in installed.steps] == [501, None]
    assert installed.steps[1]["needs_helper"] == "Brand Designer"
    assert ci.HELPER_NOT_INSTALLED.format(number=2, helper="Brand Designer") in result.warnings


def test_a_reinstall_leaves_the_workspaces_copy_as_it_is(monkeypatch):
    steps = [{"agent_id": 61, "agent_name": DIRECTOR}]
    installed, result = _install(monkeypatch, refused=True, steps=steps, remap=False)
    assert installed.steps[0]["agent_id"] == 61
    assert not [w for w in result.warnings if "has no helper" in w]
