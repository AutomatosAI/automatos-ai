"""F134 (night 4) — a playbook edit says a run in flight keeps its steps, and Auto
never makes a second active agent with an existing name.

B85: a run takes its playbook's steps when it starts. An edit made while a run
was going replied "Recipe updated successfully", and the owner took it that the
run had changed; it kept the old step. The reply now says the edit applies to
the next run. B55: asked to use an existing agent, Auto created a namesake, and
the new agent's dead model then answered nothing (B56). create_agent now refuses
a name an active agent in the workspace already has, ignoring case and spaces.
B79, B84: "update this step" re-sent the whole prompt from memory and dropped its
safety lines. update_playbook_step takes find/replace for a partial edit, a
whole-prompt overwrite names every line it dropped, and the stored step dicts are
never edited in place.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models import Agent
from core.models.core import RecipeExecution, WorkflowTemplate

WS = UUID("00000000-0000-0000-0000-0000000000c1")
IN_FLIGHT = " A run in progress keeps the steps it started with; this edit applies to the next run."


class _Rows:
    def __init__(self, first=None, count=0):
        self._first, self._count = first, count

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args):
        return self

    def first(self):
        return self._first

    def all(self):
        return [self._first] if self._first is not None else []

    def count(self):
        return self._count


class _Db:
    def __init__(self, playbook=None, runs_going=0, namesake=None):
        self.playbook, self.runs_going, self.namesake, self.added = playbook, runs_going, namesake, []

    def query(self, *entities):
        model = entities[0]
        if model is RecipeExecution:
            return _Rows(count=self.runs_going)
        if model is WorkflowTemplate:
            return _Rows(first=self.playbook)
        return _Rows(first=self.namesake)          # Agent.id, Agent.name

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def execute(self, *a, **k):  # F200's agent-count lock: this workspace has no other create
        return _LockHeld()


class _LockHeld:
    def scalar(self):
        return True


def _roast_log_playbook():
    return WorkflowTemplate(id=82, name="Roast log write-up", workspace_id=WS, steps=[
        {"order": 1, "agent_id": 7, "prompt_template": "Read the roast log. Stop and ask Gerard if a batch is missing."},
    ])


# ── B85: the edit says where it lands ───────────────────────────────────────

def test_a_step_edit_during_a_run_says_it_applies_to_the_next_run():
    from modules.tools.discovery import handlers_playbooks

    reply = asyncio.run(handlers_playbooks.update_playbook_step(
        _Db(_roast_log_playbook(), runs_going=1), WS,
        {"playbook_id": 82, "step_index": 0, "error_handling": "stop"}))
    assert reply["success"] is True
    assert reply["message"] == "Step 0 of 'Roast log write-up' updated: error_handling updated." + IN_FLIGHT


def test_with_no_run_going_the_reply_says_nothing_more():
    from modules.tools.discovery import handlers_playbooks

    reply = asyncio.run(handlers_playbooks.add_playbook_step(
        _Db(_roast_log_playbook(), runs_going=0), WS, {"playbook_id": 82, "prompt_template": "Write it up."}))
    assert reply["message"] == "Step added to playbook 'Roast log write-up' (now 2 steps)."


def test_the_note_counts_the_runs_going():
    from services.playbook_engine import next_run_note

    assert next_run_note(_Db(runs_going=2), 82) == (
        " 2 runs in progress keep the steps they started with; this edit applies to the next run.")


def test_the_uis_save_reply_carries_the_note_too():
    from pathlib import Path

    route = (Path(__file__).resolve().parents[1] / "api" / "workflow_recipes.py").read_text()
    assert '"message": "Recipe updated successfully." + next_run_note(db, recipe.id)' in route


# ── B55: one active agent per name ──────────────────────────────────────────

def test_auto_cannot_make_a_second_active_agent_with_the_same_name():
    from modules.tools.discovery import handlers_agents

    db = _Db(namesake=NS(id=284, name="BEANCOUNTER"))
    reply = asyncio.run(handlers_agents.create_agent(db, WS, {"name": "  Beancounter "}))
    assert reply == {
        "success": False,
        "existing_agent_id": 284,
        "existing_agent_ids": [284],
        "error": "An active agent is already called 'BEANCOUNTER' (id 284). "
                 "Use that agent, or give the new one a different name.",
    }
    assert db.added == []


def test_a_name_no_active_agent_has_is_created():
    from modules.tools.discovery import handlers_agents

    db = _Db(namesake=None)
    reply = asyncio.run(handlers_agents.create_agent(db, WS, {"name": "Cupping Notes"}))
    assert reply["success"] is True
    assert [type(obj) for obj in db.added] == [Agent] and db.added[0].name == "Cupping Notes"


# ── B79, B84: a partial edit keeps the rest of the prompt ───────────────────

SAFE = ("Read the roast log for the day.\nStop and ask Gerard if a batch is missing.\n"
        "Never send anything: leave a draft.")


def _playbook_with(prompt):
    return WorkflowTemplate(id=82, name="Roast log write-up", workspace_id=WS,
                            steps=[{"order": 1, "agent_id": 7, "prompt_template": prompt}])


def _edit(playbook, **params):
    from modules.tools.discovery import handlers_playbooks

    return asyncio.run(handlers_playbooks.update_playbook_step(
        _Db(playbook), WS, {"playbook_id": 82, "step_index": 0, **params}))


def test_find_replace_changes_one_passage_and_keeps_the_safety_lines():
    playbook = _playbook_with(SAFE)
    reply = _edit(playbook, find="for the day", replace="for {{date}}")
    assert reply["success"] is True and "prompt_template: one passage replaced" in reply["message"]
    assert playbook.steps[0]["prompt_template"] == SAFE.replace("for the day", "for {{date}}")


@pytest.mark.parametrize("params, said", [
    ({"find": "for the week", "replace": "x"}, "The find text is not in step 0's prompt"),
    ({"find": "a", "replace": "x"}, "The find text appears"),
    ({"find": "for the day"}, "find needs replace"),
    ({"find": "for the day", "replace": "x", "prompt_template": "y"}, "not both"),
])
def test_a_find_that_does_not_name_one_place_changes_nothing(params, said):
    playbook = _playbook_with(SAFE)
    reply = _edit(playbook, **params)
    assert reply["success"] is False and said in reply["error"]
    assert playbook.steps[0]["prompt_template"] == SAFE


def test_a_whole_prompt_overwrite_names_every_line_it_dropped():
    playbook = _playbook_with(SAFE)
    reply = _edit(playbook, prompt_template="Read the roast log for the day.\nWrite it up.")
    assert reply["success"] is True
    assert ("prompt_template replaced, and it dropped 2 lines: 'Stop and ask Gerard if a batch is missing.'; "
            "'Never send anything: leave a draft.'") in reply["message"]


def test_an_edit_builds_new_step_dicts_and_never_edits_the_stored_ones():
    original = {"order": 1, "agent_id": 7, "prompt_template": SAFE}
    playbook = WorkflowTemplate(id=82, name="Roast log write-up", workspace_id=WS, steps=[original])
    _edit(playbook, error_handling="stop")
    assert "error_handling" not in original
    assert playbook.steps[0]["error_handling"] == "stop"
