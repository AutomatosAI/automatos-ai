"""F182 (night 6) — an update given nothing to change says so, and names what it can change.

At 03:56 two update_playbook calls changed nothing and came back success, "No
changes specified", and Auto told the owner the playbook was fixed. platform_
execute now refuses the keys an action does not take (5245efa18). An update
sent only the id of what it updates reached the same success. update_playbook,
update_playbook_step, update_agent and configure_agent_heartbeat now answer it
with an error naming what each can change.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from modules.tools.discovery import handlers_agents, handlers_assignments, handlers_playbooks


class _Query:
    def __init__(self, found):
        self.found = found

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.found

    def count(self):
        return 0


class _DB:
    def __init__(self, found):
        self.found = found

    def query(self, *models):
        return _Query(self.found)

    def flush(self):
        pass

    def commit(self):
        pass


PLAYBOOK = dict(id=102, name="New Cafe Onboarding", workspace_id="ws-c1",
                steps=[{"step_id": "s1", "order": 1, "prompt_template": "Draft the welcome email."}])
AGENT = dict(id=325, name="Analyst", workspace_id="ws-c1", configuration={}, model_config={}, status="active")


@pytest.mark.parametrize("handler, params, found, can_change", [
    (handlers_playbooks.update_playbook, {"playbook_id": 102}, PLAYBOOK,
     "name, description, tags, execution_config, schedule_config, inputs"),
    (handlers_playbooks.update_playbook_step, {"playbook_id": 102, "step_index": 0}, PLAYBOOK,
     "prompt_template, find, replace, agent_id, order, error_handling, output_key"),
    (handlers_agents.update_agent, {"agent_id": 325}, AGENT,
     "new_name, description, status, model_id, system_prompt, temperature, tags, team, job_title, reports_to_id"),
    (handlers_assignments.configure_agent_heartbeat, {"agent_id": 325}, AGENT,
     "enabled, interval_minutes, prompt, auto_act, active_hours_start, active_hours_end, proactive_level, "
     "notification_channel, checklist"),
], ids=["update_playbook", "update_playbook_step", "update_agent", "configure_agent_heartbeat"])
def test_an_update_given_nothing_to_change_says_so(monkeypatch, handler, params, found, can_change):
    row = SimpleNamespace(**found)
    monkeypatch.setattr(handlers_assignments, "resolve_agent", lambda db, ws, p: (row, None))
    result = asyncio.run(handler(_DB(row), "ws-c1", dict(params)))

    action = f"platform_{handler.__name__}"
    assert result["success"] is False
    assert result["error"] == (f"Nothing was changed: the call gave {action} nothing to change. "
                               f"Pass at least one of: {can_change}.")


def test_an_update_with_a_change_still_makes_it():
    playbook = SimpleNamespace(**PLAYBOOK)
    result = asyncio.run(handlers_playbooks.update_playbook(_DB(playbook), "ws-c1",
                                                            {"playbook_id": 102, "name": "Cafe Onboarding"}))
    assert result["success"] is True and playbook.name == "Cafe Onboarding"
