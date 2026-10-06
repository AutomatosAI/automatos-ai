"""F362 (night 10c): Auto's roster names every agent, and each by its job title.

Chat 3c8c7a3e 17:17:23: "we don't currently have an agent with the specific job title 'Brand
Designer'" while #347 Brand Designer was active, in a workspace of 49 agents. Three readings
of the team each lost it:

- platform_list_agents gave each agent about 900 characters, and the tool formatter cuts a
  platform result at 20,000: Auto read the first 22 agents by id, never the newest;
- AutoBrain read its roster capped at 40 agents, so a name past them never resolved;
- the classifier's roster read an agent's ``role``, which an Agent row hasn't got: its role is
  its ``job_title``, so no line ever said "Brand Designer".
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from consumers.chatbot import auto_decisions as AD
from services.agent_roster_fit import every_agent_fits

TEAM_SIZE = 49
FORMATTER_CUT = 20000
WHY = "Can't run now: no CLI host is online to run it. Start the CLI host on the computer it is paired with."


def _listed(agent_id: int, name: str, job_title: str) -> dict:
    """One agent as platform_list_agents lists it (after says_who_can_run)."""
    return {"id": agent_id, "name": name, "type": "specialized", "status": "active", "description": "d" * 190,
            "model_id": "opus", "provider": "claude", "runtime": "cli",
            "working_directory": "/Users/owner/Development/deliverables", "temperature": None, "tools_count": 0,
            "skills_count": 2, "heartbeat_enabled": False, "has_persona": True, "tags": ["socials", "brand"],
            "team": "Socials", "job_title": job_title, "created_at": "2026-10-06T10:00:00", "can_run": False,
            "why": WHY}


def _team():
    team = [_listed(300 + i, f"Agent {i}", "Writer") for i in range(TEAM_SIZE - 1)]
    return team + [_listed(347, "Brand Designer", "Brand Designer")]


def _shown(result: dict) -> str:
    """The result as the tool formatter shows a platform result to the model, cut where it cuts."""
    text = json.dumps({k: v for k, v in result.items() if k != "success"}, default=str, indent=2)
    return f"Tool: platform_list_agents\nStatus: success\n\n{text}"[:FORMATTER_CUT]


def test_the_nights_listing_was_cut_before_the_designer():
    full = {"success": True, "agents": _team(), "count": TEAM_SIZE}
    assert '"name": "Brand Designer"' not in _shown(full)


def test_every_agent_reaches_the_model_with_its_job_title_and_whether_it_can_run():
    fitted = every_agent_fits({"success": True, "agents": _team(), "count": TEAM_SIZE})
    shown = _shown(fitted)

    assert '"name": "Brand Designer"' in shown and '"job_title": "Brand Designer"' in shown
    assert shown.count('"can_run": false') == TEAM_SIZE
    assert "platform_get_agent" in fitted["note"]


def test_a_team_too_long_even_for_that_gets_one_line_each():
    team = [_listed(i, f"Agent {i}", "Writer") for i in range(150)] + [_listed(347, "Brand Designer", "Brand Designer")]
    shown = _shown(every_agent_fits({"success": True, "agents": team, "count": len(team)}))

    assert "Brand Designer (id 347) | Brand Designer | Socials | active | cli | can't run now" in shown
    assert "#347" not in shown                                     # a '#' names a ticket (PRD-252)


def test_a_listing_that_fits_is_left_as_it_is():
    small = {"success": True, "agents": _team()[:3], "count": 3}
    assert every_agent_fits(small) is small


def _agents():
    team = [NS(id=300 + i, name=f"Agent {i}", job_title="Writer", description="writes") for i in range(TEAM_SIZE - 1)]
    return team + [NS(id=347, name="Brand Designer", job_title="Brand Designer", description="brand work",
                      configuration={"runtime": "cli"})]


def test_the_classifiers_roster_says_each_agents_job_title():
    entries = AD.roster_entries(_agents())
    assert {"name": "Brand Designer", "role": "Brand Designer"} in entries


def test_the_routing_roster_names_every_agent_past_the_described_ones():
    lines = AD.roster_lines(list(reversed(_agents())), 40)

    assert lines[0].startswith("- Agent 0 (Writer): writes")         # oldest first
    assert len(lines) == 41 and lines[-1].startswith(AD.ROSTER_REST)
    assert "Brand Designer (Brand Designer)" in lines[-1]


def test_autobrain_reads_the_whole_active_roster_and_resolves_the_49th(monkeypatch):
    from consumers.chatbot import auto

    db = MagicMock()
    db.query.return_value.filter.return_value.limit.return_value.all.return_value = _agents()
    brain = auto.AutoBrain(db=db, workspace_id="ws")

    roster = brain._active_agents()
    (cap,), _ = db.query.return_value.filter.return_value.limit.call_args

    assert cap >= TEAM_SIZE and cap == auto._ROSTER_READ_CAP
    assert brain._match_roster_agent("Brand Designer", roster, message="Give it to the Brand Designer") == (
        347, "Brand Designer")
