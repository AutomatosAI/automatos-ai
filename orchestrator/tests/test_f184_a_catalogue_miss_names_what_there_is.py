"""F184 — a catalogue miss names what there is; "nothing ready-made" needs both searches.

Night 6: Auto proposed a package before searching (02:00:17Z), searched once with
its own label, found nothing, and said nothing ready-made fitted a wholesale
roastery, never seeing the Shopify Business Analyst agent. At 02:54 it asked to
assign a 'data-analysis' skill, got a bare "Skill not found", and offered to
install the name it had made up while 'spreadsheet-qa' was installed.

- A skill is assigned from this workspace's own (its skills and the marketplace
  skills enabled for it; the lookup read every workspace's), and a miss names what
  is installed and the nearest marketplace skills, and points to the search.
- A package search and a marketplace agents search each say what they searched,
  and an empty package search says to search the agents too. The onboarding rule:
  "nothing ready-made" only after both searches came back empty in that turn.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    db = db_session
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    db.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                    "VALUES ('Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json))"), {"w": str(ws)})
    spreadsheet = Skill(name="spreadsheet-qa", skill_type="technical", description="Checks sums in a sheet")
    private = Skill(name="margin-maths", skill_type="technical", workspace_id=other)  # another workspace's own
    db.add_all([spreadsheet, private])
    db.flush()
    db.add(WorkspaceEnabledSkill(workspace_id=ws, skill_id=spreadsheet.id))
    db.flush()
    return NS(db=db, ws=ws, spreadsheet=spreadsheet, private=private)


def _assign(roastery, skill_name):
    from modules.tools.discovery.handlers_assignments import assign_skill_to_agent

    return asyncio.run(assign_skill_to_agent(roastery.db, roastery.ws, {"agent_name": "Analyst", "skill_name": skill_name}))


def _assigned(roastery):
    return roastery.db.execute(text(
        "SELECT s.name FROM agent_skills a JOIN skills s ON s.id = a.skill_id JOIN agents g ON g.id = a.agent_id "
        "WHERE g.workspace_id = CAST(:w AS uuid)"), {"w": str(roastery.ws)}).scalars().all()


# ── a skill miss names what there is ────────────────────────────────────────

def test_a_skill_that_is_not_installed_names_what_is(roastery):
    reply = _assign(roastery, "data-analysis")
    assert reply["success"] is False
    assert "Installed here: 'spreadsheet-qa'" in reply["error"]
    assert "platform_browse_marketplace_skills" in reply["error"] and reply["installed"] == ["spreadsheet-qa"]
    assert _assigned(roastery) == []


def test_another_workspaces_own_skill_is_never_assigned_here(roastery):
    reply = _assign(roastery, "margin-maths")
    assert reply["success"] is False and _assigned(roastery) == []


def test_an_installed_skill_is_assigned_by_its_name(roastery):
    reply = _assign(roastery, "Spreadsheet-QA")
    assert reply["success"] is True and _assigned(roastery) == ["spreadsheet-qa"]


# ── "nothing ready-made" rests on two searches ──────────────────────────────

def test_an_empty_package_search_says_to_search_the_agents_too(roastery, monkeypatch):
    from modules.tools.discovery.handlers_packages import NO_PACKAGE_NEXT, search_packages

    monkeypatch.setattr("services.marketplace_packages.match_by_signals", lambda signals, packages: [])
    reply = asyncio.run(search_packages(roastery.db, roastery.ws, {"text": "Specialty Food & Beverage E-commerce"}))
    assert (reply["searched"], reply["count"], reply["next"]) == ("packages", 0, NO_PACKAGE_NEXT)
    assert "platform_browse_marketplace_agents" in NO_PACKAGE_NEXT


def test_a_marketplace_agents_search_says_what_it_searched(roastery):
    from modules.tools.discovery.handlers_marketplace import browse_marketplace_agents

    reply = asyncio.run(browse_marketplace_agents(roastery.db, roastery.ws, {"search": "wholesale"}))
    assert reply["searched"] == "marketplace_agents"


def test_onboarding_allows_nothing_ready_made_only_after_both_searches():
    from modules.context.sections import onboarding

    for stage in (onboarding._STAGE_TEACH, onboarding._STAGE_PROPOSAL):
        assert "nothing ready-made" in stage and "platform_browse_marketplace_agents" in stage
