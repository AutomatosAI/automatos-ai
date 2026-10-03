"""F232 (night 6): Auto described a product it isn't.

On the local edition it sent the owner to "Settings > Billing & Plans" (#88), put a
card on "ATLAS", a helper the workspace never had (#89), said Automatos was cloud
based with logins and a Team Management page (#92), and explained a setup step it
had made up with computer vision (#99). Its prompt now says which edition it runs
in, what Settings holds and who the helpers are, and that anything else about the
product is something it doesn't know. The ATOM path, which skips the sections,
says the same.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
SETTINGS_PANEL = ORCH.parent / "frontend" / "components" / "settings" / "SettingsPanel.tsx"


@pytest.fixture
def edition(monkeypatch):
    from config import config

    def _set(local):
        monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", local)
    return _set


@pytest.fixture
def shop(db_session, seed_workspace):
    """A café with two helpers, Auto, a legacy clone, and a stranger next door."""
    ws, other = seed_workspace(), seed_workspace()
    for name, workspace, kind, system in (("Analyst", ws, "custom", False), ("Content Creator", ws, "custom", False),
                                          ("Auto", ws, "custom", True), ("Clone", ws, "ephemeral", False),
                                          ("ATLAS", other, "custom", False)):
        db_session.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type, is_system_agent) "
            "VALUES (:n, :k, CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace', :s)"),
            {"n": name, "k": kind, "w": workspace, "s": system})
    db_session.flush()
    return NS(db=db_session, ws=ws)


def test_the_local_edition_says_it_has_no_logins_plans_or_billing(edition):
    from modules.context.sections.product_facts import edition_facts

    edition(True)
    facts = edition_facts()
    assert "local edition" in facts and "no sign-in, accounts, teams, plans or billing" in facts   # #92
    assert "nothing runs in a cloud" in facts and "no logins to give anyone" in facts
    assert "Settings has these tabs: Profile, Session mode, Orchestrator," in facts
    assert "There is no Billing, Plans or Team Management tab." in facts                        # #88


def test_the_hosted_edition_says_people_sign_in_and_the_plan_sets_the_helpers(edition):
    from modules.context.sections.product_facts import edition_facts

    edition(False)
    facts = edition_facts()
    assert "hosted edition" in facts and "signs in with their own account" in facts
    assert "Settings has these tabs: Orchestrator, Webhooks," in facts and "Profile" not in facts
    assert "There is no Billing, Plans or Team Management tab." in facts


def test_the_settings_tabs_are_the_pages_own():
    from modules.context.sections.product_facts import LOCAL_SETTINGS_TABS, SETTINGS_TABS

    page = SETTINGS_PANEL.read_text(encoding="utf-8")
    labels = set(re.findall(r"label: '([^']+)'", page))
    local_only = set(re.findall(r"isLocal \? \[\{ value: '[^']+', label: '([^']+)'", page))
    assert labels - {"System Settings"} == {*LOCAL_SETTINGS_TABS, *SETTINGS_TABS}   # a platform admin's only
    assert local_only == set(LOCAL_SETTINGS_TABS)
    assert not [label for label in labels if re.search(r"billing|plan|team", label, re.I)]


def test_the_helpers_are_the_agents_page_and_nobody_else(shop):
    from modules.context.sections.product_facts import helper_names, helpers_sentence

    names = helper_names(shop.db, shop.ws)
    assert names == ["Analyst", "Content Creator"]          # not Auto, the clone, or next door's ATLAS
    sentence = helpers_sentence(names)
    assert "The helpers in this workspace (its Agents page): Analyst, Content Creator." in sentence
    assert "There is no other helper" in sentence and "or to you, never to a name that isn't listed here" in sentence  # #89


def test_past_forty_helpers_the_rest_are_counted():
    from modules.context.sections.product_facts import helpers_sentence

    sentence = helpers_sentence([f"Helper {n}" for n in range(45)])
    assert "Helper 39, +5 more (platform_list_agents lists them all)." in sentence and "Helper 40" not in sentence
    assert sentence.endswith("never to a name that platform_list_agents doesn't list.")
    assert helpers_sentence([]) == "This workspace has no helpers yet: its Agents page is empty."


def test_autos_chat_prompt_carries_the_facts(shop, edition):
    from modules.context.sections.product_facts import UNSURE, ProductFactsSection

    edition(True)
    rendered = asyncio.run(ProductFactsSection().render(NS(db_session=shop.db, workspace_id=shop.ws)))
    assert rendered.startswith("## Automatos itself\nThis is the local edition of Automatos.")
    assert UNSURE in rendered and "say you don't know and offer to check" in rendered       # #99
    assert rendered.endswith("never to a name that isn't listed here.")


def test_a_widget_turn_gets_none_of_it(shop):
    from core.security.surface import WIDGET, turn_surface
    from modules.context.sections.product_facts import ProductFactsSection, product_facts

    with turn_surface(WIDGET, ("documents:read",)):
        assert product_facts(shop.db, shop.ws) == ""
        assert asyncio.run(ProductFactsSection().render(NS(db_session=shop.db, workspace_id=shop.ws))) == ""


def test_the_section_is_on_autos_prompt_in_the_cached_prefix():
    from modules.context.modes import MODE_CONFIGS, ContextMode
    from modules.context.sections import SECTION_REGISTRY
    from modules.context.sections.product_facts import ProductFactsSection
    from modules.context.service import VOLATILE_SECTIONS

    sections = MODE_CONFIGS[ContextMode.CHATBOT].sections
    assert sections.index("product_facts") == sections.index("identity") + 1
    assert SECTION_REGISTRY["product_facts"] is ProductFactsSection
    assert "product_facts" not in VOLATILE_SECTIONS        # the edition never changes, the helpers rarely


def test_the_atom_prompt_carries_the_same_facts():
    from consumers.chatbot.atom_prompt import atom_system_prompt

    memory = "\n\n## What you remember about this user:\n- likes oat milk\n"
    prompt = atom_system_prompt(NS(name="Auto", description=" Runs the café ", persona=None),
                                identity=" You're talking to Sam.", memory_block=memory,
                                facts="## Automatos itself\nThis is the local edition of Automatos.")
    assert prompt.startswith("You are Auto, an AI assistant on the Automatos platform.\n\nGood ")
    assert ". You're talking to Sam. Read the conversation" in prompt
    assert prompt.index("## Agent Description\nRuns the café") < prompt.index("## Automatos itself")
    assert prompt.index("## Automatos itself") < prompt.index("## What you remember about this user")
    bare = atom_system_prompt(NS(name="Auto", description="", persona=None), identity="", memory_block="", facts="")
    assert bare.endswith("You adapt. That's what makes you good at this.\n")      # a widget turn: no facts

    source = (ORCH / "consumers" / "chatbot" / "service.py").read_text(encoding="utf-8")
    assert "facts=product_facts(self.db, self.workspace_id)" in source


def test_autos_doctrine_says_it_never_describes_what_it_hasnt_seen():
    from consumers.chatbot.personality import AutomatosPersonality

    guidance = AutomatosPersonality.get_tool_guidance_prompt(has_tools=True)
    assert "Describe a page, setting, plan, helper or feature of Automatos that isn't in my instructions" in guidance
    assert "I say I don't know and offer to check" in guidance


def test_autos_skill_never_offers_the_automatos_teams_agents_as_examples():
    """#89: the create-task example said "assigned_agent_name": "ATLAS", and the
    cookbook says to follow its patterns exactly. The seed is generated from
    automatos-skills v2.3.1, whose examples name placeholders."""
    skill = (ORCH / "core" / "seeds" / "platform-management-skill.md").read_text(encoding="utf-8")
    assert not re.findall(r"\b(ATLAS|SENTINEL|VECTOR|PULSE|WATCHTOWER|SCOUT|SHOPIFY_OPS)\b", skill)
    assert "\"assigned_agent_name\": \"<helper's name>\"" in skill
    assert "never pass a name you haven't seen in this workspace" in skill
