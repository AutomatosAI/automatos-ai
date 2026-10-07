"""F380 (night 11, 7 Oct): a brief for a social post always sees the actions that make one.

#2159 ("Quote card: Rosa at Lantern Kitchen", Social Media Director) answered "I cannot
directly create the social post… Please use the platform_create_social_post action…":
the agent's catalog is the top-K actions ranked against its brief, and the post's
actions had ranked out. A brief that speaks of making social posts now always shows
them, after the ranked ones, while Socials is on for the workspace; with Socials off
(a hidden category) nothing is added, and any other brief keeps its ranking as it was.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from modules.tools.discovery.socials_prior import SOCIAL_POST_ACTIONS, with_socials_actions

BRIEF_2159 = "Quote card: Rosa at Lantern Kitchen\nHer words for an Instagram card: Harbour Blend by name now."
RANKED = ["platform_search_knowledge", "platform_list_deliverables"]


def test_a_social_post_brief_gets_the_post_actions_after_its_ranking():
    shown = with_socials_actions(BRIEF_2159, RANKED, ())

    assert shown[:2] == RANKED                                           # the ranking leads
    assert set(SOCIAL_POST_ACTIONS) <= set(shown)
    assert len(shown) == len(set(shown))                                  # nothing twice
    assert with_socials_actions(BRIEF_2159, ["platform_create_social_post"], ()).count(
        "platform_create_social_post") == 1


def test_socials_off_or_another_brief_keeps_its_ranking():
    assert with_socials_actions(BRIEF_2159, RANKED, ("socials",)) == RANKED
    assert with_socials_actions("How many kilos did Lantern Kitchen order in September?", RANKED, ()) == RANKED
    assert with_socials_actions(BRIEF_2159, RANKED, ()) is not RANKED      # a new list


class _Index:
    """The semantic index, ranking the post's actions out as night 11's brief did."""

    async def rank_actions(self, query, **_kwargs):
        return [("platform_list_agents", 0.91)]


def _registry():
    from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry

    registry = ActionRegistry()
    registry._initialized = True  # no live registrar
    for name, category in (("platform_list_agents", "agents"), ("platform_create_social_post", "socials"),
                           ("platform_list_templates", "templates")):
        registry.register(ActionDefinition(
            name=name, description=f"{name} probe", category=category,
            parameters={"type": "object", "properties": {}, "required": []}))
    return registry


def _catalog(query, hidden):
    from modules.context.sections.platform_actions import PlatformActionsSection

    with patch("modules.tools.discovery.action_registry.get_action_registry", return_value=_registry()), \
            patch("modules.tools.discovery.action_semantic_index.get_action_semantic_index", return_value=_Index()):
        return asyncio.run(PlatformActionsSection()._build_filtered(query, hidden=hidden))


def test_the_agents_catalog_carries_the_post_actions_for_a_social_brief():
    catalog = _catalog(BRIEF_2159, ())

    assert "platform_list_agents" in catalog
    assert "platform_create_social_post" in catalog                       # night 11: ranked out
    assert "platform_list_templates" in catalog


def test_the_catalog_with_socials_off_or_for_another_brief_is_the_ranking():
    for catalog in (_catalog(BRIEF_2159, ("socials",)), _catalog("List my agents", ())):
        assert "platform_list_agents" in catalog
        assert "platform_create_social_post" not in catalog
        assert "platform_list_templates" not in catalog
