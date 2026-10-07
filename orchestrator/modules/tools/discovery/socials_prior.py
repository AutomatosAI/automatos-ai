"""F380 (night 11, 7 Oct): a brief for a social post always shows the Socials actions.

#2159 (Social Media Director, "Quote card: Rosa at Lantern Kitchen") answered "I cannot
directly create the social post… Please use the platform_create_social_post action…":
the agent's action catalog is the top-K actions ranked against its brief
(``PlatformActionsSection``, PRD-138), and the brief's words ranked other actions above
the ones that make and render a post. A query that speaks of making social posts
(``social_post_asks.mentions_social_posts``) now always shows the actions a post is
made with, after the ranked ones, while Socials is on for the workspace. While
it is off (PRD-251B US-B106, the hidden categories) nothing is added: off means invisible.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from modules.tools.discovery.social_post_asks import mentions_social_posts

# What a post is made with: pick the template, read its fields, draft, render, submit, check.
SOCIAL_POST_ACTIONS = (
    "platform_list_templates",
    "platform_get_template_schema",
    "platform_create_social_post",
    "platform_update_social_post",
    "platform_submit_social_post",
    "platform_get_social_post",
    "platform_list_social_posts",
)


def with_socials_actions(query: str, ranked: Sequence[str], hidden: Optional[Iterable[str]]) -> List[str]:
    """``ranked`` with the Socials post actions after it when ``query`` is social-post work
    and the workspace is shown Socials; ``ranked`` as it was otherwise. A new list either way."""
    if not mentions_social_posts(query):
        return list(ranked)
    from modules.socials.settings import SOCIALS_ACTION_CATEGORY

    if SOCIALS_ACTION_CATEGORY in set(hidden or ()):
        return list(ranked)
    return list(dict.fromkeys([*ranked, *SOCIAL_POST_ACTIONS]))


__all__ = ["SOCIAL_POST_ACTIONS", "with_socials_actions"]
