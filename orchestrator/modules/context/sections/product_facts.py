"""F232 (night 6): Auto described a product it isn't.

On the local edition, night 6:
- #88: it sent the owner to "Settings > Billing & Plans", a page neither edition has.
- #89: it put a card on "ATLAS", "an agent we have for general tasks". The workspace
  never had one; ATLAS is an example name in Auto's skill.
- #92: "Automatos is a cloud-based platform … Hana can use it from her own laptop …
  her login credentials … Settings → Team Management". The local edition has no
  logins, and no Team Management.
- #99: it explained a setup step it had made up ("choose models") with computer
  vision that inspects coffee beans.

None of these is an action claim, so F187's check can't see them. This section
gives Auto the facts it kept inventing: the edition it runs in and what that
edition has, the Settings tabs, and the names of the workspace's helpers. It also
sets the rule for everything else: a page, setting, plan or feature Auto hasn't
seen in its prompt, a tool result or the owner's documents is one it doesn't know.

Cache-stable: the edition never changes and the helpers rarely do. The ATOM chat
path, which skips the sections, appends the same text (consumers/chatbot/atom_prompt.py).

F155: a public widget turn gets none of it. The owner's set-up and helpers are
not a visitor's business.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

HELPERS_SHOWN = 40
LOCAL_EDITION = (
    "This is the local edition of Automatos. It runs on the owner's own computer and is used in a browser "
    "there. It has no sign-in, accounts, teams, plans or billing, so there are no logins to give anyone "
    "and nothing runs in a cloud."
)
HOSTED_EDITION = (
    "This is the hosted edition of Automatos: each person signs in with their own account, and the "
    "workspace's plan sets how many helpers it may have."
)
# The Settings tabs every owner has (frontend/components/settings/SettingsPanel.tsx;
# System Settings is a platform admin's only). A test holds the two lists to the page.
SETTINGS_TABS = ("Orchestrator", "Webhooks", "API Keys", "Credentials", "Channels", "Notifications",
                 "Widget SDK")
LOCAL_SETTINGS_TABS = ("Profile", "Session mode")
NO_SUCH_TAB = "There is no Billing, Plans or Team Management tab."
UNSURE = (
    "If you aren't sure Automatos has a page, a setting, a plan or a feature, say you don't know and offer "
    "to check. Never describe one you haven't seen in this prompt, a tool result or the owner's documents."
)
# Auto and any legacy ephemeral clone are the rows the Agents page leaves out (api/agents.py list_agents).
_HELPERS = (
    "SELECT name FROM agents WHERE workspace_id = CAST(:ws AS uuid) AND agent_type <> 'ephemeral' "
    "AND is_system_agent IS NOT TRUE ORDER BY id"
)


def edition_facts() -> str:
    """What this install is, and the Settings tabs it has."""
    from config import config

    local = config.IS_LOCAL_EDITION
    tabs = (*LOCAL_SETTINGS_TABS, *SETTINGS_TABS) if local else SETTINGS_TABS
    edition = LOCAL_EDITION if local else HOSTED_EDITION
    return f"{edition} Settings has these tabs: {', '.join(tabs)}. {NO_SUCH_TAB}"


def helper_names(db: Any, workspace_id: Any) -> List[str]:
    """The workspace's helpers as its Agents page lists them, oldest first."""
    from sqlalchemy import text

    return [str(row[0]) for row in db.execute(text(_HELPERS), {"ws": str(workspace_id)}).fetchall()]


def helpers_sentence(names: List[str]) -> str:
    """Who work can go to, by name, and that nobody else exists."""
    if not names:
        return "This workspace has no helpers yet: its Agents page is empty."
    shown = names[:HELPERS_SHOWN]
    hidden = len(names) - len(shown)
    more = f", +{hidden} more (platform_list_agents lists them all)" if hidden else ""
    return (f"The helpers in this workspace (its Agents page): {', '.join(shown)}{more}. There is no other "
            "helper: give work only to one of these, by its name, and never name another.")


def product_facts(db: Any, workspace_id: Any) -> str:
    """The section's text; empty on a public widget turn (F155)."""
    from core.security.surface import widget_turn

    if widget_turn():
        return ""
    lines = ["## Automatos itself", edition_facts(), UNSURE]
    if db is not None and workspace_id:
        lines.append(helpers_sentence(helper_names(db, workspace_id)))
    return "\n".join(lines)


class ProductFactsSection(BaseSection):
    """The edition, its Settings tabs and the workspace's helpers, so Auto stops inventing them."""

    name: str = "product_facts"
    priority: int = 2
    # Forty names fit whole. Past the cap the budget cuts from the end, where the names are.
    max_tokens: Optional[int] = 480

    async def render(self, ctx: SectionContext) -> str:
        try:
            return product_facts(ctx.db_session, ctx.workspace_id)
        except Exception:
            logger.exception("ProductFactsSection.render failed; the prompt goes without it")
            return ""
