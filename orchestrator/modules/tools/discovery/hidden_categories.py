"""PRD-251B US-B106 — the action categories a workspace is not shown (B3: off means invisible).

While Socials is off for a workspace (the platform master switch, or the workspace's own),
Auto's Socials actions are left out of every catalog, shortlist and dispatcher enum the model
is shown, so Auto never offers what the workspace cannot use. The execution-time refusal in
``handlers_socials`` stays as the second wall.

The registry and the semantic index take the categories as a parameter
(``exclude_categories``), so they stay pure and every existing caller is unchanged. This
module resolves the categories for a workspace: from the workspace row when the caller has
it, or from its id with one short read. A read that fails hides the category: a gate that
cannot decide must deny.

F302 (night 9): an integration's categories join them while the workspace has not connected
the integration (``integration_categories``): no Shopify actions for a workspace with no store.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

NO_HIDDEN_CATEGORIES: Tuple[str, ...] = ()
READ_FAILED_LOG = "[Socials] workspace %s's Socials switch could not be read; its Socials actions are hidden"


def hidden_categories_for(workspace: Any, db: Any = None) -> Tuple[str, ...]:
    """The categories ``workspace`` (a ``Workspace`` row, or ``None``) is not shown: Socials
    while it is off, and each integration it has not connected (read through ``db``, else the
    row's own session)."""
    from modules.socials.settings import SOCIALS_ACTION_CATEGORY, socials_actions_hidden
    from modules.tools.discovery.integration_categories import unconnected_categories

    socials = (SOCIALS_ACTION_CATEGORY,) if socials_actions_hidden(workspace) else NO_HIDDEN_CATEGORIES
    return socials + unconnected_categories(workspace, db)


def _all_hidden() -> Tuple[str, ...]:
    """What a workspace that cannot be read is not shown: every gated category."""
    from modules.socials.settings import SOCIALS_ACTION_CATEGORY
    from modules.tools.discovery.integration_categories import all_integration_categories

    return (SOCIALS_ACTION_CATEGORY,) + all_integration_categories()


def hidden_categories_for_workspace(workspace_id: Any, db: Any = None) -> Tuple[str, ...]:
    """The categories the workspace with ``workspace_id`` is not shown, read through ``db``
    when the caller has a session, else with a short session of this function's own.

    No workspace (``None``), an id that is not a UUID, or a read that fails hides the
    Socials category and every integration's (fail-closed); a failed read is logged. Inside a
    turn's ``hidden_scope`` (the chat turn sets one) the turn's answer is used: one read a turn.
    """
    scoped = _turn_answer()
    if scoped is not None:
        return scoped
    if workspace_id in (None, ""):
        return _all_hidden()
    try:
        key = uuid.UUID(str(workspace_id))
    except (TypeError, ValueError, AttributeError):
        logger.debug("[Socials] %r is not a workspace id; its Socials actions are hidden", workspace_id)
        return _all_hidden()
    try:
        if db is not None:
            return hidden_categories_for(_workspace(db, key), db)
        from core.database.database import SessionLocal

        session = SessionLocal()
        try:
            return hidden_categories_for(_workspace(session, key), session)
        finally:
            session.close()
    except Exception:  # noqa: BLE001 — a gate that cannot decide must deny
        logger.exception(READ_FAILED_LOG, workspace_id)
        return _all_hidden()


def _turn_answer() -> Optional[Tuple[str, ...]]:
    """The current turn's hidden categories, or None outside a turn scope (and with a
    stand-in registry that has none)."""
    try:
        from modules.tools.discovery.action_registry import turn_hidden
    except ImportError:
        return None
    return turn_hidden()


def _workspace(db: Any, key: uuid.UUID) -> Any:
    from core.models.workspaces import Workspace

    return db.get(Workspace, key)


def without_hidden(actions: Iterable[Any], hidden: Optional[Iterable[str]]) -> List[Any]:
    """``actions`` less every one whose category is hidden (pure)."""
    blocked = set(hidden or ())
    if not blocked:
        return list(actions)
    return [action for action in actions if getattr(action, "category", None) not in blocked]


def exclude_kwargs(hidden: Optional[Iterable[str]]) -> Dict[str, Tuple[str, ...]]:
    """``{"exclude_categories": hidden}`` while anything is hidden, else ``{}``.

    Callers spread this into the registry's and the index's calls, so a test double of
    either with the older signature is unaffected while nothing is hidden.
    """
    categories = tuple(hidden or ())
    return {"exclude_categories": categories} if categories else {}
