"""
Recipe Registry (PRD-50: US-013)
================================

Central registry mapping workflow/recipe identifiers to their
implementation classes.  The UniversalRouter and webhook dispatcher
use this to look up a recipe by ``workflow_id`` or ``recipe_name``.

Usage::

    from modules.workflows.recipes import get_recipe, RECIPE_REGISTRY

    recipe_cls = get_recipe(workflow_id=42)
    # or
    recipe_cls = get_recipe(recipe_name="jira_bug_triage")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class _BaseRecipeProtocol:
    """Structural hint — any class with ``async execute(envelope, db, workspace_id)``."""
    pass


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

# name -> recipe class
_REGISTRY_BY_NAME: Dict[str, Type[Any]] = {}
# workflow_id -> recipe class
_REGISTRY_BY_WORKFLOW_ID: Dict[int, Type[Any]] = {}


def register_recipe(
    name: str,
    recipe_class: Type[Any],
    workflow_id: Optional[int] = None,
) -> None:
    """Register a recipe class under *name* and optionally a *workflow_id*."""
    _REGISTRY_BY_NAME[name] = recipe_class
    if workflow_id is not None:
        _REGISTRY_BY_WORKFLOW_ID[workflow_id] = recipe_class
    logger.debug("Registered recipe %r (workflow_id=%s)", name, workflow_id)


def get_recipe(
    *,
    recipe_name: Optional[str] = None,
    workflow_id: Optional[int] = None,
) -> Optional[Type[Any]]:
    """Look up a recipe by *recipe_name* or *workflow_id*.

    Returns the recipe class or ``None`` if not found.
    """
    if recipe_name is not None:
        return _REGISTRY_BY_NAME.get(recipe_name)
    if workflow_id is not None:
        return _REGISTRY_BY_WORKFLOW_ID.get(workflow_id)
    return None


# Expose a read-only view for introspection
RECIPE_REGISTRY: Dict[str, Type[Any]] = _REGISTRY_BY_NAME


# ---------------------------------------------------------------------------
# Built-in recipe registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Eagerly register all built-in recipes."""
    from modules.workflows.recipes.jira_bug_triage import JiraBugTriageRecipe

    register_recipe("jira_bug_triage", JiraBugTriageRecipe)


_register_builtins()
