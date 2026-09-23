"""
PRD-251 S0.6 (D16) — the Composio deny list, platform-wide
==========================================================

Composio actions that spend real money or act outside Automatos (Higgsfield's
billing purchase and trial changes, website create/deploy/publish, contest
entries, raw app invocation) are refused before any network call: for every
caller — agents, Playbooks and the API — in every workspace, whatever the
policy plane mode and whatever the capability classifier says. Buying credits,
changing plans and deploying stay with a person, in the tool's own interface.

The list is DATA, never a code constant: the ``composio.denied_actions`` system
setting, a JSON list of action slugs, seeded by the ``prd251_socials`` migration
and edited by the super-admin in Settings → System Settings. It is read on every
call (``get_system_setting``), so removing a slug unblocks it with no restart or
deploy. Slugs match case-insensitively.

``composio_action_denial(slug)`` is the ONE check. Every Composio execution entry
point calls it before any network call; ``tests/test_prd251_composio_deny.py``
finds them by grep and holds each one to it.

* No row, or an empty value → nothing is denied (a stack without the seed).
* A value that is not a JSON list of strings → EVERY action is refused, with
  the reason, until it is fixed: a guard on real money fails closed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)

KEY_DENIED_ACTIONS = "denied_actions"  # in SettingCategory.COMPOSIO

BLOCKED_PREFIX = "This action is blocked in Automatos: "
DENIED_REASON = (
    "{slug} is on the platform's Composio deny list. Buying credits, changing plans "
    "and deploying stay with a person, in the tool's own interface."
)
UNREADABLE_REASON = (
    "the Composio deny list (system setting composio.denied_actions) is not a JSON "
    "list of action slugs, so every Composio action is refused until a super-admin "
    "fixes it in Settings → System Settings."
)
ERROR_TYPE_DENIED = "action_denied"


def parse_denied_actions(raw: Optional[str]) -> FrozenSet[str]:
    """The setting's value → the denied slugs, upper-cased.

    No value (``None``, blank, or not text — the column only ever holds text) →
    nothing denied. Raises ``ValueError`` unless the value is a JSON list of
    strings.
    """
    if not isinstance(raw, str) or not raw.strip():
        return frozenset()
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(slug, str) for slug in value):
        raise ValueError("composio.denied_actions must be a JSON list of action slugs")
    return frozenset(slug.strip().upper() for slug in value if slug.strip())


def composio_action_denial(slug: Any) -> Optional[str]:
    """``"This action is blocked in Automatos: <reason>"`` when ``slug`` may not
    run, else ``None``. Reads the setting fresh on every call."""
    # Lazy: core.llm.manager imports every LLM provider client, and this module
    # is imported by the Composio client itself.
    from core.llm.manager import get_system_setting
    from core.models.system_settings import SettingCategory

    raw = get_system_setting(SettingCategory.COMPOSIO.value, KEY_DENIED_ACTIONS, None)
    try:
        denied = parse_denied_actions(raw)
    except ValueError:
        logger.error("[ComposioDenyList] composio.denied_actions is unreadable (%r); refusing %s", raw, slug)
        return BLOCKED_PREFIX + UNREADABLE_REASON

    normalized = str(slug or "").strip().upper()
    if normalized and normalized in denied:
        logger.warning("[ComposioDenyList] refused %s", normalized)
        return BLOCKED_PREFIX + DENIED_REASON.format(slug=normalized)
    return None


def denied_result(denial: str) -> Dict[str, Any]:
    """The standard failed-execution envelope for a refused action."""
    return {"success": False, "data": None, "error": denial, "error_type": ERROR_TYPE_DENIED}
