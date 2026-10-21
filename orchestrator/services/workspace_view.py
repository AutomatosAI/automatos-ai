"""Pieces of the GET /api/workspaces/current answer (PRD-266 US-017c).

``api/workspaces.py`` is over AGENTS.md's 800-line limit and may not grow, so the
parts of the route's answer that are more than a field read live here:

* ``masked_settings``: the workspace settings with each integration token
  masked, as the route has always shown them (moved out unchanged);
* ``ai_providers_or_none``: the AI providers this workspace's data goes to
  (``services.ai_providers``, D23), or ``None`` when they cannot be read.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

from services import ai_providers

logger = logging.getLogger(__name__)

# An integration value is masked when its key names a token and it is longer
# than this; the mask keeps this many characters at each end.
TOKEN_MASK_MIN_LENGTH = 8
TOKEN_MASK_KEEP = 4
TOKEN_MASK_FILL = "..."
TOKEN_KEY_MARKER = "token"


def masked_settings(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A new copy of the workspace settings with every integration token masked.

    A string value under an ``integrations`` key that names a token, longer than
    8 characters, shows its first 4 and last 4 around ``...``; every other value
    is as it was. ``integrations`` is always present in the answer.
    """
    current = dict(settings or {})
    integrations = dict(current.get("integrations", {}))
    masked = {key: _masked_value(key, value) for key, value in integrations.items()}
    return {**current, "integrations": masked}


def _masked_value(key: str, value: Any) -> Any:
    """The value as the GET answer shows it: a token string is masked, anything else unchanged."""
    if isinstance(value, str) and len(value) > TOKEN_MASK_MIN_LENGTH and TOKEN_KEY_MARKER in key:
        return value[:TOKEN_MASK_KEEP] + TOKEN_MASK_FILL + value[-TOKEN_MASK_KEEP:]
    return value


def ai_providers_or_none(db: Session, workspace_id: Any) -> Optional[List[Dict[str, str]]]:
    """The workspace's AI providers, or ``None`` when they cannot be read (logged at ERROR).

    ``None`` means unknown and ``[]`` means none. A read failure never fails the
    route: the web app depends on the rest of its answer.
    """
    try:
        return ai_providers.workspace_ai_providers(db, workspace_id)
    except Exception:
        logger.exception("ai_providers could not be read for workspace %s", workspace_id)
        return None


__all__ = ["ai_providers_or_none", "masked_settings"]
