"""Policy plane — the master feature flag / staged mode dial (PRD-174 W4, PRD-192 S1).

``AUTOMATOS_POLICY_PLANE`` (read once in ``config.py``, per the no-``os.getenv``-
outside-config rule) drives the whole plane through a staged mode dial:

    off         — byte-for-byte today's per-router gates (no evaluation, no audit)
    shadow      — evaluate + audit every verdict; NEVER block
    destructive — enforce deny/ask only for the fail-closed risk classes
                  (destructive / external_side_effect / publish); shadow-log the rest
    on          — enforce every blocking verdict (PRD-174's original ON)

Isolated in its own module so the stdlib-only call sites (``tool_loop``) and the
unit tests can flip it without importing the heavy config graph directly — and
so a config-import failure fails *safe* (mode ``off``), never wedges execution.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Canonical stage vocabulary (PRD-192): everywhere the dial is named — config,
# docs, the shadow report — these exact strings are used. Order = rollout order.
POLICY_MODES = ("off", "shadow", "destructive", "on")

# The stages that actually block execution. shadow evaluates but never blocks.
ENFORCE_MODES = frozenset({"destructive", "on"})


def policy_plane_mode() -> str:
    """The current policy-plane stage: ``off | shadow | destructive | on``.

    Fail-safe: if config can't be read — or carries an unknown value — the mode
    is ``off`` (today's per-router gates). The dial never fails *into* a live
    stage.
    """
    try:
        from config import config

        mode = str(getattr(config, "POLICY_PLANE_MODE", "off") or "off").strip().lower()
        return mode if mode in POLICY_MODES else "off"
    except Exception:
        logger.warning(
            "[policy.flag] config read failed — policy plane treated off", exc_info=True
        )
        return "off"


def policy_plane_enabled() -> bool:
    """True when the unified policy plane is on ANY live stage (mode ≠ off).

    The registration sites (audit handler, limiter arming, F042/F043) key off
    this so they arm the moment the plane leaves ``off``. Fail-safe: if config
    can't be read the plane is OFF (falls back to today's per-router gates) —
    the flag never fails *into* the new path.
    """
    try:
        from config import config

        return bool(config.POLICY_PLANE_ENABLED)
    except Exception:
        logger.warning(
            "[policy.flag] config read failed — policy plane treated OFF", exc_info=True
        )
        return False


def enforcement_active() -> bool:
    """True in the enforce stages (``destructive`` | ``on``).

    The budget/posture readers use this to stop silently pre-deciding ALLOW on
    a read fault (PRD-192 S1): under an enforce stage they re-raise so the
    gate's single except owns the fail posture; in off/shadow the historical
    swallow stands (shadow never blocks). Fail-safe: ``False``.
    """
    return policy_plane_mode() in ENFORCE_MODES
