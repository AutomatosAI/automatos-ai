"""Policy plane — the master feature flag (PRD-174 W4).

``AUTOMATOS_POLICY_PLANE`` (read once in ``config.py``, per the no-``os.getenv``-
outside-config rule) gates the entire plane. OFF ⇒ byte-for-byte today's
per-router gates; ON ⇒ the plane (one chokepoint + the on_pre_tool seam).

Isolated in its own module so the stdlib-only call sites (``tool_loop``) and the
unit tests can flip it without importing the heavy config graph directly — and
so a config-import failure fails *safe* (plane OFF), never wedges execution.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def policy_plane_enabled() -> bool:
    """True when the unified policy plane is switched on for this deployment.

    Fail-safe: if config can't be read the plane is OFF (falls back to today's
    per-router gates) — the flag never fails *into* the new path.
    """
    try:
        from config import config

        return bool(config.POLICY_PLANE_ENABLED)
    except Exception:
        logger.warning(
            "[policy.flag] config read failed — policy plane treated OFF", exc_info=True
        )
        return False
