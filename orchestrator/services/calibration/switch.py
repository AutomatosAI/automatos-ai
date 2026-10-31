"""PRD-268 §6.11.5, K15 — calibration's platform switch.

``CALIBRATION_ENABLED`` ships off until §12-Q22 (consent to read the titles of a person's work)
is answered. It is read from config on every call, so the routes (``api/auto_calibration.py``)
and discovery's ``calibration`` capability (``services/mobile/discovery.py``) follow it at once.
"""
from __future__ import annotations

from config import config

__all__ = ["calibration_enabled"]


def calibration_enabled() -> bool:
    """K15's switch, read from config on every call: off as shipped."""
    return bool(config.CALIBRATION_ENABLED)
