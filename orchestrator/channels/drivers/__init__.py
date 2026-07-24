"""Channel drivers.

A driver knows how to talk to a single platform. The
``api.channels`` / ``api.webhooks`` / ``channels.sender`` surfaces all
go through this interface — no per-platform branching anywhere else.

Discovery
---------
``get_driver(platform)`` returns the registered driver class or raises
``UnknownPlatform``. Drivers self-register at import time via
``register_driver``. The init below imports every concrete driver so
they're discoverable as a side effect.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import (
    ChannelDriver,
    ConnectivityMode,
    DriverNotConfigured,
    SendResult,
    UnknownPlatform,
    VerifyResult,
)

_REGISTRY: Dict[str, Type[ChannelDriver]] = {}


def register_driver(platform: str, driver_cls: Type[ChannelDriver]) -> None:
    """Register ``driver_cls`` as the canonical driver for ``platform``.

    Re-registration overwrites — useful for tests, intentional in prod
    when a driver gets split into a subclass.
    """
    _REGISTRY[platform.lower()] = driver_cls


def get_driver(platform: str) -> Type[ChannelDriver]:
    """Return the driver class for ``platform`` or raise UnknownPlatform."""
    key = platform.lower()
    if key not in _REGISTRY:
        raise UnknownPlatform(
            f"No driver registered for platform {platform!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_platforms() -> list[str]:
    """All registered platforms, sorted for stable UI ordering."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Concrete driver imports — each module calls ``register_driver`` at import.
# Keep imports here so ``from channels.drivers import get_driver`` is enough
# for any caller; nobody has to know which file a driver lives in.
# ---------------------------------------------------------------------------

from . import telegram   # noqa: E402,F401
from . import slack      # noqa: E402,F401
from . import whatsapp   # noqa: E402,F401
from . import discord    # noqa: E402,F401
from . import webhook    # noqa: E402,F401


__all__ = [
    "ChannelDriver",
    "ConnectivityMode",
    "DriverNotConfigured",
    "SendResult",
    "UnknownPlatform",
    "VerifyResult",
    "get_driver",
    "list_platforms",
    "register_driver",
]
