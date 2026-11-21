"""PRD-277 US-401, US-408: what sign-in does once the boot has finished starting the app.

``finish_sign_in_start_up`` runs in the background boot (``main._boot_in_background``), in every
edition:

* It closes sign-in registration (``sign_in_flow.REGISTRY``): nothing may register an identity
  method or a hook after start-up (§5.13). The first sign-in closes it too, if it comes first.
* Only where the edition signs in with OIDC, it reads the provider's metadata and keys once, as the
  ``oidc_provider`` stage of the bootstrap report, on a worker thread. The API is already serving
  and stays up whatever happens: ``/health/bootstrap`` shows the stage as failed, with a message
  that names no address or value, and each sign-in tries the provider again and answers with a
  clear error until it can be reached.

In ``local`` and ``saas`` it closes the registry and adds no stage, so their reports are unchanged.
"""
from __future__ import annotations

import asyncio
from typing import Any

from core import editions
from core.auth import oidc_provider
from core.auth.sign_in_flow import REGISTRY
from core.models.bootstrap import BootstrapStage, run_stage


async def _read_provider() -> None:
    await asyncio.to_thread(oidc_provider.check_provider)


async def finish_sign_in_start_up(report: Any) -> None:
    """Close registration; where sign-in is OIDC, report whether the provider can be read."""
    REGISTRY.close()
    if editions.current().sign_in != editions.SIGN_IN_OIDC:
        return
    await run_stage(report, BootstrapStage.OIDC_PROVIDER, _read_provider)


__all__ = ["finish_sign_in_start_up"]
