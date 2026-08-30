"""PRD-155 S3 — explicit router mount manifest + fail-loud loader.

Replaces main.py's historic ``try/except ImportError: x_router = None`` pattern,
which silently dropped routers whose import failed (two of them — ``api.auth`` and
``api.evaluation`` — failed on *every* boot and nobody noticed). Here the set of
conditionally-mounted routers is declared explicitly:

* ``optional=False`` (default) — the router is expected to exist. If it fails to
  import, boot raises :class:`RouterMountError` naming it, unless the operator has
  set ``ALLOW_DEGRADED_BOOT=true`` (read in ``config.py``), which downgrades the
  failure to a logged skip.
* ``optional=True`` — the router is gated on an optional integration (Composio,
  S3-Vectors). Absence is logged and skipped; never fatal.

The unconditionally-imported core routers (agents, workflows, documents, …) stay
as plain imports in main.py: a plain ``import`` already fails loud, so they need
no manifest entry. Only the previously-silent routers live here.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

Importer = Callable[[str], object]


@dataclass(frozen=True)
class RouterSpec:
    """One conditionally-mounted router: ``getattr(import(module), attr)``."""

    module: str
    attr: str = "router"
    optional: bool = False

    @property
    def name(self) -> str:
        return f"{self.module}:{self.attr}"


class RouterMountError(RuntimeError):
    """Raised at startup when a required router fails to import."""


# Routers historically mounted inside `try/except ImportError` in main.py.
# Order here is the mount order; all have distinct path prefixes, so it does not
# interact with the catch-all ordering of the plain core routers in main.py.
MANIFEST_ROUTERS: tuple[RouterSpec, ...] = (
    RouterSpec("api.shopify"),
    RouterSpec("api.verticals"),  # PRD-183 S5: generic vertical provisioning (F076)
    RouterSpec("api.sites"),
    RouterSpec("api.composio", optional=True),
    RouterSpec("api.cloud_documents", optional=True),
    RouterSpec("api.admin_prompts"),
    RouterSpec("api.bug_reports", optional=True),
    RouterSpec("api.widget_email", optional=True),
    RouterSpec("api.api_keys"),
    RouterSpec("api.widgets.router"),
    RouterSpec("api.widget_marketplace"),
    RouterSpec("api.workspace_github"),
    RouterSpec("api.workspaces"),
    RouterSpec("api.activity"),
    RouterSpec("api.heartbeat"),
    RouterSpec("api.channels"),
    RouterSpec("api.voice_retell"),  # PRD-203 V·S4 — Retell streaming custom-LLM webhook
    RouterSpec("api.reports"),
    RouterSpec("api.deliverables"),
    RouterSpec("api.board_tasks"),
    RouterSpec("api.missions"),
    RouterSpec("api.missions", attr="agent_telemetry_router"),
    RouterSpec("api.assignments"),
    RouterSpec("api.scheduled_tasks"),
    # PRD-181 W11 governance + compliance surfaces.
    RouterSpec("api.approval_grants"),  # S2 — grant / revoke approvals (F060)
    RouterSpec("api.gdpr"),             # S3/S4 — GDPR export + erasure cascade
    # PRD-196 (P2-15) — governance operator surface: audit-log + status (S3),
    # policy posture + budget editors (S4). Ws-admin gated at the router.
    RouterSpec("api.governance"),
    # PRD-204 S11 -- watchlist read/cancel surface over the watch registry.
    RouterSpec("api.watches"),
    # PRD-228 -- the one new fleet-state route (GET /api/v1/fleet).
    RouterSpec("api.fleet"),
    # PRD-233 S6 -- the operator's profile (GET both editions, PUT local only).
    RouterSpec("api.profile"),
)


def load_routers(
    specs,
    *,
    allow_degraded: bool,
    importer: Importer = importlib.import_module,
):
    """Resolve each spec to its router object.

    Returns ``(mounted, failures)`` where ``mounted`` is ``[(spec, router), …]``
    and ``failures`` is ``[(spec, exception), …]``. Raises :class:`RouterMountError`
    if any *required* spec failed and ``allow_degraded`` is False.
    """
    mounted: list[tuple[RouterSpec, object]] = []
    failures: list[tuple[RouterSpec, Exception]] = []

    for spec in specs:
        try:
            module = importer(spec.module)
            router = getattr(module, spec.attr)
        except Exception as exc:  # noqa: BLE001 — a broken module is as fatal as a missing one
            failures.append((spec, exc))
            if spec.optional:
                logger.warning("Optional router %s not mounted: %s", spec.name, exc)
            continue
        mounted.append((spec, router))

    required_failures = [(s, e) for s, e in failures if not s.optional]
    if required_failures:
        names = ", ".join(s.name for s, _ in required_failures)
        if not allow_degraded:
            raise RouterMountError(
                f"Required router(s) failed to import: {names}. "
                f"Set ALLOW_DEGRADED_BOOT=true to boot without them."
            )
        for spec, exc in required_failures:
            logger.error(
                "DEGRADED BOOT: required router %s missing (ALLOW_DEGRADED_BOOT): %s",
                spec.name,
                exc,
            )

    return mounted, failures


def mount_manifest_routers(
    app,
    *,
    allow_degraded: bool,
    specs=MANIFEST_ROUTERS,
    importer: Importer = importlib.import_module,
):
    """Import + ``include_router`` every manifest spec on ``app``."""
    mounted, failures = load_routers(specs, allow_degraded=allow_degraded, importer=importer)
    for _spec, router in mounted:
        app.include_router(router)
    return mounted, failures
