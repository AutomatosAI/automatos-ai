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
    RouterSpec("api.voice_speech"),  # Auto's server voice: speech as MP3 from voice-tts, and its voices
    RouterSpec("api.reports"),
    RouterSpec("api.deliverables"),
    RouterSpec("api.deliverable_thumbnails"),  # F353 — a document card's first-page picture
    RouterSpec("api.deliverable_knowledge"),  # F354 — "Add to Knowledge" on a Deliverable, and remove
    RouterSpec("api.board_tasks"),
    RouterSpec("api.board_task_rebrief"),  # PRD-252 R2 — Discuss: "Update ticket and re-queue"
    RouterSpec("api.add_to_knowledge"),  # F305 — "Add to knowledge" on an approved card
    RouterSpec("api.add_report_to_knowledge"),  # F305 — ... and on a report
    RouterSpec("api.missions"),
    RouterSpec("api.missions", attr="agent_telemetry_router"),
    RouterSpec("api.mission_settings"),  # F282 — PATCH .../settings: the one check_each_step switch
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
    # PRD-234 S1a -- the CLI host contract (pair / heartbeat / claim / events / result).
    # Every handler 404s unless CLI_RUNTIME_ENABLED (local edition only).
    RouterSpec("api.cli_hosts"),
    # PRD-245 W1 -- the loopback MCP endpoint a ticket session calls Automatos
    # through. Same gate as the host routes: 404 unless CLI_RUNTIME_ENABLED.
    RouterSpec("api.session_tools"),
    # PRD-233 S6 -- the operator's profile (GET both editions, PUT local only).
    RouterSpec("api.profile"),
    # PRD-264 EUAI-10 (D16) -- the public detection route: a SHA-256 in, yes or no out,
    # rate-limited per IP. The one unauthenticated route of PRD-264.
    RouterSpec("api.provenance"),
    # PRD-266 US-016a -- content reports (POST /api/v1/reports): any member reports an
    # Auto message or a Deliverable. Authenticated; not PRD-76's api.reports.
    RouterSpec("api.content_reports"),
    # PRD-266 W1 Connect -- the Automatos Go backend: device discovery, pairing,
    # remote access (/api/v1/mobile/*).
    RouterSpec("api.mobile_devices"),
    # PRD-266 W2 Push -- the operator's Expo answer (/api/v1/mobile/push-consent, local only)
    # and the workspace's question redaction (/api/v1/mobile/push-settings).
    RouterSpec("api.mobile_push"),
    # PRD-266 US-001 (§4.1) -- the discovery document (GET /.well-known/automatos.json).
    RouterSpec("api.well_known"),
    # PRD-268 W1 -- Proactive Auto: the suggestions (list, Start, Hide) and the check-in
    # runs (/api/proactive/*).
    RouterSpec("api.proactive"),
    # PRD-268 US-C18 (§6.11.3) -- readiness: what Auto depends on, and Check again
    # (/api/readiness, /api/readiness/check).
    RouterSpec("api.readiness"),
    # PRD-270 US-270-01 -- the hosted catalogue a local install's Sync with Automatos Cloud
    # reads (GET /api/marketplace/catalog): public, rate-limited per IP, 404 in the local edition.
    RouterSpec("api.marketplace_catalog"),
    # PRD-270 US-270-07a -- the public Marketplace reads a visitor browses
    # (GET /api/public/marketplace/{type}): display fields only, rate-limited per IP, 404 locally.
    RouterSpec("api.public_marketplace"),
    # PRD-270 US-270-03b -- Sync with Automatos Cloud (POST /api/marketplace/cloud-sync) and its
    # status (GET .../status): local edition only (404 hosted), workspace owners and admins.
    RouterSpec("api.marketplace_cloud_sync"),
    # PRD-270 US-270-12c -- the bundles the hosted catalogue lists
    # (GET /api/marketplace/catalog/bundles/{revision}): public, rate-limited per IP, 404 in the local edition.
    RouterSpec("api.marketplace_catalog_items"),
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
