"""PRD-266 §4.1 (US-001) — ``GET /.well-known/automatos.json``: what this server is.

**PUBLIC.** AGENTS.md exception 1, approved by Gerard on 20 Oct 2026 (PRD-266 §4.10, tracking
issue #1253), as PRD-264 D16 did for ``/verify``: the app reads this before it shows a sign-in,
so it has no credential to send. The document carries names and capabilities only, never a
secret (``services.mobile.discovery``); the Clerk publishable key it names is public by design.

Both editions serve it, and a local install in ``LOCAL_REMOTE_ACCESS=paired`` mode serves it
too: the route takes no request context, so the closed anonymous lane never sees it (K7). A
bearer token or workspace header sent along is ignored. A misconfigured address or publishable
key is a 500 naming the setting (never its value), logged at ERROR.

A plain ``def`` with no database: it reads config only. Registered through
``RouterSpec("api.well_known")`` in ``router_manifest.py``, so ``main.py`` never changes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from services.mobile.discovery import DiscoveryUnavailable, build_discovery

DISCOVERY_PATH = "/.well-known/automatos.json"
NOT_SET_UP = "Discovery is not set up on this server: {reason}."

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mobile"])


@router.get(DISCOVERY_PATH)
def discovery_document(request: Request) -> Dict[str, Any]:
    """The §4.1 discovery document: edition, addresses, sign-in modes and capabilities."""
    try:
        return build_discovery(str(request.base_url))
    except DiscoveryUnavailable as misconfigured:
        logger.error("discovery refused: %s", misconfigured)
        raise HTTPException(status_code=500, detail=NOT_SET_UP.format(reason=misconfigured)) from misconfigured
