"""`orchestrator/services`

Rewrite services live here (kept separate from older `core/*` services).

Important: keep this package safe to import.
`api/*` modules import `services.*`, which means `services/__init__.py` is
executed at startup. Do **not** import optional modules here in a way that can
crash startup.
"""

from .metadata_sync_service import MetadataSyncService  # noqa: F401

__all__ = ["MetadataSyncService"]
