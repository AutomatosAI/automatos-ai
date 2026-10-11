"""The component checks behind GET /api/system/health.

Split out of ``api/system.py`` (#1100), where one 166-line handler ran every check
inline. Each check is a probe that returns ``(status, metrics)``; ``_check`` times
it, turns an exception into an unhealthy component and logs why. The metrics a
failed check reports never carry the exception text.
"""
from __future__ import annotations

import importlib
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import psutil
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from core.models import ComponentHealth, Document, RAGConfiguration

logger = logging.getLogger(__name__)

HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
DEGRADED = "degraded"
CHECK_FAILED = "Service check failed"
CHUNKS_TABLE = "document_chunks"
GIB = 1024 ** 3

Probe = Callable[[], Tuple[str, Dict[str, Any]]]


def _elapsed_ms(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


def _check(name: str, probe: Probe, failure: Dict[str, Any]) -> ComponentHealth:
    """Run one probe; an exception is an unhealthy component with ``failure`` as its metrics."""
    checked_at = datetime.now()
    try:
        status, metrics = probe()
    except Exception:
        logger.exception("system health: the %s check failed", name)
        status, metrics = UNHEALTHY, failure
    return ComponentHealth(name=name, status=status, last_check=checked_at, metrics=metrics)


def _database(db: Session) -> Probe:
    def probe():
        db.execute(text("SELECT 1"))
        return HEALTHY, {"connection": "active"}
    return probe


def _redis() -> Probe:
    def probe():
        from core.redis.client import get_redis_client

        start = time.time()
        if not get_redis_client().test_connection():
            return UNHEALTHY, {"ping": "failed", "error": "PING returned false"}
        return HEALTHY, {"ping": "success", "latency_ms": _elapsed_ms(start), "connection": "active"}
    return probe


def _api() -> Probe:
    """Internal readiness: the core modules load."""
    def probe():
        from core.llm.manager import get_llm_manager
        from modules.rag import get_rag_service

        start = time.time()
        get_llm_manager()
        get_rag_service()
        return HEALTHY, {"readiness": "ready", "latency_ms": _elapsed_ms(start), "core_modules": "loaded"}
    return probe


def _document_processor(db: Session) -> Probe:
    def probe():
        from modules.rag import get_rag_service

        start = time.time()
        importlib.import_module("consumers.document_processor")
        get_rag_service()
        doc_count = db.query(Document).count()
        return HEALTHY, {"status": "operational", "latency_ms": _elapsed_ms(start),
                         "documents_in_db": doc_count, "worker": "accessible"}
    return probe


def _chunk_count(db: Session) -> int:
    """Rows in the legacy chunks table, or 0 where it no longer exists (RAG is on S3 Vectors)."""
    if not inspect(db.get_bind()).has_table(CHUNKS_TABLE):
        return 0
    return db.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar() or 0


def _rag(db: Session) -> Probe:
    def probe():
        from modules.rag import get_rag_service

        start = time.time()
        get_rag_service()
        rag_config_count = db.query(RAGConfiguration).count()
        return HEALTHY, {"status": "operational", "latency_ms": _elapsed_ms(start),
                         "rag_configs": rag_config_count, "document_chunks": _chunk_count(db),
                         "service": "accessible"}
    return probe


def component_health(db: Session) -> List[ComponentHealth]:
    """Every component's health, in the order the dashboard shows them."""
    return [
        _check("database", _database(db), {"connection": "failed", "error": CHECK_FAILED}),
        _check("redis", _redis(), {"ping": "failed", "error": CHECK_FAILED, "connection": "failed"}),
        _check("api", _api(), {"readiness": "not_ready", "error": CHECK_FAILED, "core_modules": "failed"}),
        _check("document_processor", _document_processor(db),
               {"status": "error", "error": CHECK_FAILED, "worker": "unavailable"}),
        _check("rag_system", _rag(db), {"status": "error", "error": CHECK_FAILED, "service": "unavailable"}),
    ]


def overall_status(components: List[ComponentHealth]) -> str:
    return HEALTHY if all(c.status == HEALTHY for c in components) else DEGRADED


def host_metrics(cpu_percent: float) -> Dict[str, str]:
    """The host's CPU, memory and disk, formatted for the dashboard."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "memory_available": f"{memory.available / GIB:.1f}GB",
        "disk_usage": f"{disk.percent}%",
        "disk_free": f"{disk.free / GIB:.1f}GB",
    }


__all__ = ["component_health", "host_metrics", "overall_status"]
