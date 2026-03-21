"""
Context Backend Factory — PRD-108 A/B Experiment
=================================================
Returns the configured SharedContextPort wrapped in instrumentation.
"""

import logging
from typing import Optional
from config import config
from core.ports.context import SharedContextPort
from modules.context.instrumentation import InstrumentedSharedContext

logger = logging.getLogger(__name__)

# Singleton instrumented instances — one per backend type
_instances: dict[str, InstrumentedSharedContext] = {}


def get_shared_context(backend: Optional[str] = None) -> Optional[InstrumentedSharedContext]:
    """Get the configured shared context backend, wrapped in instrumentation.

    Args:
        backend: Override config. One of "vector_field", "redis", or None (use config default).

    Returns:
        InstrumentedSharedContext wrapping the configured backend, or None if unavailable.
    """
    backend = backend or getattr(config, "SHARED_CONTEXT_BACKEND", "vector_field")

    if backend in _instances:
        return _instances[backend]

    try:
        if backend == "vector_field":
            from modules.context.adapters.vector_field import VectorFieldSharedContext
            inner = VectorFieldSharedContext()
        elif backend == "redis":
            from modules.context.adapters.redis_context import RedisSharedContext
            inner = RedisSharedContext()
        else:
            logger.error("[Factory] Unknown backend: %s", backend)
            return None

        instrumented = InstrumentedSharedContext(inner, backend_name=backend)
        _instances[backend] = instrumented
        logger.info("[Factory] Initialized shared context backend: %s", backend)
        return instrumented
    except Exception as e:
        logger.warning("[Factory] Failed to initialize %s backend: %s", backend, e)
        return None
