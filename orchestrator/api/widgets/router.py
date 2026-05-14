"""
Widget API Router
==================
Aggregates all Widget API sub-routers under /api/widgets.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/widgets", tags=["Widget API"])

# Health check
@router.get("/health")
async def widget_health():
    return {"status": "ok", "version": "1.0"}

# Mount sub-routers
try:
    from api.widgets.session import router as session_router
    router.include_router(session_router)
except ImportError as e:
    logger.warning(f"Widget session router not available: {e}")

try:
    from api.widgets.config import router as config_router
    router.include_router(config_router)
except ImportError as e:
    logger.warning(f"Widget config router not available: {e}")

try:
    from api.widgets.chat import router as chat_router
    router.include_router(chat_router)
except ImportError as e:
    logger.warning(f"Widget chat router not available: {e}")

try:
    from api.widgets.documents import router as documents_router
    router.include_router(documents_router)
except ImportError as e:
    logger.warning(f"Widget documents router not available: {e}")

try:
    from api.widgets.data import router as data_router
    router.include_router(data_router)
except ImportError as e:
    logger.warning(f"Widget data router not available: {e}")

try:
    from api.widgets.blog import router as blog_router
    router.include_router(blog_router)
except ImportError as e:
    logger.warning(f"Widget blog router not available: {e}")

try:
    from api.widgets.docs import router as docs_router
    router.include_router(docs_router)
except ImportError as e:
    logger.warning(f"Widget docs router not available: {e}")

try:
    from api.widgets.callback import router as callback_router
    router.include_router(callback_router)
except ImportError as e:
    logger.warning(f"Widget callback router not available: {e}")
