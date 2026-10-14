"""PRD-259 — skins in Studio: the skin routes.

``GET /api/skins/schema`` serves the automatos-skin/v1 rule set
(``services/skins/schema.py``) so the Skin Kit can sync its copy from Studio.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from services.skins.schema import schema_document

router = APIRouter(prefix="/api/skins", tags=["Skins"])


@router.get("/schema")
def get_skin_schema(ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """The skin format's rule set: allowed tokens, contrast pairs, fonts, mascot, shape and text limits."""
    return schema_document()
