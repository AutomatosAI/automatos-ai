"""The workspace's callback signing secret (PRD-274 Wave 1, §8-8).

``GET /api/v1/callbacks/signing-secret`` reads it (made on first use) and
``POST /api/v1/callbacks/signing-secret/rotate`` replaces it. Both are for the workspace's managers
(``workspace:manage``) with a recent sign-in, as for minting an API key: the secret is what a
receiver trusts a callback by. ``services.run_callbacks`` says how callbacks are signed. Plain
``def`` routes: FastAPI runs them in its threadpool, off the event loop.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.step_up import require_step_up
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.run_callbacks import (
    EVENT, SIGNATURE_HEADER, SIGNATURE_VERSION, TOLERANCE_SECONDS, rotate_signing_secret, signing_secret,
)

router = APIRouter(prefix="/api/v1/callbacks", tags=["callbacks"])

STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)
HOW_TO_VERIFY = (f"{SIGNATURE_HEADER} is 't=<unix seconds>,{SIGNATURE_VERSION}=<hex>': HMAC-SHA256 of "
                 f"'<t>.<raw body>' with this secret. Refuse a t more than {TOLERANCE_SECONDS} seconds old.")


def _answer(secret: str) -> Dict[str, Any]:
    return {"secret": secret, "header": SIGNATURE_HEADER, "event": EVENT, "verify": HOW_TO_VERIFY}


@router.get("/signing-secret",
            dependencies=[Depends(require_workspace_permission("workspace:manage")), Depends(STEP_UP)])
def read_signing_secret(ctx: RequestContext = Depends(get_request_context_hybrid),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The secret this workspace's callbacks are signed with, and how to check one."""
    return _answer(signing_secret(db, ctx.workspace_id))


@router.post("/signing-secret/rotate",
             dependencies=[Depends(require_workspace_permission("workspace:manage")), Depends(STEP_UP)])
def rotate_callback_signing_secret(ctx: RequestContext = Depends(get_request_context_hybrid),
                                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A new secret; callbacks sent from now on are signed with it, so give it to the receiver."""
    return _answer(rotate_signing_secret(db, ctx.workspace_id))
