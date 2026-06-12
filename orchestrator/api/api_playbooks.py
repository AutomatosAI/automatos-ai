from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from modules.learning import PlaybookMiner


router = APIRouter(prefix="/api/playbooks", tags=["playbooks"])


class MineRequest(BaseModel):
    tenant_id: Optional[str] = None
    min_support: int = 5
    top_k: int = 20
    name_prefix: str = "auto"


@router.post("/mine")
def mine(body: MineRequest, db: Session = Depends(get_db), x_tenant_id: str | None = Header(default=None), ctx: RequestContext = Depends(get_request_context_hybrid)):
    required = config.PLAYBOOKS_REQUIRE_TENANT
    tenant_id = body.tenant_id or x_tenant_id
    if required and not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id required")
    miner = PlaybookMiner(db=db, tenant_id=tenant_id)
    generated = miner.persist_top(top_k=body.top_k, min_support=body.min_support, name_prefix=body.name_prefix)
    return {"generated": generated}


@router.get("")
def list_playbooks(tenant_id: Optional[str] = None, db: Session = Depends(get_db), x_tenant_id: str | None = Header(default=None), ctx: RequestContext = Depends(get_request_context_hybrid)):
    required = config.PLAYBOOKS_REQUIRE_TENANT
    tenant = tenant_id or x_tenant_id
    if required and not tenant:
        raise HTTPException(status_code=400, detail="tenant_id required")
    # Minimal listing to verify persistence; expand as needed
    if tenant:
        rows = db.execute("SELECT id, name, tenant_id, support, created_at FROM playbooks WHERE tenant_id = :t ORDER BY created_at DESC LIMIT 100", {"t": tenant}).fetchall()
    else:
        rows = db.execute("SELECT id, name, tenant_id, support, created_at FROM playbooks ORDER BY created_at DESC LIMIT 100").fetchall()
    return {"items": [dict(r._mapping) for r in rows]}


