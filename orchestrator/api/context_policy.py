from __future__ import annotations

from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database.database import get_db
from modules.search.policies import ContextPolicy, SlotName
from modules.search.services.context_assembler import ContextAssembler
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext


router = APIRouter(prefix="/api/policy", tags=["context-policy"])


class PolicyUpsertRequest(BaseModel):
    policy: ContextPolicy
    tenant_id: Optional[str] = None
    domain: Optional[str] = None
    agent_id: Optional[str] = None


@router.get("/{policy_id}")
def read_policy(policy_id: str, tenant_id: Optional[str] = Query(None), ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    assembler = ContextAssembler(db)
    policy = assembler.get_policy(policy_id=policy_id, tenant_id=tenant_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@router.put("/{policy_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def upsert_policy(policy_id: str, body: PolicyUpsertRequest, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    if body.policy.policy_id != policy_id:
        raise HTTPException(status_code=400, detail="policy_id mismatch")
    assembler = ContextAssembler(db)
    policy = assembler.upsert_policy(policy=body.policy, tenant_id=body.tenant_id, domain=body.domain, agent_id=body.agent_id)
    return {"status": "ok", "policy": policy}


class AssembleRequest(BaseModel):
    q: str
    slot_values: Dict[SlotName, str]
    tenant_id: Optional[str] = None


@router.post("/{policy_id}/assemble", dependencies=[Depends(require_workspace_permission("documents:read"))])
def assemble(policy_id: str, body: AssembleRequest, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    assembler = ContextAssembler(db)
    policy = assembler.get_policy(policy_id=policy_id, tenant_id=body.tenant_id)
    if not policy:
        policy = ContextPolicy.default(policy_id)
    result = assembler.assemble(q=body.q, policy=policy, slot_values=body.slot_values)
    return result


_active_variants: Dict[str, str] = {}


class ABSetRequest(BaseModel):
    policy_a: str
    policy_b: str
    active: str  # "A" | "B"


@router.post("/abtest/set", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def set_abtest(body: ABSetRequest, ctx: RequestContext = Depends(get_request_context_hybrid)):
    if body.active not in ("A", "B"):
        raise HTTPException(status_code=400, detail="active must be 'A' or 'B'")
    _active_variants[f"{body.policy_a}|{body.policy_b}"] = body.active
    return {"status": "ok", "active": body.active}


@router.get("/abtest/get")
def get_abtest(policy_a: str, policy_b: str, ctx: RequestContext = Depends(get_request_context_hybrid)):
    active = _active_variants.get(f"{policy_a}|{policy_b}", "A")
    return {"active": active}


