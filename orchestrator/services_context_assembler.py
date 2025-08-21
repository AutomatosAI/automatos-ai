from __future__ import annotations

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from .models_context_policy import ContextPolicyModel
from .context_policy import ContextPolicy, assemble_context, SlotName


class ContextAssembler:
    def __init__(self, db: Session):
        self.db = db

    # Persistence
    def get_policy(self, policy_id: str, tenant_id: Optional[str] = None) -> Optional[ContextPolicy]:
        q = self.db.query(ContextPolicyModel).filter(ContextPolicyModel.policy_id == policy_id)
        if tenant_id:
            q = q.filter(ContextPolicyModel.tenant_id == tenant_id)
        row = q.order_by(ContextPolicyModel.version.desc()).first()
        if not row:
            return None
        return ContextPolicy(policy_id=row.policy_id, slots=row.slots, max_total_chars=row.max_total_chars, version=row.version)

    def upsert_policy(self, policy: ContextPolicy, tenant_id: Optional[str] = None, domain: Optional[str] = None, agent_id: Optional[str] = None) -> ContextPolicy:
        row = ContextPolicyModel(
            policy_id=policy.policy_id,
            domain=domain,
            agent_id=agent_id,
            tenant_id=tenant_id,
            slots={k: v.model_dump() for k, v in policy.slots.items()},
            max_total_chars=policy.max_total_chars,
            version=policy.version,
        )
        self.db.add(row)
        self.db.commit()
        return policy

    # Assembly
    def assemble(self, q: str, policy: ContextPolicy, slot_values: Dict[SlotName, str]) -> Dict[str, Any]:
        assembled = assemble_context(q=q, policy=policy, slot_values=slot_values)
        return assembled.model_dump()


