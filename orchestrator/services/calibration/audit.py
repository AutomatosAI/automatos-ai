"""PRD-268 §6.11.5, K15 — the calibration's audit rows, each written after its commit (W1's K15).

Consent, decline, accept and each reject write one workspace audit row, with the person as its
actor (they clicked). A row names the calibration and counts what happened; it never carries a
line, a title or anything else Auto read. ``AuditService.log`` commits it and never raises.
"""
from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy.orm import Session

__all__ = ["ACCEPTED", "CONSENTED", "DECLINED", "REJECTED", "audit"]

CONSENTED = "proactive:calibration_consented"
DECLINED = "proactive:calibration_declined"
ACCEPTED = "proactive:calibration_accepted"
REJECTED = "proactive:calibration_line_rejected"
RESOURCE = "calibration"
ACTOR_PERSON = "user"


def audit(db: Session, workspace_id: Any, user_id: int, action: str, calibration_id: Any,
          details: Mapping[str, Any]) -> None:
    """One audit row for the person's calibration, after its commit."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(workspace_id=str(workspace_id), user_id=int(user_id), action=action,
                         actor_type=ACTOR_PERSON, resource_type=RESOURCE, resource_id=str(calibration_id),
                         details=dict(details))
