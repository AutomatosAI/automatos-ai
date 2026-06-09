"""PRD-142 Wave 4 (W4-S11): HARNESS structured-store models.

learning_outcomes is extended into the HARNESS OUTCOME store, and harness_prescriptions
is the new PRESCRIPTION store (Role 2, §12.2). These models live on the
knowledge_system Base (NOT the main create_all Base used by init_test_db), so they
never touch the CI test schema — these are pure model-shape tests (no DB).
"""
from uuid import uuid4

from modules.memory.storage.knowledge_system import HarnessPrescription, LearningOutcome


def test_learning_outcome_has_harness_outcome_fields():
    ws = uuid4()
    lo = LearningOutcome(
        agent_id=7,
        workspace_id=ws,
        run_id="2026-06-09-001",
        change_type="heartbeat_tune",
        risk_score=2,
        status="applied",
        current_value_before={"interval_minutes": 30},
    )
    assert lo.workspace_id == ws
    assert lo.run_id == "2026-06-09-001"
    assert lo.change_type == "heartbeat_tune"
    assert lo.risk_score == 2
    assert lo.status == "applied"
    assert lo.current_value_before == {"interval_minutes": 30}
    # original learning fields remain
    assert hasattr(lo, "success_rate_before")
    assert hasattr(lo, "application_count")


def test_harness_prescription_shape():
    ws = uuid4()
    rx = HarnessPrescription(
        workspace_id=ws,
        run_id="2026-06-09-001",
        prescription_id="rx-1",
        target_type="agent",
        target_id=7,
        target_name="SCOUT",
        change_type="tool_assignment_remove",
        risk_score=3,
        proposed_value={"app_name": "GMAIL"},
        current_value_before={"app_name": "GMAIL"},
        rationale="kept failing for this agent",
    )
    assert rx.__tablename__ == "harness_prescriptions"
    assert rx.workspace_id == ws
    assert rx.prescription_id == "rx-1"
    assert rx.change_type == "tool_assignment_remove"
    assert rx.risk_score == 3
    assert rx.proposed_value == {"app_name": "GMAIL"}


def test_harness_prescription_has_no_foreign_keys():
    # No FKs (plain columns, like routing_rules) so create_all ordering is never
    # a concern and the model can't drag an unrelated table into a fresh schema.
    fks = [c.name for c in HarnessPrescription.__table__.columns if c.foreign_keys]
    assert fks == []


def test_harness_prescription_is_workspace_scoped():
    # Tenant isolation (A4): workspace_id is required, indexed.
    col = HarnessPrescription.__table__.columns["workspace_id"]
    assert col.nullable is False
    assert col.index is True
