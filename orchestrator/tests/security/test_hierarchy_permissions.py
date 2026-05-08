"""Tests for core.security.hierarchy_permissions — PRD-140 Phase 1.

Covers:
  - System agent bypass (Auto / CTO can edit anything)
  - Skill always-deny for non-system actors (escalates to Auto)
  - In-subtree allow / cross-subtree deny
  - Self-edit allow
  - Unknown target_type → deny
  - Missing actor → deny
  - Unresolvable target owner (e.g. orphaned playbook) → deny with
    escalation_target='auto'

Uses an in-memory SQLite DB with the minimal columns the helper queries
(``agents.id`` / ``agents.is_system_agent`` / ``agents.reports_to_id``,
``board_tasks.id`` / ``assigned_agent_id``, ``workflow_recipes.id`` /
``created_by_agent_id``). Recursive CTE works on SQLite ≥ 3.8.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.security.hierarchy_permissions import (
    can_actor_modify,
    TARGET_AGENT,
    TARGET_PLAYBOOK,
    TARGET_SKILL,
    TARGET_TASK,
    TARGET_TOOL_ASSIGNMENT,
)


# ----------------------------------------------------------------- fixtures


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY,
                is_system_agent INTEGER NOT NULL DEFAULT 0,
                reports_to_id INTEGER REFERENCES agents(id)
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE board_tasks (
                id INTEGER PRIMARY KEY,
                assigned_agent_id INTEGER
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE workflow_recipes (
                id INTEGER PRIMARY KEY,
                created_by_agent_id INTEGER
            )
            """
        ))
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def _seed_org(db):
    """Seed a small org chart:

        AUTO (1, system)
            ├── VECTOR (2)
            │     ├── PULSE (3)
            │     └── SOCIAL OPS (4)
            │             └── SOCIAL PUBLISHER (5)
            └── ATLAS (6)
                  └── FIXER (7)

    AUTO is the only system agent. VECTOR's subtree includes 2,3,4,5.
    ATLAS's subtree includes 6,7. PULSE and FIXER are in disjoint subtrees.
    """
    rows = [
        (1, 1, None),   # AUTO — system
        (2, 0, 1),      # VECTOR → AUTO
        (3, 0, 2),      # PULSE → VECTOR
        (4, 0, 2),      # SOCIAL OPS → VECTOR
        (5, 0, 4),      # SOCIAL PUBLISHER → SOCIAL OPS
        (6, 0, 1),      # ATLAS → AUTO
        (7, 0, 6),      # FIXER → ATLAS
    ]
    for id_, sys_, parent in rows:
        db.execute(
            text("INSERT INTO agents (id, is_system_agent, reports_to_id) VALUES (:i, :s, :p)"),
            {"i": id_, "s": sys_, "p": parent},
        )
    db.commit()


# ----------------------------------------------------------------- tests


class TestSystemBypass:
    def test_auto_can_edit_any_agent(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=1, target_type=TARGET_AGENT, target_id=7)
        assert d.allowed
        assert "system" in d.reason

    def test_auto_can_edit_skills(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=1, target_type=TARGET_SKILL, target_id=99)
        assert d.allowed
        assert "system" in d.reason


class TestNoActor:
    def test_none_actor_treated_as_system(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=None, target_type=TARGET_AGENT, target_id=3)
        assert d.allowed
        assert "no_actor" in d.reason


class TestSkillAlwaysDeny:
    def test_non_system_cannot_edit_skill(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_SKILL, target_id=99)
        assert not d.allowed
        assert d.escalation_target == "auto"
        assert "skill" in d.reason

    def test_skill_create_with_no_target_id_still_denied(self, db):
        _seed_org(db)
        d = can_actor_modify(
            db, actor_agent_id=2, target_type=TARGET_SKILL, target_id=None,
            change_type="create",
        )
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestSubtreeScope:
    def test_vector_can_edit_pulse(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=3)
        assert d.allowed

    def test_vector_can_edit_deep_descendant(self, db):
        # VECTOR (2) → SOCIAL OPS (4) → SOCIAL PUBLISHER (5)
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=5)
        assert d.allowed

    def test_vector_cannot_edit_atlas_subtree(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=7)
        assert not d.allowed
        assert d.escalation_target == "auto"

    def test_pulse_cannot_edit_sibling(self, db):
        # PULSE and SOCIAL OPS both report to VECTOR but neither manages the other.
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=3, target_type=TARGET_AGENT, target_id=4)
        assert not d.allowed

    def test_self_edit_allowed(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=2)
        assert d.allowed


class TestTaskScoping:
    def test_task_assigned_in_subtree_allowed(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO board_tasks (id, assigned_agent_id) VALUES (10, 3)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=10)
        assert d.allowed

    def test_task_assigned_outside_subtree_denied(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO board_tasks (id, assigned_agent_id) VALUES (11, 7)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=11)
        assert not d.allowed

    def test_task_with_no_assignee_denies_with_escalation(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO board_tasks (id, assigned_agent_id) VALUES (12, NULL)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=12)
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestPlaybookScoping:
    def test_playbook_owned_in_subtree_allowed(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (20, 4)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=20)
        assert d.allowed

    def test_playbook_owned_outside_subtree_denied(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (21, 7)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=21)
        assert not d.allowed

    def test_playbook_with_no_owner_escalates(self, db):
        _seed_org(db)
        db.execute(text(
            "INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (22, NULL)"
        ))
        db.commit()
        d = can_actor_modify(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=22)
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestToolAssignmentScoping:
    def test_assigning_tool_to_subtree_agent_allowed(self, db):
        _seed_org(db)
        d = can_actor_modify(
            db, actor_agent_id=2, target_type=TARGET_TOOL_ASSIGNMENT, target_id=3,
            change_type="assign",
        )
        assert d.allowed

    def test_assigning_tool_to_outside_agent_denied(self, db):
        _seed_org(db)
        d = can_actor_modify(
            db, actor_agent_id=2, target_type=TARGET_TOOL_ASSIGNMENT, target_id=7,
            change_type="assign",
        )
        assert not d.allowed


class TestEdgeCases:
    def test_unknown_target_type_denied(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=2, target_type="invalid", target_id=1)
        assert not d.allowed
        assert "unknown_target_type" in d.reason

    def test_actor_not_in_db_denied(self, db):
        _seed_org(db)
        d = can_actor_modify(db, actor_agent_id=999, target_type=TARGET_AGENT, target_id=2)
        assert not d.allowed
        assert "not_found" in d.reason
