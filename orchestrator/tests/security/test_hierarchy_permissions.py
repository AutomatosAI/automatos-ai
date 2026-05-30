"""Tests for core.security.hierarchy_permissions — PRD-140 Phase 1.

Covers the hardened, deny-by-default contract:
  - Actor gate fails closed: anonymous / unknown / cross-workspace / inactive
    actors are denied (no escalation — refused, not arbitrated).
  - System bypass is narrowed: ``is_system_agent`` AND an allowlisted name.
    The flag alone is not a master key.
  - In-subtree allow / cross-subtree deny (deny escalates to Auto).
  - Self-edit allow; cross-workspace target deny.
  - Skill always-deny for non-system actors (escalates to Auto).
  - Task / playbook scoping via owning agent's subtree.
  - Unknown target_type → deny.

Uses an in-memory SQLite DB with the minimal columns the helper queries
(``agents.id/name/is_system_agent/reports_to_id/workspace_id/status``,
``board_tasks.id/assigned_agent_id``, ``workflow_recipes.id/created_by_agent_id``).
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

WS = "11111111-1111-1111-1111-111111111111"
WS2 = "22222222-2222-2222-2222-222222222222"


# ----------------------------------------------------------------- fixtures


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY,
                name TEXT,
                is_system_agent INTEGER NOT NULL DEFAULT 0,
                reports_to_id INTEGER REFERENCES agents(id),
                workspace_id TEXT,
                status TEXT DEFAULT 'active'
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
    """Seed a small org chart (all in workspace WS):

        Auto (1, system, allowlisted)
            ├── VECTOR (2)
            │     ├── PULSE (3)
            │     └── SOCIAL OPS (4)
            │             └── SOCIAL PUBLISHER (5)
            └── ATLAS (6)
                  └── FIXER (7)

    Plus edge-case rows:
        ROGUE (8)   — is_system_agent=1 but NOT allowlisted (no master key)
        SLEEPER (9) — status='inactive'
        FOREIGN (10) — workspace WS2 (cross-tenant target)
    """
    rows = [
        # id, name,                is_sys, parent, workspace, status
        (1,  "Auto",               1, None, WS, "active"),
        (2,  "VECTOR",             0, 1,    WS, "active"),
        (3,  "PULSE",              0, 2,    WS, "active"),
        (4,  "SOCIAL OPS",         0, 2,    WS, "active"),
        (5,  "SOCIAL PUBLISHER",   0, 4,    WS, "active"),
        (6,  "ATLAS",              0, 1,    WS, "active"),
        (7,  "FIXER",              0, 6,    WS, "active"),
        (8,  "ROGUE",              1, 2,    WS, "active"),
        (9,  "SLEEPER",            0, 1,    WS, "inactive"),
        (10, "FOREIGN",            0, None, WS2, "active"),
    ]
    for id_, name, sys_, parent, ws, status in rows:
        db.execute(
            text(
                "INSERT INTO agents (id, name, is_system_agent, reports_to_id, workspace_id, status) "
                "VALUES (:i, :n, :s, :p, :w, :st)"
            ),
            {"i": id_, "n": name, "s": sys_, "p": parent, "w": ws, "st": status},
        )
    db.commit()


def _check(db, **kwargs):
    kwargs.setdefault("workspace_id", WS)
    return can_actor_modify(db, **kwargs)


# ----------------------------------------------------------------- tests


class TestSystemBypass:
    def test_auto_can_edit_any_agent(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=1, target_type=TARGET_AGENT, target_id=7)
        assert d.allowed
        assert "system" in d.reason
        assert d.bypass and d.bypass_kind == "system_actor"

    def test_auto_can_edit_skills(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=1, target_type=TARGET_SKILL, target_id=99)
        assert d.allowed
        assert "system" in d.reason

    def test_system_flag_without_allowlisted_name_is_not_a_master_key(self, db):
        # ROGUE (8) has is_system_agent=1 but is NOT on the allowlist, so it
        # gets ordinary subtree authority, not a bypass. FIXER (7) is outside
        # ROGUE's (empty) subtree → denied.
        _seed_org(db)
        d = _check(db, actor_agent_id=8, target_type=TARGET_AGENT, target_id=7)
        assert not d.allowed
        assert not d.bypass
        assert d.escalation_target == "auto"


class TestActorGate:
    def test_none_actor_denied_no_escalation(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=None, target_type=TARGET_AGENT, target_id=3)
        assert not d.allowed
        assert d.reason == "anonymous_actor"
        assert d.escalation_target is None

    def test_actor_not_in_db_denied(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=999, target_type=TARGET_AGENT, target_id=2)
        assert not d.allowed
        assert "not_found" in d.reason
        assert d.escalation_target is None

    def test_cross_workspace_actor_denied(self, db):
        # VECTOR lives in WS; a call scoped to WS2 must be refused outright.
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=3, workspace_id=WS2)
        assert not d.allowed
        assert d.reason == "cross_workspace_actor"
        assert d.escalation_target is None

    def test_inactive_actor_denied(self, db):
        # SLEEPER (9) is inactive — even a self-edit is refused.
        _seed_org(db)
        d = _check(db, actor_agent_id=9, target_type=TARGET_AGENT, target_id=9)
        assert not d.allowed
        assert d.reason == "actor_inactive"
        assert d.escalation_target is None


class TestSkillAlwaysDeny:
    def test_non_system_cannot_edit_skill(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_SKILL, target_id=99)
        assert not d.allowed
        assert d.escalation_target == "auto"
        assert "skill" in d.reason

    def test_skill_create_with_no_target_id_still_denied(self, db):
        _seed_org(db)
        d = _check(
            db, actor_agent_id=2, target_type=TARGET_SKILL, target_id=None,
            change_type="create",
        )
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestSubtreeScope:
    def test_vector_can_edit_pulse(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=3)
        assert d.allowed

    def test_vector_can_edit_deep_descendant(self, db):
        # VECTOR (2) → SOCIAL OPS (4) → SOCIAL PUBLISHER (5)
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=5)
        assert d.allowed

    def test_vector_cannot_edit_atlas_subtree(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=7)
        assert not d.allowed
        assert d.escalation_target == "auto"

    def test_pulse_cannot_edit_sibling(self, db):
        # PULSE and SOCIAL OPS both report to VECTOR but neither manages the other.
        _seed_org(db)
        d = _check(db, actor_agent_id=3, target_type=TARGET_AGENT, target_id=4)
        assert not d.allowed

    def test_self_edit_allowed(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=2)
        assert d.allowed

    def test_cross_workspace_target_denied(self, db):
        # FOREIGN (10) lives in WS2 — VECTOR (WS) may not touch it.
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type=TARGET_AGENT, target_id=10)
        assert not d.allowed
        assert d.reason == "cross_workspace_target"
        assert d.escalation_target is None


class TestTaskScoping:
    def test_task_assigned_in_subtree_allowed(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO board_tasks (id, assigned_agent_id) VALUES (10, 3)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=10)
        assert d.allowed

    def test_task_assigned_outside_subtree_denied(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO board_tasks (id, assigned_agent_id) VALUES (11, 7)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=11)
        assert not d.allowed

    def test_task_with_no_assignee_denies_with_escalation(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO board_tasks (id, assigned_agent_id) VALUES (12, NULL)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_TASK, target_id=12)
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestPlaybookScoping:
    def test_playbook_owned_in_subtree_allowed(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (20, 4)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=20)
        assert d.allowed

    def test_playbook_owned_outside_subtree_denied(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (21, 7)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=21)
        assert not d.allowed

    def test_playbook_with_no_owner_escalates(self, db):
        _seed_org(db)
        db.execute(text("INSERT INTO workflow_recipes (id, created_by_agent_id) VALUES (22, NULL)"))
        db.commit()
        d = _check(db, actor_agent_id=2, target_type=TARGET_PLAYBOOK, target_id=22)
        assert not d.allowed
        assert d.escalation_target == "auto"


class TestToolAssignmentScoping:
    def test_assigning_tool_to_subtree_agent_allowed(self, db):
        _seed_org(db)
        d = _check(
            db, actor_agent_id=2, target_type=TARGET_TOOL_ASSIGNMENT, target_id=3,
            change_type="assign",
        )
        assert d.allowed

    def test_assigning_tool_to_outside_agent_denied(self, db):
        _seed_org(db)
        d = _check(
            db, actor_agent_id=2, target_type=TARGET_TOOL_ASSIGNMENT, target_id=7,
            change_type="assign",
        )
        assert not d.allowed


class TestEdgeCases:
    def test_unknown_target_type_denied(self, db):
        _seed_org(db)
        d = _check(db, actor_agent_id=2, target_type="invalid", target_id=1)
        assert not d.allowed
        assert "unknown_target_type" in d.reason
