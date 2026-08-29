"""PRD-222 W1S3/US-003 — the platform_update_onboarding tool (3-file pattern).

Contract tests (schema truth) + handler tests (delegation, clean errors). The
handler is exercised against a MagicMock session + fake workspace — no DB.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

from modules.tools.discovery.action_registry import ActionRegistry
from modules.tools.discovery.actions_onboarding import register_onboarding_actions
from modules.tools.discovery.handlers_onboarding import update_onboarding

TOOL = "platform_update_onboarding"


class _FakeWorkspace:
    def __init__(self, onboarding=None):
        self.onboarding = onboarding


def _db_returning(workspace):
    """MagicMock session whose query(...).filter(...).first() yields workspace."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    return db


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Contract — schema truth
# --------------------------------------------------------------------------- #


def _action_def():
    reg = ActionRegistry()
    register_onboarding_actions(reg)
    return reg._actions[TOOL]


def test_tool_registered_with_pinned_name():
    assert _action_def().name == TOOL


def test_required_is_empty_both_params_optional():
    # US-003 AC3 / US-011 rule: no field with a handler default sits in required[].
    assert _action_def().parameters["required"] == []


def test_advance_to_enum_excludes_not_started():
    enum = _action_def().parameters["properties"]["advance_to"]["enum"]
    assert "not_started" not in enum
    assert {"questions", "proposal", "powerup", "completed", "skipped"} <= set(enum)


def test_description_states_at_least_one_rule():
    desc = _action_def().description.lower()
    assert "at least one" in desc
    assert "segment" in desc and "advance_to" in desc


def test_registered_via_register_all_actions():
    # Proves the wiring in platform_actions.register_all_actions (full init).
    reg = ActionRegistry()
    assert reg.get(TOOL) is not None


def test_registered_in_executor_handler_map():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(MagicMock(), uuid4())
    assert TOOL in executor._handlers


# --------------------------------------------------------------------------- #
# Handler — delegation + clean errors
# --------------------------------------------------------------------------- #


def test_handler_errors_when_both_params_absent():
    res = _run(update_onboarding(_db_returning(_FakeWorkspace()), uuid4(), {}))
    assert res["success"] is False
    assert "at least one" in res["error"].lower()


def test_handler_advances_and_returns_snapshot():
    ws = _FakeWorkspace({"stage": "not_started", "stages": {}, "segment": {}})
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"advance_to": "questions"}))
    assert res["success"] is True
    assert res["data"] == {"stage": "questions", "trial": None}
    assert ws.onboarding["stage"] == "questions"


def test_handler_backward_transition_returns_clean_error():
    ws = _FakeWorkspace({"stage": "proposal", "stages": {}, "segment": {}})
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"advance_to": "questions"}))
    assert res["success"] is False
    assert "transition" in res["error"].lower()
    # No crash, state untouched.
    assert ws.onboarding["stage"] == "proposal"


def test_handler_same_stage_advance_is_idempotent_success():
    # Live-test 2026-08-29: the LLM re-asserts the stage it's already in (e.g.
    # advance_to="building" while building). That must be a benign no-op success,
    # NOT the "non-forward transition X -> X" error Auto used to surface + loop on.
    ws = _FakeWorkspace({"stage": "building", "stages": {"building": "t"}, "segment": {}})
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"advance_to": "building"}))
    assert res["success"] is True
    assert res["data"]["stage"] == "building"
    assert ws.onboarding["stage"] == "building"  # unchanged, no error


def test_handler_same_stage_advance_still_records_segment():
    # A redundant advance paired with real segment answers: drop the no-op
    # advance but still persist the segment (and report success).
    ws = _FakeWorkspace({"stage": "teach", "stages": {"teach": "t"}, "segment": {}})
    res = _run(update_onboarding(
        _db_returning(ws), uuid4(),
        {"advance_to": "teach", "segment": {"goal": "book appts"}},
    ))
    assert res["success"] is True
    assert ws.onboarding["stage"] == "teach"
    assert ws.onboarding["segment"] == {"goal": "book appts"}


def test_same_stage_advance_with_unusable_segment_succeeds():
    # THE PROD FAILURE (2026-08-29, caught by the persona harness):
    #   "Tool platform_update_onboarding failed: set_segment requires at least
    #    one of business/goal/comfort"
    # A same-stage advance drops to a no-op, then a bare-string segment reached
    # set_segment and raised, failing the whole call. Must be a benign success.
    ws = _FakeWorkspace({"stage": "teach", "stages": {"teach": "t"}, "segment": {}})
    res = _run(update_onboarding(
        _db_returning(ws), uuid4(),
        {"advance_to": "teach", "segment": "a barber shop in Leeds"},
    ))
    assert res["success"] is True
    assert ws.onboarding["stage"] == "teach"


def test_advance_with_unrecognized_segment_keys_still_advances():
    ws = _FakeWorkspace({"stage": "teach", "stages": {}, "segment": {}})
    res = _run(update_onboarding(
        _db_returning(ws), uuid4(),
        {"advance_to": "proposal", "segment": {"industry": "dental", "notes": "x"}},
    ))
    assert res["success"] is True
    assert ws.onboarding["stage"] == "proposal"


def test_unusable_segment_alone_gets_the_honest_error_not_an_internal_one():
    ws = _FakeWorkspace({"stage": "teach", "stages": {}, "segment": {}})
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"segment": "just prose"}))
    assert res["success"] is False
    assert "at least one" in res["error"].lower()
    assert "set_segment" not in res["error"]  # never leak the internal helper


def test_handler_records_segment_only():
    ws = _FakeWorkspace({"stage": "questions", "stages": {}, "segment": {}})
    res = _run(
        update_onboarding(
            _db_returning(ws),
            uuid4(),
            {"segment": {"business": "barber", "comfort": "novice"}},
        )
    )
    assert res["success"] is True
    assert ws.onboarding["segment"] == {"business": "barber", "comfort": "novice"}
    assert ws.onboarding["stage"] == "questions"  # segment-only never advances


def test_handler_advance_with_segment_in_one_call():
    ws = _FakeWorkspace({"stage": "not_started", "stages": {}, "segment": {}})
    res = _run(
        update_onboarding(
            _db_returning(ws),
            uuid4(),
            {"advance_to": "teach", "segment": {"goal": "book appts"}},
        )
    )
    assert res["success"] is True
    assert ws.onboarding["stage"] == "teach"
    assert ws.onboarding["segment"] == {"goal": "book appts"}


def test_handler_workspace_not_found():
    res = _run(update_onboarding(_db_returning(None), uuid4(), {"advance_to": "questions"}))
    assert res["success"] is False
    assert "not found" in res["error"].lower()
