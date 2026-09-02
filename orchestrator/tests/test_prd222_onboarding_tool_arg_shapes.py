"""PRD-222 — platform_update_onboarding survives the argument shapes the model
actually sends (live-test 2026-09-02, prod, Gemini 2.5 Flash).

The not_started freeze, trace-backed: the model called the spine tool with
``keys=['value', 'segment']`` — the stage name under the wrong key and a
stringified segment — was told "at least one is required", retried
identically (dedup-skipped) and narrated on with nothing recorded. The same
run answered a plan of 'Basic' with "Enterprise is coming soon".

Pure normaliser tests + handler tests against a MagicMock session — no DB.
"""
from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock
from uuid import uuid4

from modules.tools.discovery.handlers_onboarding import _normalise_params, update_onboarding

_LOGGER = "modules.tools.discovery.handlers_onboarding"


class _FakeWorkspace:
    def __init__(self, stage: str = "not_started"):
        self.id = uuid4()
        self.onboarding = {"stage": stage, "stages": {}, "segment": {}}
        self.plan = None
        self.plan_limits = {}


def _db_returning(workspace):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    return db


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# The normaliser — pure, returns a new dict, names every coercion
# --------------------------------------------------------------------------- #


def test_normaliser_returns_a_new_dict_and_never_mutates_input():
    params = {"segment": json.dumps({"comfort": "novice"}), "value": "teach", "plan": " Basic "}
    before = dict(params)
    out, notes = _normalise_params(params)
    assert params == before and out is not params
    assert out["advance_to"] == "teach"
    assert out["segment"] == {"comfort": "novice"}
    assert out["plan"] == "basic"
    assert "value" not in out
    assert len(notes) == 3  # every coercion is named


def test_normaliser_value_only_counts_when_it_names_a_stage():
    out, notes = _normalise_params({"value": "banana"})
    assert "advance_to" not in out and "value" not in out
    assert any("'value' ignored" in n for n in notes)
    out, _ = _normalise_params({"advance_to": "proposal", "value": "teach"})
    assert out["advance_to"] == "proposal"  # an explicit advance_to always wins


def test_normaliser_folds_flat_answer_keys_into_segment():
    out, notes = _normalise_params({"business": "a barber shop", "goal": "bookings"})
    assert out["segment"] == {"business": "a barber shop", "goal": "bookings"}
    assert "business" not in out and "goal" not in out
    assert any("arrived flat" in n for n in notes)


def test_normaliser_keeps_bare_text_for_mapping_and_drops_junk_dicts_with_a_shape_note():
    out, notes = _normalise_params({"segment": "We are Lumen & Lark, a jewellery brand"})
    assert out["segment"] is None
    assert out["_bare_answer"] == (None, "We are Lumen & Lark, a jewellery brand")
    note = next(n for n in notes if "bare-text" in n)
    assert "Lumen" not in note
    out, notes = _normalise_params({"segment": {"industry": "dental", "notes": "x"}})
    assert out["segment"] is None
    assert "keys=['industry', 'notes']" in next(n for n in notes if "nothing usable" in n)


def test_normaliser_lowercases_advance_to():
    out, notes = _normalise_params({"advance_to": "Teach "})
    assert out["advance_to"] == "teach"
    assert any("advance_to case" in n for n in notes)


def test_normaliser_tolerates_non_dict_params():
    out, notes = _normalise_params("garbage")
    assert not out.get("advance_to") and out.get("segment") is None and notes == []


# --------------------------------------------------------------------------- #
# Handler — the shapes that froze prod now record, junk is still honest
# --------------------------------------------------------------------------- #


def test_handler_records_a_json_string_segment():
    ws = _FakeWorkspace("questions")
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"segment": json.dumps({"comfort": "novice"})}))
    assert res["success"] is True
    assert ws.onboarding["segment"]["comfort"] == "novice"


def test_handler_advances_when_the_stage_arrives_under_value():
    # The exact prod shape: keys=['value', 'segment'] with an unusable segment.
    ws = _FakeWorkspace("questions")
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"value": "teach", "segment": "junk"}))
    assert res["success"] is True
    assert res["data"]["stage"] == "teach"


def test_handler_folds_flat_answers():
    ws = _FakeWorkspace("not_started")
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"business": "a barber shop in Leeds"}))
    assert res["success"] is True
    assert ws.onboarding["segment"]["business"] == "a barber shop in Leeds"


def test_handler_bare_text_segment_is_recorded_as_the_first_unanswered_question(caplog):
    ws = _FakeWorkspace("not_started")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        res = _run(update_onboarding(_db_returning(ws), ws.id, {"segment": "We are Lumen & Lark"}))
    assert res["success"] is True
    assert ws.onboarding["segment"] == {"business": "We are Lumen & Lark"}
    warned = [r.getMessage() for r in caplog.records if "bare-text answer recorded as 'business'" in r.getMessage()]
    assert warned and "Lumen" not in warned[0]  # never the user's text


def test_handler_junk_dict_segment_alone_is_still_the_honest_error_and_is_logged(caplog):
    ws = _FakeWorkspace("not_started")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        res = _run(update_onboarding(_db_returning(ws), ws.id, {"segment": {"industry": "jewellery"}}))
    assert res["success"] is False
    assert "at least one is required" in res["error"]
    warned = [r.getMessage() for r in caplog.records if "argument shape coerced" in r.getMessage()]
    assert warned and "keys=['industry']" in warned[0]


def test_handler_is_silent_when_the_shape_is_already_right(caplog):
    ws = _FakeWorkspace("not_started")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        res = _run(update_onboarding(_db_returning(ws), ws.id, {"advance_to": "questions"}))
    assert res["success"] is True
    assert not [r for r in caplog.records if "argument shape coerced" in r.getMessage()]


def test_handler_plan_case_is_forgiven():
    ws = _FakeWorkspace("proposal")
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"plan": " Basic "}))
    assert res["success"] is True
    assert ws.plan == "basic"


def test_handler_enterprise_copy_is_only_for_enterprise():
    ws = _FakeWorkspace("proposal")
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"plan": "Gold"}))
    assert res["success"] is False
    assert "isn't a plan tier" in res["error"] and "coming soon" not in res["error"]
    res = _run(update_onboarding(_db_returning(ws), ws.id, {"plan": "Enterprise"}))
    assert res["success"] is False and "coming soon" in res["error"]


# --------------------------------------------------------------------------- #
# Schema — the model is told an object, never a string
# --------------------------------------------------------------------------- #


def test_schema_says_object_never_string():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_onboarding import register_onboarding_actions

    reg = ActionRegistry()
    register_onboarding_actions(reg)
    desc = reg._actions["platform_update_onboarding"].parameters["properties"]["segment"]["description"].lower()
    assert "json object" in desc and "never a string" in desc
