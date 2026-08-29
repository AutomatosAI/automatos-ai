"""PRD-222 W1S1 — onboarding state machine + the single-migration guard.

Pure tests: no Postgres, no live services. The state machine operates on a
fake workspace object and asserts the rebuild-don't-mutate contract directly.
The migration chain is walked statically via Alembic's ``ScriptDirectory``
(no DB connection). @integration DB-bound coverage is CI's job on push.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from services.onboarding_state import (
    CHECKLIST_KEY,
    FIRST_INTEGRATION_KEY,
    INITIAL_STAGE,
    SEGMENT_KEYS,
    STAGE_ORDER,
    InvalidStageTransition,
    academy_url_for_comfort,
    advance_onboarding_stage,
    build_checklist,
    current_stage,
    get_checklist_state,
    get_onboarding,
    is_onboarding_active,
    record_first_integration_connected,
    set_segment,
    update_checklist,
)

NEW_REVISION = "prd222_w1s1_onboarding_jsonb"
PREBRANCH_HEAD = "prd207_su_capture"


class _FakeWorkspace:
    """Duck-typed stand-in for the Workspace ORM row (no DB)."""

    def __init__(self, onboarding=None):
        self.onboarding = onboarding


class _RecordingDB:
    """Minimal SQLAlchemy-session stand-in that records add/commit calls."""

    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1


def _assert_iso(value: str) -> None:
    # Raises ValueError if not a parseable ISO-8601 timestamp.
    datetime.fromisoformat(value)


# --------------------------------------------------------------------------- #
# Migration chain — exactly one head, chained onto the pre-branch head.
# --------------------------------------------------------------------------- #


def _script_dir():
    from alembic.script import ScriptDirectory

    alembic_dir = Path(__file__).resolve().parents[1] / "alembic"
    return ScriptDirectory(str(alembic_dir))


def test_exactly_one_alembic_head():
    heads = _script_dir().get_heads()
    assert len(heads) == 1, f"expected exactly one alembic head, got {heads}"


def test_new_migration_chains_onto_prebranch_head():
    # Asserts this migration's PARENT only. Do NOT assert get_heads() ==
    # [NEW_REVISION] — that freezes the world at this migration and fails the
    # moment any later revision chains on top (it broke PRD-223's chain the
    # same day it merged). Single-head-ness is test_exactly_one_alembic_head's
    # job.
    sd = _script_dir()
    rev = sd.get_revision(NEW_REVISION)
    assert rev.down_revision == PREBRANCH_HEAD


# --------------------------------------------------------------------------- #
# Default / read
# --------------------------------------------------------------------------- #


def test_default_onboarding_is_not_started():
    ws = _FakeWorkspace(None)
    assert current_stage(ws) == INITIAL_STAGE
    doc = get_onboarding(ws)
    assert doc == {"stage": "not_started", "stages": {}, "segment": {}}
    assert is_onboarding_active(ws) is True


def test_get_onboarding_returns_a_copy():
    ws = _FakeWorkspace({"stage": "questions", "stages": {}, "segment": {}})
    doc = get_onboarding(ws)
    doc["stage"] = "MUTATED"
    doc["stages"]["x"] = "y"
    assert ws.onboarding["stage"] == "questions"  # source untouched
    assert ws.onboarding["stages"] == {}


def test_get_onboarding_preserves_unknown_keys():
    # Trial (W1S9) and any future top-level keys must survive a read round-trip.
    ws = _FakeWorkspace(
        {"stage": "boom", "trial": {"granted_usd": 5.0, "state": "active"}}
    )
    doc = get_onboarding(ws)
    assert doc["trial"] == {"granted_usd": 5.0, "state": "active"}


# --------------------------------------------------------------------------- #
# Forward transitions + funnel timestamps + rebuild-don't-mutate
# --------------------------------------------------------------------------- #


def test_forward_advance_stamps_timestamp_commits_and_rebuilds():
    ws = _FakeWorkspace({"stage": "not_started", "stages": {}, "segment": {}})
    original = ws.onboarding
    db = _RecordingDB()

    doc = advance_onboarding_stage(db, ws, "questions")

    assert doc["stage"] == "questions"
    _assert_iso(doc["stages"]["questions"])
    _assert_iso(doc["started_at"])
    _assert_iso(doc["updated_at"])
    assert db.commits == 1 and ws in db.added
    # rebuild-don't-mutate: a NEW dict object was assigned; the original is intact.
    assert ws.onboarding is not original
    assert original == {"stage": "not_started", "stages": {}, "segment": {}}


def test_multi_step_forward_jump_allowed():
    ws = _FakeWorkspace(None)
    advance_onboarding_stage(None, ws, "proposal")  # skips intermediate stages
    assert current_stage(ws) == "proposal"
    assert "proposal" in ws.onboarding["stages"]


def test_full_spine_each_stage_stamped():
    ws = _FakeWorkspace(None)
    for stage in STAGE_ORDER[1:]:  # not_started is the implicit start
        advance_onboarding_stage(None, ws, stage)
        assert ws.onboarding["stage"] == stage
        _assert_iso(ws.onboarding["stages"][stage])
    # every non-initial stage recorded a timestamp: the funnel record.
    assert set(ws.onboarding["stages"]) == set(STAGE_ORDER[1:])


# --------------------------------------------------------------------------- #
# Invalid transitions raise
# --------------------------------------------------------------------------- #


def test_backward_transition_raises():
    ws = _FakeWorkspace({"stage": "proposal", "stages": {}, "segment": {}})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "questions")


def test_same_stage_transition_raises():
    ws = _FakeWorkspace({"stage": "teach", "stages": {}, "segment": {}})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "teach")


def test_unknown_target_stage_raises():
    ws = _FakeWorkspace(None)
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "banana")


def test_completed_is_terminal():
    ws = _FakeWorkspace({"stage": "completed", "stages": {}, "segment": {}})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "powerup")
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "skipped")


def test_skipped_is_terminal():
    ws = _FakeWorkspace({"stage": "skipped", "stages": {}, "segment": {}})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, "questions")


# --------------------------------------------------------------------------- #
# skipped reachable from any non-terminal stage; terminal stamps completed_at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("start", [s for s in STAGE_ORDER if s != "completed"])
def test_skipped_reachable_from_any_nonterminal(start):
    ws = _FakeWorkspace({"stage": start, "stages": {}, "segment": {}})
    advance_onboarding_stage(None, ws, "skipped")
    assert current_stage(ws) == "skipped"
    _assert_iso(ws.onboarding["stages"]["skipped"])
    _assert_iso(ws.onboarding["completed_at"])
    assert is_onboarding_active(ws) is False


def test_completed_stamps_completed_at():
    ws = _FakeWorkspace({"stage": "powerup", "stages": {}, "segment": {}})
    advance_onboarding_stage(None, ws, "completed")
    _assert_iso(ws.onboarding["completed_at"])
    assert is_onboarding_active(ws) is False


# --------------------------------------------------------------------------- #
# Segment answers
# --------------------------------------------------------------------------- #


def test_segment_persists_on_advance():
    ws = _FakeWorkspace(None)
    advance_onboarding_stage(
        None,
        ws,
        "questions",
        segment={"business": "barber shop", "goal": "book appts", "comfort": "novice"},
    )
    seg = ws.onboarding["segment"]
    assert seg == {"business": "barber shop", "goal": "book appts", "comfort": "novice"}
    assert set(seg) <= set(SEGMENT_KEYS)


def test_set_segment_without_advancing_stage():
    ws = _FakeWorkspace(
        {"stage": "questions", "stages": {"questions": "2026-01-01T00:00:00+00:00"}, "segment": {}}
    )
    original = ws.onboarding
    set_segment(None, ws, {"comfort": "very technical"})
    assert ws.onboarding["segment"]["comfort"] == "very technical"
    assert ws.onboarding["stage"] == "questions"  # stage unchanged
    assert ws.onboarding is not original  # still rebuilt, never mutated


def test_set_segment_merges_and_ignores_unknown_keys():
    ws = _FakeWorkspace({"stage": "questions", "segment": {"business": "cafe"}})
    set_segment(None, ws, {"goal": "invoicing", "nonsense": "x"})
    assert ws.onboarding["segment"] == {"business": "cafe", "goal": "invoicing"}


# --------------------------------------------------------------------------- #
# Boundary hardening (live-test 2026-08-29): the LLM sometimes passes `segment`
# as a bare STRING through platform_update_onboarding instead of the object the
# schema asks for. A string used to reach `segment.get(k)` and raise
# "'str' object has no attribute 'get'", which failed the WHOLE advance and
# stalled every onboarding at `teach`. A non-dict segment must be IGNORED so the
# stage advance still lands (the answers were captured on the question turns).
# --------------------------------------------------------------------------- #


def test_advance_to_proposal_with_string_segment_does_not_crash():
    # The exact live failure: advancing teach -> proposal with a free-text
    # string segment must succeed, not raise AttributeError.
    ws = _FakeWorkspace({"stage": "teach", "segment": {"business": "jeweller"}})
    advance_onboarding_stage(None, ws, "proposal", segment="a jewellery brand on Shopify")
    assert ws.onboarding["stage"] == "proposal"
    # The prior real segment is preserved; the junk string contributed nothing.
    assert ws.onboarding["segment"] == {"business": "jeweller"}


@pytest.mark.parametrize("junk", ["a string", 42, ["list"], True])
def test_clean_segment_ignores_non_dict(junk):
    from services.onboarding_state import _clean_segment

    assert _clean_segment(junk) == {}


def test_recommend_plan_survives_string_segment():
    from services.plan_tiers import recommend_plan

    plan, _reason = recommend_plan("we're a barber shop")  # bare string, not a dict
    assert plan in {"basic", "pro", "business"}


def test_set_segment_empty_raises():
    ws = _FakeWorkspace(None)
    with pytest.raises(ValueError):
        set_segment(None, ws, {})
    with pytest.raises(ValueError):
        set_segment(None, ws, {"comfort": None})


# --------------------------------------------------------------------------- #
# PRD-222 US-019 — first_integration_connected funnel stamp (once per workspace)
# --------------------------------------------------------------------------- #


def test_first_integration_stamps_once_and_rebuilds():
    ws = _FakeWorkspace({"stage": "building", "stages": {}, "segment": {}})
    original = ws.onboarding
    db = _RecordingDB()

    fired = record_first_integration_connected(db, ws)

    assert fired is True
    _assert_iso(ws.onboarding[FIRST_INTEGRATION_KEY])
    _assert_iso(ws.onboarding["updated_at"])
    assert db.commits == 1 and ws in db.added
    # rebuild-don't-mutate: a NEW dict was assigned; the original is untouched.
    assert ws.onboarding is not original
    assert FIRST_INTEGRATION_KEY not in original


def test_first_integration_is_idempotent_exactly_once():
    ws = _FakeWorkspace({"stage": "building", "stages": {}, "segment": {}})

    assert record_first_integration_connected(None, ws) is True
    stamp = ws.onboarding[FIRST_INTEGRATION_KEY]

    # A second connection (2nd app, or a re-fired callback) must NOT re-stamp.
    assert record_first_integration_connected(None, ws) is False
    assert ws.onboarding[FIRST_INTEGRATION_KEY] == stamp


def test_first_integration_reconnect_after_disconnect_never_refires():
    # Even if the workspace later drops to 0 connections and reconnects, the
    # once-per-workspace guard means the event never fires a second time.
    ws = _FakeWorkspace({"stage": "completed", FIRST_INTEGRATION_KEY: "2026-08-01T00:00:00+00:00"})
    assert record_first_integration_connected(None, ws) is False


def test_first_integration_preserves_other_onboarding_keys():
    ws = _FakeWorkspace(
        {"stage": "boom", "segment": {"business": "cafe"}, "trial": {"state": "active"}}
    )
    record_first_integration_connected(None, ws)
    assert ws.onboarding["stage"] == "boom"
    assert ws.onboarding["segment"] == {"business": "cafe"}
    assert ws.onboarding["trial"] == {"state": "active"}
    _assert_iso(ws.onboarding[FIRST_INTEGRATION_KEY])


# --------------------------------------------------------------------------- #
# PRD-222 US-020 — the post-setup checklist (derived completion + dismissal)
# --------------------------------------------------------------------------- #


def _item(checklist, item_id):
    return next((i for i in checklist["items"] if i["id"] == item_id), None)


def _base_checklist(**overrides):
    kw = dict(
        connections_count=0, missions_count=0, members_count=1, plan_seats=5, comfort=None
    )
    kw.update(overrides)
    return build_checklist(**kw)


def test_checklist_connect_second_app_needs_two_connections():
    assert _item(_base_checklist(connections_count=1), "connect_second_app")["done"] is False
    assert _item(_base_checklist(connections_count=2), "connect_second_app")["done"] is True


def test_checklist_run_first_mission_needs_one_mission():
    assert _item(_base_checklist(missions_count=0), "run_first_mission")["done"] is False
    assert _item(_base_checklist(missions_count=1), "run_first_mission")["done"] is True


def test_checklist_invite_teammate_needs_second_member():
    # owner alone = 1 member → not done; a teammate → 2 → done.
    assert _item(_base_checklist(members_count=1), "invite_teammate")["done"] is False
    assert _item(_base_checklist(members_count=2), "invite_teammate")["done"] is True


def test_checklist_invite_absent_on_single_seat_plans():
    single = _base_checklist(plan_seats=1)
    assert _item(single, "invite_teammate") is None
    # multi-seat plans DO offer it
    assert _item(_base_checklist(plan_seats=5), "invite_teammate") is not None


def test_checklist_academy_item_is_manual_and_comfort_matched():
    novice = _base_checklist(comfort="brand new")
    course = _item(novice, "take_course")
    assert course["manual"] is True
    assert course["done"] is False  # nothing dismissed yet
    assert course["href"].endswith("/abf")
    # a stored academy_done flag checks it off
    done = build_checklist(
        connections_count=0, missions_count=0, members_count=1, plan_seats=1,
        comfort="very technical", stored={"academy_done": True},
    )
    course2 = _item(done, "take_course")
    assert course2["done"] is True
    assert course2["href"].endswith("/apa")


def test_academy_url_for_comfort_mapping():
    assert academy_url_for_comfort("novice").endswith("/abf")
    assert academy_url_for_comfort("brand new").endswith("/abf")
    assert academy_url_for_comfort(None).endswith("/abf")
    assert academy_url_for_comfort("very technical").endswith("/apa")
    assert academy_url_for_comfort("APA").endswith("/apa")


def test_checklist_dismissed_flag_reflected():
    assert _base_checklist()["dismissed"] is False
    dismissed = build_checklist(
        connections_count=0, missions_count=0, members_count=1, plan_seats=1,
        stored={"dismissed": True},
    )
    assert dismissed["dismissed"] is True


def test_checklist_completed_and_total_counts():
    cl = build_checklist(
        connections_count=2, missions_count=1, members_count=2, plan_seats=5,
        stored={"academy_done": True},
    )
    # 4 items (connect, mission, invite, course), all done
    assert cl["total_count"] == 4
    assert cl["completed_count"] == 4


def test_update_checklist_persists_flags_and_rebuilds():
    ws = _FakeWorkspace({"stage": "completed", "stages": {}, "segment": {}})
    original = ws.onboarding
    db = _RecordingDB()

    stored = update_checklist(db, ws, dismissed=True)

    assert stored == {"dismissed": True}
    assert ws.onboarding[CHECKLIST_KEY] == {"dismissed": True}
    assert db.commits == 1 and ws in db.added
    # rebuild-don't-mutate: NEW dict assigned; original untouched.
    assert ws.onboarding is not original
    assert CHECKLIST_KEY not in original


def test_update_checklist_merges_flags_across_writes():
    ws = _FakeWorkspace({"stage": "completed", CHECKLIST_KEY: {"dismissed": True}})
    update_checklist(None, ws, academy_done=True)
    # academy_done added WITHOUT clobbering the prior dismissed flag
    assert ws.onboarding[CHECKLIST_KEY] == {"dismissed": True, "academy_done": True}
    assert get_checklist_state(ws) == {"dismissed": True, "academy_done": True}
