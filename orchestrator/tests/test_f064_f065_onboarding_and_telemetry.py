"""F064 — one definition of "onboarding"; F065 — numbered passes are telemetry.

F064: AutoBrain kept its own copy of the onboarding check, so the age-out that
releases a stuck workspace never reached it — the operator workspace sat at a
non-terminal stage from 2 Sep and every Auto turn took the full-context Tier 0
path, the classifier effectively off.

F065: "pass N" titles slipped the telemetry filter and full graph rebuilds
re-extracted them.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from services import onboarding_state
from services.knowledge_flywheel import title_is_telemetry


def _ws(stage, stages=None, last_reset_at=None):
    doc = {"stage": stage, "stages": stages or {}}
    if last_reset_at:
        doc["last_reset_at"] = last_reset_at
    return NS(onboarding=doc, settings={"onboarding": doc})


def test_a_live_known_stage_is_onboarding():
    ws = _ws("powerup", last_reset_at="2099-01-01T00:00:00+00:00")
    assert onboarding_state.is_onboarding_active(ws) is True


def test_a_run_stuck_past_the_stale_window_is_not():
    ws = _ws("powerup", stages={"questions": "2026-09-02T12:33:21+00:00"},
             last_reset_at="2026-09-02T12:33:14+00:00")
    assert onboarding_state.is_onboarding_active(ws) is False


def test_a_corrupt_or_unknown_stage_classifies_normally():
    assert onboarding_state.is_onboarding_active(_ws("not-a-real-stage")) is False
    assert onboarding_state.is_onboarding_active(_ws(None)) is False


def test_autobrain_reads_the_shared_definition():
    from consumers.chatbot.auto import AutoBrain

    stuck = _ws("powerup", last_reset_at="2026-09-02T12:33:14+00:00")

    class _Db:
        def query(self, _m):
            return self

        def filter(self, *_a):
            return self

        def first(self):
            return stuck

    brain = AutoBrain.__new__(AutoBrain)
    brain._db = _Db()
    brain._workspace_id = "ws-c1"
    assert brain._onboarding_active() is False, "the age-out must reach AutoBrain too"


def test_numbered_passes_are_telemetry_but_words_containing_pass_are_not():
    for title in ("pass 3", "Supplier review — pass 12", "PASS #4", "second pass 2 of the brief"):
        assert title_is_telemetry(title), title
    for title in ("Passport renewal checklist", "Rotate the admin password", "Compass roasting notes",
                  "Passion fruit syrup supplier"):
        assert not title_is_telemetry(title), title
