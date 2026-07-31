"""PRD-222 W1S2/US-002 — the onboarding snapshot embedded on the workspace surface.

Pure tests for ``onboarding_state.public_snapshot`` — the exact ``{stage, trial}``
object ``GET /api/workspaces/current`` embeds under the response's ``onboarding``
key (and the ``platform_update_onboarding`` tool returns in US-003). Endpoint
integration (auth + DB) is CI's job; here we pin the serializer contract and its
null-safety without a database.
"""
from __future__ import annotations

from services.onboarding_state import advance_onboarding_stage, public_snapshot


class _FakeWorkspace:
    def __init__(self, onboarding=None):
        self.onboarding = onboarding


def test_snapshot_defaults_to_not_started_with_null_trial():
    snap = public_snapshot(_FakeWorkspace(None))
    assert snap == {"stage": "not_started", "trial": None}


def test_snapshot_reports_current_stage():
    ws = _FakeWorkspace(None)
    advance_onboarding_stage(None, ws, "proposal")
    assert public_snapshot(ws)["stage"] == "proposal"


def test_snapshot_trial_is_null_when_absent():
    ws = _FakeWorkspace({"stage": "questions", "stages": {}, "segment": {}})
    assert public_snapshot(ws)["trial"] is None


def test_snapshot_exposes_only_client_safe_trial_fields():
    ws = _FakeWorkspace(
        {
            "stage": "powerup",
            "trial": {
                "granted_usd": 5.0,
                "spent_usd": 1.63,
                "state": "active",
                # internal bookkeeping that must NOT leak to the client:
                "platform_key_id": "secret-internal",
                "last_request_id": "req_999",
            },
        }
    )
    snap = public_snapshot(ws)
    assert snap["stage"] == "powerup"
    assert snap["trial"] == {"granted_usd": 5.0, "spent_usd": 1.63, "state": "active"}
    # only the three client-safe keys — no internal fields surfaced.
    assert set(snap["trial"]) == {"granted_usd", "spent_usd", "state"}


def test_snapshot_defaults_spent_to_zero_when_missing():
    ws = _FakeWorkspace(
        {"stage": "boom", "trial": {"granted_usd": 5.0, "state": "active"}}
    )
    assert public_snapshot(ws)["trial"]["spent_usd"] == 0
