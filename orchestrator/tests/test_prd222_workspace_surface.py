"""PRD-222 W1S2/US-002 — the onboarding snapshot embedded on the workspace surface.

Pure tests for ``onboarding_state.public_snapshot`` — the exact ``{stage}``
object ``GET /api/workspaces/current`` embeds under the response's ``onboarding``
key (and the ``platform_update_onboarding`` tool returns in US-003). Endpoint
integration (auth + DB) is CI's job; here we pin the serializer contract and its
null-safety without a database.

PRD-265 W1 moved the trial out of this snapshot: it is credits in the ledger
now (``GET /api/billing/summary``, ``services.credits.ledger.read_balance``),
not a USD dict carried on the onboarding document. ``public_snapshot`` returns
``{"stage": ...}`` only — the trial-dict tests below are removed with a note.
"""
from __future__ import annotations

from services.onboarding_state import advance_onboarding_stage, public_snapshot


class _FakeWorkspace:
    def __init__(self, onboarding=None):
        self.onboarding = onboarding


def test_snapshot_defaults_to_not_started():
    assert public_snapshot(_FakeWorkspace(None)) == {"stage": "not_started"}


def test_snapshot_reports_current_stage():
    ws = _FakeWorkspace(None)
    advance_onboarding_stage(None, ws, "proposal")
    assert public_snapshot(ws)["stage"] == "proposal"


def test_snapshot_carries_only_the_stage_key():
    # PRD-265: no trial field at all — not null, not present under any key —
    # whatever the onboarding document happens to still carry on disk.
    ws = _FakeWorkspace({"stage": "powerup", "trial": {"granted_usd": 5.0, "state": "active"}})
    snap = public_snapshot(ws)
    assert snap == {"stage": "powerup"}
    assert set(snap) == {"stage"}


# REMOVED: test_snapshot_trial_is_null_when_absent,
# test_snapshot_exposes_only_client_safe_trial_fields,
# test_snapshot_defaults_spent_to_zero_when_missing — all asserted the deleted
# USD trial dict shape on this snapshot; the trial now lives in the credit
# ledger, read through billing_summary, never through public_snapshot.
