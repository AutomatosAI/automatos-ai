"""PRD-222 — the onboarding spine's tools must survive semantic narrowing.

Live failure (Harbourline retest #2, 2026-08-29, after the #646 ATOM fix): the
flow engaged, Auto collected the segment, said "I'll update our system with
your preferences" — and stopped. The turn's user text was "Yes please.", the
semantic top-K ranked the dispatcher enum on that text, and
``platform_update_onboarding`` fell out of the surface. The section instructs
Auto to call tools by name; a surface that doesn't carry them dead-ends the
model (the blueprint-rules wall class).

Fix: ``_apply_onboarding_prior`` in tool_router — while the workspace is
mid-onboarding, ``ONBOARDING_PRIOR_ACTIONS`` is folded into the narrowed enum
through the same gate filter and cap as the PRD-221 page prior.
"""
from __future__ import annotations

import pytest

from modules.tools.discovery.actions_onboarding import ONBOARDING_PRIOR_ACTIONS
from modules.tools.tool_router import _apply_onboarding_prior

WS_ID = "11111111-1111-1111-1111-111111111111"


class _WS:
    def __init__(self, stage):
        self.onboarding = {"stage": stage, "stages": {}, "segment": {}}


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._ws


class _FakeSession:
    def __init__(self, ws):
        self._ws = ws

    def query(self, _model):
        return _FakeQuery(self._ws)


class _BoomSession:
    def query(self, _model):
        raise RuntimeError("db down")


_NARROWED = (["platform_list_agents"], "semantic_top_k", False)


# --------------------------------------------------------------------------- #
# The registry guard — a renamed tool must fail THIS test, not dead-end Auto
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ONBOARDING_PRIOR_ACTIONS)
def test_every_prior_action_is_registered(name):
    from modules.tools.discovery.action_registry import get_action_registry

    assert get_action_registry().get(name) is not None, (
        f"{name} is pinned for onboarding but not registered — the section "
        "instructs Auto to call it by name"
    )


# --------------------------------------------------------------------------- #
# The fold — active onboarding unions the spine's actions into the enum
# --------------------------------------------------------------------------- #


def test_active_onboarding_folds_spine_actions_into_enum():
    allowed, reason, _ = _apply_onboarding_prior(
        _NARROWED, _FakeSession(_WS("questions")), WS_ID,
        is_admin=False, is_super_admin=False,
    )
    assert "platform_list_agents" in allowed  # ranked set survives, order-stable
    for name in ONBOARDING_PRIOR_ACTIONS:
        assert name in allowed
    assert reason == "onboarding_prior"


def test_plain_user_gets_the_spine_actions():
    # The gate filter must clear every spine action for a non-admin principal —
    # onboarding is a plain-user flow.
    allowed, _, _ = _apply_onboarding_prior(
        _NARROWED, _FakeSession(_WS("building")), WS_ID,
        is_admin=False, is_super_admin=False,
    )
    assert set(ONBOARDING_PRIOR_ACTIONS) <= set(allowed)


# --------------------------------------------------------------------------- #
# No-op set — terminal, full-enum, missing, and error paths stay untouched
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", ["completed", "skipped"])
def test_terminal_stage_leaves_narrowing_untouched(stage):
    out = _apply_onboarding_prior(
        _NARROWED, _FakeSession(_WS(stage)), WS_ID,
        is_admin=False, is_super_admin=False,
    )
    assert out == _NARROWED


def test_full_enum_needs_no_prior():
    # allowed is None → every action already exposed; nothing to fold.
    full = (None, "flag SEMANTIC_TOOL_ROUTING=False", False)
    out = _apply_onboarding_prior(
        full, _FakeSession(_WS("questions")), WS_ID,
        is_admin=False, is_super_admin=False,
    )
    assert out == full


def test_missing_workspace_or_session_untouched():
    assert _apply_onboarding_prior(
        _NARROWED, _FakeSession(None), WS_ID, is_admin=False, is_super_admin=False
    ) == _NARROWED
    assert _apply_onboarding_prior(
        _NARROWED, None, WS_ID, is_admin=False, is_super_admin=False
    ) == _NARROWED
    assert _apply_onboarding_prior(
        _NARROWED, _FakeSession(_WS("questions")), None,
        is_admin=False, is_super_admin=False,
    ) == _NARROWED


def test_load_error_fails_soft_to_unchanged_narrowing():
    out = _apply_onboarding_prior(
        _NARROWED, _BoomSession(), WS_ID, is_admin=False, is_super_admin=False
    )
    assert out == _NARROWED
