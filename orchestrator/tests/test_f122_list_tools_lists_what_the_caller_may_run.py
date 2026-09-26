"""F122 — platform_list_tools lists what the caller may run.

The listing read registry.get_all() with no role check, so any operator's
platform_list_tools returned every super_admin_only action with its
description: the observability tier, system health and the autonomy dial.
That broke PRD-143's fail-closed discovery rule on the one listing tool. The
handler cannot see the caller. The executor can, because it runs the
super-admin and admin gates on caller_context. For this action it now injects
both answers into params from the same two predicates the gates read
(strip-then-inject, like _created_by and _turn_id). The handler lists a gated
action only when told the caller passes its gate; nothing injected means
neither.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from modules.tools.discovery import get_action_registry
from modules.tools.discovery.action_registry import ActionDefinition, action_is_available
from modules.tools.discovery.platform_executor import PlatformActionExecutor

SU_PROBE, ADMIN_PROBE = "platform_f122_su_probe", "platform_f122_admin_probe"
# F145: a caller is an admin by who the call is made for (core.security.driving_user):
# user 7 is this workspace's owner, user 8 a member.
OPERATOR = {"system_role": "user", "driving_user_id": "8"}
OWNER = {"system_role": "user", "driving_user_id": "7"}
SUPER_ADMIN = {"system_role": "super_admin", "driving_user_id": "7"}


@pytest.fixture(autouse=True)
def members(monkeypatch):
    """The membership read, stubbed: user 7 holds the owner row."""
    import core.security.driving_user as driving_user

    monkeypatch.setattr(driving_user, "driver_is_workspace_admin", lambda db, ws, ctx: bool(ctx) and (
        ctx.get("system_role") == "super_admin" or ctx.get("driving_user_id") == "7"))


@pytest.fixture
def registry(monkeypatch):
    """The live registry plus one su-only and one admin-only probe. The
    admin_only tier is empty today (PRD-143 Rev 2), so the admin half needs a
    probe to mean anything."""
    live = get_action_registry()
    live.get_all()                                 # registered before the probes go in
    for name, flag in ((SU_PROBE, "super_admin_only"), (ADMIN_PROBE, "admin_only")):
        monkeypatch.setitem(live._actions, name, ActionDefinition(
            name=name, description="F122 probe", category="t", permission_level="read",
            parameters={"type": "object", "properties": {}, "required": []}, **{flag: True}))
    return live


def _executor(*, full_autonomy=False, inherits_admin=False):
    ex = PlatformActionExecutor(MagicMock(), uuid4())
    for probe in (SU_PROBE, ADMIN_PROBE):
        ex._handlers[probe] = AsyncMock(return_value={"success": True})
    return ex, (patch.object(PlatformActionExecutor, "_full_autonomy", return_value=full_autonomy),
                patch.object(PlatformActionExecutor, "_agent_inherits_admin", return_value=inherits_admin))


def _listed(ex, caller_context, **params):
    result = asyncio.run(ex.execute("platform_list_tools", {"category": "platform", **params}, caller_context))
    assert result["success"] is True, result
    return {t["name"] for t in result["tools"]}


def _runs(ex, action, caller_context):
    return asyncio.run(ex.execute(action, {}, caller_context)).get("permission_denied") is not True


def test_an_operator_listing_has_no_gated_action(registry):
    su_actions = {a.name for a in registry.get_all() if a.super_admin_only and action_is_available(a)}
    assert su_actions - {SU_PROBE}                 # the real observability tier, not only the probe
    ex, (autonomy, inherits) = _executor()
    with autonomy, inherits:
        listed = _listed(ex, OPERATOR)
    assert "platform_list_tools" in listed and not listed & (su_actions | {ADMIN_PROBE})


def test_a_workspace_owner_listing_has_the_admin_actions_but_not_su(registry):
    ex, (autonomy, inherits) = _executor()
    with autonomy, inherits:
        listed = _listed(ex, OWNER)
    assert ADMIN_PROBE in listed and SU_PROBE not in listed


def test_a_super_admin_listing_has_both(registry):
    ex, (autonomy, inherits) = _executor()
    with autonomy, inherits:
        listed = _listed(ex, SUPER_ADMIN)
    assert {SU_PROBE, ADMIN_PROBE, "platform_get_system_health"} <= listed


def test_a_role_claimed_in_the_params_is_stripped(registry):
    ex, (autonomy, inherits) = _executor()
    with autonomy, inherits:
        listed = _listed(ex, OPERATOR, _caller_is_super_admin=True, _caller_is_admin=True)
    assert not listed & {SU_PROBE, ADMIN_PROBE}


@pytest.mark.parametrize("inherits_admin", [True, False])
def test_headless_lists_no_su_and_admin_as_the_gate_decides(registry, inherits_admin):
    ex, (autonomy, inherits) = _executor(inherits_admin=inherits_admin)
    with autonomy, inherits:
        listed = _listed(ex, None)
    assert SU_PROBE not in listed and (ADMIN_PROBE in listed) is inherits_admin


def test_the_handler_alone_lists_neither(registry):
    """Called without the executor, nothing is injected: fail-closed, the catalog's default."""
    from modules.tools.discovery.handlers_tools_llms import list_tools

    result = asyncio.run(list_tools(None, uuid4(), {"category": "platform"}))
    assert not {t["name"] for t in result["tools"]} & {SU_PROBE, ADMIN_PROBE}


@pytest.mark.parametrize("full_autonomy", [False, True])
@pytest.mark.parametrize("caller", [OPERATOR, OWNER, SUPER_ADMIN, {"system_role": "admin"},
                                    {"system_role": "super_admin"}, None])
def test_the_listing_is_what_the_caller_can_run(registry, caller, full_autonomy):
    """Each gated probe is listed exactly when the gate lets this caller run it."""
    ex, (autonomy, inherits) = _executor(full_autonomy=full_autonomy)
    with autonomy, inherits:
        listed = _listed(ex, caller)
        for probe in (SU_PROBE, ADMIN_PROBE):
            assert (probe in listed) is _runs(ex, probe, caller), (probe, caller, full_autonomy)
