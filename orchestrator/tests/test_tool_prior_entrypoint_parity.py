"""Both tool-surface entry points must apply the same priors.

LIVE FINDING (2026-08-29). ``get_tools_for_agent_async`` applied the PRD-221
page prior AND the PRD-222/#647 onboarding prior after semantic narrowing.
Its SYNC twin, ``get_tools_for_agent``, applied NEITHER — so any caller routed
through the sync entry received an unpinned surface: onboarding's own tools
absent from the dispatcher enum while the OnboardingSection was still
instructing Auto to call them by name.

That is the same shape as the stripped dispatcher (#654): one composition,
two entry points, only one hardened. A capability that depends on which
function a caller happens to reach is a capability that fails intermittently.
"""
from __future__ import annotations

import inspect

from modules.tools import tool_router


def _src(fn) -> str:
    return inspect.getsource(fn)


def test_sync_entry_applies_both_priors():
    src = _src(tool_router.get_tools_for_agent)
    assert "_apply_onboarding_prior" in src, (
        "sync entry does not apply the onboarding prior — onboarding tools "
        "would be missing from the dispatcher enum for its callers"
    )
    assert "_apply_page_prior" in src, "sync entry does not apply the page prior"


def test_async_entry_still_applies_both_priors():
    src = _src(tool_router.get_tools_for_agent_async)
    assert "_apply_onboarding_prior" in src
    assert "_apply_page_prior" in src


def test_both_entries_narrow_before_priors():
    """Priors UNION into a narrowed set; applying them before narrowing would
    let the ranking drop them again."""
    for fn in (tool_router.get_tools_for_agent, tool_router.get_tools_for_agent_async):
        src = _src(fn)
        narrow_at = max(src.find("_narrow_dispatcher_actions_sync"),
                        src.find("_narrow_dispatcher_actions_async"))
        prior_at = src.find("_apply_onboarding_prior")
        assert narrow_at != -1 and prior_at != -1, fn.__name__
        assert narrow_at < prior_at, f"{fn.__name__}: prior must run AFTER narrowing"
