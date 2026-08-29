"""A deliberate STOP must reach Auto as its reason, never as "Unknown error".

FOUND IN PRODUCTION (2026-08-29, persona harness + Railway logs). Auto called
``platform_install_package`` and the tool came back:

    Tool platform_install_package failed: Unknown error

The server had returned a perfectly good reason — the D6 one-package-during-
onboarding copy — but under the ``message`` key, while the router reads only
``error``. So the reason was discarded, and Auto, handed nothing, INVENTED a
cause for the user: "I can't proceed without the Shopify connection being
active first." It had no evidence for that.

Two PRD-230 stops were affected, and the second one matters most:
  * ``onboarding_restricted`` — D6, one package during onboarding
  * ``over_quota``            — D9, the honest plan conversation, whose stated
    contract is "NEVER a silent block". Losing its message made it exactly the
    silent block D9 forbids.

PRD-143 S15 had already fixed this class for ``requires_confirmation``; this
generalises that precedent to the marker set instead of one hard-coded key.
"""
from __future__ import annotations

import pytest

from modules.tools.tool_router import STOP_MARKERS, select_failure_message


# --------------------------------------------------------------------------- #
# The regression: each STOP surfaces its own copy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("marker", STOP_MARKERS)
def test_every_stop_marker_surfaces_its_message(marker):
    result = {"success": False, marker: True, "message": "the honest reason"}
    assert select_failure_message(result) == "the honest reason"


def test_d6_one_package_copy_reaches_auto():
    # The exact prod shape from handlers_packages.install_package_tool.
    result = {
        "success": False,
        "onboarding_restricted": True,
        "message": "You can install one package during onboarding.",
    }
    out = select_failure_message(result)
    assert out == "You can install one package during onboarding."
    assert out != "Unknown error"


def test_d9_over_quota_is_never_a_silent_block():
    # D9's contract: over-quota is an honest plan conversation, never a silent
    # block. If this returns "Unknown error", D9 is violated at the seam.
    result = {
        "success": False,
        "over_quota": True,
        "message": "That package needs 6 agents; your plan allows 5. Upgrade?",
    }
    assert "Upgrade?" in select_failure_message(result)


# --------------------------------------------------------------------------- #
# The gate: no arbitrary handler text leaks
# --------------------------------------------------------------------------- #


def test_message_without_a_stop_marker_is_not_surfaced():
    # An ordinary failure carrying an incidental 'message' must NOT have it
    # promoted — the marker gate is what keeps internals out of the LLM.
    result = {"success": False, "message": "internal detail nobody vetted"}
    assert select_failure_message(result) == "Unknown error"


def test_error_key_always_wins():
    result = {
        "success": False,
        "error": "Package not found: nope",
        "over_quota": True,
        "message": "should not be chosen",
    }
    assert select_failure_message(result) == "Package not found: nope"


# --------------------------------------------------------------------------- #
# Degenerate shapes never raise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("junk", [None, "a string", 42, []])
def test_non_dict_results_degrade(junk):
    assert select_failure_message(junk) == "Unknown error"


def test_stop_marker_with_no_message_degrades():
    assert select_failure_message({"over_quota": True}) == "Unknown error"


def test_handler_shapes_still_carry_their_markers():
    """The router's markers must match what the package handlers actually emit —
    a renamed key would silently reopen the bug."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "modules" / "tools" / "discovery" / "handlers_packages.py"
    ).read_text()
    assert '"onboarding_restricted": True' in src
    assert '"over_quota": True' in src
