"""HARNESS's per-workspace mode (PRD-263 US-016, FR-9).

``orchestrator.harness.mode`` in a workspace's settings is the only per-workspace
HARNESS switch:

* ``report_only`` (the default): the weekly run writes its audit report and changes
  nothing;
* ``propose``: an owner's opt-in that behaves exactly like ``report_only`` until
  proposals exist (US-054);
* ``disabled``: HARNESS does not run for the workspace.

The Settings API accepts only HARNESS_MODES, and every reader goes through
harness_mode(). It is also the one read-time mapping of values stored before
PRD-263, recorded on the US-001 maintainer issue (§11-Q28): ``full_auto`` and
``manual`` read as ``report_only`` (``propose`` is an opt-in made under its new
meaning), and the retired switch's ``disabled: true`` or ``enabled: false`` reads as
``disabled``. A mode chosen from HARNESS_MODES is newer than either retired key and
wins over them. No stored value is rewritten.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

REPORT_ONLY = "report_only"
PROPOSE = "propose"
DISABLED = "disabled"
HARNESS_MODES = (REPORT_ONLY, PROPOSE, DISABLED)
DEFAULT_HARNESS_MODE = REPORT_ONLY


def harness_mode(workspace_settings: Optional[Mapping[str, Any]]) -> str:
    """The workspace's HARNESS mode, one of HARNESS_MODES, from its stored settings."""
    harness = _stored_harness(workspace_settings)
    mode = harness.get("mode")
    if mode in HARNESS_MODES:
        return mode
    if harness.get("disabled") is True or harness.get("enabled") is False:
        return DISABLED
    return DEFAULT_HARNESS_MODE


def _stored_harness(workspace_settings: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """``orchestrator.harness`` from the settings, or {} when any level is missing or not a mapping."""
    orchestrator = (workspace_settings or {}).get("orchestrator")
    harness = orchestrator.get("harness") if isinstance(orchestrator, Mapping) else None
    return harness if isinstance(harness, Mapping) else {}
