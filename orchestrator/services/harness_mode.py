"""HARNESS's per-workspace mode (PRD-263 US-016, FR-9).

``orchestrator.harness.mode`` in a workspace's settings is the only per-workspace
HARNESS switch:

* ``report_only`` (the default): the weekly run writes its audit report and changes
  nothing;
* ``propose``: an owner's opt-in. While the installation's
  HARNESS_SELF_MANAGEMENT_ENABLED is on, each change HARNESS proposes comes to the
  owner as an Approvals card (services.harness_grants, US-054); off, it runs like
  ``report_only``;
* ``disabled``: HARNESS does not run for the workspace.

The Settings API accepts only HARNESS_MODES, and every reader goes through
harness_mode(). It is also the one read-time mapping of values stored before
PRD-263, recorded on the US-001 maintainer issue (§11-Q28): ``full_auto`` and
``manual`` read as ``report_only`` (``propose`` is an opt-in made under its new
meaning), and the retired switch's ``disabled: true`` or ``enabled: false`` reads as
``disabled``. A mode chosen from HARNESS_MODES is newer than either retired key and
wins over them. No stored value is rewritten.

It also holds the HARNESS block of the Settings API
(``/api/workspaces/current/orchestrator``): its view, the write's validation and
the write's merge, kept here rather than in api/workspaces.py, which is over the
800-line limit. Weekly on HARNESS_SWEEP_CRON is the one schedule (§11-Q12), and the
retired keys are neither validated, written nor returned.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

REPORT_ONLY = "report_only"
PROPOSE = "propose"
DISABLED = "disabled"
HARNESS_MODES = (REPORT_ONLY, PROPOSE, DISABLED)
DEFAULT_HARNESS_MODE = REPORT_ONLY
HARNESS_SCHEDULE = "weekly"
HARNESS_SCHEDULES = (HARNESS_SCHEDULE,)
RETIRED_HARNESS_KEYS = ("enabled", "disabled")


def harness_mode(workspace_settings: Optional[Mapping[str, Any]]) -> str:
    """The workspace's HARNESS mode, one of HARNESS_MODES, from its stored settings."""
    harness = _stored_harness(workspace_settings)
    mode = harness.get("mode")
    if mode in HARNESS_MODES:
        return mode
    if harness.get("disabled") is True or harness.get("enabled") is False:
        return DISABLED
    return DEFAULT_HARNESS_MODE


def harness_settings_view(workspace_settings: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The HARNESS block the Settings API returns: the one schedule and the mode read
    through harness_mode(), never a retired key."""
    return {"schedule": HARNESS_SCHEDULE, "mode": harness_mode(workspace_settings)}


def validate_harness_update(harness: Any) -> None:
    """Raise ValueError naming the field when a HARNESS write carries a schedule or a
    mode outside its vocabulary. Only the write is checked, so a stored legacy value
    never blocks a save; the retired keys are not checked, and a missing or non-object
    block is not a HARNESS write."""
    if not (harness and isinstance(harness, Mapping)):
        return
    if "schedule" in harness and harness["schedule"] not in HARNESS_SCHEDULES:
        raise ValueError(f"harness.schedule must be one of {list(HARNESS_SCHEDULES)}")
    if "mode" in harness and harness["mode"] not in HARNESS_MODES:
        raise ValueError(f"harness.mode must be one of {list(HARNESS_MODES)}")


def merge_harness_update(stored: Mapping[str, Any], harness: Mapping[str, Any]) -> Dict[str, Any]:
    """The stored HARNESS block with the write's keys over it, key by key, minus the
    write's retired keys. A stored retired key is kept: nothing stored is rewritten."""
    written = {key: value for key, value in harness.items() if key not in RETIRED_HARNESS_KEYS}
    return {**stored, **written}


def _stored_harness(workspace_settings: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """``orchestrator.harness`` from the settings, or {} when any level is missing or not a mapping."""
    orchestrator = (workspace_settings or {}).get("orchestrator")
    harness = orchestrator.get("harness") if isinstance(orchestrator, Mapping) else None
    return harness if isinstance(harness, Mapping) else {}
