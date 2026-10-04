"""What a playbook's ``schedule_config`` may be: its types, and what is wrong with one.

F271 (night 7b, B14): the owner saved a playbook's schedule as ``{type: 'none'}`` to
turn its timer off. Nothing changed, and the timer fired again. 'none' is not a
schedule type, so the save was refused, but the refusal said only "type must be
'manual', 'cron', or 'trigger'", never how to turn a timer off, and a refusal the
owner does not read looks like a save. A type the scheduler does not know is now
refused (422, ``api/workflow_recipes.py``) with the two ways that do turn a timer
off: ``enabled: false``, which keeps its time, and the type ``manual``.

Pure: the model's own check (``WorkflowTemplate.validate_schedule_config``) and the
playbook routes read the same rules.
"""
from __future__ import annotations

from typing import Any, Optional

# The types a playbook's schedule can have. The playbook editor offers the same
# three (frontend/components/workflows/playbook-schedule-config.tsx).
SCHEDULE_TYPES = ("manual", "cron", "trigger")
# How much of an unknown type the refusal repeats back.
SHOWN_TYPE_CHARS = 40
NO_SUCH_TYPE = (
    "A schedule's type is manual, cron or trigger; there is no '{type}', so nothing was saved. "
    "To turn a timer off, save its schedule with \"enabled\": false (it keeps its time for when "
    "you turn it back on), or with \"type\": \"manual\" to run the playbook only when you start it."
)


def schedule_problem(schedule_config: Any) -> Optional[str]:
    """What is wrong with the shape of ``schedule_config``, or None when nothing is.
    An empty one is fine: the playbook has no schedule."""
    if not schedule_config:
        return None
    if not isinstance(schedule_config, dict):
        return "schedule_config must be an object"
    if "type" not in schedule_config:
        return "schedule_config must have 'type' field"
    schedule_type = schedule_config["type"]
    if schedule_type not in SCHEDULE_TYPES:
        return "type must be 'manual', 'cron', or 'trigger'"
    if schedule_type == "cron" and "cron_expression" not in schedule_config:
        return "cron type requires cron_expression field"
    if schedule_type == "trigger" and "trigger_config" not in schedule_config:
        return "trigger type requires trigger_config field"
    return None


def unknown_type_refusal(schedule_config: Any) -> Optional[str]:
    """The refusal for a schedule whose type is none of ``SCHEDULE_TYPES``, saying how
    to turn a timer off. None for every other schedule: ``schedule_problem`` judges it."""
    if not isinstance(schedule_config, dict) or "type" not in schedule_config:
        return None
    if schedule_config["type"] in SCHEDULE_TYPES:
        return None
    return NO_SUCH_TYPE.format(type=str(schedule_config["type"])[:SHOWN_TYPE_CHARS])
