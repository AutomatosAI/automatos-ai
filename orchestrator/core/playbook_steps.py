"""F292 (night 8): a playbook's steps say their order on every read, and a write may leave it out.

The steps Auto adds (``platform_add_playbook_step``) carry ``step_number`` but no
``order``: playbooks 102, 103, 104 and 109 in the night's workspace. So the
playbook's own read gave steps without the field its edit requires, and sending
them back (``PUT /api/workflow-recipes/custom-81546f72``) answered 400 "Invalid
steps: Step 0 missing required field: order". The owner had to guess the field.

A step's order is its place in the playbook. Every read now gives it: the step's
own ``order``, else its ``step_number``, else its place in the list. A write
fills it the same way, and never stores the agent summary the read adds to each
step (``agent``, made from ``agent_id`` each time it is read). Playbook 101
carried one after an edit sent the read back.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Added to each step by the read (api.workflow_recipes._enrich_steps_with_agents).
READ_ONLY_STEP_KEYS = frozenset({"agent"})


def _whole_number(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def step_order(step: Dict[str, Any], position: int) -> int:
    """``step``'s order: its own, else its step number, else its place (from 1)."""
    return _whole_number(step.get("order")) or _whole_number(step.get("step_number")) or position + 1


def steps_in_order(steps: Any) -> Any:
    """``steps`` as a read gives them, each step with its ``order`` (new dicts).
    Anything that is not a list of steps is given back as it is."""
    if not isinstance(steps, list):
        return steps
    return [{**step, "order": step_order(step, n)} if isinstance(step, dict) else step
            for n, step in enumerate(steps)]


def steps_as_written(steps: Any) -> Any:
    """``steps`` as a write stores them: each with its ``order``, none with the
    keys only a read adds. A step that is not an object is left for the
    validator to refuse."""
    ordered = steps_in_order(steps)
    if not isinstance(ordered, list):
        return ordered
    return [{k: v for k, v in step.items() if k not in READ_ONLY_STEP_KEYS} if isinstance(step, dict) else step
            for step in ordered]
