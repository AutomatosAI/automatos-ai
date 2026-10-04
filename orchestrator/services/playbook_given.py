"""F288 (night 8, part): a playbook run's details reach its steps.

Auto ran New Cafe Onboarding (playbook 102) for Larder & Loaf with input_data
{cafe_name, owner_name, owner_email, first_order, delivery_day}. A run's details reach a
step only through a placeholder in its prompt ({input.<name>}, {{name}} or a bare
{input}: api.recipe_executor.fill_step_placeholders), and no step of playbook 102 has
one. Step 1 asks "Please provide the following details for the new cafe: cafe_name,
contact_person, contact_email, usual_harbour_blend_kg…" and step 2 fills
{{cafe_details.…}} from step 1's answer. So the run asked the owner for what Auto had
passed, then wrote the template's Thursday and "the crew" (#0366; #0440 asked again as
#1411; #0284 wrote "0 kg" and #0303 wrote about another coffee).

When a run carries details that no step's prompt names, every agent step's prompt now
ends with them, under "What the owner gave for this run". A run that a trigger or a
webhook started is left as it was: its details are not the owner's words, and the run
already shows a trigger's content to its steps ("Trigger Context").
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from core.services.playbook_inputs import INPUT_KEY, derived_inputs

GIVEN_HEADING = "## What the owner gave for this run"
GIVEN_ASK = ("Use these wherever this step needs them. Where the step's own wording says otherwise (an example, "
             "a default day, a name or an amount), these win. Never ask the owner for any of them.")
# A bare {input}: substitute_playbook_input gives the step every detail of the run.
BARE_INPUT = "{" + INPUT_KEY + "}"
# F130's {{name}} blank, which api.recipe_executor fills from the run's detail of that name.
_NAMED_BLANK = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
# A trigger's payload (api/composio.py, api/webhooks.py); the run shows its content itself.
TRIGGER_KEYS = ("content", "metadata")
# RecipeExecution.triggered_by of a run a trigger or a webhook started (api/composio.py,
# api/webhooks.py, api/workflow_recipes.py): its details are not the owner's words.
TRIGGER_STARTS = ("composio_trigger", "workspace_webhook", "webhook")
# Each detail is cut to this many characters.
DETAIL_CHARS = 500
STEP_OVERRIDES_KEY = "step_overrides"   # PRD-204 S7: a rerun's own wording for some steps


def started_by_a_trigger(run: Any, input_data: Dict[str, Any]) -> bool:
    """A run a trigger or a webhook started: its details are a payload, not the owner's."""
    return getattr(run, "triggered_by", None) in TRIGGER_STARTS or "content" in input_data


def _absent(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value in ([], {})


def unnamed_details(input_data: Dict[str, Any], prompts: Iterable[str]) -> Dict[str, Any]:
    """The run's details that no step's prompt names; none when a step reads them all
    through a bare {input}."""
    texts = [str(text or "") for text in prompts]
    if any(BARE_INPUT in text for text in texts):
        return {}
    named = set(derived_inputs([{"prompt_template": text} for text in texts]))
    named |= {name for text in texts for name in _NAMED_BLANK.findall(text)}
    return {key: value for key, value in input_data.items()
            if key not in named and key not in TRIGGER_KEYS and not _absent(value)}


def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= DETAIL_CHARS else text[:DETAIL_CHARS - 1] + "…"


def given_block(input_data: Dict[str, Any], prompts: Iterable[str]) -> Optional[str]:
    """What every agent step of the run is given: the details no step's prompt names,
    one ``name: value`` line each; None when there are none."""
    details = unnamed_details(input_data, prompts)
    if not details:
        return None
    return "\n".join([GIVEN_HEADING, GIVEN_ASK, *(f"{key}: {_shown(value)}" for key, value in details.items())])


def _rerun_prompts(run: Any) -> List[str]:
    """The wording a rerun gave some of its steps (PRD-204 S7)."""
    metadata = run.execution_metadata if isinstance(run.execution_metadata, dict) else {}
    overrides = metadata.get(STEP_OVERRIDES_KEY)
    if not isinstance(overrides, dict):
        return []
    return [str(o["prompt_template"]) for o in overrides.values() if isinstance(o, dict) and o.get("prompt_template")]


def run_prompts(db: Any, run: Any) -> List[str]:
    """The prompts of a run's steps: its playbook's, in the run's workspace, and the
    wording a rerun gave some of them."""
    from core.models.core import WorkflowTemplate

    playbook = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == run.recipe_id,
                                                 WorkflowTemplate.workspace_id == run.workspace_id).first()
    steps = [step for step in getattr(playbook, "steps", None) or [] if isinstance(step, dict)]
    return [str(step.get("prompt_template") or "") for step in steps] + _rerun_prompts(run)


def given_for_run(db: Any, run: Any, input_data: Any) -> Optional[str]:
    """What the owner gave for ``run`` that no step's prompt names; None without a run or
    details, and for a run that a trigger or a webhook started."""
    if run is None or not isinstance(input_data, dict) or not input_data:
        return None
    if started_by_a_trigger(run, input_data):
        return None
    return given_block(input_data, run_prompts(db, run))


__all__ = ["GIVEN_HEADING", "given_block", "given_for_run", "unnamed_details"]
