"""``platform_create_mission`` as the owner asks for one (F262, night 7b).

Asked to start a mission that "stops after every step for my approval", Auto failed
three times on the tool's own arguments and never made it; the owner made #0188 on the
Missions page. Its first call carried what the owner said, in the words they said it:
``wait_for_me``, ``tags`` and the three things as the goal, with its own idea of
staffing ({"coordinator": "Auto", "agents": [...]}). Its second carried the three things
as ``steps``. This reads those as the mission's own settings before the tool runs:

- ``wait_for_me`` is ``config.check_each_step``: every step waits for the owner's check;
- ``steps`` (text, or objects with a name and what to do) join the goal in order, and a
  step that says to pause first makes the mission wait for the owner;
- ``tags`` go on the mission's card (``config.card_tags``), so the owner finds it by tag;
- staffing that names no agent's work (a list of names, or an object) pins nobody: the
  plan staffs each step by what it needs, and the answer says so.

Night 8 (F282, F287, F288, F289): the check of each step is one setting
(``owner_checks.with_step_checks``) whichever key the call used, and on whenever the
owner's own words ask for it; the agents the owner's words give work to are pinned to
it (``mission_owner_words``); a playbook the owner named, a copy of a mission still
waiting, and a goal that isn't theirs make no mission (``mission_create_checks``). The
answer says whether the steps wait for the owner, so Auto can't say they do when they
don't (#0250: "Yes, it will", with no setting).
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from modules.coordination.owner_checks import checks_each_step, with_step_checks
from modules.tools.discovery.mission_owner_words import staffing_names_the_work

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

WAIT_FOR_ME, STEPS, TAGS, STAFFING, CONFIG, GOAL = "wait_for_me", "steps", "tags", "staffing", "config", "goal"
CARD_TAGS = "card_tags"
MAX_CARD_TAGS, MAX_TAG_CHARS = 10, 60
# What a step object says it is, and what it asks for, in the keys models use.
STEP_NAME_KEYS = ("name", "title", "step")
STEP_WORK_KEYS = ("objective", "description", "does", "task", "output", "prompt_template", "prompt", "instructions")
STEP_AGENT_KEYS = ("agent", "agent_name", "agent_id", "assigned_agent", "assigned_agent_name")
STEP_WAITS_KEYS = ("pause_before_start", "pause_after", WAIT_FOR_ME, "wait", "requires_approval")
STEPS_HEADING = "The owner's steps, in order:"
STAFFING_NOT_USED = ("staffing named no agent's work, so it pinned nobody: each step goes to the agent that fits "
                     "it. To pin one, give each agent its work, in the owner's words: "
                     "[{\"agent\": \"Analyst\", \"does\": \"works out the coffee\"}].")


CHECKS_EACH_STEP = " Every step waits in Review for the owner's check before the next one starts."
RUNS_UNCHECKED = (" Its steps run without waiting for the owner's check; if the owner asked to check each step, "
                  "switch it on with platform_update_mission_plan and check_each_step: true.")


def asks_as_the_owner_says(handler: Handler) -> Handler:
    """Read the call, and the owner's own words, as the mission's settings (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from modules.tools.discovery.mission_create_checks import refusal_for_mission
        from modules.tools.discovery.mission_owner_words import asks_for_checks, chosen_staffing, owners_words

        params = params or {}
        said = owners_words(db, workspace_id, params)
        refusal = refusal_for_mission(db, workspace_id, params, said)
        if refusal:
            return {"success": False, "error": refusal}
        asked, note = as_the_mission_settings(params)
        staffing = chosen_staffing(db, workspace_id, params, said)
        if staffing:
            asked, note = {**asked, STAFFING: staffing}, None
        if asks_for_checks(said):
            asked = {**asked, CONFIG: with_step_checks(asked.get(CONFIG), on=True)}
        return _answered(await handler(db, workspace_id, asked), asked, note)
    return wrapped


def _answered(out: Any, asked: Dict[str, Any], note: Optional[str]) -> Any:
    """The tool's answer, saying whether the steps wait for the owner and who was pinned."""
    if not (isinstance(out, dict) and out.get("success")):
        return out
    checks = checks_each_step(asked.get(CONFIG))
    said = {"checks_each_step": checks,
            "message": f"{out.get('message', '')}{CHECKS_EACH_STEP if checks else RUNS_UNCHECKED}"}
    if asked.get(STAFFING):
        said["staffed_by_the_owner"] = [entry["agent"] for entry in asked[STAFFING]]
    return {**out, **said, **({"staffing_note": note} if note else {})}


def as_the_mission_settings(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """New params with the owner's words in the mission's own settings, and the note
    for staffing that pinned nobody (None when there was none)."""
    steps, step_waits = _steps_text(params.get(STEPS))
    config = dict(params.get(CONFIG) or {}) if isinstance(params.get(CONFIG), dict) else {}
    config = with_step_checks(config, on=True if (params.get(WAIT_FOR_ME) is True or step_waits) else None)
    tags = _tags(params.get(TAGS))
    if tags:
        config[CARD_TAGS] = tags
    asked = {k: v for k, v in params.items() if k not in (WAIT_FOR_ME, STEPS, TAGS, CONFIG)}
    if steps:
        asked[GOAL] = f"{str(params.get(GOAL) or '').strip()}\n\n{steps}".strip()
    if config:
        asked[CONFIG] = config
    staffing = params.get(STAFFING)
    if staffing and not staffing_names_the_work(staffing):
        return {k: v for k, v in asked.items() if k != STAFFING}, STAFFING_NOT_USED
    return asked, None


def _steps_text(steps: Any) -> Tuple[str, bool]:
    """The owner's steps as numbered lines, and whether one of them says to wait first."""
    if not isinstance(steps, list) or not steps:
        return "", False
    lines: List[str] = []
    waits = False
    for n, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            waits = waits or any(step.get(key) is True for key in STEP_WAITS_KEYS)
            name = next((str(step[k]).strip() for k in STEP_NAME_KEYS if step.get(k)), "")
            work = next((str(step[k]).strip() for k in STEP_WORK_KEYS if step.get(k)), "")
            text = f"{name}: {work}" if name and work else (name or work)
        else:
            text = str(step).strip()
        if text:
            lines.append(f"{n}. {text}")
    return (STEPS_HEADING + "\n" + "\n".join(lines)) if lines else "", waits


def _tags(tags: Any) -> List[str]:
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []
    kept = [str(t).strip()[:MAX_TAG_CHARS] for t in tags if str(t).strip()]
    return list(dict.fromkeys(kept))[:MAX_CARD_TAGS]


__all__ = ["CARD_TAGS", "STAFFING_NOT_USED", "as_the_mission_settings", "asks_as_the_owner_says"]
