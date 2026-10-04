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
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

WAIT_FOR_ME, STEPS, TAGS, STAFFING, CONFIG, GOAL = "wait_for_me", "steps", "tags", "staffing", "config", "goal"
CHECK_EACH_STEP, CARD_TAGS = "check_each_step", "card_tags"
MAX_CARD_TAGS, MAX_TAG_CHARS = 10, 60
# What a step object says it is, and what it asks for, in the keys models use.
STEP_NAME_KEYS = ("name", "title", "step")
STEP_WORK_KEYS = ("objective", "description", "does", "task", "output")
STEP_WAITS_KEYS = ("pause_before_start", "pause_after", WAIT_FOR_ME, "wait", "requires_approval")
STEPS_HEADING = "The owner's steps, in order:"
STAFFING_NOT_USED = ("staffing named no agent's work, so it pinned nobody: each step goes to the agent that fits "
                     "it. To pin one, give each agent its work, in the owner's words: "
                     "[{\"agent\": \"Analyst\", \"does\": \"works out the coffee\"}].")


def asks_as_the_owner_says(handler: Handler) -> Handler:
    """Read wait_for_me, steps, tags and loose staffing as the mission's settings (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        asked, note = as_the_mission_settings(params or {})
        out = await handler(db, workspace_id, asked)
        if note and isinstance(out, dict) and out.get("success"):
            return {**out, "staffing_note": note}
        return out
    return wrapped


def as_the_mission_settings(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """New params with the owner's words in the mission's own settings, and the note
    for staffing that pinned nobody (None when there was none)."""
    steps, step_waits = _steps_text(params.get(STEPS))
    config = dict(params.get(CONFIG) or {}) if isinstance(params.get(CONFIG), dict) else {}
    if params.get(WAIT_FOR_ME) is True or step_waits:
        config[CHECK_EACH_STEP] = True
    tags = _tags(params.get(TAGS))
    if tags:
        config[CARD_TAGS] = tags
    asked = {k: v for k, v in params.items() if k not in (WAIT_FOR_ME, STEPS, TAGS, CONFIG)}
    if steps:
        asked[GOAL] = f"{str(params.get(GOAL) or '').strip()}\n\n{steps}".strip()
    if config:
        asked[CONFIG] = config
    staffing = params.get(STAFFING)
    if staffing and not _names_the_work(staffing):
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


def _names_the_work(staffing: Any) -> bool:
    """Staffing the coordinator can pin: a list of {agent, does}."""
    return isinstance(staffing, list) and all(isinstance(e, dict) and e.get("agent") and e.get("does")
                                              for e in staffing)


__all__ = ["CARD_TAGS", "STAFFING_NOT_USED", "as_the_mission_settings", "asks_as_the_owner_says"]
